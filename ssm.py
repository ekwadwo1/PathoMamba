#!/usr/bin/env python3
"""
pathomamba/ssm.py

Phase 4.2: SDF-modulated selective-scan (Path B, hand-written).

WHY HAND-WRITTEN (not mamba_ssm kernel):
modulation of the discretization step Delta by the SDF (Eq.2), inside the
SSM, as "the sole architectural change." The fused CUDA kernel bakes Delta
into its scan and cannot accept an external SDF term. So we implement the
selective-scan recurrence in PyTorch, giving us control of Delta.

This is a FAITHFUL selective-scan following Mamba [9]:
  Eq.1:  h_t = Abar h_{t-1} + Bbar x_t,   Abar = exp(Delta * A)
  Eq.2:  Delta_p = Softplus(W_img x_p + sigma(W_sdf D_p) + dt_bias)
  A: S4D-real initialized (negative reals -> stable, decaying memory)
  B: input-dependent (selective)

NOTE on dt_bias vs eps: The Eq.2 writes "+ eps" after the
softplus. We instead use a learnable per-channel dt_bias INSIDE the softplus
(standard Mamba dt-init). This makes Delta start small so memory is long
(probe-confirmed: without it, Delta~1.8 -> memory dies in ~12 steps).
dt_bias plays eps's role (a learnable floor). PHASE 9 RECONCILE: update Eq.2
to show the bias inside the softplus, or document this equivalence.

Operates on a 1D sequence [B, L, C]. The 6-way 3D wrapper (4.3) unrolls the
volume into sequences along +-D, +-H, +-W and calls this per direction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFModulatedSSM(nn.Module):
    """
    Selective SSM over a 1D sequence, with Delta modulated by an SDF prior.

    Args:
        d_model : channel dimension C of the sequence
        d_state : SSM state size N (per channel)
    Inputs to forward:
        x   : [B, L, C]  sequence features
        sdf : [B, L, 1]  per-token SDF value (distance to tumor boundary)
    Returns:
        y   : [B, L, C]
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # --- A: S4D-real init. Shape [C, N]. Stored as log; A = -exp(A_log)
        #     guarantees negative real parts (stable, decaying memory). ---
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))    # [C, N]

        # --- B, C projections: input-dependent (selective). ---
        self.W_B = nn.Linear(d_model, d_state, bias=False)
        self.W_C = nn.Linear(d_model, d_state, bias=False)

        # --- Eq.2 Delta modulation: W_img, W_sdf as per-voxel linear projs
        #     (a 1x1x1 conv on a flattened sequence == a Linear over channels). ---
        self.W_img = nn.Linear(d_model, d_model)   # W_img x_p
        self.W_sdf = nn.Linear(1, d_model)         # W_sdf D_p (scalar -> C)

        # --- Mamba dt-init: bias Delta to start SMALL for long-range memory.
        #     Target initial Delta in [dt_min, dt_max] via inverse-softplus. ---
        dt_min, dt_max = 1e-3, 0.1
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)         # [C]

        # zero-init W_img so Delta starts dominated by dt_bias (small), not by
        # random projections of x. Image/SDF dependence is LEARNED from there.
        nn.init.zeros_(self.W_img.weight)
        nn.init.zeros_(self.W_img.bias)

        # Skip/D term (standard Mamba residual path), per-channel.
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x, sdf):
        B, L, C = x.shape
        N = self.d_state

        # --- Eq.2: Delta_p = Softplus(W_img x_p + sigma(W_sdf D_p) + dt_bias) ---
        # dt_bias makes Delta start small (long memory); W_img zero-init so the
        # bias dominates at init. sigma(W_sdf D_p) is the SDF modulation.
        delta = F.softplus(
            self.W_img(x) + torch.sigmoid(self.W_sdf(sdf)) + self.dt_bias
        )                                                    # [B, L, C], > 0

        # --- A (S4D-real, negative), input-dependent B, C ---
        A = -torch.exp(self.A_log)                           # [C, N], < 0
        B_sel = self.W_B(x)                                  # [B, L, N]
        C_sel = self.W_C(x)                                  # [B, L, N]

        # --- Discretize (zero-order hold):
        #     Abar = exp(delta * A),  Bbar = delta * B ---
        dA = torch.exp(delta.unsqueeze(-1) * A)              # [B, L, C, N]
        dB = delta.unsqueeze(-1) * B_sel.unsqueeze(2)        # [B, L, C, N]

        # --- Sequential scan: h_t = Abar h_{t-1} + Bbar x_t ; y_t = <C_t, h_t> ---
        h = torch.zeros(B, C, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)   # [B,C,N]
            y_t = torch.einsum("bcn,bn->bc", h, C_sel[:, t])       # [B,C]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                                 # [B,L,C]

        # --- skip connection (D term) ---
        y = y + x * self.D
        return y

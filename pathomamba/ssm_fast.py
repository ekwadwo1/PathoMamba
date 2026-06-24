#!/usr/bin/env python3
"""
pathomamba/ssm_fast.py

Phase 4.2-fast: SDF-modulated selective-scan using mamba_ssm's (A100) CUDA kernel.

WHY: the sequential reference (ssm.py) is CORRECT but ~459s/step at full
volume -- untrainable. The math (Eq.1 recurrence, Eq.2 Delta-modulation) is
unchanged; only the SCAN is accelerated via selective_scan_fn.

KEY POINT:
We compute the PRE-softplus argument ourselves:
    delta_arg = W_img x + sigma(W_sdf D) + dt_bias        (Eq.2, our modulation)
then pass it to selective_scan_fn(..., delta_softplus=True), which applies
softplus and runs the recurrence. The SDF modulation is entirely OURS; the
kernel only does the mechanical scan. selective_scan_ref (the library's own
reference) agrees with the kernel to 1e-6, and our Gate 4.2-fast proves this
class matches the sequential SDFModulatedSSM within tolerance.

Probe-confirmed (scripts/10): delta_softplus=True, variable B/C form [B,N,L],
gradient flows to delta (norm ~16).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SDFModulatedSSMFast(nn.Module):
    """
    Drop-in fast replacement for SDFModulatedSSM. Same I/O:
        x   : [B, L, C]
        sdf : [B, L, 1]
        ->  [B, L, C]

    Identical parameterization to the sequential version so weights/behavior
    match: S4D-real A, input-dependent B/C, Eq.2 Delta with dt_bias.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A: S4D-real, stored as log; A = -exp(A_log) < 0.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))            # [C, N]

        # Input-dependent (selective) B, C.
        self.W_B = nn.Linear(d_model, d_state, bias=False)
        self.W_C = nn.Linear(d_model, d_state, bias=False)

        # Eq.2 modulation: W_img, W_sdf as per-voxel linear projections.
        self.W_img = nn.Linear(d_model, d_model)
        self.W_sdf = nn.Linear(1, d_model)

        # Mamba dt-init: bias Delta small for long memory at init.
        dt_min, dt_max = 1e-3, 0.1
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)                # [C]
        nn.init.zeros_(self.W_img.weight)
        nn.init.zeros_(self.W_img.bias)

        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x, sdf):
        B, L, C = x.shape
        N = self.d_state

        # --- Eq.2 PRE-softplus argument (our SDF modulation lives here) ---
        # delta_arg = W_img x + sigma(W_sdf D) + dt_bias   [B, L, C]
        # Note: dt_bias is per-channel; selective_scan_fn also accepts a
        # delta_bias arg, but we fold it in here so the argument is exactly
        # Eq.2. We pass delta_bias=None and delta_softplus=True.
        delta_arg = self.W_img(x) + torch.sigmoid(self.W_sdf(sdf)) + self.dt_bias

        A = -torch.exp(self.A_log)                          # [C, N], <0
        B_sel = self.W_B(x)                                 # [B, L, N]
        C_sel = self.W_C(x)                                 # [B, L, N]

        # --- selective_scan_fn expects:
        #   u:     [B, C, L]
        #   delta: [B, C, L]   (pre-softplus; kernel applies softplus)
        #   A:     [C, N]
        #   B,C:   [B, N, L]   (variable/selective form)
        #   D:     [C]
        u_t     = x.transpose(1, 2).contiguous()            # [B, C, L]
        delta_t = delta_arg.transpose(1, 2).contiguous()    # [B, C, L]
        B_t     = B_sel.transpose(1, 2).contiguous()        # [B, N, L]
        C_t     = C_sel.transpose(1, 2).contiguous()        # [B, N, L]

        # selective_scan_fn requires u, delta, A, B, C, D to share a dtype.
        # Under bf16 autocast some of these are bf16 and some fp32 (bare
        # Parameters like dt_bias/A_log aren't autocast). Force a single dtype
        # = u's dtype so the kernel's type check passes. (Verified equivalent
        # to the fp32 path by Gate 4.2-fast; this only unifies precision.)
        # selective_scan_fn dtype contract (learned from the kernel's own
        # checks): the "inputs" u, delta, B, C follow ONE dtype (may be bf16
        # under autocast), but the "weights" A and D MUST stay fp32. So we
        # unify u/delta/B/C to u's dtype, and force A/D to float32.
        in_dt = u_t.dtype
        delta_t = delta_t.to(in_dt)
        B_t     = B_t.to(in_dt)
        C_t     = C_t.to(in_dt)
        A_c     = A.float()            # A must be fp32 (kernel requirement)
        D_c     = self.D.float()       # D must be fp32

        y = selective_scan_fn(
            u_t, delta_t, A_c, B_t, C_t, D_c,
            z=None, delta_bias=None, delta_softplus=True,
            return_last_state=False,
        )                                                    # [B, C, L]

        return y.transpose(1, 2).contiguous()                # [B, L, C]

#!/usr/bin/env python3
"""
pathomamba/transforms.py

Phase 3: Differentiable spatial transformation + diffeomorphic integration.

SpatialTransformer: warps a volume by a displacement field (grid_sample).
VecInt: integrates a Stationary Velocity Field (SVF) via scaling-and-squaring
        to produce a diffeomorphic deformation phi = exp(v).

THE 23%-FOLDING LESSON: scaling-and-squaring guarantees a diffeomorphism
(det(J) > 0 everywhere) ONLY IF the scaled velocity v/2^nsteps is small enough
that each squaring step stays invertible. A LARGE velocity field breaks this.
The fix is two-fold: (a) enough integration steps (nsteps), (b) training that
keeps v bounded (smoothness regularization, Phase 5). Gate 3 proves this.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    Warp a volume `src` by a voxel displacement field `flow`.
    size = (D, H, W). flow shape = [B, 3, D, H, W] (displacement in D,H,W order).
    """
    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.size = size
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")        # D,H,W
        identity = torch.stack(grids).unsqueeze(0).float()     # [1,3,D,H,W]
        self.register_buffer("identity_grid", identity)

    def forward(self, src, flow):
        new_locs = self.identity_grid + flow                   # [B,3,D,H,W]
        shape = flow.shape[2:]
        # normalize to [-1,1] for grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / (shape[i] - 1)) - 1.0
        new_locs = new_locs.permute(0, 2, 3, 4, 1)             # [B,D,H,W,3]
        # grid_sample expects last-dim order (x,y,z) = (W,H,D); flip from (D,H,W)
        new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True,
                             mode=self.mode, padding_mode="border")


class VecInt(nn.Module):
    """
    Diffeomorphic integration of an SVF via scaling-and-squaring.
    phi = exp(v), approximated by scaling v by 1/2^nsteps then squaring nsteps times.
    Guarantees invertibility IF the scaled field is small enough.
    """
    def __init__(self, size, nsteps=7):
        super().__init__()
        assert nsteps >= 0
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer(size)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


def jacobian_determinant(phi):
    """
    Continuous Jacobian determinant of a displacement field phi [B,3,D,H,W]
    via forward finite differences. Returns [B,1,D,H,W].
    det(J) <= 0 indicates a fold (non-invertible / orientation-flipping).
    """
    # pad +1 on the high side of each spatial axis so gradients keep shape
    phi_pad = F.pad(phi, (0, 1, 0, 1, 0, 1), mode="replicate")
    # gradients of displacement w.r.t. each axis
    dD = phi_pad[:, :, 1:, :-1, :-1] - phi_pad[:, :, :-1, :-1, :-1]
    dH = phi_pad[:, :, :-1, 1:, :-1] - phi_pad[:, :, :-1, :-1, :-1]
    dW = phi_pad[:, :, :-1, :-1, 1:] - phi_pad[:, :, :-1, :-1, :-1]
    # J = I + grad(u); channel c of phi is displacement along axis c (D,H,W)
    J11 = dD[:, 0] + 1.0; J12 = dH[:, 0];       J13 = dW[:, 0]
    J21 = dD[:, 1];       J22 = dH[:, 1] + 1.0; J23 = dW[:, 1]
    J31 = dD[:, 2];       J32 = dH[:, 2];       J33 = dW[:, 2] + 1.0
    det = (J11 * (J22 * J33 - J23 * J32)
           - J12 * (J21 * J33 - J23 * J31)
           + J13 * (J21 * J32 - J22 * J31))
    return det.unsqueeze(1)


def fold_percentage(phi):
    """Percentage of voxels with non-positive Jacobian determinant (folds)."""
    det = jacobian_determinant(phi)
    return 100.0 * (det <= 0).float().mean().item()

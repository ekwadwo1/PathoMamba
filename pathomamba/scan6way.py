#!/usr/bin/env python3
"""
pathomamba/scan6way.py

Phase 4.3: Omni-Directional Stiffness-Modulated (OSM) block.

Wraps the verified 1D SDFModulatedSSM (4.2) into 6-way isotropic scanning
over a 3D feature volume, as the paper's Figure 1B:
unroll into 6 sequences (+-D, +-H, +-W), run the SSM per direction, merge.

Resolves the 1D causal bias of SSMs: a single scan direction sees the volume
in one order; 6-way makes biomechanical context omni-directional.

Runs at the DOWNSAMPLED bottleneck (e.g. 20x30x30 for a 160x240x240 input),
where the sequence length (~18k) makes 6 scans tractable. Full-volume 6-way
would be ~9M tokens x6 -- infeasible. This is why the SSM lives at the
bottleneck, per the design decision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathomamba.ssm_fast import SDFModulatedSSMFast


class OSMBlock(nn.Module):
    """
    Omni-Directional Stiffness-Modulated block over a 3D feature volume.

    Inputs:
        x   : [B, C, D, H, W]  bottleneck features
        sdf : [B, 1, D, H, W]  SDF downsampled to bottleneck resolution
    Output:
        [B, C, D, H, W]  (residual added)
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        # One SSM shared across directions? No -- each direction gets its own,
        # so forward/backward and per-axis dynamics can differ. 6 separate SSMs.
        self.ssms = nn.ModuleList([
            SDFModulatedSSMFast(d_model, d_state) for _ in range(6)
        ])
        
        # merge the 6 direction outputs back to d_model
        self.merge = nn.Linear(d_model * 6, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _unroll(x, axis, reverse):
        """
        Flatten [B,C,D,H,W] into a sequence [B, L, C] scanning along `axis`
        (0=D,1=H,2=W). reverse=True scans backward. Returns (seq, restore_fn).
        """
        B, C, D, H, W = x.shape
        dims = [D, H, W]
        # move scan axis to the front of the spatial dims, flatten the rest
        # we permute so the scan axis is the LAST spatial dim, then flatten
        # all spatial into L with scan axis varying slowest is fragile; instead
        # we move scan axis to dim -1 and flatten (others, scan) -> tokens
        perm = [0, 1] + [2 + a for a in range(3) if a != axis] + [2 + axis]
        xp = x.permute(*perm).contiguous()       # [B, C, other1, other2, scan]
        Bc, Cc = xp.shape[0], xp.shape[1]
        o1, o2, sc = xp.shape[2], xp.shape[3], xp.shape[4]
        seq = xp.view(Bc, Cc, o1 * o2, sc)       # [B,C, O, scan]
        # we want to scan ALONG `scan`, treating each (O) as independent? The
        # paper scans the whole volume as one sequence. We flatten O*scan into L
        # with scan varying fastest so the SSM walks along the axis within each
        # spatial line, lines concatenated.
        seq = seq.permute(0, 2, 3, 1).contiguous()   # [B, O, scan, C]
        seq = seq.view(Bc, o1 * o2 * sc, Cc)         # [B, L, C]
        if reverse:
            seq = torch.flip(seq, dims=[1])
        meta = (perm, (B, C, D, H, W), (o1, o2, sc), reverse)
        return seq, meta

    @staticmethod
    def _reroll(seq, meta):
        """Inverse of _unroll: [B, L, C] -> [B, C, D, H, W]."""
        perm, orig_shape, (o1, o2, sc), reverse = meta
        B, C, D, H, W = orig_shape
        if reverse:
            seq = torch.flip(seq, dims=[1])
        Bc = seq.shape[0]; Cc = seq.shape[2]
        xp = seq.view(Bc, o1, o2, sc, Cc).permute(0, 4, 1, 2, 3).contiguous()
        # invert the original permutation
        inv = [0] * 5
        for i, p in enumerate(perm):
            inv[p] = i
        return xp.permute(*inv).contiguous()

    def forward(self, x, sdf):
        B, C, D, H, W = x.shape
        outs = []
        # 6 directions: (axis, reverse)
        directions = [(0, False), (0, True),    # +-D
                      (1, False), (1, True),    # +-H
                      (2, False), (2, True)]    # +-W
        for i, (axis, rev) in enumerate(directions):
            seq, meta = self._unroll(x, axis, rev)        # [B,L,C]
            sdf_seq, _ = self._unroll(sdf, axis, rev)     # [B,L,1]  (C=1)
            y = self.ssms[i](seq, sdf_seq)                # [B,L,C]
            y3d = self._reroll(y, meta)                   # [B,C,D,H,W]
            outs.append(y3d)
        # merge 6 directions: concat on channel, project back
        merged = torch.cat(outs, dim=1)                   # [B, 6C, D,H,W]
        merged = merged.permute(0, 2, 3, 4, 1)            # [B,D,H,W,6C]
        merged = self.merge(merged)                       # [B,D,H,W,C]
        merged = self.norm(merged)
        merged = merged.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,D,H,W]
        return merged + x                                 # residual

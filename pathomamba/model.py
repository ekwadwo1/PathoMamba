#!/usr/bin/env python3
"""
pathomamba/model.py

Phase 4: PathoMamba architecture. Built incrementally with gates.

4.1 (THIS FILE, first pass): U-Net encoder/decoder skeleton + zero-init SVF
    head + diffeomorphic integration. NO Mamba yet (placeholder bottleneck).
    Gate: forward pass shapes correct; SVF exactly zero at init (identity).

Later sub-steps add: real Mamba selective-scan (4.2), 6-way scan (4.3),
SDF-modulated Delta per rebuttal Eq.2 (4.4).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathomamba.transforms import VecInt, SpatialTransformer
from pathomamba.scan6way import OSMBlock


class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm3d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _PlaceholderBottleneck(nn.Module):
    """Temporary identity bottleneck for 4.1. Replaced by Mamba in 4.2-4.4."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x, sdf_down):
        return self.conv(x) + x  # residual identity-ish placeholder


class PathoMamba(nn.Module):
    def __init__(self, vol_shape=(160, 240, 240), in_ch=2, base=16):
        super().__init__()
        self.vol_shape = vol_shape

        # Encoder: in_ch=2 -> [t1ce_T0, t1ce_T1]
        self.enc1 = ConvBlock3D(in_ch, base, stride=1)     # full res
        self.down1 = ConvBlock3D(base, base*2, stride=2)   # /2
        self.down2 = ConvBlock3D(base*2, base*4, stride=2) # /4
        self.down3 = ConvBlock3D(base*4, base*8, stride=2) # /8  <- bottleneck

        # Bottleneck (placeholder now; real 6-way SDF-modulated Mamba later)
        # self.bottleneck = _PlaceholderBottleneck(base*8)
        self.bottleneck = OSMBlock(base*8, d_state=16)

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock3D(base*4 + base*4, base*4)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock3D(base*2 + base*2, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock3D(base + base, base)

        # SVF head: predicts stationary velocity field
        self.svf = nn.Conv3d(base, 3, 3, padding=1)
        # NEAR-zero init: tiny random weights so phi starts ALMOST identity
        # (near-zero folds, safe per Gate 3) but OFF the exact identity saddle.
        # At exactly v=0 every loss term sits at a zero-gradient point
        # (probe scripts/14: identity init -> 2/80 params get gradient, all
        # losses exactly 0). A small init gives the optimizer signal to start.
        nn.init.normal_(self.svf.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.svf.bias)

        self.integrate = VecInt(vol_shape, nsteps=7)  # rebuttal: 7 steps
        self.transformer = SpatialTransformer(vol_shape)

    def forward(self, img_T0, img_T1, sdf_T0):
        x = torch.cat([img_T0, img_T1], dim=1)   # [B,2,D,H,W]
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)                       # bottleneck features

        # SDF downsampled to bottleneck resolution (for modulation, used in 4.4)
        sdf_down = F.interpolate(sdf_T0, size=d3.shape[2:],
                                 mode="trilinear", align_corners=True)
        b = self.bottleneck(d3, sdf_down)

        u3 = self.dec3(torch.cat([self.up3(b), d2], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d1], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), e1], dim=1))

        svf = self.svf(u1)                        # [B,3,D,H,W]
        phi = self.integrate(svf)                 # diffeomorphic deformation (T0->T1)
        # Warp the MOVING image (T0) toward the FIXED image (T1). phi maps
        # T0-space to T1-space, consistent with mTRE warping L_T0 -> compare L_T1.
        warped = self.transformer(img_T0, phi)
        return warped, phi, svf

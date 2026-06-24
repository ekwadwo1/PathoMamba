#!/usr/bin/env python3
"""
pathomamba/metrics.py

Phase 6: REAL mTRE -- the metric.

mTRE (mean Target Registration): after registration, how far apart are
corresponding anatomical landmarks? We warp the baseline landmarks L_T0 by the
deformation phi, then measure Euclidean distance (mm) to the follow-up L_T1.

Landmarks are stored as VOXEL coordinates on the padded grid (Phase 1).
Spacing is 1mm isotropic, so voxel distance == mm distance directly.
"""
import torch
import torch.nn.functional as F


def warp_landmarks(landmarks_vox, phi):
    """
    Warp landmark VOXEL coordinates through a deformation field.

    landmarks_vox : [N, 3] voxel coords (X,Y,Z order, on the padded grid)
    phi           : [1, 3, D, H, W] deformation (displacement in D,H,W order)

    Returns [N, 3] warped voxel coords.

    The deformation phi gives, at each grid voxel, the displacement to apply.
    To warp a landmark at position p, we sample phi at p (trilinear) and add
    that displacement. Note phi channels are (D,H,W) = (Z,Y,X) displacement,
    while landmarks are (X,Y,Z) -- we handle the axis correspondence carefully.
    """
    N = landmarks_vox.shape[0]
    D, H, W = phi.shape[2:]

    # landmarks are (X,Y,Z) voxel = (W-index, H-index, D-index).
    # phi is indexed [.,:,D,H,W] with channels (dD, dH, dW) displacement.
    # Build normalized sampling grid for grid_sample to read phi at landmark pts.
    # grid_sample expects coords in (x,y,z)=(W,H,D) normalized to [-1,1].
    lx = landmarks_vox[:, 0]  # X = W-index
    ly = landmarks_vox[:, 1]  # Y = H-index
    lz = landmarks_vox[:, 2]  # Z = D-index

    gx = 2.0 * lx / (W - 1) - 1.0
    gy = 2.0 * ly / (H - 1) - 1.0
    gz = 2.0 * lz / (D - 1) - 1.0
    # grid_sample grid shape [1, N, 1, 1, 3], last dim order (x,y,z)=(W,H,D)
    grid = torch.stack([gx, gy, gz], dim=-1).view(1, N, 1, 1, 3)

    # sample each of the 3 displacement channels of phi at the landmark points
    disp = F.grid_sample(phi, grid, mode="bilinear",
                         align_corners=True, padding_mode="border")
    # disp shape [1, 3, N, 1, 1] -> [N, 3], channels (dD, dH, dW)
    disp = disp.view(3, N).t()                 # [N,3] = (dD, dH, dW)

    # displacement channels are (dD,dH,dW) = (dZ,dY,dX). Add to (X,Y,Z) landmark:
    warped = landmarks_vox.clone()
    warped[:, 0] = lx + disp[:, 2]             # X += dW (=dX)
    warped[:, 1] = ly + disp[:, 1]             # Y += dH (=dY)
    warped[:, 2] = lz + disp[:, 0]             # Z += dD (=dZ)
    return warped


def compute_mtre(landmarks_T0, landmarks_T1, phi, spacing=1.0):
    """
    Mean Target Registration Error (mm).

    landmarks_T0, landmarks_T1 : [N,3] voxel coords (paired)
    phi    : [1,3,D,H,W] deformation warping T0 space toward T1
    spacing: mm per voxel (BraTS-Reg = 1mm isotropic)

    Returns (mtre_mm, per_landmark_mm).
    """
    warped_T0 = warp_landmarks(landmarks_T0, phi)
    # Euclidean distance per landmark, in voxels -> mm via spacing
    diff = (warped_T0 - landmarks_T1) * spacing
    dist = diff.pow(2).sum(dim=1).sqrt()       # [N]
    return dist.mean().item(), dist


def initial_mtre(landmarks_T0, landmarks_T1, spacing=1.0):
    """mTRE with NO registration (identity) -- the baseline to beat."""
    diff = (landmarks_T0 - landmarks_T1) * spacing
    dist = diff.pow(2).sum(dim=1).sqrt()
    return dist.mean().item(), dist

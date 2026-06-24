#!/usr/bin/env python3
"""
pathomamba/losses.py

Phase 5: PathoMamba loss terms. Built one at a time, each gated.

Full objective: L_total = L_sim + lambda_reg * L_diff
                                    + lambda_bio * (L_TABL + L_MK)

THIS FILE (5.1): L_diff -- SDF-weighted diffusion (smoothness) regularizer.
The contribution is that smoothness is MODULATED by the SDF: rigid in healthy tissue, plastic in
  the tumor. w(D) = sigma(alpha * D), alpha=1.0, D in mm.
    w(+5)=0.993 (rigid), w(0)=0.5 (boundary), w(-5)=0.007 (plastic).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sdf_weight(sdf, alpha=1.0):
    """
    Stiffness weight w(p) = sigma(alpha * D_p), D in mm (rebuttal [2]).
    D > 0 outside tumor (healthy) -> w -> 1 (rigid).
    D < 0 inside tumor            -> w -> 0 (plastic).
    """
    return torch.sigmoid(alpha * sdf)


def l_diff(velocity, sdf, alpha=1.0):
    """
    SDF-weighted smoothness penalty on the velocity field.

    velocity : [B, 3, D, H, W]  the SVF (penalize ITS spatial gradients)
    sdf      : [B, 1, D, H, W]  signed distance (mm), negative inside tumor
    alpha    : stiffness sharpness (rebuttal: 1.0)

    L_diff = mean over voxels of w(D_p) * ||grad v_p||^2

    Gradients computed by forward finite differences along D,H,W.
    The weight is evaluated at each voxel; we average the per-axis gradient
    weight at the two voxels the difference spans (use the base voxel).
    """
    w = sdf_weight(sdf, alpha)                      # [B,1,D,H,W]

    # spatial gradients of the velocity field (forward differences)
    dvx = velocity[:, :, 1:, :, :] - velocity[:, :, :-1, :, :]   # d/dD
    dvy = velocity[:, :, :, 1:, :] - velocity[:, :, :, :-1, :]   # d/dH
    dvz = velocity[:, :, :, :, 1:] - velocity[:, :, :, :, :-1]   # d/dW

    # squared magnitude of each gradient component, summed over the 3 vel channels
    gx = (dvx ** 2).sum(dim=1, keepdim=True)        # [B,1,D-1,H,W]
    gy = (dvy ** 2).sum(dim=1, keepdim=True)        # [B,1,D,H-1,W]
    gz = (dvz ** 2).sum(dim=1, keepdim=True)        # [B,1,D,H,W-1]

    # weight each gradient by w at the base voxel of the difference
    wx = w[:, :, :-1, :, :]
    wy = w[:, :, :, :-1, :]
    wz = w[:, :, :, :, :-1]

    loss = (wx * gx).mean() + (wy * gy).mean() + (wz * gz).mean()
    return loss


class LDiff(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, velocity, sdf):
        return l_diff(velocity, sdf, self.alpha)

# ============================================================
# 5.2: L_sim -- Local Normalized Cross-Correlation (LNCC)
# ============================================================
#  9^3 window. LNCC is robust to inter-timepoint intensity
# differences (it correlates local structure, not absolute intensity).
# Returns 1 - LNCC so that LOWER is better (a loss). Perfect alignment -> 0.

def lncc_loss(warped, target, window=9, eps=1e-5, mask=None):
    """
    1 - mean local normalized cross-correlation over a window^3 neighborhood,
    averaged over FOREGROUND voxels only. Rebuttal: 9^3 window.

    PRECISION: LNCC's windowed-variance ("sum of squares minus square of sums")
    is catastrophically unstable in bf16 -- probe scripts/18 showed cc reaching
    3.5e6 and I_var going NEGATIVE under autocast, making the loss explode to
    -14825 (vs 0.59 in fp32). With BraTS's ~77% near-constant background this is
    severe. So we force fp32 here even when the rest of the model runs in bf16.

    Foreground-masked (probe scripts/12): zero-padding contaminates boundary
    windows and the 84.6% zero background would otherwise dominate; we average
    cc only where the target has signal, so L_sim measures BRAIN alignment.
    """
    # Force fp32: disable autocast and cast inputs. The variance math must not
    # run in bf16. Cost is negligible (one loss eval/step); model stays bf16.
    with torch.autocast(device_type=warped.device.type, enabled=False):
        I = warped.float()
        J = target.float()
        B, C, D, H, W = I.shape
        assert C == 1, "LNCC expects single-channel volumes"

        k = window
        pad = k // 2
        win_vol = k ** 3
        sum_filt = torch.ones(1, 1, k, k, k, device=I.device, dtype=I.dtype)

        def box(x):
            return F.conv3d(x, sum_filt, padding=pad)

        I_sum = box(I); J_sum = box(J)
        I2_sum = box(I * I); J2_sum = box(J * J); IJ_sum = box(I * J)
        I_mean = I_sum / win_vol; J_mean = J_sum / win_vol

        cross = IJ_sum - J_mean * I_sum - I_mean * J_sum + I_mean * J_mean * win_vol
        I_var = (I2_sum - 2 * I_mean * I_sum + I_mean * I_mean * win_vol).clamp_min(0)
        J_var = (J2_sum - 2 * J_mean * J_sum + J_mean * J_mean * win_vol).clamp_min(0)

        cc = (cross * cross) / (I_var * J_var + eps)
        cc = cc.clamp(0.0, 1.0)   # squared correlation is [0,1]; clip residual noise

        if mask is None:
            mask = (J > 0).to(cc.dtype)
        denom = mask.sum().clamp_min(1.0)
        masked_cc = (cc * mask).sum() / denom
        return 1.0 - masked_cc

class LSim(nn.Module):
    def __init__(self, window=9):
        super().__init__()
        self.window = window

    def forward(self, warped, target):
        return lncc_loss(warped, target, self.window)

# ============================================================
# 5.3: L_TABL -- Tumor-Aware Biomechanical Loss (component-wise)
# ============================================================
# per-connected-component Jacobian penalty driven by the
# observed log-volume ratio eta_k = log(V_T1^k / V_T0^k). Growing components
# (eta_k > 0) pushed toward |J| > 1; shrinking (eta_k < 0) toward |J| < 1.
# Handles simultaneous recurrence + cavity collapse (a single scalar eta
# would average opposing changes into a meaningless net value).

from pathomamba.transforms import jacobian_determinant


def _connected_components_3d(mask_bool):
    """
    Label connected components of a 3D boolean mask (6-connectivity) on CPU
    via scipy. Returns an integer label volume (0 = background) and the count.
    Used once per sample to define per-component biomechanical targets.
    """
    import numpy as np
    from scipy.ndimage import label
    structure = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                          [[0,1,0],[1,1,1],[0,1,0]],
                          [[0,0,0],[0,1,0],[0,0,0]]], dtype=bool)  # 6-conn
    lbl, n = label(mask_bool, structure=structure)
    return lbl, n


def l_tabl(phi, mask_T0, mask_T1, eps=1e-5, margin=0.0):
    """
    Component-wise tumor-aware biomechanical loss.

    phi      : [B, 3, D, H, W]  deformation field (displacement)
    mask_T0  : [B, 1, D, H, W]  baseline tumor-core mask
    mask_T1  : [B, 1, D, H, W]  follow-up tumor-core mask
    margin   : optional slack (delta) before penalizing

    For each connected component k of the T0 mask:
      eta_k = log(V_T1^k / V_T0^k)   (volume change of THAT region)
      if eta_k > 0 (grew):   penalize |J| < 1 inside the component (want expansion)
      if eta_k < 0 (shrank): penalize |J| > 1 inside the component (want contraction)
    Penalty: hinge on the wrong-direction Jacobian, averaged per component,
    then over components.

    NOTE: component volumes use the GLOBAL T1 volume matched spatially to each
    T0 component's location (we measure T1 mass overlapping each T0 component).
    """
    B = phi.shape[0]
    detJ = jacobian_determinant(phi)              # [B,1,D,H,W], |J| proxy (det)

    total = phi.new_tensor(0.0)
    n_terms = 0

    for b in range(B):
        m0 = (mask_T0[b, 0] > 0)
        m1 = (mask_T1[b, 0] > 0)
        if not m0.any():
            continue  # no tumor in baseline -> no biomechanical target

        lbl, n = _connected_components_3d(m0.detach().cpu().numpy())
        lbl_t = torch.from_numpy(lbl).to(phi.device)

        for k in range(1, n + 1):
            comp = (lbl_t == k)                   # [D,H,W] bool, this component
            v0 = comp.sum().float()
            if v0 < 10:                           # ignore tiny specks
                continue
            # T1 volume for THIS component: dilate the component region so that
            # GROWTH beyond the original boundary is counted (strict intersection
            # m1 & comp can never exceed v0, so it cannot see growth -- that was
            # the bug). We dilate comp by a margin and count T1 mass inside it.
            comp_np = comp.detach().cpu().numpy()
            from scipy.ndimage import binary_dilation
            comp_dil = binary_dilation(comp_np, iterations=5)   # ~5-voxel halo
            comp_dil_t = torch.from_numpy(comp_dil).to(phi.device)
            v1 = (m1 & comp_dil_t).sum().float()
            eta_k = torch.log((v1 + eps) / (v0 + eps))

            det_in = detJ[b, 0][comp]             # |J| inside ORIGINAL component
            # use a Python float for the branch to avoid tensor-comparison subtleties
            if eta_k.item() > 0:
                pen = F.relu((1.0 + margin) - det_in)   # grew -> want |J|>1
            else:
                pen = F.relu(det_in - (1.0 - margin))   # shrank -> want |J|<1
            total = total + pen.mean()
            n_terms += 1

    if n_terms == 0:
        return phi.new_tensor(0.0)
    return total / n_terms


class LTABL(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, phi, mask_T0, mask_T1):
        return l_tabl(phi, mask_T0, mask_T1, margin=self.margin)

# ============================================================
# 5.4: L_MK -- Monro-Kellie peritumoral shell penalty
# ============================================================
# second term of Eq.(4): (1/|Omega_shell|) * sum ReLU(|J|-1-delta)
# over the peritumoral SHELL (healthy rim just outside the tumor). Monro-Kellie
# doctrine: the rigid skull conserves intracranial volume, so the healthy rim
# around a growing tumor is DISPLACED, not expanded. This penalizes unphysical
# expansion (|J| > 1+delta) in that shell, enforcing |J| ~ 1 there. The shell
# is the SDF band 0 < D < shell_width (just outside the tumor boundary).

def l_mk(phi, sdf, shell_width=5.0, delta=0.05):
    """
    Monro-Kellie shell penalty.

    phi         : [B, 3, D, H, W]  deformation field
    sdf         : [B, 1, D, H, W]  signed distance (mm), <0 inside tumor,
                                    >0 outside. Shell = 0 < D < shell_width.
    shell_width : mm thickness of the peritumoral shell (default 5mm)
    delta       : slack before penalizing expansion (rebuttal: delta)

    L_MK = mean over shell voxels of ReLU(|J| - 1 - delta)
         (penalizes expansion beyond 1+delta in the healthy rim)
    """
    detJ = jacobian_determinant(phi)              # [B,1,D,H,W]
    # peritumoral shell: just OUTSIDE the tumor (0 < D < shell_width)
    shell = ((sdf > 0) & (sdf < shell_width)).to(detJ.dtype)   # [B,1,D,H,W]

    penalty = F.relu(detJ - 1.0 - delta)          # only expansion is penalized
    denom = shell.sum().clamp_min(1.0)
    return (penalty * shell).sum() / denom


class LMK(nn.Module):
    def __init__(self, shell_width=5.0, delta=0.05):
        super().__init__()
        self.shell_width = shell_width
        self.delta = delta

    def forward(self, phi, sdf):
        return l_mk(phi, sdf, self.shell_width, self.delta)

# ============================================================
# 5.5: Total objective
# ============================================================
# L_total = L_sim + lambda_reg * L_diff + lambda_bio * (L_TABL + L_MK)

class PathoMambaLoss(nn.Module):
    """
    Full PathoMamba training objective.

    L_total = L_sim + lambda_reg * L_diff + lambda_bio * (L_TABL + L_MK)

    Returns (total, components_dict) so training can log each term separately
    (essential for diagnosing which term dominates / whether TABL is active).
    """
    def __init__(self, lambda_reg=1.0, lambda_bio=0.1,
                 alpha=1.0, lncc_window=9,
                 shell_width=5.0, delta=0.05, tabl_margin=0.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_bio = lambda_bio
        self.l_sim = LSim(window=lncc_window)
        self.l_diff = LDiff(alpha=alpha)
        self.l_tabl = LTABL(margin=tabl_margin)
        self.l_mk = LMK(shell_width=shell_width, delta=delta)

    def forward(self, warped, target, velocity, phi, sdf, mask_T0, mask_T1):
        sim = self.l_sim(warped, target)
        diff = self.l_diff(velocity, sdf)
        tabl = self.l_tabl(phi, mask_T0, mask_T1)
        mk = self.l_mk(phi, sdf)

        total = sim + self.lambda_reg * diff + self.lambda_bio * (tabl + mk)
        components = {
            "total": total.detach(),
            "L_sim": sim.detach(),
            "L_diff": diff.detach(),
            "L_TABL": tabl.detach(),
            "L_MK": mk.detach(),
        }
        return total, components

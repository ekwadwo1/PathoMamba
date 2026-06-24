#!/usr/bin/env python3
"""
pathomamba/preprocessing.py

Phase 1: BraTS-Reg preprocessing for PathoMamba.

DESIGN DECISIONS (locked):
  - FULL VOLUME, pad-only to (240,240,160). NEVER crop. -> no crop-offset
    landmark bug; tumor intact by construction; SDF sees whole tumor.
  - t1ce-only registration input -> 2-channel pair [t1ce_T0, t1ce_T1].
  - SDF + eta from REAL Tumor-Core masks (Phase 0.3 validated).
  - Landmarks stored in VOXEL coordinates, converted from WORLD (mm) via
    the INVERSE AFFINE using nibabel's tested machinery (not hand-derived).
  - BOTH landmarks_T0 and landmarks_T1 saved -> real mTRE possible.

WHY (240,240,160) and not (240,240,155): U-Net downsampling needs dims
divisible by 2^depth. Z=155 is padded to 160 (add 5 zero-background
slices). Padding is SAFE: it can never clip a tumor. Cropping can.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


# ============================ CONFIG ============================
DEFAULT_RAW = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/BraTSReg_Training_Dataset"
DEFAULT_OUT = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/PathoMamba_Code/processed_data"

# Padded target: full 240x240 in-plane, Z padded 155 -> 160 for clean
# downsampling. Pad-only: no voxel is ever removed.
TARGET_SHAPE = (240, 240, 160)
# ===============================================================


def safe_glob(patient_dir, pattern):
    files = glob.glob(os.path.join(patient_dir, pattern))
    return sorted(f for f in files if not os.path.basename(f).startswith("._"))


def pad_to_target(volume, target=TARGET_SHAPE):
    """
    Symmetric zero-pad a volume up to target shape. NEVER crops.
    Returns (padded_volume, pad_before) where pad_before is the per-axis
    number of voxels added at the START of each axis (needed to shift
    landmark voxel coordinates consistently).

    Asserts every source voxel survives (target >= source on all axes).
    """
    src = np.array(volume.shape)
    tgt = np.array(target)
    assert np.all(tgt >= src), (
        f"TARGET {target} smaller than source {volume.shape} on some axis. "
        f"This would CROP and could clip a tumor. Increase TARGET_SHAPE."
    )
    total_pad = tgt - src
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before
    padded = np.pad(
        volume,
        pad_width=[(pad_before[i], pad_after[i]) for i in range(3)],
        mode="constant", constant_values=0,
    )
    assert padded.shape == target, f"pad produced {padded.shape}, expected {target}"
    return padded, pad_before


def world_to_voxel(world_coords_xyz, affine):
    """
    Convert Nx3 WORLD landmark coordinates to VOXEL indices.

    BraTS-Reg landmarks are stored in RAS world convention, but the NIfTI
    affine encodes LPS. These disagree by a sign flip on X and Y. We flip
    X,Y to match the affine's LPS convention BEFORE applying the inverse
    affine. Determined EMPIRICALLY (scripts/06): the flip lands 100% of
    landmarks in-bounds and on brain tissue across all patients; every
    other convention lands 0%. Uses nibabel's tested apply_affine.

    Returns Nx3 float voxel coordinates (sub-voxel precision retained).
    """
    if world_coords_xyz.size == 0:
        return world_coords_xyz
    # RAS -> LPS: negate X and Y to match the affine's convention
    lps = world_coords_xyz * np.array([-1.0, -1.0, 1.0])
    inv = np.linalg.inv(affine)
    return nib.affines.apply_affine(inv, lps)


def normalize_intensity(image):
    """[0, 99.5] percentile clip + min-max to [0,1] over foreground."""
    fg = image[image > 0]
    if fg.size == 0:
        return image.astype(np.float32)
    p99_5 = np.percentile(fg, 99.5)
    image = np.clip(image, 0.0, p99_5)
    lo, hi = image.min(), image.max()
    if hi - lo > 1e-5:
        image = (image - lo) / (hi - lo)
    return image.astype(np.float32)


def compute_sdf(mask):
    """
    Signed Distance Function to the tumor boundary.
    Negative INSIDE tumor, positive OUTSIDE. Matches paper convention
    (D_p < 0 inside pathology). Computed on the FULL padded mask, so the
    tumor is intact and the zero-level-set is correct.
    """
    mask_bool = mask > 0
    if not mask_bool.any():
        # No tumor (e.g. fully resected): large positive distance everywhere.
        return np.full(mask.shape, 100.0, dtype=np.float32)
    dist_out = distance_transform_edt(~mask_bool)   # >0 outside
    dist_in = distance_transform_edt(mask_bool)     # >0 inside
    sdf = dist_out - dist_in                         # outside +, inside -
    return sdf.astype(np.float32)


def to_tensor_dhw(arr):
    """
    NIfTI arrays are (X,Y,Z). PyTorch conv3d expects (D,H,W) = (Z,Y,X).
    Add channel dim -> [1, D, H, W].
    """
    arr_dhw = np.transpose(arr, (2, 1, 0))
    return torch.from_numpy(np.ascontiguousarray(arr_dhw)).unsqueeze(0).float()


def process_patient(patient_dir, out_dir, target=TARGET_SHAPE):
    pid = os.path.basename(patient_dir)

    t0_paths = safe_glob(patient_dir, "*_00_*t1ce.nii.gz")
    t1_paths = safe_glob(patient_dir, "*_01_*t1ce.nii.gz")
    m0_paths = safe_glob(patient_dir, "*_00_*tc_mask.nii.gz")
    m1_paths = safe_glob(patient_dir, "*_01_*tc_mask.nii.gz")
    lm0_paths = safe_glob(patient_dir, "*_00_*landmarks.csv")
    lm1_paths = safe_glob(patient_dir, "*_01_*landmarks.csv")

    missing = [n for n, p in [("t1ce_T0", t0_paths), ("t1ce_T1", t1_paths),
                              ("tc_mask_T0", m0_paths), ("tc_mask_T1", m1_paths),
                              ("lm_T0", lm0_paths), ("lm_T1", lm1_paths)] if not p]
    if missing:
        return {"pid": pid, "status": "skipped", "missing": missing}

    # --- load images + masks (NIfTI X,Y,Z) ---
    t0_nii = nib.load(t0_paths[0])
    t1_nii = nib.load(t1_paths[0])
    t0_img = t0_nii.get_fdata().astype(np.float32)
    t1_img = t1_nii.get_fdata().astype(np.float32)
    m0 = nib.load(m0_paths[0]).get_fdata().astype(np.uint8)
    m1 = nib.load(m1_paths[0]).get_fdata().astype(np.uint8)

    # --- record raw tumor volumes BEFORE padding (for survival assertion) ---
    v0_raw, v1_raw = int((m0 > 0).sum()), int((m1 > 0).sum())

    # --- pad everything to target (pad_before identical for matched shapes) ---
    t0_pad, pad_before = pad_to_target(t0_img, target)
    t1_pad, _ = pad_to_target(t1_img, target)
    m0_pad, _ = pad_to_target(m0, target)
    m1_pad, _ = pad_to_target(m1, target)

    # --- TUMOR SURVIVAL ASSERTION: padding must not change tumor voxel count ---
    v0_pad, v1_pad = int((m0_pad > 0).sum()), int((m1_pad > 0).sum())
    if v0_pad != v0_raw or v1_pad != v1_raw:
        return {"pid": pid, "status": "FAILED_survival",
                "v0_raw": v0_raw, "v0_pad": v0_pad,
                "v1_raw": v1_raw, "v1_pad": v1_pad}

    # --- intensity normalize images ---
    t0_norm = normalize_intensity(t0_pad)
    t1_norm = normalize_intensity(t1_pad)

    # --- SDF from REAL TC mask (T0 prior drives stiffness modulation) ---
    sdf_t0 = compute_sdf(m0_pad)

    # --- eta: log-volume ratio (component-wise handled later; scalar here) ---
    eps = 1e-5
    eta_scalar = float(np.log((v1_raw + eps) / (v0_raw + eps)))
    eta_map = np.zeros_like(m0_pad, dtype=np.float32)
    eta_map[m0_pad > 0] = eta_scalar  # eta lives on the tumor for TABL

    # --- LANDMARKS: world -> voxel (inverse affine), then shift by pad_before ---
    def load_landmarks(lm_path, affine):
        df = pd.read_csv(lm_path)
        # Some BraTS-Reg CSVs (e.g. patients 051-070) have leading spaces in
        # the header: "Landmark, X, Y, Z" -> columns named " X", " Y", " Z".
        # Strip whitespace from column names so both variants parse identically.
        df.columns = [c.strip() for c in df.columns]
        if not all(c in df.columns for c in ["X", "Y", "Z"]):
            raise KeyError(
                f"{os.path.basename(lm_path)}: expected X,Y,Z columns after "
                f"stripping, found {list(df.columns)}"
            )
        world = df[["X", "Y", "Z"]].values.astype(np.float64)  # Nx3 WORLD mm (RAS)
        voxel = world_to_voxel(world, affine)                  # Nx3 VOXEL (X,Y,Z)
        voxel_padded = voxel + pad_before                      # shift for the pad
        return torch.from_numpy(voxel_padded).float(), world, voxel

    lm0_vox, lm0_world, lm0_vox_raw = load_landmarks(lm0_paths[0], t0_nii.affine)
    lm1_vox, lm1_world, lm1_vox_raw = load_landmarks(lm1_paths[0], t1_nii.affine)

    # --- save ---
    patient_data = {
        "img_T0": to_tensor_dhw(t0_norm),     # [1,D,H,W]
        "img_T1": to_tensor_dhw(t1_norm),
        "mask_T0": to_tensor_dhw(m0_pad),
        "mask_T1": to_tensor_dhw(m1_pad),
        "sdf_T0": to_tensor_dhw(sdf_t0),
        "eta": to_tensor_dhw(eta_map),
        "eta_scalar": torch.tensor(eta_scalar),
        # landmarks in (X,Y,Z) VOXEL coords on the PADDED grid, ready for use
        "landmarks_T0": lm0_vox,   # [N,3]
        "landmarks_T1": lm1_vox,   # [N,3]
        "pad_before": torch.tensor(pad_before),
        "vol_T0": v0_raw, "vol_T1": v1_raw,
    }
    torch.save(patient_data, os.path.join(out_dir, f"{pid}.pt"))

    return {"pid": pid, "status": "ok", "v0": v0_raw, "v1": v1_raw,
            "n_lm": lm0_vox.shape[0], "eta": round(eta_scalar, 3)}


def run(raw_dir, out_dir, target=TARGET_SHAPE):
    os.makedirs(out_dir, exist_ok=True)
    patient_dirs = sorted(
        d.path for d in os.scandir(raw_dir)
        if d.is_dir() and "BraTSReg" in os.path.basename(d.path)
    )
    print(f"[*] Preprocessing {len(patient_dirs)} patients -> {out_dir}")
    print(f"[*] Target shape (pad-only): {target}\n")

    ok, skipped, failed = 0, [], []
    for pdir in tqdm(patient_dirs, desc="Preprocessing"):
        try:
            r = process_patient(pdir, out_dir, target)
            if r["status"] == "ok":
                ok += 1
            elif r["status"] == "skipped":
                skipped.append((r["pid"], r["missing"]))
            elif r["status"] == "FAILED_survival":
                failed.append(r)
                print(f"\n[FAIL-SURVIVAL] {r['pid']}: tumor voxels changed under "
                      f"padding (T0 {r['v0_raw']}->{r['v0_pad']}, "
                      f"T1 {r['v1_raw']}->{r['v1_pad']}). NOT SAVED.")
        except Exception as e:
            failed.append({"pid": os.path.basename(pdir), "error": str(e)})
            print(f"\n[ERROR] {os.path.basename(pdir)}: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"  Saved : {ok}")
    print(f"  Skipped (missing files): {len(skipped)}")
    print(f"  Failed : {len(failed)}")
    print(f"{'='*60}")
    if skipped:
        for pid, miss in skipped[:10]:
            print(f"    SKIP {pid}: missing {miss}")
    if failed:
        print("  [!] FAILURES ABOVE -- investigate before training.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    run(args.raw, args.out)

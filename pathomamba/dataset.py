#!/usr/bin/env python3
"""
pathomamba/dataset.py

Phase 2: Dataset + dataloaders for PathoMamba.

Loads the SAVED split (splits/split.json) -- never regenerates it. Yields
the full set of tensors Phase 1 produced. Landmarks are variable-length
[N,3] per patient, so the collate fn keeps them as a list (stacking would
crash on differing N).
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


EXPECTED_SHAPE = (1, 160, 240, 240)  # [C, D, H, W] = [1, Z=160, Y=240, X=240]


class BraTSRegDataset(Dataset):
    def __init__(self, processed_dir, patient_ids):
        self.processed_dir = processed_dir
        self.patient_ids = patient_ids
        self.paths = [os.path.join(processed_dir, f"{pid}.pt") for pid in patient_ids]
        for p in self.paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing processed file: {p}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=True)

        def chk(name):
            t = data[name].to(torch.float32)
            if tuple(t.shape) != EXPECTED_SHAPE:
                raise ValueError(
                    f"{self.patient_ids[idx]} '{name}': shape {tuple(t.shape)} "
                    f"!= expected {EXPECTED_SHAPE}"
                )
            return t

        return {
            "pid": self.patient_ids[idx],
            "img_T0": chk("img_T0"),
            "img_T1": chk("img_T1"),
            "mask_T0": chk("mask_T0"),
            "mask_T1": chk("mask_T1"),
            "sdf_T0": chk("sdf_T0"),
            "eta": chk("eta"),
            "eta_scalar": data["eta_scalar"].to(torch.float32),
            # variable-length [N,3] voxel landmarks on the padded grid
            "landmarks_T0": data["landmarks_T0"].to(torch.float32),
            "landmarks_T1": data["landmarks_T1"].to(torch.float32),
        }


def brats_collate(batch):
    """Stack fixed-shape volumes; keep variable-length landmarks as lists."""
    out = {
        "pid": [b["pid"] for b in batch],
        "eta_scalar": torch.stack([b["eta_scalar"] for b in batch]),
        "landmarks_T0": [b["landmarks_T0"] for b in batch],
        "landmarks_T1": [b["landmarks_T1"] for b in batch],
    }
    for k in ["img_T0", "img_T1", "mask_T0", "mask_T1", "sdf_T0", "eta"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)  # [B,1,D,H,W]
    return out


def get_dataloaders(processed_dir, split_path="splits/split.json",
                    batch_size=1, num_workers=4):
    """Build train/val loaders from the SAVED split."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run scripts/07_make_split.py first."
        )
    with open(split_path) as f:
        split = json.load(f)

    print(f"[*] Loaded split: {split['n_train']} train / {split['n_val']} val "
          f"(seed={split['seed']}, created {split['created'][:10]})")

    train_ds = BraTSRegDataset(processed_dir, split["train"])
    val_ds = BraTSRegDataset(processed_dir, split["val"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=brats_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=brats_collate,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick smoke test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed",
                    default="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/PathoMamba_Code/processed_data")
    ap.add_argument("--split", default="splits/split.json")
    args = ap.parse_args()

    train_loader, val_loader = get_dataloaders(args.processed, args.split, batch_size=2)
    batch = next(iter(train_loader))
    print("\n--- One batch ---")
    for k in ["img_T0", "img_T1", "mask_T0", "sdf_T0", "eta"]:
        print(f"  {k:10s}: {list(batch[k].shape)}  dtype={batch[k].dtype}")
    print(f"  pids        : {batch['pid']}")
    for i, (l0, l1) in enumerate(zip(batch["landmarks_T0"], batch["landmarks_T1"])):
        print(f"  patient {i}: L_T0 {list(l0.shape)}, L_T1 {list(l1.shape)}")
    print("\n[OK] Dataset + collate + saved-split loading verified.")

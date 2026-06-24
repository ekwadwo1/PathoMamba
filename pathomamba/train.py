#!/usr/bin/env python3
"""
pathomamba/train.py

Phase 6: PathoMamba training engine.

HEADLINE METRIC: held-out val mTRE. Train mTRE is that the question -- Q1 proved the model can fit any specific case.

Features:
  - AdamW + cosine LR schedule, bf16 autocast
  - landmark supervision (differentiable mTRE on TRAIN landmarks) + image losses
  - full checkpoint + auto-resume (model/opt/scheduler/epoch/best)
  - real val mTRE every N epochs (the verdict)
  - separate checkpoint of BEST-val model (captures best generalization)
  - W_sdf norm logged each eval
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from pathomamba.model import PathoMamba
from pathomamba.losses import PathoMambaLoss
from pathomamba.metrics import compute_mtre, initial_mtre, warp_landmarks
from pathomamba.transforms import fold_percentage
from pathomamba.dataset import get_dataloaders


def landmark_loss(L0, L1, phi):
    """Differentiable mTRE on landmarks."""
    warped = warp_landmarks(L0, phi)
    return (warped - L1).pow(2).sum(dim=1).sqrt().mean()


def wsdf_norm(model):
    tot = 0.0
    for name, p in model.named_parameters():
        if "W_sdf" in name and "weight" in name:
            tot += p.detach().float().norm().item() ** 2
    return tot ** 0.5


@torch.no_grad()
def evaluate(model, val_loader, dev):
    """Real mTRE on held-out val patients. Returns (mean_mtre, mean_fold,
    mean_initial, per_patient)."""
    model.eval()
    mtres, folds, inits, pids = [], [], [], []
    for batch in val_loader:
        t0 = batch["img_T0"].to(dev)
        t1 = batch["img_T1"].to(dev)
        sdf = batch["sdf_T0"].to(dev)
        L0 = batch["landmarks_T0"][0].to(dev)   # batch_size=1
        L1 = batch["landmarks_T1"][0].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, phi, _ = model(t0, t1, sdf)
        phi_f = phi.float()
        mtres.append(compute_mtre(L0, L1, phi_f)[0])
        folds.append(fold_percentage(phi_f))
        inits.append(initial_mtre(L0, L1)[0])
        pids.append(batch["pid"][0])
    mean_mtre = sum(mtres) / len(mtres)
    return (mean_mtre, sum(folds)/len(folds), sum(inits)/len(inits),
            list(zip(pids, inits, mtres)))


def save_ckpt(path, model, opt, sched, epoch, best_val):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched else None,
        "epoch": epoch,
        "best_val": best_val,
    }, path)


def train(args):
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)

    train_loader, val_loader = get_dataloaders(
        args.processed, args.split, batch_size=1, num_workers=args.workers)

    # infer volume shape from one sample
    sample = next(iter(val_loader))
    vol = tuple(sample["img_T0"].shape[2:])  # [B,1,D,H,W] -> (D,H,W)
    print(f"[*] Volume {vol} | lambda_lm={args.lambda_lm} "
          f"lambda_reg={args.lambda_reg} lambda_bio={args.lambda_bio} "
          f"lr={args.lr} epochs={args.epochs}\n")

    model = PathoMamba(vol_shape=vol).to(dev)
    crit = PathoMambaLoss(lambda_reg=args.lambda_reg,
                          lambda_bio=args.lambda_bio).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 1
    best_val = float("inf")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest = os.path.join(args.ckpt_dir, "latest.pt")
    best_path = os.path.join(args.ckpt_dir, "best_val.pt")

    # --- auto-resume ---
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ck["sched"] and sched:
            sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        print(f"[*] Resumed from epoch {ck['epoch']} (best_val={best_val:.3f})\n")

    # --- baseline: held-out mTRE BEFORE any training (the bar to beat) ---
    init_val_mtre, _, _, per = evaluate(model, val_loader, dev)
    print(f"[*] Held-out val mTRE at init (untrained): {init_val_mtre:.3f} mm")
    print(f"[*] W_sdf norm at init: {wsdf_norm(model):.4e}\n")
    print(f"{'epoch':>6} {'train_loss':>11} {'train_mTRE':>11} "
          f"{'VAL_mTRE':>10} {'val_init':>9} {'folds':>7} {'W_sdf':>10} "
          f"{'lr':>9} {'verdict':>10}")
    print("  " + "-" * 96)

    log_path = os.path.join(args.ckpt_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t_ep = time.time()
        ep_loss, ep_lm, n = 0.0, 0.0, 0
        for batch in train_loader:
            t0 = batch["img_T0"].to(dev)
            t1 = batch["img_T1"].to(dev)
            sdf = batch["sdf_T0"].to(dev)
            m0 = batch["mask_T0"].to(dev)
            m1 = batch["mask_T1"].to(dev)
            L0 = batch["landmarks_T0"][0].to(dev)
            L1 = batch["landmarks_T1"][0].to(dev)

            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                warped, phi, svf = model(t0, t1, sdf)
                img_loss, comp = crit(warped, t1, svf, phi, sdf, m0, m1)
            lm = landmark_loss(L0, L1, phi.float())   # fp32 geometry
            total = args.lambda_lm * lm + img_loss
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ep_loss += total.item(); ep_lm += lm.item(); n += 1
        sched.step()

        # --- evaluation ---
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            val_mtre, val_fold, val_init, per = evaluate(model, val_loader, dev)
            # train mTRE on a few batches (sanity, NOT the metric)
            tr_mtre, _, _, _ = evaluate(model, train_loader, dev)
            wsdf = wsdf_norm(model)
            lr_now = opt.param_groups[0]["lr"]
            beats = val_mtre < val_init - 0.05
            verdict = "BEATS" if beats else "no gain"

            print(f"{epoch:>6} {ep_loss/n:>11.4f} {tr_mtre:>11.3f} "
                  f"{val_mtre:>10.3f} {val_init:>9.3f} {val_fold:>7.2f} "
                  f"{wsdf:>10.4e} {lr_now:>9.2e} {verdict:>10}")

            log_f.write(json.dumps({
                "epoch": epoch, "train_loss": ep_loss/n, "train_mtre": tr_mtre,
                "val_mtre": val_mtre, "val_init": val_init, "val_fold": val_fold,
                "wsdf": wsdf, "lr": lr_now,
            }) + "\n"); log_f.flush()

            # save best-val model
            if val_mtre < best_val:
                best_val = val_mtre
                save_ckpt(best_path, model, opt, sched, epoch, best_val)

        save_ckpt(latest, model, opt, sched, epoch, best_val)

    log_f.close()
    print("\n" + "=" * 96)
    print(f"  TRAINING COMPLETE")
    print(f"  Held-out val mTRE at init: {init_val_mtre:.3f} mm")
    print(f"  Best held-out val mTRE:    {best_val:.3f} mm")
    print(f"  Reduction vs init:         {100*(1-best_val/init_val_mtre):+.1f}%")
    if best_val < init_val_mtre - 0.1:
        print(f"  VERDICT: GENERALIZES -- supervised training on 112 patients")
        print(f"           reduces held-out mTRE. The method works.")
    else:
        print(f"  VERDICT: DOES NOT GENERALIZE -- held-out mTRE not beaten even")
        print(f"           with full data. Memorization, not registration.")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed",
                    default="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/PathoMamba_Code/processed_data")
    ap.add_argument("--split", default="splits/split.json")
    ap.add_argument("--ckpt-dir", default="checkpoints/run1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lambda-lm", type=float, default=1.0)
    ap.add_argument("--lambda-reg", type=float, default=1.0)
    ap.add_argument("--lambda-bio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

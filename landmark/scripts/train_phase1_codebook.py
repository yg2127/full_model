#!/usr/bin/env python
"""Phase 1: VQ-VAE codebook 학습 (IR DMD, vit=None).

목적: DMD IR face_crop 의 patch vocabulary 학습 (2048 entries × 256 dim).
입력: 정상 face_crop_112 만 (가린 sample 제외).
출력: edge_heatmap (12 channels, 64×64) reconstruction.
GT: cfg.DMD.GT_SOURCE = "mediapipe" (Phase 0.5 검증 후 swap).

Loss:
    L = ‖x_hat − edge_GT‖² + L_commitment (VectorQuantizer 내부 계산)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src"))             # for `models` package (relative imports)

# vendor (yacs, torchlm)
sys.path.insert(0, "/data/shared/orformer/vendor")

from default import get_cfg
from dataset_dmd import DMDHeatmapDataset
from augmentation import phase1_train_transform, normalize_transform
from models.VQVAE import VQVAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default=str(ROOT / "artifacts" / "phase1_codebook"))
    p.add_argument("--gt-source", choices=["occface", "mediapipe"], default="mediapipe")
    p.add_argument("--dataset", choices=["DMD", "DMD_68"], default="DMD")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--T_0", type=int, default=5)
    p.add_argument("--T_mult", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-clips", type=int, default=0, help="0 = all")
    p.add_argument("--validEpoch", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-embeddings", type=int, default=2048)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--h-dim", type=int, default=128)
    p.add_argument("--res-h-dim", type=int, default=32)
    p.add_argument("--n-res-layers", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--frame-stride", type=int, default=5,
                   help="frame sub-sampling stride (1=all, 5=20% 사용)")
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--warm-start-ckpt", type=str, default="",
                   help="친구 VQVAE ckpt path (encoder + codebook 만 load, decoder 마지막 conv 는 random)")
    p.add_argument("--lr-codebook", type=float, default=-1.0,
                   help="warm-start 후 codebook 의 lr 분리. -1 이면 lr 동일.")
    return p.parse_args()


def build_dataset(cfg, subset, args, augment=False):
    train_aug = phase1_train_transform() if augment else None
    nm = normalize_transform()
    ds = DMDHeatmapDataset(
        cfg, args.dataset, subset,
        augmentation_transform=train_aug,
        normalize_transform=nm,
        edge_type="ADNet",
        ratio=4,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        mix_prob=0.0,        # codebook 학습은 정상 sample 만
    )
    if args.frame_stride > 1:
        ds.database = ds.database[::args.frame_stride]
    return ds


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = get_cfg()
    cfg.DMD.GT_SOURCE = args.gt_source
    cfg.DMD_68.GT_SOURCE = args.gt_source
    cfg.freeze()
    ds_cfg = cfg[args.dataset]
    print(f"[cfg] dataset={args.dataset} NUM_POINT={ds_cfg.NUM_POINT} NUM_EDGE={ds_cfg.NUM_EDGE} GT={args.gt_source}", flush=True)

    train_ds = build_dataset(cfg, "train", args, augment=True)
    val_ds   = build_dataset(cfg, "test",  args, augment=False)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  (stride={args.frame_stride})", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=max(1, args.workers // 2), pin_memory=True)

    model = VQVAE(
        h_dim=args.h_dim, res_h_dim=args.res_h_dim,
        output_dim=ds_cfg.NUM_EDGE,           # 12 polylines (DMD 478) / 13 (DMD 68)
        n_res_layers=args.n_res_layers,
        n_embeddings=args.n_embeddings,
        embedding_dim=args.embedding_dim,
        code_dim=args.embedding_dim,
        beta=args.beta,
        vit=None,                              # Phase 1: codebook only
    ).to(device)

    # ---- warm-start (B-2) ----
    if args.warm_start_ckpt:
        ck = torch.load(args.warm_start_ckpt, map_location=device, weights_only=False)
        state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        # decoder 의 마지막 conv 만 output channel mismatch (COFW 14 vs DMD 12) — skip
        filtered = {}
        for k, v in state.items():
            if k.startswith("vit."):
                continue                                       # Phase 1 vit=None
            if model.state_dict().get(k) is not None and model.state_dict()[k].shape != v.shape:
                print(f"  skip mismatched key {k}: ckpt={tuple(v.shape)} vs model={tuple(model.state_dict()[k].shape)}", flush=True)
                continue
            filtered[k] = v
        incompat = model.load_state_dict(filtered, strict=False)
        print(f"[warm-start] from {args.warm_start_ckpt}", flush=True)
        print(f"             loaded {len(filtered)} keys, missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}", flush=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] VQVAE params={n_params/1e6:.2f}M", flush=True)

    # optimizer: codebook 의 lr 을 더 작게 (warm-start 보존)
    if args.warm_start_ckpt and args.lr_codebook > 0:
        codebook_params = list(model.vector_quantization.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith("vector_quantization.")]
        optim = torch.optim.Adam([
            {"params": codebook_params, "lr": args.lr_codebook},
            {"params": other_params,    "lr": args.lr},
        ])
        print(f"[optim] codebook lr={args.lr_codebook}, others lr={args.lr}", flush=True)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=args.T_0, T_mult=args.T_mult, eta_min=1e-7,
    )

    metrics_path = save_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_recon", "train_commit", "train_perplexity",
            "val_recon", "val_perplexity", "lr",
        ])

    best_val = float("inf")
    no_improve = 0

    for ep in range(args.epoch):
        # ----- train -----
        model.train()
        sum_recon = sum_commit = sum_perp = n_batch = 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            _, res_in, _, meta, _, _ = batch
            res_in = res_in.to(device, non_blocking=True)
            gt_edge = meta["Edge_Heatmaps"].to(device, non_blocking=True)   # (B, E, 64, 64)

            optim.zero_grad()
            commit, x_hat, perp, *_ = model(res_in)
            recon = F.mse_loss(x_hat, gt_edge)
            loss = recon + commit
            loss.backward()
            optim.step()
            sched.step(ep + step / len(train_loader))

            sum_recon += float(recon.detach())
            sum_commit += float(commit.detach()) if commit is not None else 0.0
            sum_perp += float(perp.detach()) if perp is not None else 0.0
            n_batch += 1

            if step % 50 == 0:
                print(f"  ep{ep} step {step}/{len(train_loader)}  "
                      f"recon={float(recon):.4f}  commit={float(commit):.4f}  "
                      f"perplexity={float(perp):.1f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)

        train_recon = sum_recon / max(n_batch, 1)
        train_commit = sum_commit / max(n_batch, 1)
        train_perp = sum_perp / max(n_batch, 1)
        train_loss = train_recon + train_commit

        # ----- val -----
        val_recon = val_perp = 0.0
        if (ep + 1) % args.validEpoch == 0:
            model.eval()
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    _, res_in, _, meta, _, _ = batch
                    res_in = res_in.to(device, non_blocking=True)
                    gt_edge = meta["Edge_Heatmaps"].to(device, non_blocking=True)
                    _, x_hat, perp, *_ = model(res_in)
                    val_recon += float(F.mse_loss(x_hat, gt_edge))
                    val_perp += float(perp) if perp is not None else 0.0
                    n_val += 1
            val_recon /= max(n_val, 1)
            val_perp /= max(n_val, 1)

        elapsed = time.time() - t0
        lr_now = float(sched.get_last_lr()[0])
        print(f"[ep {ep}] train_recon={train_recon:.4f} train_commit={train_commit:.4f} "
              f"train_perp={train_perp:.1f} | val_recon={val_recon:.4f} val_perp={val_perp:.1f} "
              f"| lr={lr_now:.2e} | {elapsed:.0f}s",
              flush=True)

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, train_loss, train_recon, train_commit, train_perp,
                                    val_recon, val_perp, lr_now])

        # save best
        if val_recon < best_val:
            best_val = val_recon
            no_improve = 0
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "best_val_recon": best_val,
                "args": vars(args),
            }, save_dir / "best.pt")
            print(f"  ★ best val_recon = {best_val:.4f}  → saved best.pt", flush=True)
        else:
            no_improve += 1

        # latest
        torch.save({
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "sched_state_dict": sched.state_dict(),
            "best_val_recon": best_val,
            "args": vars(args),
        }, save_dir / "latest.pt")

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[early stop] no improve for {no_improve} epochs", flush=True)
            break

    print(f"\n=== Phase 1 done. best val_recon = {best_val:.4f} ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 2: ORFormer + Codebook 학습 (vit attached).

Phase 1 의 codebook 을 warm-start 로 사용 (codebook unfreeze).
가린 sample mix (mix_prob 0.5) 로 messenger 학습 + manifest BCE auxiliary.

Loss:
    L = ‖x_hat − edge_GT‖²
      + L_codebook_commitment
      + λ_aux · BCE(α, manifest_GT)        ← 우리 추가
      + λ_img · L_codebook_recon            ← 원논문 (이미 commit 안에 일부 반영)

α (sigmoid attention from ORTransformer) 가 manifest_mask (478,) 의 16×16 pooling 과
align 되도록 보조.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data/shared/orformer/vendor")

from default import get_cfg
from dataset_dmd import DMDHeatmapDataset
from augmentation import phase2_train_transform, normalize_transform
from models.VQVAE import VQVAE
from models.simple_vit import ORFormer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default=str(ROOT / "artifacts" / "phase2_orformer"))
    p.add_argument("--codebook-weights", type=str, required=True,
                   help="Phase 1 best.pt path")
    p.add_argument("--gt-source", choices=["occface", "mediapipe"], default="mediapipe")
    p.add_argument("--dataset", choices=["DMD", "DMD_68"], default="DMD")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-vit", type=float, default=1e-4)
    p.add_argument("--lr-codebook", type=float, default=5e-5)
    p.add_argument("--epoch", type=int, default=150)
    p.add_argument("--T_0", type=int, default=5)
    p.add_argument("--T_mult", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--validEpoch", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mix-prob", type=float, default=0.5)
    p.add_argument("--lambda-aux", type=float, default=0.1,
                   help="manifest BCE auxiliary loss weight")
    p.add_argument("--aux-warmup-epochs", type=int, default=10,
                   help="α BCE warmup — 처음 N epoch 은 weight 0")
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--early-stop-patience", type=int, default=15)
    return p.parse_args()


def manifest_to_patch_16(manifest_478: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """(B, 478) manifest mask → (B, grid*grid) by mean-pooling via canonical patch idx.

    간단 fallback: 478 점을 grid×grid 셀에 평균 분배. (정확도는 patch idx 매핑 보완 필요)
    Phase 2 에서는 일단 placeholder — canonical_idx 가 잇으면 정확한 매핑.
    """
    # placeholder: 478 → 256 (16*16) by reshape-truncate
    B = manifest_478.shape[0]
    target = grid * grid
    if manifest_478.shape[1] >= target:
        return manifest_478[:, :target]
    # zero-pad if smaller (DMD_68 의 경우)
    pad = target - manifest_478.shape[1]
    return torch.cat([manifest_478, torch.ones(B, pad, device=manifest_478.device)], dim=1)


def build_dataset(cfg, subset, args, augment):
    train_aug = phase2_train_transform() if augment else None
    nm = normalize_transform()
    mix = args.mix_prob if augment else 0.0
    ds = DMDHeatmapDataset(
        cfg, args.dataset, subset,
        augmentation_transform=train_aug,
        normalize_transform=nm,
        edge_type="ADNet", ratio=4,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        mix_prob=mix,
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
    print(f"[cfg] dataset={args.dataset} NUM_POINT={ds_cfg.NUM_POINT} NUM_EDGE={ds_cfg.NUM_EDGE} mix_prob={args.mix_prob}", flush=True)

    train_ds = build_dataset(cfg, "train", args, augment=True)
    val_ds   = build_dataset(cfg, "test",  args, augment=False)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  (stride={args.frame_stride})", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=max(1, args.workers // 2), pin_memory=True)

    # ----- model -----
    vit = ORFormer(
        image_size=16, patch_size=1, num_classes=2048,
        dim=256, depth=3, heads=8, mlp_dim=512, channels=256,
    )
    model = VQVAE(
        h_dim=128, res_h_dim=32,
        output_dim=ds_cfg.NUM_EDGE,
        n_res_layers=2,
        n_embeddings=2048, embedding_dim=256, code_dim=256,
        beta=0.25,
        vit=vit,
    ).to(device)

    # Phase 1 weight load (codebook + encoder + decoder)
    ck = torch.load(args.codebook_weights, map_location=device, weights_only=False)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    incompat = model.load_state_dict(state, strict=False)
    print(f"[load] codebook from {args.codebook_weights}", flush=True)
    print(f"       missing={len(incompat.missing_keys)}  unexpected={len(incompat.unexpected_keys)}", flush=True)

    # ----- optimizer (codebook + ORFormer 둘 다 trainable) -----
    vit_params = list(model.vit.parameters())
    code_params = [p for n, p in model.named_parameters() if not n.startswith("vit.")]
    optim = torch.optim.Adam([
        {"params": vit_params, "lr": args.lr_vit},
        {"params": code_params, "lr": args.lr_codebook},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=args.T_0, T_mult=args.T_mult, eta_min=1e-7,
    )

    n_vit = sum(p.numel() for p in vit_params if p.requires_grad)
    n_code = sum(p.numel() for p in code_params if p.requires_grad)
    print(f"[model] vit={n_vit/1e6:.2f}M  codebook(enc+dec+quant)={n_code/1e6:.2f}M", flush=True)

    metrics_path = save_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_recon", "train_commit", "train_aux",
            "train_perp", "val_recon", "val_perp",
            "alpha_normal_mean", "alpha_occluded_mean", "lr_vit", "lr_code",
        ])

    best_val = float("inf")
    no_improve = 0

    for ep in range(args.epoch):
        aux_w = args.lambda_aux if ep >= args.aux_warmup_epochs else 0.0

        model.train()
        sum_recon = sum_commit = sum_aux = sum_perp = n_batch = 0
        alpha_n = alpha_o = n_n = n_o = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            _, res_in, _, meta, _, _ = batch
            res_in = res_in.to(device, non_blocking=True)
            gt_edge = meta["Edge_Heatmaps"].to(device, non_blocking=True)
            manifest = meta["Manifest_Mask"].to(device, non_blocking=True)        # (B, 478) — 1=visible

            optim.zero_grad()
            commit, x_hat, perp, _, _, _, _, OR_portion, attn = model(res_in)
            # OR_portion : (B, 16, 16, 1) → alpha per patch
            alpha = OR_portion.squeeze(-1).view(OR_portion.shape[0], -1)         # (B, 256)

            recon = F.mse_loss(x_hat, gt_edge)
            # 우리 추가: α 가 manifest 와 align (occluded → α↑)
            target = 1.0 - manifest_to_patch_16(manifest, grid=16)               # 가린 patch = 1
            aux = F.binary_cross_entropy(alpha.clamp(1e-6, 1 - 1e-6), target)

            loss = recon + commit + aux_w * aux
            loss.backward()
            optim.step()
            sched.step(ep + step / len(train_loader))

            sum_recon += float(recon.detach())
            sum_commit += float(commit.detach()) if commit is not None else 0.0
            sum_aux += float(aux.detach())
            sum_perp += float(perp.detach()) if perp is not None else 0.0
            n_batch += 1

            # variant 별 α 통계 — DataLoader collate 가 list of strings 로 반환
            variants = meta.get("Variant", [])
            if isinstance(variants, (list, tuple)) and len(variants) > 0:
                for bi, v in enumerate(variants):
                    if v != "normal":
                        alpha_o += float(alpha[bi].mean()); n_o += 1
                    else:
                        alpha_n += float(alpha[bi].mean()); n_n += 1

            if step % 50 == 0:
                print(f"  ep{ep} step {step}/{len(train_loader)}  "
                      f"recon={float(recon):.4f} commit={float(commit):.4f} "
                      f"aux={float(aux):.4f} (w={aux_w}) perp={float(perp):.1f}",
                      flush=True)

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

        a_n = alpha_n / max(n_n, 1)
        a_o = alpha_o / max(n_o, 1)
        elapsed = time.time() - t0
        lrs = [g["lr"] for g in optim.param_groups]
        print(f"[ep {ep}] recon={sum_recon/n_batch:.4f} commit={sum_commit/n_batch:.4f} "
              f"aux={sum_aux/n_batch:.4f} perp={sum_perp/n_batch:.1f} "
              f"| val_recon={val_recon:.4f} | α_normal={a_n:.3f} α_occ={a_o:.3f} "
              f"| lr=[{lrs[0]:.2e},{lrs[1]:.2e}] | {elapsed:.0f}s",
              flush=True)

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, sum_recon/n_batch + sum_commit/n_batch + aux_w*sum_aux/n_batch,
                sum_recon/n_batch, sum_commit/n_batch, sum_aux/n_batch,
                sum_perp/n_batch, val_recon, val_perp,
                a_n, a_o, lrs[0], lrs[1],
            ])

        if val_recon < best_val:
            best_val = val_recon
            no_improve = 0
            torch.save({
                "epoch": ep, "model_state_dict": model.state_dict(),
                "best_val_recon": best_val, "args": vars(args),
            }, save_dir / "best.pt")
            print(f"  ★ best val_recon = {best_val:.4f}", flush=True)
        else:
            no_improve += 1

        torch.save({
            "epoch": ep, "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "sched_state_dict": sched.state_dict(),
            "best_val_recon": best_val, "args": vars(args),
        }, save_dir / "latest.pt")

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[early stop] no improve for {no_improve} epochs", flush=True)
            break

    print(f"\n=== Phase 2 done. best val_recon = {best_val:.4f} ===")


if __name__ == "__main__":
    main()

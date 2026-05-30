#!/usr/bin/env python
"""Phase 3: HGNet + ORFormer joint training.

친구 `train_HGNet_with_ORFormer.py` 의 simplified fork — DMD dataset 만 지원.

학습 흐름:
    face_crop (256×256) → ORFormer + VQVAE codebook
                             ↓
                          reference_heatmap (edge, B, NUM_EDGE, 64, 64)
                             ↓ concat with HGNet pre-feature
                          HGNet (4 stack hourglass)
                             ↓
                          478 (또는 68) landmark coord

Stage A (default): HGNet 만 학습, ORFormer frozen
Stage B (--finetune-orformer): HGNet + ORFormer joint, codebook frozen
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data/shared/orformer/vendor")

from default import get_cfg
from dataset_dmd import DMDHeatmapDataset
from augmentation import phase3_train_transform, normalize_transform
from heatmap_gen import denorm_points
from models.VQVAE import VQVAE
from models.simple_vit import ORFormer
from models.StackedHGNet import IntergrationStackedHGNet


# ---------- NME loss (DMD anchor) ----------
class NME_DMD(nn.Module):
    def __init__(self, anchor_l: int, anchor_r: int):
        super().__init__()
        self.anchor_l = anchor_l
        self.anchor_r = anchor_r

    def forward(self, pred, gt):
        # pred, gt : (B, N, 2) in [-1, 1] (norm_points)
        norm = torch.linalg.vector_norm(gt[:, self.anchor_l, :] - gt[:, self.anchor_r, :], dim=1)
        norm = norm[:, None]
        return torch.mean(torch.linalg.vector_norm(pred - gt, 2, dim=2) / norm, dim=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--orformer-weights", type=str, required=True,
                   help="Phase 2 best.pt path (VQVAE + ORFormer 통합 weight)")
    p.add_argument("--init-hgnet-weights", type=str, default="",
                   help="optional warm start (Stage B 또는 68 model 학습 시)")
    p.add_argument("--dataset", choices=["DMD", "DMD_68"], default="DMD")
    p.add_argument("--gt-source", choices=["occface", "mediapipe"], default="mediapipe")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--orformer-lr", type=float, default=1e-5)
    p.add_argument("--T_0", type=int, default=5)
    p.add_argument("--T_mult", type=int, default=2)
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--nstack", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="heatmap loss weight (NME + alpha·(edge+point) L2)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--validEpoch", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mix-prob", type=float, default=0.5)
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--early-stop-patience", type=int, default=20)

    p.add_argument("--finetune-orformer", action="store_true",
                   help="Stage B — joint fine-tune (default: Stage A, ORFormer frozen)")
    p.add_argument("--orformer-train-scope", choices=["vit", "full"], default="vit",
                   help="Stage B 에서 ORFormer 의 어디까지 unfreeze. vit=ORFormer 만, full=codebook 포함")
    return p.parse_args()


# ---------- model builders ----------
def build_orformer(ds_cfg, device, weights_path):
    """VQVAE + ORFormer (Phase 2 weight) → reference heatmap generator."""
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
    ck = torch.load(weights_path, map_location=device, weights_only=False)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    incompat = model.load_state_dict(state, strict=False)
    print(f"[orformer] loaded from {weights_path}", flush=True)
    print(f"           missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}", flush=True)
    return model


def build_hgnet(ds_cfg, args, device):
    """HGNet 4-stack landmark detector."""
    edge_info = [list(x) for x in ds_cfg.EDGE_INFO]
    model = IntergrationStackedHGNet(
        classes_num=[ds_cfg.NUM_POINT, ds_cfg.NUM_EDGE, ds_cfg.NUM_POINT],
        edge_info=edge_info,
        nstack=args.nstack,
    ).to(device)
    if args.init_hgnet_weights:
        state = torch.load(args.init_hgnet_weights, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"[hgnet] warm start from {args.init_hgnet_weights}", flush=True)
    return model


def configure_orformer_finetune(orformer_model, train_scope: str):
    for p in orformer_model.parameters():
        p.requires_grad = False
    if train_scope == "vit":
        for p in orformer_model.vit.parameters():
            p.requires_grad = True
    elif train_scope == "full":
        for p in orformer_model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(train_scope)


# ---------- dataset ----------
def build_dataset(cfg, subset, args, augment):
    aug = phase3_train_transform() if augment else None
    nm = normalize_transform()
    ds = DMDHeatmapDataset(
        cfg, args.dataset, subset,
        augmentation_transform=aug,
        normalize_transform=nm,
        edge_type="ADNet", ratio=4,
        max_clips=args.max_clips if args.max_clips > 0 else None,
        mix_prob=args.mix_prob if augment else 0.0,
    )
    if args.frame_stride > 1:
        ds.database = ds.database[::args.frame_stride]
    return ds


# ---------- main ----------
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
    print(f"[cfg] dataset={args.dataset} NUM_POINT={ds_cfg.NUM_POINT} NUM_EDGE={ds_cfg.NUM_EDGE} "
          f"stage={'B (joint)' if args.finetune_orformer else 'A (HGNet only)'}", flush=True)

    train_ds = build_dataset(cfg, "train", args, augment=True)
    val_ds   = build_dataset(cfg, "test",  args, augment=False)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  (stride={args.frame_stride})", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              prefetch_factor=args.prefetch_factor, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=max(1, args.workers // 2), pin_memory=True,
                              persistent_workers=True)

    # ----- models -----
    orformer = build_orformer(ds_cfg, device, args.orformer_weights)
    hgnet    = build_hgnet(ds_cfg, args, device)

    if args.finetune_orformer:
        configure_orformer_finetune(orformer, args.orformer_train_scope)
        orformer.train()
    else:
        for p in orformer.parameters():
            p.requires_grad = False
        orformer.eval()

    # ----- optimizer -----
    param_groups = [{"params": hgnet.parameters(), "lr": args.lr}]
    if args.finetune_orformer:
        trainable_or = [p for p in orformer.parameters() if p.requires_grad]
        if not trainable_or:
            raise RuntimeError("--finetune-orformer requested but no trainable ORFormer params")
        param_groups.append({"params": trainable_or, "lr": args.orformer_lr})

    optim = torch.optim.Adam(param_groups, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=args.T_0, T_mult=args.T_mult, eta_min=1e-7,
    )

    anchor_l, anchor_r = ds_cfg.NME_ANCHOR
    criterion = NME_DMD(anchor_l, anchor_r).to(device)

    n_hg = sum(p.numel() for p in hgnet.parameters() if p.requires_grad)
    n_or = sum(p.numel() for p in orformer.parameters() if p.requires_grad)
    print(f"[model] hgnet={n_hg/1e6:.2f}M  orformer_trainable={n_or/1e6:.2f}M", flush=True)

    metrics_path = save_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_nme", "train_edge", "train_point",
            "val_nme", "lr", "elapsed_s",
        ])

    best_nme = float("inf")
    no_improve = 0

    for ep in range(args.epoch):
        hgnet.train()
        if args.finetune_orformer:
            orformer.train()
        else:
            orformer.eval()

        sum_nme = sum_edge = sum_point = n_batch = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_t, res_in, _, meta, _, _ = batch
            input_t = input_t.to(device, non_blocking=True)
            res_in  = res_in.to(device, non_blocking=True)
            gt_edge  = meta["Edge_Heatmaps"].to(device, non_blocking=True)
            gt_point = meta["Point_Heatmaps"].to(device, non_blocking=True)
            gt_lm    = meta["Landmarks"].to(device, non_blocking=True)        # (B, N, 2) [-1, 1]

            optim.zero_grad()

            # ORFormer reference heatmap
            if args.finetune_orformer:
                _, ref_heatmap, *_ = orformer(res_in)
            else:
                with torch.no_grad():
                    _, ref_heatmap, *_ = orformer(res_in)

            # HGNet — input + reference (concat in HGNet 의 conv)
            y, landmarks = hgnet(input_t, reference_heatmaps=ref_heatmap)

            total_loss = 0
            for si in range(args.nstack):
                pred_lm = y[3 * si]                         # (B, N, 2) [-1, 1]
                pred_e  = y[3 * si + 1]
                pred_p  = y[3 * si + 2]

                nme_loss = criterion(pred_lm, gt_lm).sum()
                e_loss = torch.mean((pred_e - gt_edge) ** 2, dim=1).sum()
                p_loss = torch.mean((pred_p - gt_point) ** 2, dim=1).sum()

                sum_nme   += float(nme_loss.detach()) / args.nstack
                sum_edge  += float(e_loss.detach())   / args.nstack
                sum_point += float(p_loss.detach())   / args.nstack

                total_loss = total_loss + nme_loss + args.alpha * (e_loss + p_loss)

            total_loss.backward()
            optim.step()
            sched.step(ep + step / len(train_loader))
            n_batch += input_t.shape[0]

            if step % 50 == 0:
                lrs = [g["lr"] for g in optim.param_groups]
                print(f"  ep{ep} step {step}/{len(train_loader)}  "
                      f"nme={float(nme_loss)/input_t.shape[0]:.4f} "
                      f"edge={float(e_loss)/input_t.shape[0]:.4f} "
                      f"point={float(p_loss)/input_t.shape[0]:.4f} "
                      f"lr={lrs[0]:.2e}", flush=True)

        train_nme   = sum_nme   / max(n_batch, 1) * 100.0
        train_edge  = sum_edge  / max(n_batch, 1)
        train_point = sum_point / max(n_batch, 1)

        # ----- val -----
        val_nme = 0.0
        if (ep + 1) % args.validEpoch == 0:
            hgnet.eval()
            orformer.eval()
            errs = []
            with torch.no_grad():
                for batch in val_loader:
                    input_t, res_in, _, meta, _, _ = batch
                    input_t = input_t.to(device, non_blocking=True)
                    res_in  = res_in.to(device, non_blocking=True)
                    gt_lm   = meta["Landmarks"].to(device, non_blocking=True)
                    _, ref_heatmap, *_ = orformer(res_in)
                    _, landmarks = hgnet(input_t, reference_heatmaps=ref_heatmap)
                    err = criterion(landmarks, gt_lm)
                    errs.append(float(err.mean()))
            val_nme = float(np.mean(errs)) * 100.0 if errs else 0.0

        elapsed = time.time() - t0
        lrs = [g["lr"] for g in optim.param_groups]
        print(f"[ep {ep}] train_nme={train_nme:.4f} train_edge={train_edge:.4f} "
              f"train_point={train_point:.4f} | val_nme={val_nme:.4f} "
              f"| lr={lrs[0]:.2e} | {elapsed:.0f}s", flush=True)

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, train_nme, train_edge, train_point, val_nme, lrs[0], elapsed,
            ])

        # save best
        if val_nme < best_nme:
            best_nme = val_nme
            no_improve = 0
            torch.save({
                "epoch": ep,
                "hgnet_state_dict": hgnet.state_dict(),
                "orformer_state_dict": orformer.state_dict() if args.finetune_orformer else None,
                "best_nme": best_nme,
                "args": vars(args),
            }, save_dir / "best.pt")
            print(f"  ★ best val_nme = {best_nme:.4f}", flush=True)
        else:
            no_improve += 1

        torch.save({
            "epoch": ep,
            "hgnet_state_dict": hgnet.state_dict(),
            "orformer_state_dict": orformer.state_dict() if args.finetune_orformer else None,
            "optim_state_dict": optim.state_dict(),
            "sched_state_dict": sched.state_dict(),
            "best_nme": best_nme,
            "args": vars(args),
        }, save_dir / "latest.pt")

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[early stop] no improve for {no_improve} epochs", flush=True)
            break

    print(f"\n=== Phase 3 done. best val_nme = {best_nme:.4f} ===")


if __name__ == "__main__":
    main()

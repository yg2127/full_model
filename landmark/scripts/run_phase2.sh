#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yg/fusion/pretrain_v4"
SAVE_DIR="$ROOT/artifacts/phase2_orformer"
# Phase 2 D 의 ep0 best (α discrimination 성공) 를 warm-start 로 사용
CODEBOOK="$ROOT/artifacts/phase2_orformer_d_ep01_backup/best.pt"
mkdir -p "$SAVE_DIR"

export PYTHONPATH="/data/shared/orformer/vendor:${PYTHONPATH:-}"

exec /data/shared/envs/scuppy/bin/python -u "$ROOT/scripts/train_phase2_orformer.py" \
    --save-dir "$SAVE_DIR" \
    --codebook-weights "$CODEBOOK" \
    --gt-source mediapipe \
    --dataset DMD \
    --batch-size 64 \
    --lr-vit 1e-4 \
    --lr-codebook 5e-5 \
    --epoch 150 \
    --T_0 5 --T_mult 2 \
    --workers 12 \
    --frame-stride 5 \
    --mix-prob 0.5 \
    --lambda-aux 0.5 \
    --aux-warmup-epochs 0 \
    --validEpoch 1 \
    --early-stop-patience 15 \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/train.log"

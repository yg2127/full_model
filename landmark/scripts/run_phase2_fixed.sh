#!/usr/bin/env bash
set -euo pipefail

# Phase 2 FIXED — codebook + decoder frozen (lr=0), ORFormer messenger 만 학습.
# Root cause: D 학습이 codebook 을 1-entry collapse 시켜 decoder 출력 saturate (max 0.0002).
# Phase 1 best.pt (perp 40, 224 active entries) 부터 다시 시작.

ROOT="/home/yg/fusion/pretrain_v4"
SAVE_DIR="$ROOT/artifacts/phase2_orformer_fixed"
# 핵심: Phase 1 best (살아있는 codebook+decoder) 에서 warm-start
CODEBOOK="$ROOT/artifacts/phase1_codebook/best.pt"
mkdir -p "$SAVE_DIR"

export PYTHONPATH="/data/shared/orformer/vendor:${PYTHONPATH:-}"

exec /data/shared/envs/scuppy/bin/python -u "$ROOT/scripts/train_phase2_orformer.py" \
    --save-dir "$SAVE_DIR" \
    --codebook-weights "$CODEBOOK" \
    --gt-source mediapipe \
    --dataset DMD \
    --batch-size 64 \
    --lr-vit 1e-4 \
    --lr-codebook 0 \
    --epoch 30 \
    --T_0 5 --T_mult 2 \
    --workers 12 \
    --frame-stride 5 \
    --mix-prob 0.5 \
    --lambda-aux 0.5 \
    --aux-warmup-epochs 0 \
    --validEpoch 1 \
    --early-stop-patience 10 \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/train.log"

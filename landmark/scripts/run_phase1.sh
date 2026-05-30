#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yg/fusion/pretrain_v4"
SAVE_DIR="$ROOT/artifacts/phase1_codebook"
mkdir -p "$SAVE_DIR"

export PYTHONPATH="/data/shared/orformer/vendor:${PYTHONPATH:-}"

FRIEND_CKPT="/data/shared/orformer/capstone/artifacts/orformer_runs/orformer/COFW/cofw_full_cofw_orformer_lr0.0001_T05_Tmult2_epoch300_batch64_alpha50/best_model.pt"

exec /data/shared/envs/scuppy/bin/python -u "$ROOT/scripts/train_phase1_codebook.py" \
    --save-dir "$SAVE_DIR" \
    --gt-source mediapipe \
    --dataset DMD \
    --batch-size 64 \
    --lr 1e-4 \
    --lr-codebook 1e-6 \
    --epoch 30 \
    --T_0 5 --T_mult 2 \
    --workers 12 \
    --frame-stride 5 \
    --validEpoch 1 \
    --early-stop-patience 8 \
    --beta 0.25 \
    --warm-start-ckpt "$FRIEND_CKPT" \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/train.log"

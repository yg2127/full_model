#!/usr/bin/env bash
# Phase 3 Stage A — HGNet 478 학습. ORFormer frozen.
# Stage B 는 같은 script 를 --finetune-orformer 와 함께 호출 (init-hgnet-weights 는 Stage A best).
set -euo pipefail

ROOT="/home/yg/fusion/pretrain_v4"
SAVE_DIR="$ROOT/artifacts/phase3a_hgnet_478"
ORFORMER_CKPT="$ROOT/artifacts/phase2_orformer/best.pt"
mkdir -p "$SAVE_DIR"

export PYTHONPATH="/data/shared/orformer/vendor:${PYTHONPATH:-}"

HGNET_WARMSTART="$ROOT/artifacts/phase3a_backup/hgnet_ep0.pt"

exec /data/shared/envs/scuppy/bin/python -u "$ROOT/scripts/train_phase3_hgnet.py" \
    --save-dir "$SAVE_DIR" \
    --orformer-weights "$ORFORMER_CKPT" \
    --init-hgnet-weights "$HGNET_WARMSTART" \
    --dataset DMD \
    --gt-source mediapipe \
    --batch-size 16 \
    --lr 1e-3 \
    --epoch 200 \
    --T_0 5 --T_mult 2 \
    --nstack 4 \
    --alpha 0.05 \
    --workers 12 \
    --frame-stride 3 \
    --mix-prob 0.5 \
    --validEpoch 1 \
    --early-stop-patience 20 \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/train.log"

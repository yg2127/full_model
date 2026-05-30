#!/usr/bin/env bash
set -euo pipefail
# Phase 3a v2 — Phase 2 FIXED 의 살아있는 ORFormer + 이전 HGNet ep3 warm-start
ROOT="/home/yg/fusion/pretrain_v4"
SAVE_DIR="$ROOT/artifacts/phase3a_hgnet_478_v2"
mkdir -p "$SAVE_DIR"
export PYTHONPATH="/data/shared/orformer/vendor:${PYTHONPATH:-}"

exec /data/shared/envs/scuppy/bin/python -u "$ROOT/scripts/train_phase3_hgnet.py" \
    --save-dir "$SAVE_DIR" \
    --orformer-weights "$ROOT/artifacts/phase2_orformer_fixed/best.pt" \
    --init-hgnet-weights "$ROOT/artifacts/phase3a_hgnet_478/hgnet_ep3_for_warmstart.pt" \
    --dataset DMD --gt-source mediapipe \
    --batch-size 16 --lr 1e-3 \
    --epoch 100 --T_0 5 --T_mult 2 \
    --nstack 4 --alpha 0.05 \
    --workers 8 --frame-stride 3 --mix-prob 0.5 \
    --validEpoch 1 --early-stop-patience 15 \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/train.log"

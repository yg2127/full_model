#!/usr/bin/env bash
set -euo pipefail
# Phase 3a v3 — workers 16, batch 32, prefetch 4, lr √2 scaling
# 친구들 학습 멈춘 후 launch. 이전 v2 의 ep0 best 에서 warm-start.
ROOT="/home/yg/fusion/pretrain_v4"
SAVE_DIR="$ROOT/artifacts/phase3a_hgnet_478_v3"
mkdir -p "$SAVE_DIR"
export PYTHONPATH="/data/shared/orformer/vendor:${PYTHONPATH:-}"

# v2 ep0 best 의 hgnet_state_dict 만 추출
WARMSTART="$ROOT/artifacts/phase3a_hgnet_478_v2/hgnet_ep0_for_warmstart.pt"
if [ ! -f "$WARMSTART" ]; then
    /data/shared/envs/scuppy/bin/python -c "
import torch
ck = torch.load('$ROOT/artifacts/phase3a_hgnet_478_v2/best.pt', map_location='cpu', weights_only=False)
torch.save(ck['hgnet_state_dict'], '$WARMSTART')
print('extracted hgnet_ep0_for_warmstart.pt (val_nme', ck['best_nme'], ')')
"
fi

exec /data/shared/envs/scuppy/bin/python -u "$ROOT/scripts/train_phase3_hgnet.py" \
    --save-dir "$SAVE_DIR" \
    --orformer-weights "$ROOT/artifacts/phase2_orformer_fixed/best.pt" \
    --init-hgnet-weights "$WARMSTART" \
    --dataset DMD --gt-source mediapipe \
    --batch-size 32 --lr 1.4e-3 \
    --epoch 100 --T_0 5 --T_mult 2 \
    --nstack 4 --alpha 0.05 \
    --workers 16 --prefetch-factor 4 \
    --frame-stride 3 --mix-prob 0.5 \
    --validEpoch 1 --early-stop-patience 15 \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/train.log"

#!/usr/bin/env bash
# Phase 3 Stage B — HGNet 478 + ORFormer joint fine-tune.
# Stage A best 의 hgnet 으로 warm start, codebook frozen 상태에서 ViT 도 함께 fine-tune.
set -euo pipefail

ROOT="/home/yg/fusion/pretrain_v4"
SAVE_DIR="$ROOT/artifacts/phase3a_hgnet_478_jointFT"
ORFORMER_CKPT="$ROOT/artifacts/phase2_orformer/best.pt"
HGNET_INIT="$ROOT/artifacts/phase3a_hgnet_478/best.pt"
mkdir -p "$SAVE_DIR"

export PYTHONPATH="/data/shared/orformer/vendor:${PYTHONPATH:-}"

# init-hgnet-weights 는 .pt 파일 안의 hgnet_state_dict 만 추출해서 별도 저장이 필요할 수 있음.
# train_phase3_hgnet.py 의 load 가 model_state_dict 만 시도하므로, Stage A best.pt 를 그대로 넣음.
# (script 가 model_state_dict 키 없으면 state 그대로 시도 — Stage A best 는 hgnet_state_dict 키라 별도 추출 필요)
# 단순화: 학습 시작 전 추출
PURE_HGNET="$SAVE_DIR/hgnet_from_phase3a.pt"
/data/shared/envs/scuppy/bin/python -c "
import torch
ck = torch.load('$HGNET_INIT', map_location='cpu', weights_only=False)
torch.save(ck['hgnet_state_dict'], '$PURE_HGNET')
print(f'extracted hgnet_state_dict → $PURE_HGNET')
"

exec /data/shared/envs/scuppy/bin/python -u "$ROOT/scripts/train_phase3_hgnet.py" \
    --save-dir "$SAVE_DIR" \
    --orformer-weights "$ORFORMER_CKPT" \
    --init-hgnet-weights "$PURE_HGNET" \
    --dataset DMD \
    --gt-source mediapipe \
    --batch-size 12 \
    --lr 2e-4 \
    --orformer-lr 1e-5 \
    --epoch 150 \
    --T_0 5 --T_mult 2 \
    --nstack 4 \
    --alpha 0.05 \
    --workers 4 \
    --frame-stride 3 \
    --mix-prob 0.5 \
    --validEpoch 1 \
    --early-stop-patience 20 \
    --finetune-orformer \
    --orformer-train-scope vit \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/train.log"

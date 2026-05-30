#!/usr/bin/env bash
# Phase 3 Stage B — HGNet 68 학습. ORFormer + codebook 은 Stage A (478) 의 best 그대로 사용.
# 두 model 은 ORFormer/codebook share (face encoder 동일), HGNet head 만 다름.
set -euo pipefail

ROOT="/home/yg/fusion/pretrain_v4"
SAVE_DIR="$ROOT/artifacts/phase3b_hgnet_68"
# 478 stage A 의 best 또는 jointFT best (jointFT 있으면 그걸 우선)
ORFORMER_CKPT="$ROOT/artifacts/phase3a_hgnet_478_jointFT/best.pt"
if [ ! -f "$ORFORMER_CKPT" ]; then
    ORFORMER_CKPT="$ROOT/artifacts/phase2_orformer/best.pt"
fi
mkdir -p "$SAVE_DIR"

export PYTHONPATH="/data/shared/orformer/vendor:${PYTHONPATH:-}"

# ORFormer ckpt 가 hgnet+orformer 통합이면 orformer 부분만 추출
PURE_ORFORMER="$SAVE_DIR/orformer_from_phase3a.pt"
/data/shared/envs/scuppy/bin/python -c "
import torch
ck = torch.load('$ORFORMER_CKPT', map_location='cpu', weights_only=False)
if 'orformer_state_dict' in ck and ck['orformer_state_dict'] is not None:
    torch.save({'model_state_dict': ck['orformer_state_dict']}, '$PURE_ORFORMER')
    print('extracted orformer_state_dict (joint FT) → $PURE_ORFORMER')
else:
    # Phase 2 best.pt 라 model_state_dict 가 이미 orformer 통합
    torch.save(ck, '$PURE_ORFORMER')
    print('used phase2 best directly → $PURE_ORFORMER')
"

exec /data/shared/envs/scuppy/bin/python -u "$ROOT/scripts/train_phase3_hgnet.py" \
    --save-dir "$SAVE_DIR" \
    --orformer-weights "$PURE_ORFORMER" \
    --dataset DMD_68 \
    --gt-source mediapipe \
    --batch-size 16 \
    --lr 1e-3 \
    --epoch 150 \
    --T_0 5 --T_mult 2 \
    --nstack 4 \
    --alpha 0.05 \
    --workers 4 \
    --frame-stride 3 \
    --mix-prob 0.5 \
    --validEpoch 1 \
    --early-stop-patience 20 \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/train.log"

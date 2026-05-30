"""DMD pretrain augmentation transforms.

친구 코드의 torchlm LandmarksCompose 패턴을 따라가되, DMD 의 IR 1ch 특성에 맞춤:
  - RandomGrayscale 제거 (이미 grayscale)
  - RandomMask (occlusion augment) → ORFormer 학습용
  - Rotate / Translate / Scale → 일반 geometric

Phase 1 (codebook): mild augment (가린 patch X)
Phase 2 (ORFormer): RandomMask 추가
Phase 3 (HGNet):    Phase 2 와 동일
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

_VENDOR = "/data/shared/orformer/vendor"
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import torchlm
import torchvision.transforms as T


def phase1_train_transform(image_size: int = 256):
    """Codebook 학습용 — geometric only, 가림 augment X. 끝에 size 강제."""
    return torchlm.LandmarksCompose([
        torchlm.LandmarksRandomRotate(angle=(-15, 15), prob=0.5),
        torchlm.LandmarksRandomTranslate(translate=(-0.04, 0.04), prob=0.5),
        torchlm.LandmarksRandomScale(scale=(-0.05, 0.05), prob=0.5),
        torchlm.LandmarksResize((image_size, image_size), keep_aspect=True),
    ])


def phase2_train_transform(image_size: int = 256):
    """ORFormer 학습용 — geometric + random mask occlusion."""
    return torchlm.LandmarksCompose([
        torchlm.LandmarksRandomMask(mask_ratio=0.2, prob=0.5),
        torchlm.LandmarksRandomRotate(angle=(-20, 20), prob=0.5),
        torchlm.LandmarksRandomTranslate(translate=(-0.04, 0.04), prob=0.5),
        torchlm.LandmarksRandomScale(scale=(-0.05, 0.05), prob=0.5),
        torchlm.LandmarksResize((image_size, image_size), keep_aspect=True),
    ])


def phase3_train_transform():
    """HGNet joint — Phase 2 와 동일."""
    return phase2_train_transform()


def normalize_transform():
    """ImageNet normalize (친구 코드와 호환). 입력은 (H, W, 3) uint8."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == "__main__":
    import numpy as np
    aug = phase1_train_transform()
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    pts = np.random.rand(478, 2) * 256
    out_img, out_pts = aug(img, pts)
    print(f"aug: img {out_img.shape} dtype={out_img.dtype}, pts {out_pts.shape}")

    nm = normalize_transform()
    t = nm(img)
    print(f"normalize: {t.shape} {t.dtype} range {t.min():.2f}~{t.max():.2f}")

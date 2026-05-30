"""Pretrain V4 의 yacs config. 친구 _C.{WFLW,COFW,300W} 패턴 확장.

`_C.DMD`     — 478 FaceMesh landmark 사용 (default)
`_C.DMD_68`  — 300W 68-point subset (478 → 68 mapping 후 학습)
"""
import os
import sys
from pathlib import Path

# 친구 vendor 의 yacs 사용 (scuppy env 에 yacs 미설치).
_VENDOR = "/data/shared/orformer/vendor"
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from yacs.config import CfgNode as CN

# face_edge_info 가 같은 src/data/ 안. import path setup.
_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR.parent / "src"
sys.path.insert(0, str(_SRC))
from data.face_edge_info import (  # noqa: E402
    build_edge_info_478,
    build_edge_info_68,
    MP478_TO_300W68,
    NME_ANCHOR_478,
    NME_ANCHOR_68,
)


_C = CN()

_C.MODEL = CN()
_C.MODEL.IMG_SIZE = 256

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.HYPERPARAMETERS = CN()


# ---------- DMD (478 FaceMesh landmark) ----------
_C.DMD = CN()
_C.DMD.ROOT = os.environ.get(
    "DMD_FACE_CROP_ROOT",
    "/data/shared/DMD_landmarks/face_crops_112",
)
_C.DMD.ROOT_OCCLUDED = os.environ.get(
    "DMD_FACE_CROP_OCCLUDED_ROOT",
    "/data/shared/DMD_landmarks/face_crops_112_occluded",
)
_C.DMD.FACEMESH_ROOT = os.environ.get(
    "DMD_FACEMESH_ROOT",
    "/data/shared/DMD_landmarks/facemesh",
)
# GT 출처 — "occface" (canonical normalize 필요, V3 와 호환) 또는 "mediapipe" (face_crop 좌표계 native)
_C.DMD.GT_SOURCE = os.environ.get("DMD_GT_SOURCE", "occface")
_C.DMD.MEDIAPIPE_CACHE_ROOT = os.environ.get(
    "DMD_MP_CACHE",
    "/data/shared/DMD_landmarks/face_crops_112_facemesh",
)
_C.DMD.MANIFEST_PATH = os.environ.get(
    "DMD_OCC_MANIFEST",
    "/data/shared/DMD_landmarks/facemesh_occluded_subsets_v3/occlusion_manifest_full.jsonl",
)
_C.DMD.VARIANT_MASKED_INDICES = os.environ.get(
    "DMD_VARIANT_MASKED",
    "/data/shared/scuppy/Classification_model_V1/constants/variant_masked_indices.json",
)

_C.DMD.NUM_POINT = 478
_C.DMD.FRACTION = 1.2
_C.DMD.EDGE_INFO = [list(x) for x in build_edge_info_478()]   # yacs 호환 (tuple → list)
_C.DMD.NUM_EDGE = len(_C.DMD.EDGE_INFO)

# 478 점은 좌우 대칭이 명확. flip_mapping 은 향후 augmentation 용 — 일단 빈 list (flip 끄고).
_C.DMD.FLIP_MAPPING: list = []

_C.DMD.SCALE = 0.05
_C.DMD.ROTATION = 15
_C.DMD.TRANSLATION = 0.05
_C.DMD.OCCLUSION_MEAN = 0.20
_C.DMD.OCCLUSION_STD  = 0.08
_C.DMD.DATA_FORMAT = "L"           # IR 1-channel
_C.DMD.FLIP = False                # left/right asymmetry 위험 — 일단 끔
_C.DMD.CHANNEL_TRANSFER = False    # IR 이라 의미 없음
_C.DMD.OCCLUSION = True            # 6 variant + random occlusion augment

_C.DMD.NME_ANCHOR = list(NME_ANCHOR_478)    # (33, 263) outer eye corners
_C.DMD.MIX_PROB = 0.5              # 가린 sample 비율 (training 시)
_C.DMD.VARIANTS = [
    "sunglasses_both_100",
    "sunglasses_left_100",
    "sunglasses_right_100",
    "lower_face_without_nose_100",
    "left_face_half_100",
    "right_face_half_100",
]


# ---------- DMD_68 (300W subset) ----------
# 478 FaceMesh GT 를 MP478_TO_300W68 mapping 으로 변환해서 학습.
_C.DMD_68 = _C.DMD.clone()
_C.DMD_68.NUM_POINT = 68
_C.DMD_68.EDGE_INFO = [list(x) for x in build_edge_info_68()]
_C.DMD_68.NUM_EDGE = len(_C.DMD_68.EDGE_INFO)
_C.DMD_68.NME_ANCHOR = list(NME_ANCHOR_68)   # (36, 45)
# subset mapping — dataloader 가 이걸 보고 478 → 68 idx 추출.
_C.DMD_68.SUBSET_MAPPING = list(MP478_TO_300W68)


# ---------- 친구 dataset config (호환용, 그대로 fork) ----------
# WFLW / COFW / 300W 는 친구 코드의 default.py 와 동일. 우리는 안 쓰지만 import 호환을 위해 dummy 정의.

_C.WFLW = CN()
_C.WFLW.NUM_POINT = 98
_C.WFLW.NUM_EDGE = 15
_C.WFLW.ROOT = ""

_C.W300 = CN()
_C.W300.NUM_POINT = 68
_C.W300.NUM_EDGE = 13
_C.W300.ROOT = ""

_C.COFW = CN()
_C.COFW.NUM_POINT = 29
_C.COFW.NUM_EDGE = 14
_C.COFW.ROOT = ""


def get_cfg():
    return _C.clone()


cfg = _C


if __name__ == "__main__":
    print("[DMD 478]")
    print(f"  NUM_POINT = {_C.DMD.NUM_POINT}")
    print(f"  NUM_EDGE  = {_C.DMD.NUM_EDGE}")
    print(f"  ROOT      = {_C.DMD.ROOT}")
    print(f"  variants  = {len(_C.DMD.VARIANTS)}")
    print(f"  NME_ANCHOR= {_C.DMD.NME_ANCHOR}")
    print()
    print("[DMD 68]")
    print(f"  NUM_POINT = {_C.DMD_68.NUM_POINT}")
    print(f"  NUM_EDGE  = {_C.DMD_68.NUM_EDGE}")
    print(f"  subset    = {len(_C.DMD_68.SUBSET_MAPPING)} indices")

"""DMD_heatmap_Dataset — 친구 `COFW_heatmap_Dataset` 패턴을 DMD IR + 6 variant 가림에 맞춤.

friend 의 dataloader 출력 형식 (input/resized_input/resized_occluded_input/meta/image/resized_image)
을 그대로 따라가서 친구 `train_HGNet_with_ORFormer.py` 와 호환.

GT source (현재 — Phase 0):
    face_crops_112_occface/.../*.npz 의 `landmarks_xy` (canonical normalize → face_crop coord)
    TODO: 정확도 검증 후 face_crops_112 위에서 mediapipe FaceMesh recompute 할지 결정.

가린 sample:
    face_crops_112_occluded/{variant}/.../*.npz 의 이미지 (6 variant zero-paint)
    landmark GT 는 정상 영상 좌표를 그대로 사용 (messenger 학습 압력).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from face_edge_info import MP478_TO_300W68  # noqa: E402
from heatmap_gen import generate_edgemap, generate_pointmap, norm_points  # noqa: E402


# ---------- canonical normalize ----------
_GIM = Path("/data/shared/scuppy/Gaze_image_model")
_CANONICAL = np.load(_GIM / "constants/canonical_face_mean.npz")
CANON_LO = _CANONICAL["affine_lo"].astype(np.float32)   # (2,)
CANON_HI = _CANONICAL["affine_hi"].astype(np.float32)   # (2,)


# ---------- path helpers ----------
def _rel_from_normal(face_npz: Path, root_normal: Path) -> Path:
    return face_npz.relative_to(root_normal)


def _occface_path(rel: Path, root_occface: Path) -> Path:
    return root_occface / str(rel).replace("_crops112.npz", "_occface.npz")


def _occluded_path(rel: Path, root_occluded: Path, variant: str) -> Path:
    return root_occluded / variant / rel


def _occface_occluded_path(rel: Path, variant: str) -> Path:
    base = Path("/data/shared/DMD_landmarks/face_crops_112_occface_occluded")
    return base / variant / str(rel).replace("_crops112.npz", "_occface.npz")


# ---------- landmark coord conversion ----------
def _occface_to_face_crop_coord(occface_lm_xy: np.ndarray, crop_size: int) -> np.ndarray:
    """occface cache 의 normalized (lo, hi) coord → face_crop 좌표 (0 ~ crop_size)."""
    norm = (occface_lm_xy - CANON_LO) / (CANON_HI - CANON_LO + 1e-6)
    return norm * crop_size


class DMDHeatmapDataset(Dataset):
    """Friend `COFW_heatmap_Dataset` 패턴.

    Args:
        cfg: yacs CfgNode — `cfg.DMD` 또는 `cfg.DMD_68` 사용.
        subset: "train" 또는 "test" — split 은 cfg.DMD.SPLIT_FILE 따라 (현재 미구현, 전체 사용).
        augmentation_transform: torchlm LandmarksCompose 또는 None
        normalize_transform: torchvision Compose (ToTensor + Normalize)
        edge_type: "ADNet" 또는 "VQVAE" (heatmap 스타일)
        ratio: input → resized_input downsample ratio (friend 의 4 = 256/64)
        max_clips: debugging 용 — None 이면 전체
    """

    def __init__(
        self,
        cfg,
        dataset_cfg_name: str = "DMD",       # "DMD" 또는 "DMD_68"
        subset: str = "train",
        augmentation_transform=None,
        normalize_transform=None,
        edge_type: str = "ADNet",
        ratio: int = 4,
        max_clips: Optional[int] = None,
        mix_prob: Optional[float] = None,
        rng_seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.ds_cfg = cfg[dataset_cfg_name]              # CfgNode (DMD 또는 DMD_68)
        self.dataset_name = dataset_cfg_name
        self.subset = subset
        self.augmentation_transform = augmentation_transform
        self.normalize_transform = normalize_transform
        self.edge_type = edge_type
        self.ratio = ratio
        self.Image_size = cfg.MODEL.IMG_SIZE              # 256
        self.crop_size = 112                              # face_crop_112 원본
        self.number_landmarks = self.ds_cfg.NUM_POINT     # 478 or 68
        self.Fraction = self.ds_cfg.FRACTION
        self.edge_info = [tuple(x) for x in self.ds_cfg.EDGE_INFO]
        self.variants = list(self.ds_cfg.VARIANTS)
        self.mix_prob = mix_prob if mix_prob is not None else self.ds_cfg.MIX_PROB
        self.rng = np.random.default_rng(rng_seed)

        self.root_normal = Path(self.ds_cfg.ROOT)
        self.root_occluded = Path(self.ds_cfg.ROOT_OCCLUDED)
        self.root_occface = Path("/data/shared/DMD_landmarks/face_crops_112_occface")
        self.root_mediapipe = Path(self.ds_cfg.MEDIAPIPE_CACHE_ROOT)
        self.gt_source = self.ds_cfg.GT_SOURCE.lower()
        assert self.gt_source in ("occface", "mediapipe"), self.gt_source

        # variant → masked landmark indices (478 기준)
        with open(self.ds_cfg.VARIANT_MASKED_INDICES) as f:
            self.variant_masked: Dict[str, List[int]] = json.load(f)

        # subset 478 → 68 mapping (DMD_68 일 때만 사용)
        self.subset_mapping: Optional[List[int]] = None
        if dataset_cfg_name == "DMD_68":
            self.subset_mapping = list(self.ds_cfg.SUBSET_MAPPING)
            assert len(self.subset_mapping) == 68

        self.database: List[Dict[str, Any]] = self._build_database(max_clips=max_clips)

    # ------------------------------------------------------------------
    def _build_database(self, max_clips: Optional[int] = None) -> List[Dict[str, Any]]:
        """모든 clip 의 valid (detected) frame 을 enumerate."""
        clips = sorted(self.root_normal.rglob("*_crops112.npz"))
        if max_clips is not None:
            clips = clips[:max_clips]

        # train/test naive split — clip 단위 8:2
        n = len(clips)
        if self.subset == "train":
            clips = clips[: int(n * 0.8)]
        elif self.subset == "test":
            clips = clips[int(n * 0.8):]
        elif self.subset == "all":
            pass
        else:
            raise ValueError(f"unknown subset: {self.subset}")

        db = []
        for clip_path in clips:
            rel = _rel_from_normal(clip_path, self.root_normal)
            # GT cache 존재 확인 (출처에 따라)
            if self.gt_source == "mediapipe":
                gt_cache = self.root_mediapipe / str(rel).replace("_crops112.npz", "_facemesh.npz")
            else:
                gt_cache = _occface_path(rel, self.root_occface)
            if not gt_cache.exists():
                continue
            # detected frame 만 (yolo + GT cache 의 detected 교집합)
            with np.load(clip_path, allow_pickle=True) as d:
                yolo_det = d["detected"].astype(bool)
            with np.load(gt_cache, allow_pickle=True) as d:
                gt_det = d["detected"].astype(bool) if "detected" in d.files else yolo_det
            valid_mask = yolo_det & gt_det[:len(yolo_det)] if len(gt_det) >= len(yolo_det) else yolo_det
            valid = np.where(valid_mask)[0]
            for fi in valid:
                db.append({"rel": str(rel), "frame": int(fi)})
        return db

    def __len__(self) -> int:
        return len(self.database)

    # ------------------------------------------------------------------
    def _select_variant(self, idx: int) -> str:
        if self.subset != "train":
            return "normal"
        if self.rng.random() >= self.mix_prob:
            return "normal"
        return self.variants[self.rng.integers(0, len(self.variants))]

    def _load_face_and_landmarks(
        self, rec: Dict[str, Any], variant: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """face_crop 112×112 grayscale + landmark (478, 2) in 112 coord + manifest mask (478,)."""
        rel = Path(rec["rel"])
        fi = rec["frame"]

        if variant == "normal":
            face_npz = self.root_normal / rel
        else:
            face_npz = self.root_occluded / variant / rel
        with np.load(face_npz, allow_pickle=True) as d:
            face = d["images"][fi]                # (112, 112) uint8

        # landmark GT — 항상 정상 영상의 좌표 (가린 sample 도 same GT → messenger 학습 압력)
        if self.gt_source == "mediapipe":
            mp_p = self.root_mediapipe / str(rel).replace("_crops112.npz", "_facemesh.npz")
            with np.load(mp_p, allow_pickle=True) as d:
                landmarks_112 = d["landmarks_xy"][fi].astype(np.float32)   # already in 112 coord
        else:
            occface_p = _occface_path(rel, self.root_occface)
            with np.load(occface_p, allow_pickle=True) as d:
                lm_xy = d["landmarks_xy"][fi]
            landmarks_112 = _occface_to_face_crop_coord(lm_xy, self.crop_size)

        # manifest mask
        if variant == "normal":
            manifest = np.ones(478, dtype=np.float32)
        else:
            masked = set(self.variant_masked.get(variant, []))
            manifest = np.array([0.0 if i in masked else 1.0 for i in range(478)], dtype=np.float32)

        return face, landmarks_112, manifest

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        rec = self.database[idx]
        variant = self._select_variant(idx)

        face_112, landmarks_112, manifest_478 = self._load_face_and_landmarks(rec, variant)

        # 112 grayscale → 256 RGB (3ch by replication, friend 와 호환)
        face_256_gray = cv2.resize(face_112, (self.Image_size, self.Image_size), interpolation=cv2.INTER_LINEAR)
        face_256_rgb = np.stack([face_256_gray] * 3, axis=-1)             # (H, W, 3) uint8
        landmarks_256 = landmarks_112 * (self.Image_size / self.crop_size)  # (478, 2) in 256 coord

        # subset (478 → 68) — DMD_68 일 때
        if self.subset_mapping is not None:
            landmarks_256 = landmarks_256[self.subset_mapping]            # (68, 2)
            manifest_478 = manifest_478[self.subset_mapping]               # (68,)

        # affine trans (face_crop 는 이미 crop 됐으므로 identity-like)
        # friend trans 와 호환되도록 [scale_x, 0, 0; 0, scale_y, 0] 형식 (2, 3)
        s = float(self.Image_size) / self.crop_size
        trans = np.array([[s, 0.0, 0.0], [0.0, s, 0.0]], dtype=np.float32)

        # augmentation (friend 는 LandmarksCompose 사용)
        if self.augmentation_transform is not None:
            face_256_rgb, landmarks_256 = self.augmentation_transform(face_256_rgb, landmarks_256)

        image = copy.deepcopy(face_256_rgb)
        occluded_input = copy.deepcopy(face_256_rgb)        # variant 자체가 가린 거 (variant != normal)

        # heatmap GT (in 256 coord, downsample to 64)
        edgemap = generate_edgemap(
            landmarks_256, self.edge_info, image_size=self.Image_size,
            scale=1.0 / self.ratio, edge_type=self.edge_type,
        )
        pointmap = generate_pointmap(
            landmarks_256, image_size=self.Image_size,
            scale=1.0 / self.ratio,
        )

        # resized (64×64) inputs for ORFormer / VQVAE
        res_size = self.Image_size // self.ratio
        resized_input = cv2.resize(face_256_rgb, (res_size, res_size), interpolation=cv2.INTER_LINEAR)
        resized_occluded_input = cv2.resize(occluded_input, (res_size, res_size), interpolation=cv2.INTER_LINEAR)
        resized_image = copy.deepcopy(resized_occluded_input)

        # normalize
        if self.normalize_transform is not None:
            input_t = self.normalize_transform(face_256_rgb)
            resized_input_t = self.normalize_transform(resized_input)
            resized_occluded_input_t = self.normalize_transform(resized_occluded_input)
        else:
            input_t = torch.from_numpy(face_256_rgb).permute(2, 0, 1).float() / 255.0
            resized_input_t = torch.from_numpy(resized_input).permute(2, 0, 1).float() / 255.0
            resized_occluded_input_t = torch.from_numpy(resized_occluded_input).permute(2, 0, 1).float() / 255.0

        # norm_points: 256 coord → [-1, 1]
        resized_Points = torch.from_numpy(landmarks_256 / self.ratio).float()   # in 64 coord
        landmarks_norm = norm_points(resized_Points, res_size, res_size)

        meta = {
            "Annotated_Points": landmarks_256.astype(np.float32),       # original GT (256 coord)
            "Points":           landmarks_256.astype(np.float32),
            "Landmarks":        landmarks_norm,                         # [-1, 1]
            "Edge_Heatmaps":    edgemap,                                 # (E, 64, 64)
            "Point_Heatmaps":   pointmap,                                # (N, 64, 64)
            "Manifest_Mask":    torch.from_numpy(manifest_478).float(),  # (478,) or (68,)
            "Variant":          variant,
            "BBox":             np.array([0, 0, self.crop_size, self.crop_size], dtype=np.float32),
            "trans":            trans,
            "Scale":            self.Fraction,
            "rel":              rec["rel"],
            "frame":            rec["frame"],
        }
        return input_t.float(), resized_input_t.float(), resized_occluded_input_t.float(), meta, image, resized_image


# ---------- smoke test ----------
if __name__ == "__main__":
    sys.path.insert(0, str(_THIS.parent.parent / "configs"))
    from default import get_cfg
    cfg = get_cfg()

    ds = DMDHeatmapDataset(cfg, "DMD", "all", max_clips=2)
    print(f"dataset size: {len(ds)}")
    if len(ds) == 0:
        print("EMPTY — check ROOT paths")
    else:
        out = ds[0]
        input_t, res_in, res_occ, meta, image, res_img = out
        print(f"input:                {tuple(input_t.shape)}")
        print(f"resized_input:        {tuple(res_in.shape)}")
        print(f"resized_occluded:     {tuple(res_occ.shape)}")
        print(f"image (np):           {image.shape}")
        print(f"resized_image (np):   {res_img.shape}")
        print(f"meta['Landmarks']:    {tuple(meta['Landmarks'].shape)}")
        print(f"meta['Edge_Heatmaps']:  {tuple(meta['Edge_Heatmaps'].shape)}")
        print(f"meta['Point_Heatmaps']: {tuple(meta['Point_Heatmaps'].shape)}")
        print(f"meta['Manifest_Mask']:  {tuple(meta['Manifest_Mask'].shape)}  sum={meta['Manifest_Mask'].sum().item():.0f}/478")
        print(f"meta['Variant']:        {meta['Variant']}")

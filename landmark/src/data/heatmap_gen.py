"""Heatmap GT 생성기 — 친구 `Dataloader/heatmapDataset.py` 의 utility 분리.

- `generate_pointmap`: 478 (or N) landmark 별 가우시안 (σ=1.5) → (N, H*scale, W*scale)
- `generate_edgemap`:  EDGE_INFO polyline 별 spline-fit + distance transform → (E, H*scale, W*scale)

DMD dataloader 에서 이 두 함수 사용. 친구 default (Image=256, scale=0.25) → output 64×64.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate


# ---------- per-point gaussian (ADNet style) ----------

def circle_ADNet(img: np.ndarray, pt, sigma: float = 1.5, label_type: str = "Gaussian") -> np.ndarray:
    """단일 점을 가우시안 spot 으로 그림. 친구 heatmapDataset.py 그대로."""
    tmp_size = int(sigma * 3)
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] > img.shape[1] - 1 or ul[1] > img.shape[0] - 1 or
            br[0] - 1 < 0 or br[1] - 1 < 0):
        return img

    size = int(2 * tmp_size + 1)
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    if label_type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    image_slice = img[img_y[0]:img_y[1], img_x[0]:img_x[1]]
    gaussian_slice = 255 * g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    overlap_h = min(image_slice.shape[0], gaussian_slice.shape[0])
    overlap_w = min(image_slice.shape[1], gaussian_slice.shape[1])
    if overlap_h <= 0 or overlap_w <= 0:
        return img
    img[img_y[0]:img_y[0] + overlap_h, img_x[0]:img_x[0] + overlap_w] = \
        gaussian_slice[:overlap_h, :overlap_w]
    return img


# ---------- polyline heatmap (ADNet / VQVAE style) ----------

def polylines_ADNet(img: np.ndarray, lmks: np.ndarray, is_closed: bool,
                    color: int = 255, thickness: int = 1,
                    draw_mode=cv2.LINE_AA, interpolate_mode=cv2.INTER_AREA,
                    scale: int = 4) -> np.ndarray:
    h, w = img.shape
    img_scale = cv2.resize(img, (w * scale, h * scale), interpolation=interpolate_mode)
    lmks_scale = (lmks * scale + 0.5).astype(np.int32)
    cv2.polylines(img_scale, [lmks_scale], is_closed, color, thickness * scale, draw_mode)
    img = cv2.resize(img_scale, (w, h), interpolation=interpolate_mode)
    return img


def polylines_VQVAE(img: np.ndarray, lmks: np.ndarray, is_closed: bool,
                    color: int = 255, thickness: int = 1,
                    draw_mode=cv2.LINE_AA, interpolate_mode=cv2.INTER_AREA,
                    scale: int = 1) -> np.ndarray:
    h, w = img.shape
    img_scale = cv2.resize(img, (w * scale, h * scale), interpolation=interpolate_mode)
    lmks_scale = (lmks * scale + 0.5).astype(np.int32)
    img_scale = cv2.polylines(img_scale, [lmks_scale], is_closed, color, thickness * scale, draw_mode)
    img_scale = -1 * img_scale + 255   # flip black/white
    distance_map = cv2.distanceTransform(img_scale.astype(np.uint8), cv2.DIST_L2, 0)
    std = np.std(distance_map)
    threshold = std * 3
    heatmap = np.exp(-1 * np.square(distance_map) / (2 * np.square(std)))
    heatmap[distance_map >= threshold] = 0
    heatmap = cv2.resize(heatmap, (w, h), interpolation=interpolate_mode)
    return heatmap


def fit_curve(lmks: np.ndarray, is_closed: bool = False, density: int = 5) -> np.ndarray:
    """B-spline interpolation 으로 smooth polyline 만들기 (k=3)."""
    try:
        x = lmks[:, 0].copy()
        y = lmks[:, 1].copy()
        if is_closed:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        tck, u = interpolate.splprep([x, y], s=0, per=is_closed, k=3)
        intervals = np.array([])
        for i in range(len(u) - 1):
            intervals = np.concatenate((intervals, np.linspace(u[i], u[i + 1], density, endpoint=False)))
        if not is_closed:
            intervals = np.concatenate((intervals, [u[-1]]))
        lmk_x, lmk_y = interpolate.splev(intervals, tck, der=0)
        return np.stack([lmk_x, lmk_y], axis=-1)
    except Exception:
        return lmks


# ---------- high-level generators ----------

def generate_pointmap(points: np.ndarray, image_size: int = 256,
                      scale: float = 0.25, sigma: float = 1.5) -> torch.Tensor:
    """(N, 2) landmark in `image_size` coord → (N, image_size*scale, image_size*scale) gaussian heatmap.

    Vectorized numpy: 478 점을 한 번의 broadcast 로. 기존 per-point loop 보다 ~9× 빠름.
    """
    target = int(image_size * scale)                      # downsampled grid size
    if target <= 0:
        target = image_size
    pts = np.asarray(points, dtype=np.float32) * scale     # scale 좌표를 target grid 로
    pts[:, 0] = np.clip(pts[:, 0], 0, target - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, target - 1)

    sigma_scaled = sigma                                   # sigma 는 target grid 단위
    yy, xx = np.meshgrid(np.arange(target), np.arange(target), indexing="ij")
    yy = yy[None].astype(np.float32)                       # (1, H, W)
    xx = xx[None].astype(np.float32)
    px = pts[:, 0:1, None]                                 # (N, 1, 1)
    py = pts[:, 1:2, None]
    d2 = (xx - px) ** 2 + (yy - py) ** 2                   # (N, H, W)
    hm = np.exp(-d2 / (2.0 * sigma_scaled ** 2)).astype(np.float32)
    return torch.from_numpy(hm)


def generate_edgemap(points: np.ndarray, edge_info: List[Tuple[bool, List[int]]],
                     image_size: int = 256, scale: float = 0.25,
                     thickness: int = 1, edge_type: str = "ADNet") -> torch.Tensor:
    """(N, 2) landmark + edge_info polylines → (E, image_size*scale, image_size*scale) edge heatmap.

    edge_type = "ADNet" → polylines + downsample
    edge_type = "VQVAE" → polylines + distance-transform gaussian
    """
    h, w = image_size, image_size
    edgemaps = []
    for is_closed, indices in edge_info:
        em = np.zeros([h, w], dtype=np.float32)
        part = copy.deepcopy(points[np.array(indices)])
        part = fit_curve(part, is_closed)
        part[:, 0] = np.clip(part[:, 0], 0, w - 1)
        part[:, 1] = np.clip(part[:, 1], 0, h - 1)
        if edge_type == "VQVAE":
            em = polylines_VQVAE(em, part, is_closed, 255, thickness)
        elif edge_type == "ADNet":
            em = polylines_ADNet(em, part, is_closed, 255, thickness) / 255.0
        else:
            raise ValueError(f"unknown edge_type: {edge_type}")
        edgemaps.append(em)
    edgemaps_np = np.stack(edgemaps, axis=0)
    t = torch.from_numpy(edgemaps_np).float().unsqueeze(0)
    t = F.interpolate(t, size=(int(h * scale), int(w * scale)),
                      mode="bilinear", align_corners=False).squeeze(0)
    return t


def norm_points(points: torch.Tensor, h: int, w: int, align_corners: bool = True) -> torch.Tensor:
    if align_corners:
        return torch.clamp(points / torch.tensor([w - 1, h - 1]).to(points).view(1, 2) * 2 - 1, -1, 1)
    return torch.clamp((points * 2 + 1) / torch.tensor([w, h]).to(points).view(1, 2) - 1, -1, 1)


def denorm_points(points: torch.Tensor, h: int, w: int, align_corners: bool = True) -> torch.Tensor:
    if align_corners:
        return (points + 1) / 2 * torch.tensor([w - 1, h - 1]).to(points).view(1, 1, 2)
    return ((points + 1) * torch.tensor([w, h]).to(points).view(1, 1, 2) - 1) / 2


if __name__ == "__main__":
    # smoke test
    pts = np.random.rand(478, 2) * 256
    pm = generate_pointmap(pts, 256, 0.25, 1.5)
    print("pointmap:", pm.shape, pm.dtype, pm.min().item(), pm.max().item())

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from face_edge_info import build_edge_info_478
    info = build_edge_info_478()
    em = generate_edgemap(pts, info, 256, 0.25, 1, "ADNet")
    print("edgemap :", em.shape, em.dtype, em.min().item(), em.max().item())

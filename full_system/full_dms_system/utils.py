from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2


def ensure_bgr_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("frame is None")
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be np.ndarray, got {type(frame)}")
    if frame.ndim != 3:
        raise ValueError(f"frame must be HxWxC, got shape={frame.shape}")
    if frame.shape[2] == 3:
        return frame
    if frame.shape[2] == 4:
        return frame[:, :, :3]
    raise ValueError(f"unsupported channel count: {frame.shape}")


def clamp_xyxy(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    b = np.asarray(xyxy, dtype=np.float32).copy()
    b[0] = np.clip(b[0], 0, max(width - 1, 0))
    b[2] = np.clip(b[2], 0, max(width, 0))
    b[1] = np.clip(b[1], 0, max(height - 1, 0))
    b[3] = np.clip(b[3], 0, max(height, 0))
    if b[2] <= b[0] or b[3] <= b[1]:
        return np.zeros((4,), dtype=np.float32)
    return b.astype(np.float32)


def expand_bbox_xyxy(
    bbox: np.ndarray,
    frame_shape: Tuple[int, int, int],
    pad_ratio: float = 0.2,
    square: bool = False,
    scale_factor: Optional[float] = None,
    y_shift_ratio: float = 0.0,
) -> np.ndarray:
    """Expand an x1,y1,x2,y2 bbox.

    - pad_ratio expands width/height independently by pad_ratio per side when square=False.
    - square=True makes a square box centered at bbox center. If scale_factor is given, side=max(w,h)*scale_factor;
      otherwise side=max(w,h)*(1+2*pad_ratio).
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2) + bh * y_shift_ratio
    if square:
        side = max(bw, bh) * (float(scale_factor) if scale_factor is not None else (1.0 + 2.0 * pad_ratio))
        out = np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)
    else:
        out = np.array([x1 - bw * pad_ratio, y1 - bh * pad_ratio, x2 + bw * pad_ratio, y2 + bh * pad_ratio], dtype=np.float32)
    return clamp_xyxy(out, width=w, height=h)


def crop_resize_gray(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray, size: int) -> np.ndarray:
    b = np.asarray(bbox_xyxy, dtype=np.float32)
    x1, y1, x2, y2 = [int(round(x)) for x in b]
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("empty crop")
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def crop_resize_bgr(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray, size: int | None = None) -> np.ndarray:
    b = np.asarray(bbox_xyxy, dtype=np.float32)
    x1, y1, x2, y2 = [int(round(x)) for x in b]
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("empty crop")
    if size is not None:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def crop_landmarks_to_frame_xy(lm_xy: np.ndarray, crop_bbox_xyxy: np.ndarray, crop_size: int = 112) -> np.ndarray:
    """Convert landmark coordinates in crop pixel space to full frame pixel space."""
    xy = np.asarray(lm_xy, dtype=np.float32)
    x1, y1, x2, y2 = np.asarray(crop_bbox_xyxy, dtype=np.float32)
    sx = (x2 - x1) / float(crop_size)
    sy = (y2 - y1) / float(crop_size)
    out = xy.copy()
    out[:, 0] = x1 + out[:, 0] * sx
    out[:, 1] = y1 + out[:, 1] * sy
    return out.astype(np.float32)


def make_zero_facemesh() -> np.ndarray:
    return np.zeros((478, 3), dtype=np.float32)


def make_zero_pose() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((17, 2), dtype=np.float32), np.zeros((17,), dtype=np.float32)

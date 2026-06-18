from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .utils import ensure_bgr_frame, expand_bbox_xyxy, crop_resize_gray

REGION_NAMES = [
    "left_eye_visible",
    "right_eye_visible",
    "nose_visible",
    "mouth_visible",
]


class VisibilityResNet18(nn.Module):
    """Gray 256x256 crop input -> 4 visibility logits."""

    def __init__(self, num_labels: int = 4):
        super().__init__()
        m = models.resnet18(weights=None)
        old_conv = m.conv1
        m.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        m.fc = nn.Linear(m.fc.in_features, num_labels)
        self.net = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_visibility_model(ckpt_path: Union[str, Path], device: Union[str, torch.device]) -> nn.Module:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"visibility ckpt not found: {ckpt_path}")
    model = VisibilityResNet18(num_labels=4)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    clean_sd = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        if not nk.startswith("net."):
            nk = "net." + nk
        clean_sd[nk] = v

    model_sd = model.state_dict()
    filtered = {k: v for k, v in clean_sd.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    model_sd.update(filtered)
    model.load_state_dict(model_sd, strict=True)
    loaded_ratio = len(filtered) / max(1, len(model.state_dict()))
    if loaded_ratio < 0.90:
        raise RuntimeError(f"Checkpoint load ratio too low: {loaded_ratio:.2%}; architecture mismatch likely.")
    model.to(device)
    model.eval()
    return model


class OccCNNRealtimeWrapper:
    """YOLO-bbox based real-time face occlusion/visibility estimator.

    Unlike the offline occ-cache script, this does not write NPZ/JSON. It returns
    a frame-level vector immediately.
    """

    def __init__(
        self,
        ckpt_path: Optional[Union[str, Path]],
        device: Optional[Union[str, int]] = None,
        crop_size: int = 256,
        bbox_scale_factor: float = 1.35,
        bbox_y_shift_ratio: float = 0.03,
        default_visible_prob: float = 0.5,
        default_crop_valid: float = 0.0,
    ):
        self.device = (f"cuda:{device}" if isinstance(device, int) else (device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")))
        self.crop_size = int(crop_size)
        self.bbox_scale_factor = float(bbox_scale_factor)
        self.bbox_y_shift_ratio = float(bbox_y_shift_ratio)
        self.default_visible_prob = float(default_visible_prob)
        self.default_crop_valid = float(default_crop_valid)
        self.model = None
        if ckpt_path:
            self.model = load_visibility_model(ckpt_path, self.device)

    def __call__(self, frame_bgr: np.ndarray, face_bbox: np.ndarray, face_detected: bool) -> Dict[str, Any]:
        return self.extract(frame_bgr, face_bbox, face_detected)

    def extract(self, frame_bgr: np.ndarray, face_bbox: np.ndarray, face_detected: bool) -> Dict[str, Any]:
        frame_bgr = ensure_bgr_frame(frame_bgr)
        if self.model is None or not face_detected:
            return self._fallback()
        try:
            crop_bbox = expand_bbox_xyxy(
                face_bbox,
                frame_bgr.shape,
                square=True,
                scale_factor=self.bbox_scale_factor,
                y_shift_ratio=self.bbox_y_shift_ratio,
            )
            if float(crop_bbox[2] - crop_bbox[0]) <= 1 or float(crop_bbox[3] - crop_bbox[1]) <= 1:
                return self._fallback()
            gray = crop_resize_gray(frame_bgr, crop_bbox, self.crop_size)
            x = gray.astype(np.float32) / 255.0
            ten = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(ten)
                probs = torch.sigmoid(logits).detach().cpu().numpy()[0].astype(np.float32)
            return {
                "probs": probs,
                "crop_valid": True,
                "crop_bbox": crop_bbox.astype(np.float32),
                "regions": REGION_NAMES,
            }
        except Exception:
            return self._fallback()

    def _fallback(self) -> Dict[str, Any]:
        return {
            "probs": np.full((4,), self.default_visible_prob, dtype=np.float32),
            "crop_valid": bool(self.default_crop_valid > 0.5),
            "crop_bbox": np.zeros((4,), dtype=np.float32),
            "regions": REGION_NAMES,
        }

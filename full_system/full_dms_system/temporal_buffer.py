from __future__ import annotations

from collections import deque
from typing import Any, Dict
import numpy as np


class TemporalDMSBuffer:
    def __init__(self, window_size: int = 48):
        self.window_size = int(window_size)
        self.body = deque(maxlen=self.window_size)
        self.body_conf = deque(maxlen=self.window_size)
        self.face_lm = deque(maxlen=self.window_size)
        self.face_detected = deque(maxlen=self.window_size)
        self.face_bbox = deque(maxlen=self.window_size)
        self.face_det_score = deque(maxlen=self.window_size)
        self.face_bbox_detected = deque(maxlen=self.window_size)
        self.occ = deque(maxlen=self.window_size)

    def __len__(self) -> int:
        return len(self.body)

    @property
    def ready(self) -> bool:
        return len(self.body) >= self.window_size

    def append(
        self,
        body_keypoints: np.ndarray,
        body_conf: np.ndarray,
        face_lm: np.ndarray,
        face_detected: bool,
        face_bbox: np.ndarray,
        face_det_score: float,
        face_bbox_detected: bool,
        occ_feature: np.ndarray,
    ) -> None:
        self.body.append(np.asarray(body_keypoints, dtype=np.float32))
        self.body_conf.append(np.asarray(body_conf, dtype=np.float32))
        self.face_lm.append(np.asarray(face_lm, dtype=np.float32))
        self.face_detected.append(bool(face_detected))
        self.face_bbox.append(np.asarray(face_bbox, dtype=np.float32))
        self.face_det_score.append(float(face_det_score))
        self.face_bbox_detected.append(bool(face_bbox_detected))
        self.occ.append(np.asarray(occ_feature, dtype=np.float32))

    def as_arrays(self) -> Dict[str, np.ndarray]:
        if not self.ready:
            raise RuntimeError(f"buffer not ready: {len(self)}/{self.window_size}")
        return {
            "body_seq": np.stack(self.body).astype(np.float32),
            "body_conf_seq": np.stack(self.body_conf).astype(np.float32),
            "face_lm_seq": np.stack(self.face_lm).astype(np.float32),
            "face_detected_seq": np.asarray(self.face_detected, dtype=bool),
            "face_bbox_seq": np.stack(self.face_bbox).astype(np.float32),
            "face_det_score_seq": np.asarray(self.face_det_score, dtype=np.float32),
            "face_bbox_detected_seq": np.asarray(self.face_bbox_detected, dtype=bool),
            "occ_seq": np.stack(self.occ).astype(np.float32),
        }

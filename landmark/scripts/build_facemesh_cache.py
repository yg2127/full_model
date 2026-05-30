#!/usr/bin/env python
"""DMD face_crops_112 위에서 MediaPipe FaceMesh 478 landmark 재추출.

Input  : /data/shared/DMD_landmarks/face_crops_112/.../*_crops112.npz
Output : /data/shared/DMD_landmarks/face_crops_112_facemesh/.../*_facemesh.npz
         - landmarks_xy : (T, 478, 2) float32, in 112 coord
         - detected     : (T,) bool — MediaPipe detection 성공 여부
         - meta         : dict

face_crop_112 (작은 112×112) 직접 inference 는 detection 률 ↓ → upscale 256 후 추출.
출력 좌표는 다시 112 단위로 환원.

Usage:
    python build_facemesh_cache.py --workers 8 --upscale 256
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

INPUT_ROOT = Path("/data/shared/DMD_landmarks/face_crops_112")
OUTPUT_ROOT = Path("/data/shared/DMD_landmarks/face_crops_112_facemesh")


def _process_one(args) -> Tuple[str, int, int, float]:
    """한 npz 파일을 처리. (rel_path, T, n_detected, elapsed) 반환."""
    npz_path, upscale = args
    rel = npz_path.relative_to(INPUT_ROOT)
    out_path = OUTPUT_ROOT / str(rel).replace("_crops112.npz", "_facemesh.npz")
    if out_path.exists():
        return (str(rel), 0, 0, 0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as d:
        images = d["images"]
        meta_in = d["meta"].item() if isinstance(d["meta"], np.ndarray) else d["meta"]
        # detected mask (yolo) — mediapipe inference 도 yolo detected 일 때만 시도
        yolo_det = d["detected"].astype(bool) if "detected" in d.files else np.ones(images.shape[0], dtype=bool)

    T = images.shape[0]
    crop_size = images.shape[1]

    landmarks_xy = np.zeros((T, 478, 2), dtype=np.float32)
    detected = np.zeros(T, dtype=bool)

    start = time.time()

    # mediapipe 는 worker 내부에서 import (spawn 시 lazy)
    import mediapipe as mp_pkg
    face_mesh = mp_pkg.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.1,
    )

    try:
        for fi in range(T):
            if not yolo_det[fi]:
                continue
            img_gray = images[fi]
            img_up = cv2.resize(img_gray, (upscale, upscale), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_up, cv2.COLOR_GRAY2RGB)
            res = face_mesh.process(img_rgb)
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                coords = np.array([[lm.x * upscale, lm.y * upscale] for lm in lms[:478]],
                                  dtype=np.float32)
                coords = coords * (crop_size / upscale)
                landmarks_xy[fi] = coords
                detected[fi] = True
    finally:
        face_mesh.close()

    meta_out = {
        "source_npz": str(rel),
        "total_frames": int(T),
        "crop_size": int(crop_size),
        "upscale": int(upscale),
        "mesh_model": "mediapipe_face_mesh_refined",
        "min_detection_confidence": 0.1,
        "source_video": meta_in.get("source_video", "") if isinstance(meta_in, dict) else "",
    }

    np.savez_compressed(
        out_path,
        landmarks_xy=landmarks_xy,
        detected=detected,
        meta=meta_out,
    )

    return (str(rel), int(T), int(detected.sum()), float(time.time() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--upscale", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 = all clips")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    clips = sorted(INPUT_ROOT.rglob("*_crops112.npz"))
    print(f"total clips: {len(clips)}", flush=True)
    if args.limit > 0:
        clips = clips[: args.limit]
        print(f"limit: {args.limit}", flush=True)

    todo = []
    for c in clips:
        rel = c.relative_to(INPUT_ROOT)
        out = OUTPUT_ROOT / str(rel).replace("_crops112.npz", "_facemesh.npz")
        if not out.exists():
            todo.append(c)
    print(f"to process: {len(todo)} (skipping {len(clips) - len(todo)} done)", flush=True)
    if not todo:
        return

    pool_args = [(c, args.upscale) for c in todo]
    t0 = time.time()
    total_T = 0
    total_det = 0

    # spawn 으로 mediapipe 깔끔 init
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, (rel, T, det, el) in enumerate(pool.imap_unordered(_process_one, pool_args)):
            if T == 0:
                continue
            total_T += T
            total_det += det
            done = i + 1
            avg_fps = total_T / max(time.time() - t0, 1e-3)
            eta = (len(todo) - done) * (T / max(avg_fps, 1)) / args.workers
            print(
                f"[{done}/{len(todo)}] {Path(rel).stem[-45:]}  "
                f"T={T} det={det}/{T} ({100*det/T:.1f}%) {el:.1f}s  "
                f"avg {avg_fps:.0f}fps  eta {eta/60:.1f}min",
                flush=True,
            )

    elapsed = time.time() - t0
    print()
    print("=== done ===")
    print(f"  time: {elapsed/60:.1f} min")
    print(f"  total frames: {total_T}")
    print(f"  detected: {total_det}/{total_T} ({100*total_det/max(total_T,1):.1f}%)")


if __name__ == "__main__":
    main()

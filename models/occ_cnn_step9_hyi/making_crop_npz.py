from __future__ import annotations

import os

# ============================================================
# CPU thread control
# ============================================================
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

cv2.setNumThreads(1)
torch.set_num_threads(2)


# ============================================================
# Hard-coded paths
# ============================================================

MANIFEST_PATHS = [
    Path("/home/hyi/Code/Step9_extract_crop_npz/clip_manifest_train.json"),
    Path("/home/hyi/Code/Step9_extract_crop_npz/clip_manifest_val.json"),
    Path("/home/hyi/Code/Step9_extract_crop_npz/clip_manifest_test_clean.json"),
    Path("/home/hyi/Code/Step9_extract_crop_npz/clip_manifest_test_masked.json"),
]

# clean roots
CLEAN_FACEMESH_ROOT = Path("/data/shared/DMD_landmarks/facemesh")
CLEAN_VIDEO_ROOT = Path("/data/shared/DMD")

# masked roots
MASKED_FACEMESH_ROOT = Path(
    "/data/shared/Occlusion_subset_dataset/"
    "region_occlusion_video_dataset_v3_original_fixedmask_yolo_face_facemesh_canonical/facemesh"
)

MASKED_VIDEO_CANONICAL_ROOT = Path(
    "/data/shared/Occlusion_subset_dataset/"
    "region_occlusion_video_dataset_v3_original_fixedmask/videos_canonical"
)

# output root: MediaPipe crop version
OUT_ROOT = Path(
    "/data/shared/Occlusion_subset_dataset/"
    "region_occlusion_video_dataset_v3_original_fixedmask_occ_pred_mediapipe"
)

OCC_NPZ_ROOT = OUT_ROOT / "occ_npz"
MAP_JSON = OUT_ROOT / "face_npz_to_occ_npz.json"
SUMMARY_JSON = OUT_ROOT / "occ_generation_summary.json"
FAIL_JSONL = OUT_ROOT / "occ_generation_failures.jsonl"

VIS_CNN_CKPT = Path("/home/hyi/Code/Step9_extract_crop_npz/best.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 512

# 5프레임마다 1개만 OCC CNN 추론
OCC_FRAME_STRIDE = 5

SAVE_DEBUG_CROPS = False
DEBUG_CROP_DIR = OUT_ROOT / "debug_crops"
DEBUG_CROP_LIMIT = 100

SAVE_FLOAT16 = True

# crop/image config
SAVE_SIZE = 256
MIN_FACE_BOX_SIZE = 25
MAX_FACE_BOX_RATIO = 0.98
FACE_PAD_FACTOR = 1.35
CENTER_X_SHIFT_RATIO = 0.00
CENTER_Y_SHIFT_RATIO = 0.03

# MediaPipe FaceDetection config
MIN_DETECTION_CONFIDENCE = 0.4
MEDIAPIPE_MODEL_SELECTION = 1

# crop 실패 시 중립값
NEUTRAL_PROB = 0.5

REGION_NAMES = [
    "left_eye_visible",
    "right_eye_visible",
    "nose_visible",
    "mouth_visible",
]


# ============================================================
# Model
# ============================================================

class VisibilityResNet18(nn.Module):
    """
    gray 256x256 crop input
    output: 4 logits
    """

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


def load_visibility_model(ckpt_path: Path, device: str) -> nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"visibility ckpt not found: {ckpt_path}")

    model = VisibilityResNet18(num_labels=4)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

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
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module."):]

        if not nk.startswith("net."):
            nk = "net." + nk

        clean_sd[nk] = v

    model_sd = model.state_dict()

    filtered_sd = {}
    skipped = []

    for k, v in clean_sd.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            filtered_sd[k] = v
        else:
            skipped.append(k)

    model_sd.update(filtered_sd)
    model.load_state_dict(model_sd, strict=True)

    loaded_ratio = len(filtered_sd) / max(1, len(model.state_dict()))

    print(f"[CKPT] loaded: {ckpt_path}")
    print(f"[CKPT] loaded keys: {len(filtered_sd)} / {len(model.state_dict())} ({loaded_ratio:.2%})")
    print(f"[CKPT] skipped keys: {len(skipped)}")

    if skipped:
        print("[CKPT] skipped sample:", skipped[:10])

    if loaded_ratio < 0.90:
        raise RuntimeError(
            f"Checkpoint load ratio too low: {loaded_ratio:.2%}. "
            f"Architecture/key mismatch likely."
        )

    model.to(device)
    model.eval()
    return model


# ============================================================
# Manifest loading
# ============================================================

def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8").strip()

    if not txt:
        return []

    if txt[0] == "[":
        return json.loads(txt)

    return [json.loads(line) for line in txt.splitlines() if line.strip()]


def collect_needed_frames(
    manifest_paths: List[Path],
) -> Tuple[Dict[str, Set[int]], Dict[str, int]]:
    needed: Dict[str, Set[int]] = defaultdict(set)

    stats = {
        "num_manifests": 0,
        "num_clips": 0,
        "num_bad_rows": 0,
        "occ_frame_stride": OCC_FRAME_STRIDE,
    }

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        rows = load_json_or_jsonl(manifest_path)
        stats["num_manifests"] += 1
        stats["num_clips"] += len(rows)

        print(f"[MANIFEST] {manifest_path.name}: {len(rows)} clips")

        for r in rows:
            try:
                face_npz = str(r["face_npz"])
                s = int(r["face_start"])
                e = int(r["face_end"])

                if e < s:
                    stats["num_bad_rows"] += 1
                    continue

                needed[face_npz].update(range(s, e + 1, OCC_FRAME_STRIDE))

            except Exception:
                stats["num_bad_rows"] += 1

    return needed, stats


# ============================================================
# Face npz helpers
# ============================================================

LANDMARK_KEY_CANDIDATES = [
    "landmarks",
    "facemesh",
    "face_landmarks",
    "points",
    "coords",
    "arr_0",
]


def find_landmark_key(npz: np.lib.npyio.NpzFile) -> str:
    for k in LANDMARK_KEY_CANDIDATES:
        if k in npz.files:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.ndim >= 3:
                return k

    for k in npz.files:
        arr = npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim >= 3:
            return k

    raise ValueError(f"No landmark-like array found. keys={npz.files}")


def to_tvc(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D landmark array, got shape={arr.shape}")

    _, a, b = arr.shape

    # common: (T, V, C)
    if a >= 100 and b <= 16:
        return arr.copy(), "TVC"

    # possible: (T, C, V)
    if b >= 100 and a <= 16:
        return np.transpose(arr, (0, 2, 1)).copy(), "TCV"

    raise ValueError(f"Ambiguous landmark shape: {arr.shape}")


def load_face_npz_frame_count(src_npz: Path) -> int:
    """
    Face npz의 frame count만 얻는다.
    MediaPipe crop 버전에서는 landmark를 bbox 계산에 사용하지 않는다.
    """
    with np.load(str(src_npz), allow_pickle=True) as data:
        key = find_landmark_key(data)
        arr = data[key]
        lm, _ = to_tvc(arr)

    return int(lm.shape[0])


# ============================================================
# Crop helpers: MediaPipe FaceDetection bbox 기반 자동 crop
# ============================================================

def detect_face_bbox_mediapipe(
    face_detector: Any,
    frame_bgr: np.ndarray,
) -> Optional[Tuple[int, int, int, int, Dict[str, Any]]]:
    """
    MediaPipe FaceDetection으로 현재 frame의 얼굴 bbox를 검출하고,
    padding을 적용한 square-ish face crop bbox를 만든다.

    반환:
        (x1, y1, x2, y2, info) 또는 None
    """
    frame_h, frame_w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_detector.process(frame_rgb)

    if not result.detections:
        return None

    best = max(
        result.detections,
        key=lambda d: float(d.score[0]) if d.score else 0.0,
    )

    det_score = float(best.score[0]) if best.score else 0.0
    box = best.location_data.relative_bounding_box

    x = float(box.xmin * frame_w)
    y = float(box.ymin * frame_h)
    bw = float(box.width * frame_w)
    bh = float(box.height * frame_h)

    if bw < MIN_FACE_BOX_SIZE or bh < MIN_FACE_BOX_SIZE:
        return None

    if bw > frame_w * MAX_FACE_BOX_RATIO or bh > frame_h * MAX_FACE_BOX_RATIO:
        return None

    cx = x + bw / 2.0 + bw * CENTER_X_SHIFT_RATIO
    cy = y + bh / 2.0 + bh * CENTER_Y_SHIFT_RATIO

    side = max(bw, bh) * FACE_PAD_FACTOR

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < MIN_FACE_BOX_SIZE or crop_h < MIN_FACE_BOX_SIZE:
        return None

    info = {
        "detector": "mediapipe_face_detection",
        "det_score": det_score,
        "face_raw_bbox_xyxy": [x, y, x + bw, y + bh],
        "face_crop_xyxy": [x1, y1, x2, y2],
        "face_crop_w": int(crop_w),
        "face_crop_h": int(crop_h),
        "face_pad_factor": FACE_PAD_FACTOR,
        "center_x_shift_ratio": CENTER_X_SHIFT_RATIO,
        "center_y_shift_ratio": CENTER_Y_SHIFT_RATIO,
        "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
        "mediapipe_model_selection": MEDIAPIPE_MODEL_SELECTION,
    }

    return x1, y1, x2, y2, info


def crop_frame_to_gray256(
    frame_bgr: np.ndarray,
    crop_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = crop_xyxy
    crop = frame_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        raise RuntimeError("empty face crop")

    crop_256 = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop_256, cv2.COLOR_BGR2GRAY)

    return gray


def gray_to_tensor(gray: np.ndarray) -> torch.Tensor:
    if gray.shape != (SAVE_SIZE, SAVE_SIZE):
        gray = cv2.resize(gray, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)

    x = gray.astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0)  # (1, H, W)

    return x


# ============================================================
# Video matching
# ============================================================

def is_masked_face_npz(face_npz: Path) -> bool:
    return str(face_npz).startswith(str(MASKED_FACEMESH_ROOT))


def is_clean_face_npz(face_npz: Path) -> bool:
    return str(face_npz).startswith(str(CLEAN_FACEMESH_ROOT))


def clean_face_npz_to_video(face_npz: Path) -> Path:
    rel = face_npz.relative_to(CLEAN_FACEMESH_ROOT)
    video_dir = CLEAN_VIDEO_ROOT / rel.parent

    name = face_npz.name
    candidates = []

    if name.endswith("_ir_face_facemesh.npz"):
        base = name.replace("_ir_face_facemesh.npz", "")
        candidates.extend(
            [
                video_dir / f"{base}_ir_face.mp4",
                video_dir / f"{base}_ir_body.mp4",
                video_dir / f"{base}_rgb_body.mp4",
            ]
        )

    if video_dir.exists():
        stem = name.replace("_ir_face_facemesh.npz", "")
        candidates.extend(sorted(video_dir.glob(f"{stem}*.mp4")))

    for p in candidates:
        if p.exists() or p.is_symlink():
            return p

    raise FileNotFoundError(f"clean video not found for face_npz={face_npz}")


def masked_face_npz_to_video(face_npz: Path) -> Path:
    rel = face_npz.relative_to(MASKED_FACEMESH_ROOT)

    video_name = face_npz.name.replace("_ir_face_facemesh.npz", "_ir_face.mp4")
    video = MASKED_VIDEO_CANONICAL_ROOT / rel.parent / video_name

    if video.exists() or video.is_symlink():
        return video

    raise FileNotFoundError(
        f"masked canonical video not found for face_npz={face_npz}, expected={video}"
    )


def face_npz_to_video(face_npz: Path) -> Path:
    if is_masked_face_npz(face_npz):
        return masked_face_npz_to_video(face_npz)

    if is_clean_face_npz(face_npz):
        return clean_face_npz_to_video(face_npz)

    raise ValueError(f"Unknown face_npz root: {face_npz}")


def face_npz_to_occ_npz(face_npz: Path) -> Path:
    if is_masked_face_npz(face_npz):
        rel = face_npz.relative_to(MASKED_FACEMESH_ROOT)
        variant = "masked"
    elif is_clean_face_npz(face_npz):
        rel = face_npz.relative_to(CLEAN_FACEMESH_ROOT)
        variant = "clean"
    else:
        raise ValueError(f"Unknown face_npz root: {face_npz}")

    out_name = face_npz.name.replace("_ir_face_facemesh.npz", "_ir_face_occ.npz")
    return OCC_NPZ_ROOT / variant / rel.parent / out_name


# ============================================================
# Frame reading and inference
# ============================================================

def infer_batch(
    model: nn.Module,
    batch_tensors: List[torch.Tensor],
    device: str,
) -> np.ndarray:
    if not batch_tensors:
        return np.zeros((0, 4), dtype=np.float32)

    x = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    return probs


def save_occ_npz(
    out_path: Path,
    probs: np.ndarray,
    crop_valid: np.ndarray,
    computed: np.ndarray,
    t_face: int,
    face_npz: Path,
    video_path: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_probs = probs.astype(np.float16) if SAVE_FLOAT16 else probs.astype(np.float32)

    np.savez_compressed(
        str(out_path),
        probs=save_probs,
        crop_valid=crop_valid.astype(np.uint8),
        computed=computed.astype(np.uint8),
        frame_idx=np.arange(t_face, dtype=np.int32),
        regions=np.array(REGION_NAMES),
        source_face_npz=np.array(str(face_npz)),
        source_video=np.array(str(video_path)),
        occ_frame_stride=np.array(OCC_FRAME_STRIDE, dtype=np.int32),
        crop_method=np.array("mediapipe_face_detection_auto_crop"),
        mediapipe_model_selection=np.array(MEDIAPIPE_MODEL_SELECTION, dtype=np.int32),
        min_detection_confidence=np.array(MIN_DETECTION_CONFIDENCE, dtype=np.float32),
    )


def process_one_face_npz(
    face_npz: Path,
    needed_frames: Set[int],
    model: nn.Module,
    device: str,
    debug_state: Dict[str, int],
) -> Dict[str, Any]:
    """
    face_npz는 video matching과 frame length 산출용으로만 사용한다.
    crop bbox는 landmark가 아니라 MediaPipe FaceDetection 결과로 만든다.
    """
    t_face = load_face_npz_frame_count(face_npz)

    valid_needed = sorted([int(x) for x in needed_frames if 0 <= int(x) < t_face])

    probs = np.full((t_face, 4), NEUTRAL_PROB, dtype=np.float32)
    crop_valid = np.zeros((t_face,), dtype=np.uint8)
    computed = np.zeros((t_face,), dtype=np.uint8)

    video_path = face_npz_to_video(face_npz)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = min(t_face, video_frames)

    needed_in_range = [fi for fi in valid_needed if fi < max_frame]
    out_of_range_frames = [fi for fi in valid_needed if fi >= max_frame]

    for fi in out_of_range_frames:
        computed[fi] = 1

    needed_set = set(needed_in_range)

    batch_tensors: List[torch.Tensor] = []
    batch_indices: List[int] = []

    num_crop_fail = 0
    num_read_fail = 0
    num_out_of_range = len(out_of_range_frames)

    if len(needed_set) == 0:
        cap.release()

        out_path = face_npz_to_occ_npz(face_npz)
        save_occ_npz(
            out_path=out_path,
            probs=probs,
            crop_valid=crop_valid,
            computed=computed,
            t_face=t_face,
            face_npz=face_npz,
            video_path=str(video_path),
        )

        return {
            "face_npz": str(face_npz),
            "video_path": str(video_path),
            "occ_npz": str(out_path),
            "t_face": t_face,
            "video_frames": video_frames,
            "needed_frames": len(needed_frames),
            "valid_needed_frames": len(valid_needed),
            "computed_frames": int(computed.sum()),
            "crop_valid_frames": int(crop_valid.sum()),
            "crop_fail": 0,
            "read_fail": 0,
            "out_of_range": int(num_out_of_range),
        }

    max_needed = max(needed_set)

    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=MEDIAPIPE_MODEL_SELECTION,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    cur_idx = 0

    while cur_idx <= max_needed:
        ok, frame = cap.read()

        if not ok or frame is None:
            for fi in needed_set:
                if fi >= cur_idx and computed[fi] == 0:
                    computed[fi] = 1
                    num_read_fail += 1
            break

        if cur_idx not in needed_set:
            cur_idx += 1
            continue

        bbox = detect_face_bbox_mediapipe(
            face_detector=face_detector,
            frame_bgr=frame,
        )

        if bbox is None:
            num_crop_fail += 1
            computed[cur_idx] = 1
            cur_idx += 1
            continue

        x1, y1, x2, y2, _ = bbox

        try:
            gray = crop_frame_to_gray256(frame, (x1, y1, x2, y2))
            tensor = gray_to_tensor(gray)

            if SAVE_DEBUG_CROPS and debug_state["saved"] < DEBUG_CROP_LIMIT:
                DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)
                dbg_name = f"{debug_state['saved']:05d}_{face_npz.stem}_f{cur_idx:06d}.jpg"
                cv2.imwrite(str(DEBUG_CROP_DIR / dbg_name), gray)
                debug_state["saved"] += 1

            batch_tensors.append(tensor)
            batch_indices.append(cur_idx)

            if len(batch_tensors) >= BATCH_SIZE:
                pred = infer_batch(model, batch_tensors, device)

                for idx, p in zip(batch_indices, pred):
                    probs[idx] = p
                    crop_valid[idx] = 1
                    computed[idx] = 1

                batch_tensors.clear()
                batch_indices.clear()

        except Exception:
            num_crop_fail += 1
            computed[cur_idx] = 1

        cur_idx += 1

    if batch_tensors:
        pred = infer_batch(model, batch_tensors, device)

        for idx, p in zip(batch_indices, pred):
            probs[idx] = p
            crop_valid[idx] = 1
            computed[idx] = 1

        batch_tensors.clear()
        batch_indices.clear()

    cap.release()
    face_detector.close()

    out_path = face_npz_to_occ_npz(face_npz)

    save_occ_npz(
        out_path=out_path,
        probs=probs,
        crop_valid=crop_valid,
        computed=computed,
        t_face=t_face,
        face_npz=face_npz,
        video_path=str(video_path),
    )

    return {
        "face_npz": str(face_npz),
        "video_path": str(video_path),
        "occ_npz": str(out_path),
        "t_face": t_face,
        "video_frames": video_frames,
        "needed_frames": len(needed_frames),
        "valid_needed_frames": len(valid_needed),
        "computed_frames": int(computed.sum()),
        "crop_valid_frames": int(crop_valid.sum()),
        "crop_fail": int(num_crop_fail),
        "read_fail": int(num_read_fail),
        "out_of_range": int(num_out_of_range),
    }


# ============================================================
# Main
# ============================================================

def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    print("[INFO] device:", DEVICE)
    print("[INFO] torch num threads:", torch.get_num_threads())
    print("[INFO] OpenCV threads limited to 1")
    print("[INFO] OCC_FRAME_STRIDE:", OCC_FRAME_STRIDE)

    print("[INFO] manifests:")
    for p in MANIFEST_PATHS:
        print("  -", p)

    print("[INFO] ckpt:", VIS_CNN_CKPT)
    print("[INFO] out_root:", OUT_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OCC_NPZ_ROOT.mkdir(parents=True, exist_ok=True)

    if FAIL_JSONL.exists():
        FAIL_JSONL.unlink()
    if MAP_JSON.exists():
        MAP_JSON.unlink()
    if SUMMARY_JSON.exists():
        SUMMARY_JSON.unlink()

    needed, manifest_stats = collect_needed_frames(MANIFEST_PATHS)

    print("\n[NEEDED]")
    print("unique face_npz:", len(needed))
    print("total needed frame refs unique-per-video:", sum(len(v) for v in needed.values()))

    clean_count = sum(1 for k in needed if str(k).startswith(str(CLEAN_FACEMESH_ROOT)))
    masked_count = sum(1 for k in needed if str(k).startswith(str(MASKED_FACEMESH_ROOT)))

    print("clean face_npz :", clean_count)
    print("masked face_npz:", masked_count)

    model = load_visibility_model(VIS_CNN_CKPT, DEVICE)

    face_to_occ: Dict[str, str] = {}
    per_file_results: List[Dict[str, Any]] = []

    debug_state = {"saved": 0}
    failures = 0

    for face_npz_str, frames in tqdm(sorted(needed.items()), desc="process face_npz"):
        face_npz = Path(face_npz_str)

        try:
            result = process_one_face_npz(
                face_npz=face_npz,
                needed_frames=frames,
                model=model,
                device=DEVICE,
                debug_state=debug_state,
            )

            per_file_results.append(result)
            face_to_occ[str(face_npz)] = result["occ_npz"]

        except Exception as e:
            failures += 1
            row = {
                "face_npz": str(face_npz),
                "error": f"{type(e).__name__}: {e}",
                "needed_frames": len(frames),
            }
            write_jsonl(FAIL_JSONL, row)

    with MAP_JSON.open("w", encoding="utf-8") as f:
        json.dump(face_to_occ, f, ensure_ascii=False, indent=2)

    total_computed_frames = int(sum(x["computed_frames"] for x in per_file_results))
    total_crop_valid_frames = int(sum(x["crop_valid_frames"] for x in per_file_results))
    total_crop_fail = int(sum(x["crop_fail"] for x in per_file_results))
    total_read_fail = int(sum(x["read_fail"] for x in per_file_results))
    total_out_of_range = int(sum(x["out_of_range"] for x in per_file_results))

    crop_valid_rate = (
        total_crop_valid_frames / total_computed_frames
        if total_computed_frames > 0
        else 0.0
    )

    summary = {
        "device": DEVICE,
        "ckpt": str(VIS_CNN_CKPT),
        "out_root": str(OUT_ROOT),
        "occ_npz_root": str(OCC_NPZ_ROOT),
        "occ_frame_stride": OCC_FRAME_STRIDE,
        "crop_method": "mediapipe_face_detection_auto_crop",
        "mediapipe_model_selection": MEDIAPIPE_MODEL_SELECTION,
        "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
        "face_pad_factor": FACE_PAD_FACTOR,
        "manifest_stats": manifest_stats,
        "num_face_npz_total": len(needed),
        "num_face_npz_clean": clean_count,
        "num_face_npz_masked": masked_count,
        "total_needed_frames": int(sum(len(v) for v in needed.values())),
        "num_success": len(per_file_results),
        "num_failures": failures,
        "total_computed_frames": total_computed_frames,
        "total_crop_valid_frames": total_crop_valid_frames,
        "crop_valid_rate": crop_valid_rate,
        "total_crop_fail": total_crop_fail,
        "total_read_fail": total_read_fail,
        "total_out_of_range": total_out_of_range,
        "map_json": str(MAP_JSON),
        "fail_jsonl": str(FAIL_JSONL),
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
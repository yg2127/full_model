"""
hyi fixed split 의 masked sample 들에 대해 face_crop 112×112 npz 추출.
output: face_path 와 같은 디렉토리에 *_crops112.npz 저장.

input:
  - masked_video_path  (mp4)
  - face5pt_path       (yolo_face bbox + det_score)
output:
  - face_path.parent / face_path.name.replace("_facemesh.npz", "_crops112.npz")
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np

CROP_SIZE = 112
PAD_RATIO = 0.10

def crop_frame(frame_bgr, bbox, detected):
    if not detected or bbox is None:
        return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    x1, y1, x2, y2 = bbox.astype(np.float32)
    H, W = frame_bgr.shape[:2]
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    side = max(x2-x1, y2-y1) * (1 + PAD_RATIO)
    if side <= 1: return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    half = side * 0.5
    sx1 = int(max(0, cx-half)); sy1 = int(max(0, cy-half))
    sx2 = int(min(W, cx+half)); sy2 = int(min(H, cy+half))
    if sx2 <= sx1 or sy2 <= sy1: return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    patch = frame_bgr[sy1:sy2, sx1:sx2]
    if patch.size == 0: return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    sh, sw = gray.shape
    if sh != sw:
        side_int = max(sh, sw)
        padded = np.zeros((side_int, side_int), dtype=gray.dtype)
        oy = (side_int-sh)//2; ox = (side_int-sw)//2
        padded[oy:oy+sh, ox:ox+sw] = gray
        gray = padded
    return cv2.resize(gray, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA).astype(np.uint8)

def process_item(it, det_thres=0.25):
    fp = Path(it["face_path"])
    out = fp.parent / fp.name.replace("_facemesh.npz", "_crops112.npz")
    if out.exists(): return ("skip", str(out))
    mvp = Path(it["masked_video_path"]); face5 = Path(it["face5pt_path"])
    if not mvp.exists(): return ("missing_video", str(mvp))
    if not face5.exists(): return ("missing_face5pt", str(face5))

    cap = cv2.VideoCapture(str(mvp))
    if not cap.isOpened(): return ("video_open_fail", str(mvp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with np.load(face5, allow_pickle=True) as d:
        bbox = d["bbox"].astype(np.float32)
        ds = d["det_score"].astype(np.float32)
        det = d["detected"].astype(bool)

    T = min(total, bbox.shape[0])
    images = np.zeros((T, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    detected = np.zeros((T,), dtype=bool)
    for i in range(T):
        ret, frame = cap.read()
        if not ret: break
        is_det = bool(det[i]) and float(ds[i]) >= det_thres
        images[i] = crop_frame(frame, bbox[i] if is_det else None, is_det)
        detected[i] = is_det
    cap.release()

    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"source_video": str(mvp), "n_frames": int(T), "fps": float(fps),
            "src_w": W, "src_h": H, "mask_region": it.get("mask_region"),
            "mask_appearance": it.get("mask_appearance"), "sample_key": it.get("sample_key")}
    np.savez_compressed(out, images=images, detected=detected, meta=meta)
    return ("done", f"{out.name} T={T} det={int(detected.sum())}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", default="/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json")
    ap.add_argument("--keys", nargs="+", default=["train_masked","val_masked","test_masked"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    split = json.load(open(args.split_json))
    items = []
    seen_keys = set()
    for k in args.keys:
        for it in split["items"][k]:
            sk = it["sample_key"]
            if sk in seen_keys: continue
            seen_keys.add(sk); items.append(it)
    if args.limit > 0: items = items[:args.limit]
    print(f"[start] {len(items)} unique masked samples", flush=True)

    t0 = time.time(); stats = {"done":0,"skip":0,"fail":0}
    for i, it in enumerate(items):
        status, msg = process_item(it)
        if status == "done": stats["done"] += 1
        elif status == "skip": stats["skip"] += 1
        else: stats["fail"] += 1
        if (i+1) % 5 == 0 or i == len(items)-1:
            el = time.time()-t0; eta = el/(i+1)*(len(items)-i-1)
            print(f"  [{i+1}/{len(items)}] {status}: {msg}  | done={stats['done']} skip={stats['skip']} fail={stats['fail']}  el={el:.0f}s eta={eta:.0f}s", flush=True)
        elif status not in ("done","skip"):
            print(f"  [{i+1}/{len(items)}] {status}: {msg}", flush=True)

    print(f"\n[final] {stats}  total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

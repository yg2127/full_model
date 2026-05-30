"""
Step D — Phase 2 fixed ORFormer 로 각 영상 frame 별 region visibility 추출.
hyi 의 occ npz 와 같은 형식으로 저장 → 우리 ORFormer α 가 hyi 의 mediapipe CNN OCC 자리에 그대로 들어감.

Output:
  cache_root / "alpha_npz" / <variant_or_clean> / <relative_path> / <name>_alpha.npz
    probs       : (T, 4) float16 — [left_eye, right_eye, nose, mouth] visibility prob = 1 - α
    crop_valid  : (T,)  uint8
    computed    : (T,)  uint8 (모든 frame inference 했으니 1)
    frame_idx   : (T,)  int32
    regions     : (4,)  ['left_eye_visible','right_eye_visible','nose_visible','mouth_visible']
    meta...

매핑 JSON (hyi 의 face_npz_to_occ_npz.json 와 동일 형식):
  cache_root / "face_npz_to_alpha_npz.json"
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, cv2

PRETRAIN_V4 = Path("/home/yg/fusion/pretrain_v4")
sys.path[:0] = [str(PRETRAIN_V4/"src"), str(PRETRAIN_V4/"src/data"),
                str(PRETRAIN_V4/"configs"), "/data/shared/orformer/vendor"]
from models.VQVAE import VQVAE
from models.simple_vit import ORFormer
from default import get_cfg
import torchvision.transforms as T
import models.quantizer as _q

# 16×16 patch grid 의 region 매핑 (face crop 이 center-aligned 라 가정)
# 추후 face_regions7 같은 anatomical region 정의로 교체 가능
REGIONS = ["left_eye_visible", "right_eye_visible", "nose_visible", "mouth_visible"]
REGION_PATCHES = {
    "left_eye_visible":  (slice(4, 9),  slice(2, 7)),    # rows 4-8, cols 2-6
    "right_eye_visible": (slice(4, 9),  slice(9, 14)),
    "nose_visible":      (slice(7, 11), slice(6, 10)),
    "mouth_visible":     (slice(11, 15), slice(4, 12)),
}

def build_model(orformer_ckpt: str, device):
    cfg = get_cfg(); ds_cfg = cfg.DMD
    vit = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3,
                   heads=8, mlp_dim=512, channels=256)
    m = VQVAE(h_dim=128, res_h_dim=32, output_dim=ds_cfg.NUM_EDGE, n_res_layers=2,
              n_embeddings=2048, embedding_dim=256, code_dim=256, beta=0.25, vit=vit).to(device).eval()
    ck = torch.load(orformer_ckpt, map_location=device, weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    return m

@torch.no_grad()
def get_alpha_batch(model, face_arr_u8, norm, device, bs=64):
    """face_arr_u8: (T, 112, 112) uint8 → (T, 16, 16) float32 occlusion α (0~1)."""
    T = len(face_arr_u8)
    alphas = np.zeros((T, 16, 16), dtype=np.float32)
    for s in range(0, T, bs):
        e = min(s+bs, T)
        batch = []
        for i in range(s, e):
            face_256 = cv2.resize(face_arr_u8[i], (256, 256))
            res_64 = cv2.resize(np.stack([face_256]*3, -1), (64, 64))
            batch.append(norm(res_64))
        inp = torch.stack(batch).to(device)
        out = model(inp)
        # out[8] = attention_weights = α (B, 16, 16, 1)
        alpha = out[8].squeeze(-1).cpu().numpy()
        alphas[s:e] = alpha
    return alphas

def alpha_to_region_probs(alphas):
    """alphas: (T, 16, 16) → (T, 4) region visibility prob (= 1 - mean α in region patches)."""
    T = alphas.shape[0]
    probs = np.zeros((T, len(REGIONS)), dtype=np.float32)
    for ri, name in enumerate(REGIONS):
        rs, cs = REGION_PATCHES[name]
        region_alpha = alphas[:, rs, cs].reshape(T, -1).mean(axis=1)  # (T,)
        probs[:, ri] = 1.0 - region_alpha
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", default="/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json")
    ap.add_argument("--orformer-ckpt", default="/home/yg/fusion/pretrain_v4/artifacts/phase2_orformer_fixed/best.pt")
    ap.add_argument("--cache-root", default="/data/shared/scuppy/yg/orformer_alpha_cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        _q.device = torch.device("cpu")
    model = build_model(args.orformer_ckpt, device)
    norm = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    split = json.load(open(args.split_json))
    cache_root = Path(args.cache_root); cache_root.mkdir(parents=True, exist_ok=True)
    alpha_dir = cache_root / "alpha_npz"; alpha_dir.mkdir(exist_ok=True)

    # 모든 unique face_path 수집
    seen = {}
    keys = ("train_clean_all","val_clean_all","test_clean_all",
            "train_masked","val_masked","test_masked")
    for k in keys:
        for it in split["items"][k]:
            fp = it["face_path"]
            if fp in seen: continue
            seen[fp] = it
    items = list(seen.values())
    if args.limit > 0: items = items[:args.limit]
    print(f"[start] {len(items)} unique samples (clean+masked all)", flush=True)

    face_to_alpha = {}
    t0 = time.time(); stats = {"done":0,"skip":0,"fail":0}
    for i, it in enumerate(items):
        fp = Path(it["face_path"])
        # clean: facemesh root → face_crops_112 root 로 swap
        # masked: 같은 디렉토리에 우리가 만든 *_crops112.npz 존재
        CLEAN_FM = "/data/shared/DMD_landmarks/facemesh"
        CLEAN_CROP = "/data/shared/DMD_landmarks/face_crops_112"
        if str(fp).startswith(CLEAN_FM):
            crop_npz = Path(str(fp).replace(CLEAN_FM, CLEAN_CROP).replace("_facemesh.npz", "_crops112.npz"))
        else:
            crop_npz = fp.parent / fp.name.replace("_facemesh.npz", "_crops112.npz")
        if not crop_npz.exists():
            stats["fail"] += 1
            print(f"  [{i+1}/{len(items)}] SKIP no_crop: {crop_npz}", flush=True)
            continue
        # 출력 path
        rel = fp.parent.relative_to(fp.parents[len(fp.parents)-3]) if False else fp.parent.name  # 그냥 평탄화
        # 더 명확: alpha_npz 의 mirror 구조
        # variant prefix 식별: face_path 가 hyi 의 occlusion v3 안이면 'masked', 아니면 'clean'
        prefix = "masked" if "Occlusion_subset_dataset" in str(fp) else "clean"
        out_path = alpha_dir / prefix / fp.relative_to(fp.anchor).parent / fp.name.replace("_facemesh.npz", "_alpha.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        face_to_alpha[str(fp)] = str(out_path)

        if out_path.exists():
            stats["skip"] += 1
            if (i+1) % 20 == 0:
                el = time.time()-t0
                print(f"  [{i+1}/{len(items)}] skip existing {out_path.name}  el={el:.0f}s", flush=True)
            continue

        with np.load(crop_npz) as d:
            face_arr = d["images"]   # (T, 112, 112) uint8
        alphas = get_alpha_batch(model, face_arr, norm, device)
        probs = alpha_to_region_probs(alphas).astype(np.float16)  # (T, 4)
        T_ = len(face_arr)
        np.savez_compressed(out_path,
            probs=probs,
            crop_valid=np.ones(T_, dtype=np.uint8),
            computed=np.ones(T_, dtype=np.uint8),
            frame_idx=np.arange(T_, dtype=np.int32),
            regions=np.array(REGIONS, dtype="<U17"),
            source_face_npz=str(fp), source_video=it.get("masked_video_path") or "",
            occ_frame_stride=np.int32(1),
            crop_method="hgnet_face_crop_112",
            source="orformer_phase2_fixed",
        )
        stats["done"] += 1
        el = time.time()-t0; eta = el/(i+1)*(len(items)-i-1)
        print(f"  [{i+1}/{len(items)}] {out_path.name}  T={T_}  done={stats['done']} skip={stats['skip']}  el={el:.0f}s eta={eta:.0f}s", flush=True)

    # 매핑 JSON 저장
    map_path = cache_root / "face_npz_to_alpha_npz.json"
    json.dump(face_to_alpha, open(map_path, "w"), indent=2)
    print(f"\n[final] {stats}  total {time.time()-t0:.0f}s")
    print(f"매핑 JSON: {map_path}  ({len(face_to_alpha)} entries)")

if __name__ == "__main__":
    main()

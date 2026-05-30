"""
Step 1 — hyi fixed split JSON 의 각 sample 의 face_path 옆에 *_hgnet478.npz 생성.
실행: PYTHONPATH=. python scripts/build_hgnet_cache_from_hyi_split.py

(README 의 코드 그대로 옮긴 것. Phase 3a v3 best.pt 가 준비된 후 실행)
"""
import json, sys, torch, cv2, numpy as np
from pathlib import Path

PRETRAIN_V4 = Path("/data/shared/scuppy/pretrain_v4")
sys.path[:0] = [str(PRETRAIN_V4/"src"), str(PRETRAIN_V4/"src/data"),
                str(PRETRAIN_V4/"configs"), "/data/shared/orformer/vendor"]
from models.VQVAE import VQVAE
from models.simple_vit import ORFormer
from models.StackedHGNet import IntergrationStackedHGNet
from heatmap_gen import denorm_points
from default import get_cfg
import torchvision.transforms as T

cfg = get_cfg(); ds_cfg = cfg.DMD
DEVICE = torch.device("cuda")
NORM = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

ORFORMER_CKPT = Path("/home/yg/fusion/pretrain_v4/artifacts/phase2_orformer_fixed/best.pt")
HGNET_CKPT    = Path("/home/yg/fusion/pretrain_v4/artifacts/phase3a_hgnet_478_v3/best.pt")
SPLIT_JSON    = Path("/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json")

vit = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3,
               heads=8, mlp_dim=512, channels=256)
orformer = VQVAE(h_dim=128, res_h_dim=32, output_dim=ds_cfg.NUM_EDGE, n_res_layers=2,
                 n_embeddings=2048, embedding_dim=256, code_dim=256, beta=0.25, vit=vit).to(DEVICE).eval()
orformer.load_state_dict(torch.load(str(ORFORMER_CKPT), map_location=DEVICE, weights_only=False)["model_state_dict"], strict=False)

edge_info = [list(x) for x in ds_cfg.EDGE_INFO]
hgnet = IntergrationStackedHGNet(
    classes_num=[ds_cfg.NUM_POINT, ds_cfg.NUM_EDGE, ds_cfg.NUM_POINT],
    edge_info=edge_info, nstack=4
).to(DEVICE).eval()
hgnet.load_state_dict(torch.load(str(HGNET_CKPT), map_location=DEVICE, weights_only=False)["hgnet_state_dict"], strict=True)

@torch.no_grad()
def hgnet_infer_batch(face_u8_arr, bs=32):
    """face_u8_arr: (T, 112, 112) uint8 → (T, 478, 2) float32 — batch inference."""
    T = len(face_u8_arr)
    out = np.zeros((T, 478, 2), dtype=np.float32)
    for s in range(0, T, bs):
        e = min(s+bs, T)
        inp_list, res_list = [], []
        for i in range(s, e):
            face = cv2.resize(face_u8_arr[i], (256,256))
            rgb = np.stack([face]*3, -1)
            inp_list.append(NORM(rgb))
            res_list.append(NORM(cv2.resize(rgb, (64,64))))
        inp = torch.stack(inp_list).to(DEVICE)
        res = torch.stack(res_list).to(DEVICE)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, ref_hm, *_ = orformer(res)
            _, lm = hgnet(inp, reference_heatmaps=ref_hm)
        out[s:e] = denorm_points(lm.float(), 64, 64).cpu().numpy() * (112/64)
    return out

split = json.load(open(SPLIT_JSON))
seen = set()
keys = ("train_clean_all","val_clean_all","test_clean_all",
        "train_masked","val_masked","test_masked")
n_total = sum(len(split["items"][k]) for k in keys)
done = 0
for k in keys:
    for it in split["items"][k]:
        fp = Path(it["face_path"])
        if fp in seen: continue
        seen.add(fp)
        # clean: facemesh root → face_crops_112 root 로 swap
        # masked: 같은 디렉토리에 우리가 만든 *_crops112.npz 존재
        CLEAN_FM = "/data/shared/DMD_landmarks/facemesh"
        CLEAN_CROP = "/data/shared/DMD_landmarks/face_crops_112"
        if str(fp).startswith(CLEAN_FM):
            crop_npz = Path(str(fp).replace(CLEAN_FM, CLEAN_CROP).replace("_facemesh.npz", "_crops112.npz"))
        else:
            crop_npz = fp.parent / fp.name.replace("_facemesh.npz", "_crops112.npz")
        if not crop_npz.exists():
            print(f"  SKIP (no face_crops): {crop_npz}"); continue
        # output: hg cache 는 fixed split 의 face_path 옆에 (face_npz_swap 으로 학습 시 lookup)
        out = fp.parent / fp.name.replace("_facemesh.npz", "_hgnet478.npz")
        if out.exists():
            done += 1; continue
        with np.load(crop_npz) as d: arr = d["images"]
        import time as _t; _ts = _t.time()
        print(f"  >> infer {crop_npz.name} T={len(arr)}", flush=True)
        lm_xy = hgnet_infer_batch(arr, bs=256)  # (T,478,2) — max throughput
        print(f"     infer done {_t.time()-_ts:.1f}s", flush=True)
        # hyi facemesh 호환: (T,478,3) with z=0, key='landmarks'
        landmarks = np.concatenate([lm_xy, np.zeros((*lm_xy.shape[:-1], 1), dtype=lm_xy.dtype)], axis=-1)
        meta = {"source": "hgnet_phase3a_v3", "n_frames": int(len(arr)),
                "ckpt": "phase3a_hgnet_478_v3/best.pt",
                "hyi_split_face_path": str(fp)}
        np.savez_compressed(out, landmarks=landmarks, detected=np.ones(len(arr), bool), meta=meta)
        done += 1
        print(f"  [{done}] {out.name}  T={len(arr)}", flush=True)
print(f"done {done} / {n_total}")

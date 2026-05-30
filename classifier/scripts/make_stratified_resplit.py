"""
stratified clip-level resplit — source(gaze/distraction) 별 비율 보존.
gaze 영상이 적으므로(29개) test gaze 충분 확보 위해 gaze는 5:2.5:2.5, distraction은 8:1:1.
같은 sample_key 의 clean+masked는 같은 split.

실행: python scripts/make_stratified_resplit.py --seed 42
"""
import json, argparse, random
from pathlib import Path
from collections import Counter

SRC = "/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json"

def split_keys(keys, ratios, rng):
    ks = list(keys); rng.shuffle(ks)
    n = len(ks); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
    return ks[:n_tr], ks[n_tr:n_tr+n_va], ks[n_tr+n_va:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gaze-ratios", nargs=3, type=float, default=[0.5, 0.25, 0.25])
    ap.add_argument("--distr-ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--out", default="/data/shared/scuppy/yg/Ablation/AblationB/configs/stratified_resplit_v1.json")
    args = ap.parse_args()

    d = json.load(open(args.src))
    clean_by_key, masked_by_key = {}, {}
    for k in ("train_clean_all","val_clean_all","test_clean_all"):
        for it in d["items"][k]: clean_by_key[it["sample_key"]] = it
    for k in ("train_masked","val_masked","test_masked"):
        for it in d["items"][k]: masked_by_key[it["sample_key"]] = it

    # source 별 분리
    gaze_keys  = sorted(k for k,it in clean_by_key.items() if it["source"]=="gaze")
    distr_keys = sorted(k for k,it in clean_by_key.items() if it["source"]=="distraction")
    print(f"gaze={len(gaze_keys)} distraction={len(distr_keys)}")

    rng = random.Random(args.seed)
    g_tr,g_va,g_te = split_keys(gaze_keys, args.gaze_ratios, rng)
    d_tr,d_va,d_te = split_keys(distr_keys, args.distr_ratios, rng)
    train_keys, val_keys, test_keys = g_tr+d_tr, g_va+d_va, g_te+d_te
    rng.shuffle(train_keys); rng.shuffle(val_keys); rng.shuffle(test_keys)
    print(f"train={len(train_keys)} (gaze {len(g_tr)}) | val={len(val_keys)} (gaze {len(g_va)}) | test={len(test_keys)} (gaze {len(g_te)})")

    def ss(it, sp): it=dict(it); it["split"]=sp; return it
    items = {}
    items["train_clean_all"] = [ss(clean_by_key[k],"train") for k in train_keys]
    items["val_clean_all"]   = [ss(clean_by_key[k],"val")   for k in val_keys]
    items["test_clean_all"]  = [ss(clean_by_key[k],"test")  for k in test_keys]
    items["train_clean_paired"]=items["train_clean_all"]
    items["val_clean_paired"]  =items["val_clean_all"]
    items["test_clean_paired"] =items["test_clean_all"]
    items["train_masked"]=[ss(masked_by_key[k],"train") for k in train_keys if k in masked_by_key]
    items["val_masked"]  =[ss(masked_by_key[k],"val")   for k in val_keys   if k in masked_by_key]
    items["test_masked"] =[ss(masked_by_key[k],"test")  for k in test_keys  if k in masked_by_key]
    items["train_clean_masked_1to1"]=items["train_clean_all"]+items["train_masked"]
    items["val_clean_masked_1to1"]  =items["val_clean_all"]+items["val_masked"]

    out = {
        "split_name": f"stratified_resplit_seed{args.seed}",
        "version": 1, "label_names": d["label_names"], "paths": d["paths"],
        "subjects": {"note":"stratified clip-level (gaze 5:2.5:2.5, distr 8:1:1). subject leakage 있음 — 실사용 참고용"},
        "items": items, "protocols": d["protocols"],
        "notes": [
            "stratified clip-level split: source별 비율 보존.",
            f"gaze {args.gaze_ratios}, distraction {args.distr_ratios}.",
            "test gaze 영상 충분 확보(=fixed split 동급) — gaze F1 안정.",
            "단 subject leakage 있음 (clip-level). baseline 비교 불가, 실사용 참고용.",
            f"source={args.src}, seed={args.seed}",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out,"w"), indent=1, ensure_ascii=False)
    # test gaze 수 검증
    tg = sum(1 for it in items["test_clean_all"] if it["source"]=="gaze")
    print(f"saved: {args.out}")
    print(f"  test_clean={len(items['test_clean_all'])} (gaze {tg}), test_masked={len(items['test_masked'])}")
    print(f"  train_clean_masked_1to1={len(items['train_clean_masked_1to1'])}")

if __name__ == "__main__":
    main()

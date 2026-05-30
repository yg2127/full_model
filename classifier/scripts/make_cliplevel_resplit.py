"""
hyi fixed split (subject-disjoint) JSON 의 item 들을 clip-level random 8:1:1 로 재분할.
같은 sample_key 의 clean+masked 는 같은 split 에 둠 (영상 leakage 방지).
출력: 동일 schema 의 새 JSON → 기존 학습 코드 (use_fixed_items_manifest) 그대로 사용 가능.

실행: python scripts/make_cliplevel_resplit.py --seed 42
"""
import json, argparse, random
from pathlib import Path
from collections import defaultdict

SRC = "/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])  # train/val/test
    ap.add_argument("--out", default="/data/shared/scuppy/yg/Ablation/AblationB/configs/cliplevel_resplit_v1.json")
    args = ap.parse_args()

    d = json.load(open(args.src))

    # 1. 모든 unique sample_key 의 clean / masked item 수집
    clean_by_key, masked_by_key = {}, {}
    for k in ("train_clean_all", "val_clean_all", "test_clean_all"):
        for it in d["items"][k]:
            clean_by_key[it["sample_key"]] = it
    for k in ("train_masked", "val_masked", "test_masked"):
        for it in d["items"][k]:
            masked_by_key[it["sample_key"]] = it

    keys = sorted(clean_by_key.keys())
    print(f"unique sample_key: {len(keys)}  (clean), masked 매칭: {len(masked_by_key)}")

    # 2. clip-level random shuffle + 8:1:1 분할 (subject 무관)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n = len(keys)
    n_tr = int(n * args.ratios[0])
    n_va = int(n * args.ratios[1])
    train_keys = keys[:n_tr]
    val_keys   = keys[n_tr:n_tr+n_va]
    test_keys  = keys[n_tr+n_va:]
    print(f"split: train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}")

    def set_split(it, sp):
        it = dict(it); it["split"] = sp; return it

    items = {}
    # clean
    items["train_clean_all"] = [set_split(clean_by_key[k], "train") for k in train_keys]
    items["val_clean_all"]   = [set_split(clean_by_key[k], "val")   for k in val_keys]
    items["test_clean_all"]  = [set_split(clean_by_key[k], "test")  for k in test_keys]
    # clean_paired = clean_all (모든 clean 에 masked 짝 있음)
    items["train_clean_paired"] = items["train_clean_all"]
    items["val_clean_paired"]   = items["val_clean_all"]
    items["test_clean_paired"]  = items["test_clean_all"]
    # masked
    items["train_masked"] = [set_split(masked_by_key[k], "train") for k in train_keys if k in masked_by_key]
    items["val_masked"]   = [set_split(masked_by_key[k], "val")   for k in val_keys   if k in masked_by_key]
    items["test_masked"]  = [set_split(masked_by_key[k], "test")  for k in test_keys  if k in masked_by_key]
    # 1to1
    items["train_clean_masked_1to1"] = items["train_clean_all"] + items["train_masked"]
    items["val_clean_masked_1to1"]   = items["val_clean_all"]   + items["val_masked"]

    out = {
        "split_name": f"cliplevel_resplit_seed{args.seed}",
        "version": 1,
        "label_names": d["label_names"],
        "paths": d["paths"],
        "subjects": {"note": "clip-level random split (subject 무관) — subject leakage 있음, upper-bound 참고용"},
        "items": items,
        "protocols": d["protocols"],
        "notes": [
            "clip-level random 8:1:1 split (NOT subject-disjoint).",
            "같은 sample_key 의 clean+masked 는 같은 split — 영상 leakage 없음.",
            "단 같은 subject 가 train/test 양쪽에 있을 수 있음 (subject leakage) → 실사용 시나리오 참고용, baseline 비교 불가.",
            f"source split: {args.src}, seed={args.seed}",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"saved: {args.out}")
    print(f"  train_clean_masked_1to1: {len(items['train_clean_masked_1to1'])}")
    print(f"  val_clean_masked_1to1:   {len(items['val_clean_masked_1to1'])}")
    print(f"  test_clean_paired:       {len(items['test_clean_paired'])}")
    print(f"  test_masked:             {len(items['test_masked'])}")

if __name__ == "__main__":
    main()

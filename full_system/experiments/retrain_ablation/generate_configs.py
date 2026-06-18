#!/usr/bin/env python
"""재학습 Ablation config 생성기 (단일 소스).

base 모델(model4_occgateRAW_explicitRegionScalarMaskGate, Full_System 탑재)에서
모듈을 1개씩만 제거한 6개 변형 config 를 `modules/<variant>/config.yaml` 로 생성한다.
각 변형은 base 와 '딱 한 가지'만 다르다 (아래 MODULES 표 참조).

사용:  /data/shared/envs/scuppy/bin/python generate_configs.py
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
CLS  = Path("/data/shared/scuppy/Full_System/vendor/classifier")
BASE_CFG = CLS / "configs/model4_occgateRAW_explicitRegionScalarMaskGate_seed42_loss045.yaml"
SEED = 42

# ── 모듈별 ablation 정의: (이름, 한 줄 설명, config 편집 함수) ───────────────
def e_full(c):     pass                                                  # baseline
def e_no_body(c):  c["ablation"] = {"zero_pose": True}                    # 신체(pose) 분기 제거
def e_no_face(c):  c["ablation"] = {"zero_face": True}                    # 얼굴 랜드마크 분기 제거
def e_no_occ(c):                                                         # Occ 차폐 신호 제거
    c["model"]["fusion"]["region_gate_condition_occ"] = False
    c["model"]["fusion"]["scalar_gate_condition_occ"] = False
def e_no_hgnet(c): c["face"]["npz_swap"]["enabled"] = False              # HGNet 복원 제거(raw MediaPipe)
def e_no_gate(c):  c["model"]["fusion"] = {"kind": "concat_condition",   # 차폐-인지 fusion 게이트 제거
                                           "occ_hidden_dim": 64, "occ_dropout": 0.1}

MODULES = [
    ("full",     "baseline (제거 없음)",                              e_full),
    ("no_body",  "신체(pose) 분기 제거  [ablation.zero_pose=true]",   e_no_body),
    ("no_face",  "얼굴 랜드마크 분기 제거  [ablation.zero_face=true]", e_no_face),
    ("no_occ",   "Occ 차폐 신호 제거  [gate occ-condition off]",       e_no_occ),
    ("no_hgnet", "HGNet 복원 제거  [face.npz_swap.enabled=false]",     e_no_hgnet),
    ("no_gate",  "차폐-인지 fusion 게이트 제거  [fusion=concat_condition]", e_no_gate),
]

def main():
    base = yaml.safe_load(open(BASE_CFG))
    runs = HERE / "runs"
    for name, desc, edit in MODULES:
        c = copy.deepcopy(base)
        c["seed"] = SEED
        c["paths"]["save_root"]   = str(runs / f"{name}_seed{SEED}")
        c["paths"]["results_root"] = str(runs)
        edit(c)
        mod_dir = HERE / "modules" / name
        mod_dir.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(c, open(mod_dir / "config.yaml", "w"), sort_keys=False)
        (mod_dir / "README.md").write_text(
            f"# {name}\n\n**제거 모듈**: {desc}\n\n"
            f"- base = `model4_occgateRAW_explicitRegionScalarMaskGate` 에서 **이 한 가지만** 변경\n"
            f"- 학습:  `python -m src.training.train --config modules/{name}/config.yaml` (vendor/classifier 에서)\n"
            f"- 결과:  `runs/{name}_seed{SEED}/summary.json` (test_clean / test_masked 의 head별 clip_f1_macro)\n",
            encoding="utf-8")
        print(f"[ok] modules/{name}/config.yaml  — {desc}")
    print("\n6개 변형 config 생성 완료 →", HERE / "modules")

if __name__ == "__main__":
    main()

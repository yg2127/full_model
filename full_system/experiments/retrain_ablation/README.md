# 재학습 Ablation — 같은 모델, 모듈 1개씩 제거 후 재학습

base 모델(`model4_occgateRAW_explicitRegionScalarMaskGate`, Full_System 탑재)에서
**모듈을 하나씩만 제거**하고 **처음부터 재학습**한 뒤, GT 기반 test F1 을 비교하는 실험.
(입력 zero / agreement 같은 대용지표가 아니라, 모듈 없이 새로 학습한 모델의 실제 성능 비교)

## 폴더 구조 (ablation 학습코드 한곳)

```
retrain_ablation/
├─ README.md                  ← (이 파일) 개요·실행법·모듈 표
├─ base_config.yaml           ← base 모델 config 스냅샷 (참조용)
├─ generate_configs.py        ← 모듈 정의 단일 소스 → modules/<v>/config.yaml 생성
├─ train_all.sh               ← 학습 런처 (modules/ 사용).  bash train_all.sh [변형…]
├─ make_compare_notebook.py   ← compare.ipynb 생성기
├─ compare.ipynb              ← 결과 비교/시각화 (학습 완료 후 실행)
├─ modules/                   ← ★ 모듈별 정리
│  ├─ full/      { config.yaml, README.md }
│  ├─ no_body/   { config.yaml, README.md }
│  ├─ no_face/   { config.yaml, README.md }
│  ├─ no_occ/    { config.yaml, README.md }
│  ├─ no_hgnet/  { config.yaml, README.md }
│  └─ no_gate/   { config.yaml, README.md }
└─ runs/<v>_seed42/           ← 학습 산출물 (best.pt, summary.json, confusion CSV …)
```

> `configs/`, `run_all.sh`, `run_all.log` 은 **최초 학습 실행(진행 중)** 이 쓰던 평면 복사본이다.
> `configs/<v>_seed42.yaml` == `modules/<v>/config.yaml` (동일). 최초 학습이 끝나면 `configs/` 와
> `run_all.sh` 는 지워도 되고, 이후로는 `modules/` + `train_all.sh` 만 쓰면 된다.

## 모듈별 ablation (base 에서 '딱 한 가지'만 변경)

| 변형 | 제거 모듈 | config 변경 |
|---|---|---|
| `full`     | (없음, baseline)        | — |
| `no_body`  | 신체(pose) 분기          | `ablation.zero_pose=true` |
| `no_face`  | 얼굴 랜드마크 분기        | `ablation.zero_face=true` |
| `no_occ`   | Occ 차폐 신호            | gate `region/scalar_gate_condition_occ=false` |
| `no_hgnet` | HGNet 복원              | `face.npz_swap.enabled=false` (raw MediaPipe) |
| `no_gate`  | 차폐-인지 fusion 게이트   | `model.fusion.kind=concat_condition` |

- 통제: 동일 seed(42)·동일 데이터 split(fixed manifest)·동일 하이퍼파라미터.
- `no_occ` 주: explicit gate 가 occ.enabled=true 를 요구하므로 occ 는 켜두되 게이트가 occ 를
  조건으로 쓰지 않도록 꺼서 occ '신호'를 제거한다 (잔여 영향 ≈0).

## 실행

```bash
# 1) (선택) config 재생성
/data/shared/envs/scuppy/bin/python generate_configs.py

# 2) 학습 (vendor/classifier 의 학습 파이프라인 사용, 순차)
bash train_all.sh                 # 6개 전부
bash train_all.sh no_occ no_gate  # 일부만

# 3) 결과 비교 (모든 runs/*/summary.json 읽어 F1 / F1 drop 시각화)
jupyter nbconvert --to notebook --execute --inplace compare.ipynb
```

각 학습은 끝에서 자동으로 `test_clean` / `test_masked` 를 평가해
`runs/<v>_seed42/summary.json` 에 head별 `clip_f1_macro` 를 저장한다.
`compare.ipynb` 가 `full` 대비 각 변형의 **F1 drop**(= 모듈별 진짜 기여도)을 표·그래프로 보여준다.

## 학습 파이프라인 (외부, 공유)

학습 코드 본체는 `Full_System/vendor/classifier/` (DMS 분류기) 이며 수정하지 않는다.
이 폴더는 그 위에서 **ablation config + 런처 + 비교**만 모아둔 래퍼다.

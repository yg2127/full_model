# no_body

**제거 모듈**: 신체(pose) 분기 제거  [ablation.zero_pose=true]

- base = `model4_occgateRAW_explicitRegionScalarMaskGate` 에서 **이 한 가지만** 변경
- 학습:  `python -m src.training.train --config modules/no_body/config.yaml` (vendor/classifier 에서)
- 결과:  `runs/no_body_seed42/summary.json` (test_clean / test_masked 의 head별 clip_f1_macro)

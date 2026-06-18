# no_face

**제거 모듈**: 얼굴 랜드마크 분기 제거  [ablation.zero_face=true]

- base = `model4_occgateRAW_explicitRegionScalarMaskGate` 에서 **이 한 가지만** 변경
- 학습:  `python -m src.training.train --config modules/no_face/config.yaml` (vendor/classifier 에서)
- 결과:  `runs/no_face_seed42/summary.json` (test_clean / test_masked 의 head별 clip_f1_macro)

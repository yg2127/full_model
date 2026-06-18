# no_gate

**제거 모듈**: 차폐-인지 fusion 게이트 제거  [fusion=concat_condition]

- base = `model4_occgateRAW_explicitRegionScalarMaskGate` 에서 **이 한 가지만** 변경
- 학습:  `python -m src.training.train --config modules/no_gate/config.yaml` (vendor/classifier 에서)
- 결과:  `runs/no_gate_seed42/summary.json` (test_clean / test_masked 의 head별 clip_f1_macro)

# no_occ

**제거 모듈**: Occ 차폐 신호 제거  [gate occ-condition off]

- base = `model4_occgateRAW_explicitRegionScalarMaskGate` 에서 **이 한 가지만** 변경
- 학습:  `python -m src.training.train --config modules/no_occ/config.yaml` (vendor/classifier 에서)
- 결과:  `runs/no_occ_seed42/summary.json` (test_clean / test_masked 의 head별 clip_f1_macro)

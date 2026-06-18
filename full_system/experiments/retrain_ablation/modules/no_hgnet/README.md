# no_hgnet

**제거 모듈**: HGNet 복원 제거  [face.npz_swap.enabled=false]

- base = `model4_occgateRAW_explicitRegionScalarMaskGate` 에서 **이 한 가지만** 변경
- 학습:  `python -m src.training.train --config modules/no_hgnet/config.yaml` (vendor/classifier 에서)
- 결과:  `runs/no_hgnet_seed42/summary.json` (test_clean / test_masked 의 head별 clip_f1_macro)

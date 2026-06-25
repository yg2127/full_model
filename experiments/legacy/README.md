# Legacy — 구 model4 산출물 (superseded)

최종 모델 **OcclusionGateNet**(= `full_system/`, `explicit_region_scalar_mask_gate` 변형)으로 대체된
초기 model4 변형의 결과를 참고용으로 보존한다.

- `model4_occgateRAW_taskGated_occCNN_seed42/` — 구 model4 (`task_gated_late` fusion) 학습/평가 결과.
  gaze clip-F1 clean 0.600 / masked 0.546. 최종본(0.6355 / 0.5831)에 의해 갱신됨.

> 코드는 제거하지 않았다(최종 모델이 `classifier/`·`landmark/` 코어와 `task_feature_fusion` 베이스를
> 공유하므로). 여기엔 superseded 된 **결과물**만 모아둔다.

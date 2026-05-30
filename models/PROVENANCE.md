# 체크포인트 Provenance — 각 모델이 어떤 학습코드/설정으로 만들어졌나

ckpt 메타(`args`/`note`) + 학습 스크립트에서 역추적한 기록. **model4 사용여부**까지 표기.
경로는 (repo 내 학습코드) / (원본 ckpt 위치) 순.

---

## 1. ORFormer — `models/orformer/best.pt`  ✅ model4 사용 (occgateRAW 생성)
- **학습코드**: `landmark/scripts/train_phase2_orformer.py`  (실행: `landmark/scripts/run_phase2_fixed.sh`)
- **설정(args)**: codebook_weights=`phase1_codebook/best.pt`, gt_source=mediapipe, dataset=DMD, batch 64, lr_vit 1e-4, **lr_codebook 0.0**(codebook frozen), T_0 5/T_mult 2, epoch 30 → best epoch 5
- 원본: `pretrain_v4/artifacts/phase2_orformer_fixed/best.pt`
- 역할: HGNet 의 edge reference heatmap 제공

## 2. VQ-VAE codebook — `models/codebook/best.pt`  ⚠️ 여분(orformer best.pt 에 포함)
- **학습코드**: `landmark/scripts/train_phase1_codebook.py`  (실행: `run_phase1.sh`)
- **설정**: gt_source=mediapipe, batch 64, lr 1e-4, n_embeddings 2048, embedding_dim 256, epoch 30
- 원본: `pretrain_v4/artifacts/phase1_codebook/best.pt`

## 3. HGNet phase3a (best, NME 4.44) — `models/hgnet_phase3a/best.pt`  ⚠️ model4 미사용
- **학습코드**: `landmark/scripts/train_phase3_hgnet.py`  (실행: `run_phase3a_478.sh`)
- **설정**: orformer_weights=`phase2_orformer/best.pt`(_fixed 아님!), init=`phase3a_backup/hgnet_ep0.pt`, gt mediapipe → **best_nme 4.444 (epoch 3)**
- 원본: `pretrain_v4/artifacts/phase3a_hgnet_478/best.pt`
- 비고: NME 가장 좋지만 **model4 occgateRAW 는 v3 로 생성됨**(아래). 우리 inference_demo/finetune init 용.

## 4. HGNet v2 (NME 4.93) — `models/hgnet_phase3a_v2/best.pt`  ⚠️ model4 미사용
- **학습코드**: `train_phase3_hgnet.py`  (실행: `run_phase3a_478_v2.sh`)
- **설정**: orformer=`phase2_orformer_fixed/best.pt`, init=`phase3a_478/hgnet_ep3_for_warmstart.pt` → best_nme 4.928 (epoch 0)
- 원본: `pretrain_v4/artifacts/phase3a_hgnet_478_v2/best.pt`

## 5. HGNet v3 (NME 4.96) — `models/hgnet_phase3a_v3/best.pt`  ✅✅ **model4 occgateRAW 좌표 생성에 실제 사용**
- **학습코드**: `train_phase3_hgnet.py`  (실행: `run_phase3a_478_v3.sh`)
- **설정**: orformer=`phase2_orformer_fixed/best.pt`, init=`v2/hgnet_ep0_for_warmstart.pt` → best_nme 4.963 (epoch 0)
- 원본: `pretrain_v4/artifacts/phase3a_hgnet_478_v3/best.pt`
- **근거**: `*_hgnet478.npz` 메타 `ckpt: phase3a_hgnet_478_v3/best.pt`. 캐시 생성코드 `classifier/scripts/build_hgnet_cache_from_hyi_split.py` 의 HGNET_CKPT 가 v3.

## 6. 분류기 model4 (ST-GCN + face branch) — `models/classifier_model4/best.pt`  ✅ model4 본체
- **학습코드**: `classifier/src/training/train.py --config configs/generated/model4_occgateRAW_taskGated_occCNN_seed42.yaml`
- **설정(ckpt 내 config 저장됨)**: fusion=task_gated_late, face=occgateRAW(npz_swap), occ=`_occ_pred`, loss α 1.0/0.5/0.3/0.2, best_score 0.3/0.45/0.15/0.1 → best epoch 10, best_score 0.784
- 원본: `AblationB/results/model4_occgateRAW_taskGated_occCNN_seed42/best.pt`

## 7. occ CNN (model4 의 x_occ 가림확률 생성)  ❗ **권한거부 — 미포함**
- **실제 ckpt**: `/home/hyi/Code/Step9_extract_crop_npz/best.pt`  (hyi 홈, 권한거부)
- **근거**: model4 occ 소스 = `_occ_pred/face_npz_to_occ_npz.json`; 그 캐시의 `occ_generation_summary.json` → `"ckpt": "/home/hyi/Code/Step9_extract_crop_npz/best.pt"`
- **생성코드(Step9)**: `/home/hyi/Code/Step9_extract_crop_npz/` (권한거부)
- **모델정의(접근가능)**: `scuppy/external_scripts/hyi_masking/Step3_full_dir/3_task_train.py` 의 `TinyRegionCNN`
- model4 추론 시엔 이 ckpt 로 만든 `_occ_pred` 캐시(npz)를 읽음(분류기는 occ CNN 미로드).

## 8. occ CNN (내 재학습본) — `models/occ_cnn_retrain_mine/best.pt`  ❌ **model4 무관**
- **학습코드**: `pipeline/train_occ_cnn.py`
- **설정(note)**: face_crops_112 도메인, var_label 7종, val_macro_f1 0.958 (epoch 17)
- 원본: `scuppy/yg/occ_cnn_v1/best.pt`
- 비고: hyi Step9 대용으로 내가 따로 학습. **model4 의 occ_pred 와 다른 모델** — 혼동 주의.

---

## model4 가 실제 의존하는 ckpt 요약
| ckpt | 학습코드 | model4 |
|---|---|---|
| 분류기 model4 | `classifier/src/training/train.py` | ✅ 본체 |
| HGNet **v3** | `landmark/scripts/train_phase3_hgnet.py` (run_phase3a_478_v3.sh) | ✅ occgateRAW 좌표 |
| ORFormer | `landmark/scripts/train_phase2_orformer.py` | ✅ HGNet reference |
| facemesh (mediapipe) | (pip, 학습 아님) | ✅ clean GT |
| occ CNN **hyi Step9** | `/home/hyi/Code/Step9_extract_crop_npz/` (권한거부) | ✅ x_occ (occ_pred 캐시) |

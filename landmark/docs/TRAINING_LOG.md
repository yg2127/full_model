# Pretrain V4 학습 일지

> 친구 ORFormer 공식 구현 fork → DMD IR + 6 variant 가림 dataset 으로 적용한 학습 기록.

## Timeline 요약

| 단계 | 결과 | 핵심 사건 |
|---|---|---|
| Phase 0 | ✅ 완료 | 코드 fork, dataset 어댑터, edge_info, heatmap_gen |
| Phase 0.5 | ✅ 완료 | occface GT 무용성 확인 → mediapipe cache 생성 |
| Mediapipe cache | ✅ 완료 | 1.53M frames, 86.1% detection, 18.6min |
| Phase 1 (Fix A) | ❌ 실패 | codebook scratch — perplexity 1.0 collapse |
| Phase 1 (B-2 warm-start) | ✅ 성공 | friend codebook 사용, val_recon 0.00108 |
| 속도 최적화 | ✅ | pointmap vectorize 17× ↑, workers 4→12 |
| Phase 2 (초기 0.1/warmup 10) | ❌ 실패 | α discrimination 안 됨, collapse |
| **Phase 2 D (aux 0.5/warmup 0)** | 🔄 진행 중 | step 0 에서 perp 65 활성 |

## Phase 0 — 인프라 (2026-05-15)

친구 코드 (`/data/shared/orformer/ORFormer/`) 를 fork:

- `src/models/` ← StackedHGNet, VQVAE, simple_vit, encoder, decoder, quantizer, residual
- `src/utils/` ← heatmap, transform, get_transforms
- **수정 1개**: `quantizer.py` 의 vit-mode 에서 commitment loss 활성화 + STE 추가 (codebook unfreeze 위해)

우리 작성:
- `configs/default.py` — `_C.DMD` (478 점, 12 edge), `_C.DMD_68` (68 점, 13 edge), GT_SOURCE switch
- `src/data/face_edge_info.py` — MediaPipe FACEMESH_* → polyline 자동 build, 478→68 mapping
- `src/data/heatmap_gen.py` — point + edge heatmap 생성
- `src/data/dataset_dmd.py` — `DMDHeatmapDataset` (친구 COFW 패턴, GT swap 지원)

## Phase 0.5 — GT 정확도 검증 (2026-05-15)

**원래 GT (occface predictor)** 의 정확도 의심 → MediaPipe FaceMesh 재추출과 비교.

`notebooks/phase0_5_verify.ipynb` 작성, 19 cells:
- variant 별 face + landmark overlay
- edge heatmap 12 polylines 시각화
- point heatmap aggregate
- manifest mask 일치 확인
- 68 subset visualization
- **occface vs MediaPipe 픽셀 차이 측정**

**결과**:
```
clip 별 평균 L1 difference:
  clip A: 14.2 / 11.4 / 16.2 px
  clip B: 17.5 / 17.2 / 18.9 px
  clip C: 22.9 / 21.3 / 10.5 px
  overall avg: 16.7 px @ 112 coord ≈ 38 px @ 256 coord
```

→ face 의 ~15% 영역 어긋남. occface predictor 가 IR 에 부적합 (memory `occface-predictor-manifest-hard-mask-visibility-supervision` 와 일치).

**결정**: MediaPipe cache 새로 생성, GT_SOURCE=mediapipe.

## MediaPipe Cache 생성 (2026-05-15)

`scripts/build_facemesh_cache.py`:
- 입력: `face_crops_112/.../*_crops112.npz`
- mediapipe FaceMesh refined (478 lm)
- face_crop 112 → 256 upscale (mediapipe 작은 사이즈에 약함) → inference → 좌표 112 로 환원
- spawn multiprocessing, 8 workers

**결과**: 18.6 min 소요, 1.53M frames, **86.1% detection rate**.

## Phase 1 — VQ-VAE Codebook 학습

### 시도 1: Scratch (Fix A) — 실패 (2026-05-16)

**setup**:
- codebook init: scratch (uniform ±1/2048)
- commitment β: 0.25
- lr: 1e-4, batch 64, workers 4

**결과**: 50 step 만에 **perplexity 1.0 collapse**. dead codebook.

**Fix A 시도**:
- codebook init scale ↑: `normal(std=1/√256) ≈ 0.0625` (125× ↑)
- β: 0.25 → 0.5

**Fix A 결과**: 여전히 50 step 만에 collapse. init scale + β 로는 부족.

**진단**:
- VQ-VAE codebook 학습은 본래 매우 불안정
- 원논문 author 도 batch 128 × 수백 epoch GPU 로 학습
- 우리의 작은 setup 으론 부족
- edge_heatmap GT 가 sparse → decoder 가 codebook 무시하고 trivial recon 가능

### 시도 2: Warm-start (B-2) — 성공 ✅ (2026-05-17)

**전략**: 친구의 author pretrained codebook (`cofw_full_cofw_orformer_*/best_model.pt`) 로 warm-start.

**setup 변경**:
- `--warm-start-ckpt` 옵션 추가 (encoder + codebook + decoder 부분 load, decoder 의 마지막 conv 만 NUM_EDGE 14→12 mismatch 로 skip + random init)
- codebook lr: 1e-6 (frozen 비슷, scale 보존)
- 나머지 lr: 1e-4
- β: 0.25

**Step 0 → 100 추세**:
```
step  recon    commit   perplexity
  0   0.355    146.6    6.1     ← warm-start 시작 (visible RGB encoder ↔ IR mismatch)
 50   0.035    5.97     6.5
100   0.0019   2.00     5.9     ← commit 73× 감소, encoder IR 적응 중
500   0.0011   0.060    9.7
1000  0.0010   0.009    11~12   ← 안정화
```

### 속도 최적화 (2026-05-17)

**문제**: 학습이 매우 느림 (38분 / 50step 측정).

**진단**:
- workers 4, GPU util 11% (다른 사용자 점유 + dataloader CPU bound)
- `generate_pointmap` 의 per-point loop: **270 ms / sample**
- disk IO: 5 MB/s (idle, bottleneck X)
- 진짜 bottleneck = CPU heatmap generation

**대처**:
1. `generate_pointmap` numpy vectorize → **16 ms/sample** (17× ↓)
2. workers 4 → 12 (CPU 풀 활용)
3. cache pre-generation 검토 → 불필요 (vectorize 만으로 충분)

### Phase 1 최종 결과 (2026-05-18 02:51)

**ep0 metrics** (6.5h 소요):
```
train_recon  = 0.0039
train_commit = 0.504
train_perp   = 9.97

val_recon    = 0.00108  ✓
val_perp     = 10.36    ✓
```

**Ep1 추세**:
- step 200: perp 13.9
- step 500: perp 14.8 (peak)

**판단**: ep1 도 좋은 추세였지만 Phase 2 더 흥미로워서 stop. best.pt 활용.

## Phase 2 — ORFormer + Codebook + α Learning

### 시도 1: 기본 setup — 실패 (2026-05-18)

**setup**:
- Phase 1 best.pt warm-start
- ORFormer (ViT) depth 3, 3.42M params 새로 init
- codebook unfreeze (lr 5e-5)
- ORFormer lr 1e-4
- mix_prob 0.5 (가린 sample 포함)
- batch 32, workers 12
- **lambda-aux 0.1, warmup 10 ep**

**진행**:
- Ep0 (5.4h): perp 5.3, α_normal=0.076, α_occ=0.075
- Ep1 (4.8h): perp 3.2, α_normal=0.039, α_occ=0.039
- Ep2 진행 중: perp 3.3 plateau

**핵심 문제** (양쪽 ep 모두):
- **α_normal ≈ α_occ** — ORFormer 가 가린/정상 sample 구분 못 함
- α 가 0 으로 수렴 — messenger 사용 안 함, 일반 ViT 처럼 작동
- perp 3.2 plateau — codebook 활성 entry 3 개

**원인**:
- aux warmup 중 weight=0 → α supervision 없음
- implicit α 학습이 trivial direction 으로 collapse (α → 0)
- 5h × 10 ep warmup = 50 시간 의미 없는 학습 진행

### 시도 2: 옵션 D — 진행 중 (2026-05-18 16:13~)

**옵션 비교 후 D 선택**:
- B: warmup 1ep + weight 0.2 — 보수적
- C: warmup 0 + weight 0.3
- **D: warmup 0 + weight 0.5** ★ 강한 manifest 가이드
- F: Phase 2 포기 → V3 style — last resort

**변경 사항**:
```
aux warmup:  10 ep → 0       (즉시 supervision)
aux weight:  0.1  → 0.5      (implicit 보다 manifest dominant)
나머지: 동일
```

**Step 0 결과**:
```
recon  = 0.0010
commit = 0.0013
aux    = 0.6799   ★ w=0.5 적용 (gradient flow 시작)
perp   = 65.2     ★ 이전 51.7 보다 ↑ (codebook 더 다양하게 활용)
```

**Ep0 결과 (2026-05-18 23:00 경, 7.5h 소요)**:
```
train_recon  = 0.00101
train_commit = 0.00108
train_aux    = 0.042       (이전 Phase 2 의 0.39 와 비교 9× ↓)
train_perp   = 28.7        (이전 Phase 2 의 3.2 와 비교 9× ↑)
val_recon    = 0.00108

★ α_normal   = 0.0106
★ α_occluded = 0.1992      ← 19× 차이, discrimination 명확!
```

→ **D 옵션 성공 확정**. α_occ - α_normal = 0.188 > 평가기준 0.10. messenger mechanism 정상 작동.

**Ep1 진행 중** (step 4050/7059, 57%):
- aux 0.002~0.03 (더 작음)
- perp 14~36 (안정)
- 추세 유지 → ep1 더 좋은 α discrimination 기대

### 시도 3: D3 (batch 64 + workers 12, warm-start from D ep0 best) — 진행 중 (2026-05-19)

**의도**: D ep0 best 를 warm-start 로 batch 늘려 학습 속도 ↑.

**setup 변경 (D2 → D3)**:
- D2 (workers 16) → thrashing (load 33) → 효과 미미
- D3: workers 12 + batch 64 로 결정

**Ep0 결과 (5.6h 소요)**:
```
train_recon  = 0.0010
train_commit = 0.0009
train_aux    = 0.021       ← D ep0 의 0.042 의 절반
train_perp   = 29.2
val_recon    = 0.0011

★ α_normal   = 0.006       ← D ep0 의 0.011 → 45% 더 작아짐
★ α_occluded = 0.204       ← D ep0 의 0.199 와 비슷
★ 비율       = 34×          ← D ep0 의 18× 의 2배
```

→ **warm-start 효과 명확**. α discrimination 이 더 강해짐 (18× → 34×).

**ep 시간**: D (7.5h) → D3 (5.6h), batch 2× 효과로 ~25% 가속.

**ep1 진행 중** (step 200 시점):
- aux 0.032, perp 40.7

## 자원 / 환경 메모

### GPU 상황 변화

| 시점 | GPU util | 점유 사용자 |
|---|---|---|
| Phase 1 시작 (5/16) | 92% (hyi) | hyi 의 gate_fusion |
| Phase 1 중간 (5/17) | 0~95% 변동 | hyi 끝남, 다른 사용자 시작 |
| Phase 2 시작 (5/18) | 11~85% | 여러 사용자 |

### 자원 측정

- 메모리: 119GB 총량, Phase 1+2 학습 시 ~40GB 사용 (workers 12 + main)
- 시스템 load: 17~25 (20 core 시스템, 다른 사용자 포함)
- Disk IO: 5 MB/s (idle, bottleneck 아님)

### 학습 속도 (workers 12 기준)

- Phase 1: 6.5h / ep (3529 step, batch 64)
- Phase 2: 5.3h / ep (7059 step, batch 32)

## 핵심 통찰

1. **VQ-VAE codebook scratch 학습 = 매우 어려움**. Phase 1 의 가장 큰 교훈. 친구의 author pretrained codebook 사용 = 정답.

2. **occface predictor 완전 무용** (Phase 0.5). MediaPipe cache 새로 생성 = 정공법.

3. **CPU heatmap generation 이 실제 bottleneck** (GPU 아님). pointmap vectorize 가 17× 가속.

4. **ORFormer 의 implicit α 학습 = collapse 위험** (Phase 2 초기 실패). manifest BCE auxiliary 가 필수. weight + warmup 매우 조심스럽게 조정 필요.

5. **friend codebook (visible RGB) → IR domain gap 은 commit loss 로 측정 가능**. step 0 의 commit 146 → step 100 commit 2 → 성공적 IR 적응.

## 다음 step 결정 매트릭스

| Phase 2 D 결과 | 다음 |
|---|---|
| α_occ > α_normal + 0.1 (성공) | Phase 2 계속, ep5~10 까지 학습 |
| α 둘 다 0 (실패) | F (V3 style) — codebook + messenger 제거, 단순 ViT + manifest BCE |
| 중간 (차이 약함) | aux weight 0.5 → 0.8 로 더 강하게 |

## 자산 위치

| | 경로 |
|---|---|
| Phase 1 best | `artifacts/phase1_codebook/best.pt` (val_recon 0.00108) |
| Phase 1 backup | `artifacts/phase1_codebook/best_ep0_backup.pt` |
| Phase 2 D (진행 중) | `artifacts/phase2_orformer/` |
| Mediapipe cache | `/data/shared/DMD_landmarks/face_crops_112_facemesh/` |
| Friend codebook | `/data/shared/orformer/.../cofw_full_cofw_orformer_*/best_model.pt` |
| Phase 0.5 notebook | `notebooks/phase0_5_verify.ipynb` |

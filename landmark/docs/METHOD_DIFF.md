# 원논문 vs 친구 코드 vs 우리 코드 — 학습 방법 차이

## TL;DR (한 줄)

- **원논문**: 가린 일반 얼굴 사진 (visible RGB) 의 landmark 찾기
- **친구**: 원논문 + IR 운전자 영상 mix 로 도메인 전이
- **우리**: IR 운전자 영상만 + 의도적으로 정확히 가린 sample 로 처음부터 학습

## 비유로 먼저

세 사람 모두 "가린 얼굴에서 landmark 점 찾기" 라는 같은 문제를 푸는데:

| | 누가 가르치는가 | 어떤 학생인가 |
|---|---|---|
| **원논문** | 정상 visible 얼굴 → 점이 어디 있는지 명확하게 알려줌 | 학생은 처음부터 잘 배움 |
| **친구** | 원논문이 가르친 학생 + 운전자 IR 영상도 살짝 보여줌 | 학생이 "비슷한 빛의 얼굴은 RGB 였는데 IR 도 비슷하네" 라며 부분 적응 |
| **우리** | 처음부터 IR 영상만 보여줌. 가린 얼굴은 "여기가 가렸음" 까지 명확히 표시 | 학생은 IR 만 알지만 가린 패턴을 정확히 학습 |

## 1. 데이터 측면

| | 데이터셋 | 채널 | 가림 방식 |
|---|---|---|---|
| **원논문** | WFLW (10k) / COFW (1.8k) / 300W (3.8k) | RGB visible | random rectangle 40~100% |
| **친구** | COFW + IVGaze mixed (1505 sample) | RGB + IR mix | COFW 자연 가림 + IVGaze 운전자 자연 가림 (선글라스 등) |
| **우리** | DMD ~150k frame | **IR only** | **6 variant zero-paint** (anatomically 정확) + random aug |

**핵심 차이**: 가림이 자연 (random or 운전자 환경) 인가 vs 의도적이고 정확 (anatomical region 별로 깨끗하게 zero-paint) 인가.

## 2. 학습 단계

### 원논문 (2-stage)

**Stage 1: VQ-VAE codebook 학습**
- 정상 얼굴 patch 들을 256-dim 으로 embed
- 2048 entry codebook 양자화 (얼굴 patch vocabulary)
- 목적: "정상 얼굴이 어떻게 생겼는지" 의 vocabulary 만들기

**Stage 2: ORFormer 학습**
- 가린 얼굴 입력 → patch tokens 추출
- ORFormer 가 두 sequence 동시 출력:
  - Regular (S_I): "내가 본 patch 그대로"
  - Messenger (S_M): "정상이라면 어떤 codebook entry 였을까"
- α (가린 정도) 가 두 sequence 의 dissimilarity 로 자동 결정
- 복원: Z_rec = α·Z_M + (1-α)·Z_I
- 출력: landmark heatmap → soft-argmax

### 친구 (3-stage)

**Stage 1+2: 원논문 그대로** (codebook + ORFormer 모두 COFW visible 로 학습)

**Stage 3: HGNet + ORFormer joint**
- 별도 detector HGNet (4 stack hourglass) 가 진짜 landmark 추출
- ORFormer 의 출력 (edge heatmap) 을 reference 로 HGNet 에 주입
- 친구 추가: IVGaze + COFW mixed dataset 으로 IR domain 으로 transfer
- **codebook frozen, ViT 만 unfreeze**

### 우리 (3-stage, plan 완전 동일하지만 내용 다름)

**Phase 1: VQ-VAE codebook — IR 로 처음부터**
- 친구는 visible RGB pretrained 그대로 쓰지만, 우리는 IR 의 face patch vocabulary 를 처음부터 학습
- 이유: visible↔IR 의 spectral 분포 다름. visible codebook entry 가 IR patch 매칭 잘 안 됨

**Phase 2: ORFormer — codebook 도 unfreeze**
- 친구는 codebook frozen 이지만, 우리는 codebook 도 같이 fine-tune
- 이유: Phase 1 에서 만든 IR codebook 이 가린 sample 에는 적응 안 됐을 수 있음
- **추가**: manifest BCE auxiliary loss (weight 0.1, warmup 10 ep)
  - 우리만의 강점 — 가린 region 정확히 표시한 manifest 가 있으므로 implicit α 학습에 약한 가이드 추가

**Phase 3: HGNet — 478 + 68 둘 다**
- Stage A: 478 model (FaceMesh full)
- Stage B: 68 model (300W subset, A 의 ORFormer warm start)
- inference 시 선택 가능

## 3. Supervision 측면

| | landmark GT | α GT (가린 정도) | codebook |
|---|---|---|---|
| **원논문** | heatmap L2 (dense, σ=3 가우시안) | **없음 (implicit)** | commitment + reconstruction |
| **친구** | NME + heatmap L2 | 없음 (implicit) | frozen (학습 안 함) |
| **우리** | NME + heatmap L2 (FaceMesh GT 사용, 가린 sample 도 정상 좌표) | **implicit + manifest BCE auxiliary (weight 0.1)** | 처음부터 학습 |

**가린 sample 의 landmark GT 가 핵심**:
- 가린 영역의 좌표가 **정상 영상의 좌표** 로 supervision → "messenger 가 가린 부분도 복원해야 한다" 는 명시적 학습 압력
- 친구 코드도 같음 (COFW 가린 sample 의 GT 가 visible 좌표)
- 원논문도 같음 (random occlusion 입혔지만 GT 는 가리기 전 정확한 좌표)

## 4. α 학습 방식 (가장 중요한 차이)

### 원논문 / 친구 — Implicit only
```python
# 모델이 알아서 학습
alpha = sigmoid(||x - ORx||²)
```
- 가린 patch 일수록 regular(x) 와 messenger(ORx) 가 달라짐 → α ↑
- 명시적 라벨 없이 자연스럽게 학습

**장점**: 깔끔, 라벨 노이즈 없음
**단점**: 학습 초기 weak signal, 수렴 느림

### 우리 — Implicit + BCE auxiliary
```python
alpha = sigmoid(||x - ORx||²)            # implicit (원논문과 동일)
L_aux = 0.1 · BCE(alpha, manifest_GT)   # auxiliary (우리 추가)
```
- manifest 의 binary 0/1 mask 를 약한 가이드로
- weight 0.1 + warmup 10 ep → 너무 강하게 영향 주지 않음

**왜 추가?**:
- 우리 V3 의 성공 이유 = manifest hard mask supervision
- V4 의 실패 이유 = occface cache 의 noisy supervision
- → manifest 는 강한 신호이므로, implicit 만 의존하지 않고 보조로 사용

**위험**: BCE weight 너무 크면 V4 처럼 trivial 수렴. weight 0.1 + warmup 이 trade-off 의 답.

## 5. 코드 수정 사항 (친구 → 우리)

### A. `quantizer.py` — vit-mode 에서 commitment loss 활성화

**친구 코드** (vit 있을 때 `loss=None` 반환):
```python
if vit is not None:
    ...
    return None, Z_q, None, ...    # ← loss 안 계산
```

**우리 수정**:
```python
if vit is not None:
    ...
    loss = MSE(z_q.detach(), z_e) + beta * MSE(z_q, z_e.detach())
    return loss, Z_q, perplexity, ...
```

→ ORFormer 학습 중에도 codebook 갱신 신호 흐름

### B. `simple_vit.py` — α 에 BCE hook

**친구 코드**:
```python
def forward(self, img):
    ...
    return self.linear_head(x), self.linear_head(ORx), alpha, attention_weights
```

**우리 수정** (alpha 가 외부에서 manifest 와 비교 가능하게):
```python
# train_phase2_orformer.py 에서
_, _, alpha, _ = orformer(img)        # (B, 16, 16) 또는 flatten
alpha_patch = alpha.view(B, 16, 16)
# manifest_GT 도 16×16 patch 해상도로 downsample
manifest_patch = pool_478_to_16x16(manifest_GT)
L_aux = 0.1 * BCE(alpha_patch, manifest_patch)
```

### C. `Dataloader/heatmapDataset.py` — DMD_heatmap_Dataset 추가

친구 코드의 COFW_heatmap_Dataset 패턴 그대로:
- `__getitem__`: image, resized_input, resized_occluded_input, meta, image_raw, resized_image_raw
- `meta`: Landmarks, Edge_Heatmaps, Point_Heatmaps, Annotated_Points, trans

**우리 추가**:
- meta 에 `Manifest_Mask` (478 bool, 가린 idx=0)
- 가린 영역의 Landmark 는 **정상 영상의 좌표** 그대로 사용 (messenger 학습)

### D. `Config/default.py` — `_C.DMD` block 추가

```python
_C.DMD = CN()
_C.DMD.ROOT = "/data/shared/DMD_landmarks/face_crops_112"
_C.DMD.NUM_POINT = 478     # 또는 68
_C.DMD.NUM_EDGE = 15
_C.DMD.FRACTION = 1.2
_C.DMD.EDGE_INFO = [
    # FaceMesh 의 LEFT_EYE / RIGHT_EYE / LIPS / NOSE / FACE_OVAL ...
    [True,  [33, 7, 163, 144, 145, ...]],   # 왼쪽 눈 윤곽
    [True,  [263, 249, 390, 373, ...]],     # 오른쪽 눈 윤곽
    [True,  [61, 84, 17, 314, ...]],        # 입술 외곽
    ...
]
_C.DMD.SCALE = 0.05
_C.DMD.ROTATION = 15
_C.DMD.OCCLUSION_MEAN = 0.2
_C.DMD.OCCLUSION_STD = 0.08
_C.DMD.DATA_FORMAT = "L"   # IR 1ch
_C.DMD.OCCLUSION = True
```

## 6. 우리 코드의 가장 큰 의의

친구 코드와 비교해서 우리가 갖는 advantage 3 가지:

1. **IR domain pure 학습** — visible→IR domain gap 회피
2. **Anatomically 정확한 가림 supervision** — 6 variant + manifest mask (자연 가림이 아니라 region 별로 깨끗)
3. **478 landmark dense supervision** — FaceMesh GT 가 COFW 29 점보다 훨씬 dense

trade-off: dataset 의 가림이 zero-paint 라 자연스럽지 않음 (학습/평가 시 zero-paint 와 실제 가림의 domain gap 가능성).

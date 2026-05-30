# Pretrain V4

ORFormer + VQ-VAE + StackedHGNet 을 DMD IR + 6 variant 가림 dataset 으로 학습.
친구 코드 (`/data/shared/orformer/capstone/`) 의 원논문 충실 구현을 fork 해서 우리 IR domain 에 맞춤.

## 빠른 reference

- 전체 plan: [PLAN.md](PLAN.md)
- 학습방법 비교 (원논문 vs 친구 vs 우리): [docs/METHOD_DIFF.md](docs/METHOD_DIFF.md)
- **학습 일지** (timeline, 결과, 결정): [docs/TRAINING_LOG.md](docs/TRAINING_LOG.md)
- 친구 원본 코드: `/data/shared/orformer/`

## 폴더 구조

```
pretrain_v4/
├── PLAN.md              # 전체 계획
├── README.md            # 이 문서
├── docs/
│   └── METHOD_DIFF.md   # 학습방법 비교
├── configs/             # 학습 config (Phase 1~3)
├── src/
│   ├── data/            # DMD dataloader, edge_info, heatmap 생성
│   ├── models/          # 친구 코드 fork (수정 사항 적용)
│   └── training/        # phase 별 train script
├── scripts/             # 학습 shell scripts
└── artifacts/           # 학습 결과 (gitignore)
```

## 학습 순서

```bash
bash scripts/run_phase1.sh       # VQVAE codebook (IR native)
bash scripts/run_phase2.sh       # ORFormer (codebook unfreeze)
bash scripts/run_phase3a_478.sh  # HGNet 478 landmark
bash scripts/run_phase3b_68.sh   # HGNet 68 landmark (warm start from A)
```

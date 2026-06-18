#!/bin/bash
# 재학습 ablation 런처 (정리된 modules/ 레이아웃 사용).
# 6개 변형을 순차 학습. 각 변형은 base 에서 모듈 1개만 제거된 모델.
#   사용:  bash train_all.sh           (전체)
#          bash train_all.sh no_occ    (특정 변형만)
set -u
PY=/data/shared/envs/scuppy/bin/python
CLS=/data/shared/scuppy/Full_System/vendor/classifier
HERE=/data/shared/scuppy/Full_System/experiments/retrain_ablation
LOG="$HERE/train_all.log"
cd "$CLS" || exit 1

MODULES=${@:-"full no_body no_face no_occ no_hgnet no_gate"}
echo "=== retrain ablation START $(date) | modules: $MODULES ===" | tee -a "$LOG"
for v in $MODULES; do
  CFG="$HERE/modules/$v/config.yaml"
  echo "" | tee -a "$LOG"; echo "######## TRAIN $v  $(date) ########" | tee -a "$LOG"
  if "$PY" -m src.training.train --config "$CFG" >>"$LOG" 2>&1; then
    echo "#### DONE $v  $(date) ####" | tee -a "$LOG"
  else
    echo "#### FAILED $v  $(date) ####" | tee -a "$LOG"
  fi
done
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"

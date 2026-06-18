#!/bin/bash
# 재학습 ablation: 6개 변형을 순차 학습 (full / no_body / no_face / no_occ / no_hgnet / no_gate), seed 42
set -u
PY=/data/shared/envs/scuppy/bin/python
CLS=/data/shared/scuppy/Full_System/vendor/classifier
CFGD=/data/shared/scuppy/Full_System/experiments/retrain_ablation/configs
LOG=/data/shared/scuppy/Full_System/experiments/retrain_ablation/run_all.log
cd "$CLS" || exit 1

echo "=== retrain ablation START $(date) ===" | tee -a "$LOG"
for v in full no_body no_face no_occ no_hgnet no_gate; do
  echo "" | tee -a "$LOG"
  echo "######## TRAIN $v  $(date) ########" | tee -a "$LOG"
  if "$PY" -m src.training.train --config "$CFGD/${v}_seed42.yaml" >>"$LOG" 2>&1; then
    echo "#### DONE $v  $(date) ####" | tee -a "$LOG"
  else
    echo "#### FAILED $v  $(date) ####" | tee -a "$LOG"
  fi
done
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"

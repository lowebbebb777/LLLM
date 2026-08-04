#!/usr/bin/env bash
# M1: 5条件アブレーション (SPEC §7 M1)。A/B/C/D/E を各3seedで学習する。
# ★M0 (run_m0.sh) が通り、安定設定が確定してから実行すること★
#
# 判定 (SPEC §4.3): C > B かつ C > D の両立で「変分原理アナロジーに効果あり」。
# 条件 E (加法形 qw·cont+disc) は C (凸結合) との結合形の対照。
# seed は最低3つで反復し分散を見る (§4.4)。評価は src/evaluate.py で別途実施。
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1 2)
CONDS=(A B C D E)

for cond in "${CONDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== condition ${cond}, seed ${seed} ==="
    out="outputs/cond_${cond}_seed${seed}"
    python3 src/train.py \
      --config "configs/cond_${cond}.yaml" \
      --seed "${seed}"
    # output_dir は config 既定。seed 別に分けたい場合は config を複製 or 環境変数化する。
  done
done

echo "全条件×3seed 完了。src/evaluate.py で HumanEval/MBPP/自作/経路独立性を評価し、"
echo "§5.3 の作法 (平均±標準偏差・効果量) で §4.3 の判定を行うこと。"

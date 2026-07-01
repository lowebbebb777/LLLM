#!/usr/bin/env bash
# M1 本番: 4条件 × 3seed アブレーション (SPEC §7-M1)。
# ★run_m1_1seed.sh でパイプラインが通ってから実行すること★
#
# 判定 (SPEC §4.3): C > B かつ C > D の両立で「変分原理アナロジーに効果あり」。
# 統制変数は config で固定 (SPEC §4.4)。学習データは全条件・全seedで同一。
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2"}       # SPEC §4.4: 最低3 seed
EPOCHS=${EPOCHS:-3}
N_TRAIN=${N_TRAIN:-4000}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B}
DATA=data/m1_train.jsonl

echo "=== 学習データ生成 (n=${N_TRAIN}) ==="
python3 scripts/gen_train_data.py --n "${N_TRAIN}" --out "${DATA}"

for cond in A B C D; do
  for seed in ${SEEDS}; do
    out="outputs/m1_${cond}_seed${seed}"
    echo "=== train ${cond} seed ${seed} (epochs=${EPOCHS}) ==="
    python3 src/train.py --config "configs/cond_${cond}.yaml" --seed "${seed}" \
      --dataset "${DATA}" --epochs "${EPOCHS}" --output-dir "${out}"
    echo "=== eval ${cond} seed ${seed} ==="
    python3 src/evaluate.py --adapter "${out}/adapter" --condition "${cond}" \
      --model "${MODEL}" --out "${out}/eval.json"
  done
done

echo "=== 集計 (平均±標準偏差, SPEC §5.3) ==="
python3 scripts/aggregate_m1.py --glob 'outputs/m1_*_seed*/eval.json'

echo ""
echo "§4.3 の判定を効果量 (evaluate.cohens_d) 込みで行うこと。"
echo "HumanEval/MBPP を追加するなら別途 human-eval ハーネスを評価に組み込む (SPEC §5.1)。"

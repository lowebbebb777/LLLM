#!/usr/bin/env bash
# 条件 E (EquilibriumLoRA) を 3 seed 学習・評価する。既存の A/B/C/D と比較する。
# ★A/B/C/D と同一の学習データ (m1_train.jsonl) を使う (統制変数)。seeded 生成なので
#   同じコマンドで再生成すれば M1 と同一データになる★
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-3}
N_TRAIN=${N_TRAIN:-4000}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B}
DATA=data/m1_train.jsonl

# M1 と同一データを保証 (存在しなければ同じ seed で再生成)
if [ ! -f "${DATA}" ]; then
  echo "=== ${DATA} が無いので再生成 (M1 と同一 seed) ==="
  python3 scripts/gen_train_data.py --n "${N_TRAIN}" --out "${DATA}"
fi

for seed in ${SEEDS}; do
  out="outputs/m1_E_seed${seed}"
  echo "=== train E seed ${seed} (equilibrium gate, epochs=${EPOCHS}) ==="
  python3 src/train.py --config configs/cond_E.yaml --seed "${seed}" \
    --dataset "${DATA}" --epochs "${EPOCHS}" --output-dir "${out}"
  echo "=== eval E seed ${seed} ==="
  python3 src/evaluate.py --adapter "${out}/adapter" --condition E \
    --model "${MODEL}" --out "${out}/eval.json"
done

echo "=== 集計 (A/B/C/D/E, 平均±標準偏差 + Cohen's d) ==="
python3 scripts/aggregate_m1.py --glob 'outputs/m1_*_seed*/eval.json'

echo ""
echo "見るべき比較 (SPEC §4.3 の拡張):"
echo "  E vs B : 拘束(釣り合い)を入れると容量一致対照を超えるか (自由ゲートCは超えなかった)"
echo "  E vs C : 等号版が自由ゲート版を超えるか (拘束の効果)"
echo "  E vs D : 平衡ゲートが固定ゲートを超えるか"

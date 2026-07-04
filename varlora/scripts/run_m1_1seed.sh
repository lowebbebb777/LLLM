#!/usr/bin/env bash
# M1 パイプライン検証 (1 seed × 4条件)。★まずこれで train→save→reload→eval→集計が
# 全条件で通ることを確認してから run_ablation.sh (3 seed 本番) に進む★
#
# 速度優先の dry-run 既定 (軽い)。本番並みにしたいなら環境変数で:
#   N_TRAIN=4000 EPOCHS=3 bash scripts/run_m1_1seed.sh
set -euo pipefail
cd "$(dirname "$0")/.."

N_TRAIN=${N_TRAIN:-1200}      # dry-run は軽く。本番は 4000 程度
EPOCHS=${EPOCHS:-1}           # dry-run は 1。本番は 3 程度
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B}
DATA=data/m1_dryrun.jsonl

echo "=== 1) 学習データ生成 (n=${N_TRAIN}, 評価セットとのリーク除去) ==="
python3 scripts/gen_train_data.py --n "${N_TRAIN}" --out "${DATA}"

for cond in A B C D; do
  out="outputs/m1_${cond}_seed0"
  echo "=== 2) train condition ${cond} (seed 0, epochs=${EPOCHS}) ==="
  python3 src/train.py --config "configs/cond_${cond}.yaml" --seed 0 \
    --dataset "${DATA}" --epochs "${EPOCHS}" --output-dir "${out}"
  echo "=== 3) eval condition ${cond} (数値整合性 + 経路独立性) ==="
  python3 src/evaluate.py --adapter "${out}/adapter" --condition "${cond}" \
    --model "${MODEL}" --out "${out}/eval.json"
done

echo "=== 4) 集計 (SPEC §4.3 判定ヒント) ==="
python3 scripts/aggregate_m1.py outputs/m1_A_seed0/eval.json outputs/m1_B_seed0/eval.json \
  outputs/m1_C_seed0/eval.json outputs/m1_D_seed0/eval.json

echo ""
echo "パイプラインが4条件すべて通ったら run_ablation.sh (3 seed) へ。"
echo "1 seed では優劣を断じないこと (SPEC §5.3)。"

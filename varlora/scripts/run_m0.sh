#!/usr/bin/env bash
# M0: 手応え確認 (SPEC §7 M0)。VariationalLoRA (条件C相当) 単体を小データ1エポックで回し、
#   [ ] VRAM に収まる   [ ] loss が下がる   [ ] 勾配が NaN にならない
#   [ ] ゲート値が 0/1 に張り付いてないか
# を実機 (RTX 3060) で確認する。★このM0が通るまで4条件 (run_ablation.sh) には進まない★
set -euo pipefail
cd "$(dirname "$0")/.."

# まずパラメータ数レポート (条件 B≈C の交絡対照が成立しているか, SPEC §2.4)
echo "=== parameter report (条件 B が C にパラメータ一致するか) ==="
python3 src/train.py --config configs/cond_C.yaml --report-only

# 条件 C を小データで 1 エポック (M0 本体)。config は M1 本番データを指すので、
# M0 では --dataset / --epochs で小さな smoke 設定に上書きする。
echo "=== M0 smoke run (condition C, smoke data, 1 epoch) ==="
python3 src/train.py --config configs/cond_C.yaml --seed 0 \
  --dataset data/m0_smoke.jsonl --epochs 1 --output-dir outputs/m0_cond_C

echo "M0 完了。上記の loss 推移 / NaN ダンプ有無 / VRAM 実挙動を確認すること。"

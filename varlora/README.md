# VariationalLoRA

有限要素法の変分原理（連続場⇄離散点の重み付き統合）を構造的アナロジーとして借用した
独自 LoRA アダプタ `VariationalLoRA` を Qwen2.5-Coder-7B の **FFN 層** に適用し、
コード生成 LLM の内部表現を改良できるかを **交絡を排した実験計画**で検証するプロジェクト。

詳細な設計・実験計画・検証基準は [`SPEC.md`](./SPEC.md) を参照（これが本プロジェクトの正典）。

## FEM ⇄ NN 対応（設計意図）

| FEM 側 | NN 側 (本実装) |
|--------|----------------|
| 連続場 σ, ε | `cont` 経路 (`cont_A → cont_B`) |
| 離散場 節点自由度 u_I | `disc` 経路 (`disc_A → disc_B`) |
| 形状関数 N_I(ξ), ΣN_I=1 | `shape_fn → softmax`（分割の単位性） |
| ガウス求積 Σ w_g·f(ξ_g) | `quad_weights` による重み付き統合 |
| 変分原理 内部+境界仕事 | `out = qw·cont + (1-qw)·disc` |

## 状態（このリポジトリで何が出来ているか）

このコードは **GPU の無いサンドボックスで実装・検証**された。したがって：

- ✅ **実装済み・CPU で検証済み**: `VariationalLoRA` 本体（§2）、FFN 注入（§3.2）、
  4 条件 A/B/C/D の学習配線（§4）、交絡対照の rank 一致計算（§2.4）、
  評価指標（pass@k / 数値整合性 / 経路独立性 / 効果量, §5）、自作評価セットのシード。
  → `tests/` の **33 テストが全て pass**（`torch` CPU のみで実行可能）。
  - 学習は自前ループ（`train.train_loop`）。`transformers.Trainer` は 4bit モデルを
    「PEFT 経由のアダプタ」がある場合のみ学習許可するため、PEFT を介さず注入する
    条件 C/D が弾かれる。自前ループで回避しつつ §6.3 の NaN ダンプ・勾配 clip と
    M0 のゲート観察（qw が 0/1 に張り付かないか）を直接実装している。
- ⏳ **ユーザーの RTX 3060 での実機確認が必要**: M0 のチェックリスト
  （VRAM に収まる / loss が下がる / 勾配が NaN にならない / ゲートが 0/1 に張り付かない）。
  これは GPU 実機でしか確認できない（SPEC §9-4 の通り、数値は実測で詰める）。
- ⛔ **意図的に未実装**: M3 (`component_lora.py`) / M4 (`newmark_lora.py`)。
  SPEC §7-M2/§9 により「M2 で効果が確認できるまで進まない」ため stub のみ
  （`NotImplementedError` + 設計メモ）。

## M0 結果（実機 RTX 3060, 条件C, 500件×1epoch）

`run_m0.sh` 実測。**M0 合格**（SPEC §7-M0 の4項目すべてクリア）。

| 項目 | 結果 |
|------|------|
| VRAM に収まる | ✅ OOM なし（4bit + PagedAdamW8bit + grad-checkpointing） |
| loss が下がる | ✅ 1.91 → ~0.5（63更新, 1epoch） |
| 勾配 NaN | ✅ 出ず（max_grad_norm=1.0） |
| ゲート 0/1 張り付き | ✅ なし。qw_mean≈0.33 で安定 |

**ゲートの学習可能性**: gate_lr を 1× → 10× にすると qw_mean の移動量が −0.001 → −0.009
とほぼ LR 比例で増加。勾配経路は生きており（死んでいない）、線形応答＝発散兆候なし。
本番 M1（多ステップ）で動的性が立ち上がるか、`gate_lr_multiplier` で調整して検証する。

## マイルストーン（SPEC §7）

1. **M0 手応え確認** ← まずここだけ。`scripts/run_m0.sh`（条件 C を小データ1エポック）
2. **M1 4条件アブレーション** A/B/C/D ×3seed。`scripts/run_ablation.sh`
3. **M2 結論①** §4.3 の判定（`C>B かつ C>D` で「アナロジーに効果あり」）
4. M3〜M5 は M2 成功を前提に1機構ずつ（本リポジトリでは未実装）

## 使い方

```bash
# 依存（torch は CUDA 版を別途。SPEC §6.4）
pip install -r requirements.txt

# CPU で実装の健全性テスト（GPU 不要）
python3 tests/test_variational_lora.py
python3 tests/test_inject.py
python3 tests/test_evaluate.py
python3 tests/test_train_config.py

# 交絡対照が成立するか（B のパラメータ数が C に一致するか）を確認
python3 src/train.py --config configs/cond_C.yaml --report-only   # 要 transformers/モデル

# M0（実機 RTX 3060）
bash scripts/run_m0.sh

# M0 通過後に M1（4条件×3seed）
bash scripts/run_ablation.sh
```

> パラメータ一致の確認は `compute_matched_rank`（`src/inject.py`）による。Qwen2.5-Coder-7B
> 実寸（hidden=3584, intermediate=18944, 28層）で **B/C ≈ 0.995**（残差は整数 rank 丸めのみ）、
> A は約半分。「効果がパラメータ増加由来か機構由来か」を分離できる設計（SPEC §4.1）。

## ディレクトリ（SPEC §8）

```
varlora/
├── SPEC.md                    # 正典（実装指示書）
├── README.md
├── requirements.txt
├── src/
│   ├── variational_lora.py    # VariationalLoRA 本体（§2, M0-M1）
│   ├── inject.py              # FFN 注入 + 条件B rank一致（§3.2, §2.4）
│   ├── train.py               # 学習ループ（全条件切替, §4, §6）
│   ├── evaluate.py            # HumanEval/MBPP/自作/経路独立性（§5）
│   ├── component_lora.py      # M3 stub（M2成功まで保留）
│   └── newmark_lora.py        # M4 stub（M2成功まで保留）
├── data/
│   ├── m0_smoke.jsonl         # M0 手応え用の小データ
│   └── numeric_stats_eval/    # 自作評価セット（§5.2, シード約14問）
├── configs/cond_{A,B,C,D}.yaml
├── scripts/{run_m0.sh, run_ablation.sh}
└── tests/                     # CPU で実行可能な単体テスト（33件）
```

## 次の一手（実装エージェントからの引き継ぎ）

1. RTX 3060 に依存をインストールし `bash scripts/run_m0.sh` を実行。M0 チェックリストを確認。
   OOM/NaN が出たら SPEC §6.2/§6.3 の調整順（seq_length↓ → grad_accum↑ → rank↓ / τ 導入 / lr↓）。
2. M0 で安定設定を確定したら、その設定を 4 つの config に反映し `run_ablation.sh`。
3. 自作評価セットを 50〜100 問へ拡充（`data/numeric_stats_eval/README.md` 参照）。
4. §4.3 の判定 → M2。効果が出たら初めて M3/M4 の stub を実装する。
```

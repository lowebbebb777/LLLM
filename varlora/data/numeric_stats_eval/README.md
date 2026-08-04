# 自作評価セット (数値・統計の内部表現) — SPEC §5.2

LLM が数値・統計を「文字列」でなく**構造**として扱えているかを測る問題群。

## スキーマ (`problems.jsonl`, 1 行 1 問)

| フィールド | 説明 |
|-----------|------|
| `id` | 一意 ID |
| `category` | `cpk` / `stats_impl` / `scale_consistency` / `discrete_continuous` |
| `kind` | `numeric` (数値解答を相対誤差で照合) / `code` (test を実行して pass/fail) |
| `prompt` | 問題文 |
| `expected` | (numeric) 期待値 |
| `rel_tol` | (numeric) 許容相対誤差 |
| `test` | (code) `assert` 群。生成関数を検証する |
| `paraphrases` | 経路独立性 (§5.2b) 用の言い換え/順序変更プロンプト群 |

## カテゴリ (SPEC §5.2)
- **cpk**: プロセス能力指数 (Cp/Cpk) の計算
- **stats_impl**: 統計手法 (回帰・分類・検定) の正しい実装
- **scale_consistency**: 数値の桁・スケールの一貫性 (12.34 を正しく数として扱うか)
- **discrete_continuous**: 離散 (カテゴリ/フラグ) と連続 (実数) が混在する処理の正しさ

## 現状: 61 問へ拡充済み (SPEC §5.2 の 50〜100 問を満たす)
| カテゴリ | 問題数 | 種別 |
|---------|-------|------|
| cpk | 14 | numeric 10 / code 4 |
| stats_impl | 17 | numeric 10 / code 7 |
| scale_consistency | 18 | numeric 14 / code 4 |
| discrete_continuous | 12 | numeric 6 / code 6 |
| **合計** | **61** | numeric 41 / code 20 |

- `numeric` 41 問はすべて 2〜3 個の `paraphrases` を持つ (経路独立性 §5.2b を測定可能)
- 拡充要件は `tests/test_evaluate.py::test_expanded_dataset_contract` で固定 (退行防止)

### 検証済みであること (前回の反省: 未検証の値をコミットしない)
全期待値・全 test は生成器 (`scripts/gen_eval.py` 相当のロジック) で機械検証済み:
- `numeric` の `expected` は Python で計算した値をそのまま格納 (手打ちしない)
- `code` の `test` は参照実装で pass することを確認、かつ**誤実装で fail する**ことも確認
  (弱い/自明なテストを排除。実際 `ols_code_001` で端点勾配が通る弱さを検出し修正した)
- `evaluate.py` のオラクル (完全正答生成器) で numeric overall=1.0 / 経路一致=1.0 を確認

### さらに拡充する場合の指針
- `numeric` には必ず 2〜3 個の `paraphrases` を付け、経路独立性 (§5.2b) を測れるようにする
- `code` の `test` は複数の `assert` で境界条件まで検証する
- 期待値は手計算または別実装で必ず検証してからコミットする (契約テストが最低限を担保)

## 指標 (`src/evaluate.py`)
- `numeric_consistency_score`: pass@1 とは別軸の「数値整合性スコア」(カテゴリ別 + 全体)
- `path_independence_score`: 言い換え耐性 (agreement / dispersion)

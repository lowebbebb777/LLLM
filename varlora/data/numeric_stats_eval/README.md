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

## ★現状はシード (種) であり未完成★
本ファイルは **約14問のシード**。SPEC §5.2 は **50〜100問程度** を要求している。
M1 本評価の前に、各カテゴリを 12〜25 問へ拡充すること。拡充時の指針:
- `numeric` には必ず 2〜3 個の `paraphrases` を付け、経路独立性 (§5.2b) を測れるようにする
- `code` の `test` は複数の `assert` で境界条件まで検証する
- 期待値は手計算または別実装で必ず検証してからコミットする

## 指標 (`src/evaluate.py`)
- `numeric_consistency_score`: pass@1 とは別軸の「数値整合性スコア」(カテゴリ別 + 全体)
- `path_independence_score`: 言い換え耐性 (agreement / dispersion)

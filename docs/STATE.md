# STATE — 現在地 / Current Position

最終更新 / Last updated: 2026-09-04 Antigravity @ 00e7cc1 (最新規約導入)

## 検証済みの事実 / Verified Facts

- Phase D0: CPU参照実装 合格 (線形参照とFLOPモデル検証)
- Phase D1: activation差分観測 実装済み (CPU上での実装検証 15テスト成功)
- D1アーキテクチャ比較定義完了 (deltafem/D1_ARCHITECTURE_COMPARISON.md)
- VariationalLoRA: M0実機検証完了、先行研究資産として固定保存 (arlora/)

## 文書の優先順位 / Doc Precedence

矛盾時: README.md > deltafem/RESEARCH_PLAN.md > docs/STATE.md。残作業の正は本ファイル。

## 残作業(優先順) / Remaining Work

1. D1 Windows実機測定: .venv 生成および 15テストと toy smoke の再現
2. Qwen2.5-Coder-0.5B での D1 実行 (last_token vs ligned_sequence)
3. block size と threshold のスイープおよび Go条件の再現判定

## アイディア棚 / Idea Shelf

docs/IDEAS.md(1件) — **着手前に関連語で grep**。通読しない

## 未検証項目 / Unverified Items

- Windows実機 (RTX 3060) 上での D1 activation差分エネルギーの再現性

## 着手禁止 / Do Not Start

- なし
# LOG — セッション履歴 / Session History

---

## 2026-09-04 Antigravity — 最新引き継ぎ規約の初期導入 (di)

- やったこと:
  - AGENTS.md: 最新の「運用の指針」「自律範囲」「終了時の義務」「ショートカット(ds/dm/di/did)」を新規配備。
  - docs/STATE.md: 新規作成。現在地・検証済みの事実・残作業を32行で整理。
  - docs/LOG.md: 新規作成。
  - docs/IDEAS.md: 新規作成。散文でない詩形式＋grep具体語タグ必須化([tag1, tag2])に対応した初期棚を配置。
  - 	ests/test_conventions.py: 規約機械チェックテスト(5件)を配備。
- 逸脱・追加・スキップ: なし
- 検証: python -m unittest tests/test_conventions.py 5件全件 PASS。
- 申し送り: なし

## 2026-11-17 GPT-6 — 共通実行規約同期

- やったこと: AGENTS.mdへ共通のリスク別Git/QA/統合条件と記録・テスト頻度を同期。
- 固有条件: README/DeltaFEM研究計画の正典、棄却可能な実験根拠、VariationalLoRA凍結、GPU費用・外部データ送信境界を維持。
- 逸脱・スキップ: README/STATE/研究コード/依存は変更なし。テストは文書のみの変更のためPR上のCIを待つ。
- 検証: AGENTS.md読み戻しを実施予定。研究実験・GPU実測はNOT_RUN。

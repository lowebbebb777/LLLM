# AGENTS.md — LLLM エージェント共通ルール / Common Agent Rules

全エージェント(Codex / Antigravity / Claude Code / Cursor)共通の入口。
**このファイルには「今やること」を書かない**(陳腐化するため)。現在地は docs/STATE.md。

## 読む順番 / Reading Order

1. README.md — 研究思想・全体像・現在の方針(最優先)
2. docs/STATE.md — 現在地と残作業
3. deltafem/RESEARCH_PLAN.md — DeltaFEM-LLM 研究計画・仕様
4. docs/LOG.md — 経緯と判断理由(必要時のみ遡る)

## ビルド・環境(Windows) / Build & Environment

`powershell
powershell -ExecutionPolicy Bypass -File .\deltafem\scripts\verify_windows.ps1
powershell -ExecutionPolicy Bypass -File .\deltafem\scripts\run_d1_windows.ps1
`

## 不変条件 / Invariants

- 正は README.md および deltafem/RESEARCH_PLAN.md。
- 実装と棄却可能な比較実験で検証する。推測や散文だけで高速化を主張しない。
- VariationalLoRA (arlora/) は一度の探索的実機検証を完了した研究資産として保存・凍結。

## 終了時の義務 / End-of-Session Duties

1. **1 Step = 1 コミット**。仕様があればメッセージに節番号、無ければ目的を1行。
2. docs/STATE.md を**上書き更新**。
3. docs/LOG.md に**1エントリ追記**。
4. 検証を実行し、結果(何をどう確かめたか)を STATE と LOG に記録。
   **新しいテストは修正前のコードで FAIL することを確認してから緑にする**(緑だけでは何も検証していないテストと区別できない)。
5. repo の事実は必ず repo 内に置く(エージェント固有メモリだけに残さない)。
6. 今やらない案は docs/IDEAS.md へ**詩**(散文でない1〜2文)を追記。STATE には置かない。

## 運用の指針 / Operating Guidance

すべて**既定値**であり、AIの判断で外してよい。外したら LOG に1行理由を書く。

1. **報告は結論ファースト**:
   挨拶・前置き・自己弁護は書かない。ただし**行数上限は設けない** — 判断理由・トレードオフ・リスク・逸脱は必要なだけ書く。「短さ」より「次走者が再現できること」を優先。
2. **探索は構造から、変更は本体を読んでから**:
   俯瞰には AST 要約や grep を使う。変更を加えるファイルは本体を読む。
3. **状態の正は docs/STATE.md 一つ**:
   補助的な JSON/メモはセッション内利用に留め commit しない。食い違ったら STATE が正。
4. **完了の判定**:
   - テストがある領域: 緑は必要条件。新しいテストは修正前のコードで FAIL することを確認してから採用。
   - テストが無い領域: 「どう確かめたか」を手順ごと STATE の「検証済みの事実」に書けば完了としてよい。
   - 緑になった瞬間にセッションを終える義務はない。残作業を続けてよい。

## 自律範囲 — 確認なしでやってよいこと

迷ったら止まるのではなく、**仮定を STATE に明記して進める**。

- 読み取り全般 / ビルド・テスト・lint・型チェックの実行
- 実装・リファクタ・テスト追加、作業ブランチの作成、ローカル commit
- 一時ファイル(scratch)の作成、サブエージェント・並列探索の起動
- 仕様の穴を埋める合理的な既定値の採用(採った仮定を STATE に1行書く)
- **規約そのものへの改善提案**(このファイルを含む。提案は LOG に書き、適用はユーザー承認後)

確認が要るもの(これだけ):

- push / PR作成 / リリース・デプロイ、外部への送信
- 破壊的操作(履歴改変、force push、ファイル・データの削除)
- 依存の追加・更新、有料APIの実行
- docs/STATE.md「着手禁止」に載っている項目

## ショートカット(合言葉)/ Shortcuts
- **ds = 引継ぎする**: docs/STATE.md を読んで「残作業」の最上位から着手し、区切りでは上記「終了時の義務」に従う。
- **dm = 引継ぎ保存**: 途中経過を docs/STATE.md(次の一手)と docs/LOG.md に書いて commit。タスク途中で別AIへ渡す前の強制セーブ。
- **di = 引継ぎ登録**: 新規repoを規約に登録。
- **did = アイディアを棚に落とす**: 直前に出た案を、今いる repo の docs/IDEAS.md へ**詩**(散文でない1〜2文)で追記。仕様も手順も書かない。STATE には置かない。横断的な着想を lowebbebb777/Ideas へ集約する場合は明示的に指示すること。

## AIとGitコミット名義（2026-10-31以降）

- Git Author / Committerは、名義使用を認められた実際の人間・正規botのみ。Claude、Codex、GPT等のAI名義やAIの `Co-Authored-By` 自動登録は禁止。
- コミット前に `git var GIT_AUTHOR_IDENT` と `git var GIT_COMMITTER_IDENT` を確認する。未確認の氏名・メールを推測しない。
- AIの作業、役割、証拠と人間の指示・承認は `docs/LOG.md` 等に記録する。**過去のGit履歴は変更しない**。
- 正本: https://github.com/lowebbebb777/dev-conventions/blob/main/docs/COMMIT_ATTRIBUTION.md

# DeltaFEM-LLM

**重み付残差法＋動的差分志向**を、FEM・領域分割法・予測子修正子法の観点からLLM推論へ移植できるかを検証する研究プロトタイプです。

現段階ではTransformer高速化を主張しません。まず、状態差分に疎性・ブロック局所性・低ランク性が実在するかを測定し、専用GPU実装へ進む価値があるかを棄却可能な形で判定します。

## 基本状態

```text
H_tilde = H_anchor + sum_j w_j * DeltaH_j
residual = H_full - H_tilde
```

- `H_anchor`: 最後に検証された全状態
- `DeltaH_j`: 局所領域・チャネル・ヘッド・KVブロック等の増分
- `w_j`: 局所寄与の重み
- `residual`: 差分和と検証済み全状態の不整合

## 現在地

### D0: 線形参照実装 — 合格

- `WeightedResidualLedger`: アンカーと重み付き差分を補償和で管理
- `IncrementalLinearOperator`: 疎な入力差分に対する厳密な線形層更新
- `CorrectionPolicy`: 残差閾値または最大増分回数による再アンカー判定
- `estimate_linear_cost`: 管理費用と定期全量計算を含むFLOP損益モデル
- `choose_active_indices`: 差分エネルギーに基づくactive自由度選択

### D1: 実モデルactivation差分観測 — 実装済み、実機測定待ち

D1はモデルを書き換えません。二つのforward結果を採取し、次を層別にJSON/CSVへ保存します。

- residual stream、attention出力、MLP出力、KV cache
- 90% / 95% / 99%差分エネルギーを保持する最小active率
- block active率
- 閾値後のchanged fraction
- stable rank、entropy effective rank、SVD energy rank
- 深層化に伴うchanged fractionとactive率の増減
- 局所プロンプト編集と逐次token生成の比較

Go条件の初期値は、**10%以下のactive成分またはブロックで90%以上の差分エネルギーを保持する領域が複数層・複数入力で再現すること**です。

D1には外部モデル不要のtoy smokeもあります。toyモデルは配線検査用であり、研究結果には数えません。

## Windows研究環境

今後の標準作業場所:

```text
C:\Users\soich\PycharmProjects\LLLM
```

リポジトリを最新化した後、PowerShellで実行します。

```powershell
cd C:\Users\soich\PycharmProjects\LLLM
powershell -ExecutionPolicy Bypass -File .\deltafem\scripts\setup_windows.ps1
```

このスクリプトはリポジトリ直下へ`.venv`を作り、既定ではWindows用PyTorch `2.11.0 / CUDA 12.8`、Transformers、テスト依存、DeltaFEMパッケージを導入します。CPU環境なら次を使います。

```powershell
powershell -ExecutionPolicy Bypass -File .\deltafem\scripts\setup_windows.ps1 -TorchBuild cpu
```

環境とtoy経路の検証:

```powershell
powershell -ExecutionPolicy Bypass -File .\deltafem\scripts\verify_windows.ps1
```

RTX 3060でD1を実行:

```powershell
powershell -ExecutionPolicy Bypass -File .\deltafem\scripts\run_d1_windows.ps1
```

既定モデルは`Qwen/Qwen2.5-Coder-0.5B-Instruct`です。初回だけHugging Faceからモデルを`.cache\huggingface`へ取得します。

別モデルの例:

```powershell
powershell -ExecutionPolicy Bypass -File .\deltafem\scripts\run_d1_windows.ps1 `
  -Model "Qwen/Qwen2.5-Coder-1.5B-Instruct" `
  -Device cuda `
  -View aligned_sequence `
  -MaxLength 128
```

結果は`deltafem\results\d1_*\d1_results.json`と`d1_metrics.csv`へ出力されます。生成結果はGit管理対象外です。採否判断に用いた要約だけを後から固定結果としてコミットします。

## 手動実行

```powershell
.\.venv\Scripts\Activate.ps1
cd .\deltafem
python -m pytest -q
python .\scripts\run_d1.py --mode toy --device cpu
python .\scripts\run_d1.py --mode hf --device cuda --model Qwen/Qwen2.5-Coder-0.5B-Instruct
```

`active_fraction`や`d1_go_candidate`は測定上のスクリーニング値です。GPU実時間高速化、精度維持、専用疎カーネルの有効性を証明するものではありません。

研究全体の仮説、棄却条件、マイルストーンは[RESEARCH_PLAN.md](./RESEARCH_PLAN.md)を参照してください。

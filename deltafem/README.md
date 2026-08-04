# DeltaFEM-LLM

**重み付残差法＋動的差分志向**を、FEM・領域分割法・予測子修正子法の観点からLLM推論へ移植できるかを検証する研究プロトタイプです。

現段階ではTransformer高速化を主張しません。まず、次の必要条件を小さなCPU参照実装で検証します。

1. 状態を「確定アンカー＋重み付き局所差分の和」として安定に保持できること
2. 線形演算では局所差分更新が全量再計算と一致すること
3. 差分が十分に疎な場合だけ、再アンカーと管理コストを含めても理論的な演算削減余地があること
4. 誤差が増えたとき、予測子–修正子で全状態へ戻せること

## 基本状態

```text
H_tilde = H_anchor + sum_j w_j * DeltaH_j
residual = H_full - H_tilde
```

- `H_anchor`: 最後に検証された全状態
- `DeltaH_j`: 局所領域・チャネル・ヘッド・KVブロック等の増分
- `w_j`: 局所寄与の重み
- `residual`: 差分和と検証済み全状態の不整合

Phase 0では残差を全量計算できる「オラクル」として扱います。後続Phaseで、ランダム射影・監視チャネル・logit marginなどの安価な残差指標へ置き換えます。

## 現在の実装

- `WeightedResidualLedger`: アンカーと重み付き差分を補償和で管理
- `IncrementalLinearOperator`: 疎な入力差分に対する厳密な線形層更新
- `CorrectionPolicy`: 残差閾値または最大増分回数による再アンカー判定
- `estimate_linear_cost`: 管理費用と定期全量計算を含むFLOP損益モデル
- `choose_active_indices`: 差分エネルギーに基づくactive自由度選択
- 合成Phase-0実験とCPU単体テスト

## 実行

```bash
cd deltafem
python3 -m pip install -r requirements.txt

PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_phase0.py \
  --output results/local-phase0.json
```

出力の`theoretical_speedup`はFLOPモデル上の値であり、GPU実測値ではありません。専用疎カーネル、メモリアクセス、起動遅延、CPU–GPU同期を含む実測は後続Phaseで行います。

研究全体の仮説、棄却条件、マイルストーンは[RESEARCH_PLAN.md](./RESEARCH_PLAN.md)を参照してください。

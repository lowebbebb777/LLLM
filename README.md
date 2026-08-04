# LLLM

**小型ローカルLLMを、限られたGPU資源でより効率よく学習・推論できるかを、実装と棄却可能な比較実験で検証する研究リポジトリです。**

現在の中心テーマは **DeltaFEM-LLM** です。

> GPUでは主に局所的な情報差分を処理し、アンカー状態・重み付き差分和・検証済み全状態の整合性を、FEM・領域分割法・予測子修正子法の発想で管理できるかを調べます。

最終更新: 2026-08-05

---

## 現在の研究方針

### Active: DeltaFEM-LLM

新しい研究対象は、**重み付残差法＋動的差分志向のLLM推論**です。

概念状態は次です。

```text
H_tilde = H_anchor + sum_j w_j * DeltaH_j
residual = H_full - H_tilde
```

- `H_anchor`: 最後に検証された全状態
- `DeltaH_j`: 局所領域・チャネル・ヘッド・KVブロック等の増分
- `w_j`: 各局所増分の寄与重み
- `residual`: 差分の和分と全分の不整合

主な研究問いは、次のとおりです。

1. LLMのactivation差分は、疎・低ランク・ブロック局所な形を保つか
2. 差分計算、assembly、監視、reanchorを含めてもGPU計算を削減できるか
3. 低次元残差監視と局所correctorで誤差を制御できるか
4. 通常生成、局所プロンプト編集、KV更新、SSM状態更新のどこで最も成立しやすいか

実装と研究計画:

- [DeltaFEM-LLM README](./deltafem/README.md)
- [DeltaFEM-LLM 研究計画](./deltafem/RESEARCH_PLAN.md)

### Phase D0: CPU参照実装 — 合格

最初のコミットでは、GPU高速化を主張する前に必要条件を検証する小さな参照実装を追加しました。

- アンカー＋重み付き差分を補償和で管理
- 疎な入力変化に対する線形層の厳密な増分更新
- 残差閾値・最大増分回数による予測子–修正子判定
- 管理費用と定期全量計算を含むFLOP損益モデル
- 差分エネルギーに基づくactive自由度選択
- 合成Phase-0実験

CPU単体テストは **10件成功**です。

合成線形層では全量計算との相対誤差が概ね`1e-16`で、変更率100%では理論上の利得が消え、変更率が低い場合だけ演算削減余地が現れることを確認しました。これは線形参照とFLOPモデルの検証であり、TransformerまたはGPUの高速化実証ではありません。

---

## VariationalLoRAの扱い

[`varlora/`](./varlora/) は、FEMの構造的アナロジーを二経路LoRAと入力依存ゲートへ応用した先行探索です。

2026-08-05時点で、次の状態として固定します。

| 項目 | 状態 |
|---|---|
| RTX 3060で学習可能 | M0の単回実機検証で確認済み |
| loss低下・NaNなし・ゲート非崩壊 | 確認済み |
| 実装・CPUテスト | 保存済み |
| 標準LoRAより高性能 | 未証明 |
| 動的ゲートの性能寄与 | 未証明 |
| A/B/C/D × 複数seedのM1 | 未実施・当面凍結 |

したがってVariationalLoRAは、**一度の探索的実機検証を完了した研究資産**として保存します。成功とも失敗とも過大評価せず、比較優位は未証明のままです。今後の主開発はDeltaFEM-LLMへ移します。

詳細:

- [VariationalLoRA 状態・実行方法](./varlora/README.md)
- [VariationalLoRA 仕様](./varlora/SPEC.md)

---

## 次の正式タスク: D1 activation差分観測

D0の次は、新しいニューラル機構を追加することではありません。小型の実モデルからactivation差分を採取し、差分計算に圧縮可能性があるかを測ります。

測定対象:

- layer別の`Delta h` top-k energy
- block sparsity
- 有効ランク
- 深層化に伴う差分の密化率
- attention / MLP / residual / KVの比較
- 逐次生成と局所プロンプト編集の比較

初期Go条件は、**10%以下のactive成分またはブロックで90%以上の差分エネルギーを保持する領域が、複数層・複数入力で再現すること**です。満たさない場合は、汎用Transformer高速化を広く主張せず、局所編集・KV・SSMなどへ適用範囲を絞ります。

---

## 実行例

### DeltaFEM-LLM

```bash
cd deltafem
python3 -m pip install -r requirements.txt

PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_phase0.py \
  --output results/local-phase0.json
```

### VariationalLoRA（保存済み先行研究）

```bash
cd varlora
python3 -m pip install -r requirements.txt
python3 tests/test_variational_lora.py
python3 tests/test_inject.py
python3 tests/test_evaluate.py
python3 tests/test_train_config.py
```

CUDA版PyTorchは環境に合わせて別途導入してください。

---

## Runovaとの関係

LLLMと[Runova](https://github.com/lowebbebb777/Runova)は、長期的には次の循環を目指します。

```text
Runova
  実タスク、失敗、修復、テスト、採用diff、状態差分を生成
          ↓
LLLM
  状態差分の構造を観測し、学習・推論機構へ反映
          ↓
改良されたローカルモデル
  次のRunova開発と実タスク処理を改善
```

- **Runova**: 外部記憶、観測、ツール実行、検証、失敗修復
- **LLLM**: 小型モデルの学習・推論構造と計算効率の研究

---

## Author / 開発者

### Soichiro Oka

#### 日本語

**Soichiro Oka** は、機械工学、CAE、数値シミュレーション、統計解析、ソフトウェア開発を専門とするエンジニア兼独立研究者です。

神戸大学大学院では、移動有限要素法を用いた数値シミュレーションを研究しました。その後、自動車用駆動ベルト、スクロール圧縮機、半導体製造部品の開発に携わり、非線形構造解析、高周波振動解析、FORTRANとCAEの連成解析、統計的品質管理、製造DXなどの技術を構築してきました。

現在は、これらの工学的経験を生かし、限られた計算資源で高度な推論能力を獲得する小型ローカルLLMの研究開発に取り組んでいます。

#### English

**Soichiro Oka** is a mechanical engineer and independent researcher specializing in CAE, numerical simulation, statistical analysis, and software development.

His graduate research at Kobe University focused on numerical simulation using the Moving Finite Element Method. He has since worked on automotive drive systems, scroll compressors, semiconductor manufacturing components, nonlinear structural analysis, high-frequency vibration modeling, coupled FORTRAN–CAE simulations, statistical quality control, and manufacturing digitalization.

He is currently applying this engineering background to compact local language models capable of advanced reasoning under limited computational resources.

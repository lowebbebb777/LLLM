# LLLM

**小型ローカルモデル自身の内部表現と推論能力を改良し、限られた計算資源でも高度なソフトウェア開発を継続できるかを検証する研究リポジトリです。**

現在の中心テーマは、Qwen2.5-Coder-7Bへ独自LoRAアダプタ **VariationalLoRA** を適用する研究です。

> 現在地を一言で表すと、**独自方式がRTX 3060上で学習可能なことは確認済み。ただし、標準LoRAより高性能であることはまだ未証明**です。

最終更新: 2026-08-04

---

## 現在の研究対象

LLLMは、LLMをゼロから事前学習するプロジェクトではありません。

既存の7B級コードモデルへ追加学習層を装着し、一般的な個人向けGPUでもモデルの能力を効率よく改善できるかを研究しています。現在の主要実装は [`varlora/`](./varlora/) 以下にあります。

### VariationalLoRA

通常のLoRAが単一の低ランク経路を持つのに対し、VariationalLoRAは二つの経路を持ち、入力依存ゲートで統合します。

```text
入力
 ├─ cont経路 ─┐
 │             ├─ 入力依存ゲートで混合 → 出力
 └─ disc経路 ─┘
```

- `cont`経路: 滑らかで全体的な表現を担当
- `disc`経路: 局所的・離散的な表現を担当
- `shape_fn` / `quad_weights`: 入力ごとの混合比を決定

概略式は次の通りです。

```python
cont = cont_B(cont_A(x))
disc = disc_B(disc_A(x))
N = softmax(shape_fn(x) / tau)
qw = sum(N * quad_weights)
out = qw * cont + (1 - qw) * disc
```

この構造は、有限要素法における連続場・離散自由度・形状関数・ガウス求積の関係から着想しています。ただし、有限要素法の具体的な支配方程式をLLMへ埋め込むものではなく、**構造的アナロジーが表現学習に有効かを比較実験で検証する研究**です。

詳細:

- [VariationalLoRA README](./varlora/README.md)
- [実装仕様・実験計画 SPEC](./varlora/SPEC.md)
- [LLLM × Runova 統合構想](./LLLM_Runova_7B高度推論モデル統合構想.md)

---

## 実験設計

独自機構を実装しただけで効果を主張しないため、次の4条件を比較します。

| 条件 | 内容 | 確認すること |
|---|---|---|
| A | 標準LoRA、基準rank | 通常のベースライン |
| B | 標準LoRA、Cと総パラメータ数を一致 | 単なるパラメータ増加の効果を分離 |
| C | VariationalLoRA、動的ゲート | 本研究の仮説 |
| D | VariationalLoRA、固定ゲート | 動的ゲート自体の寄与を分離 |

仮説成立の最低条件は、同一条件下で概ね次が再現されることです。

```text
C > B
かつ
C > D
```

- `C > B`: 改善が単なる追加パラメータ数によるものではない
- `C > D`: 入力依存の動的ゲートに意味がある

---

## 現在までに確認できたこと

### M0: 学習可能性の確認 — 合格

RTX 3060 12GB、条件C、約500件、1 epochの実機学習で、M0の確認項目を通過しました。

| 確認項目 | 結果 |
|---|---|
| VRAM内で実行できる | OOMなし。4bit量子化、PagedAdamW8bit、gradient checkpointingを使用 |
| lossが低下する | `1.91 → 約0.5`、63更新 |
| NaN / Infが発生しない | 発生なし |
| ゲートが0または1へ張り付かない | `qw_mean ≈ 0.33`で安定 |

したがって、現時点で確認できたのは次です。

> VariationalLoRAは、少なくともRTX 3060上で学習可能であり、初期実験では即座に発散・崩壊しない。

### ゲート学習率の問題と修正

最初の実機学習では、二つのLoRA経路は学習している一方、ゲート値がほとんど変化しませんでした。

原因は、ゲート部分が数百パラメータ程度しかないのに対し、経路側は約6200万パラメータあり、同一学習率ではゲートの更新が埋もれていたことです。

現在は、ゲート用パラメータを別optimizer groupへ分離し、次の設定を導入しています。

```yaml
gate_lr_multiplier: 10.0
```

学習率倍率を1倍から10倍へ変更したとき、`qw_mean`の移動量は約`-0.001`から`-0.009`へ増加しました。この結果から、ゲートへの勾配経路が生きていることと、倍率に対して概ね線形に反応することは確認できています。

ただし、これは**ゲートが性能改善へ寄与した証明ではありません**。

### 実装・テスト

現在までに次が実装されています。

- VariationalLoRA本体
- Qwen2.5-Coder-7BのFFN層への注入
- 4bit QLoRA学習
- A/B/C/D条件の切り替え
- 条件BとCのパラメータ数を近似一致させるrank計算
- gradient clipping
- NaN / Inf検出と状態ダンプ
- ゲート統計の記録
- pass@k評価
- 数値整合性評価
- 経路独立性評価
- 効果量計算

最新の実装コミットでは、CPU単体テスト **35件成功**と記録されています。

---

## まだ証明されていないこと

M0合格は「方式が動く」という確認であり、「方式が優れている」という確認ではありません。

以下は未証明です。

- VariationalLoRAが標準LoRAより高性能か
- 動的ゲートが性能改善へ寄与するか
- 数値・統計問題の整合性が改善するか
- 問題の言い換えに対して結論が安定するか
- Runovaの実行軌跡を学習し、7Bモデル自身へ開発能力を内在化できるか
- 7B級モデルを主体とした長時間の自律開発が成立するか

したがって、現時点で「高度推論7Bモデルが完成した」とは扱いません。

---

## 次の正式タスク: M1

次の価値ある実験は、新しい機構の追加ではなく、A/B/C/D比較によるM1です。

### M1で行うこと

1. A/B/C/Dを各3seed以上実行する
2. データ、epoch、optimizer、評価条件を固定する
3. 条件BとCの学習可能パラメータ数を記録する
4. `gate_lr_multiplier`を実験変数として明示する
5. seedごとの設定・出力・結果を保存する
6. pass@k、数値整合性、経路独立性、効果量を集計する
7. `C > B`かつ`C > D`が再現されるか判定する

M1の前に、数値・統計評価セットを50〜100問へ拡張し、複数の言い換えとholdoutを用意する必要があります。

### マイルストーン

| 段階 | 内容 | 状態 |
|---|---|---|
| M0 | 小規模学習でVRAM・loss・NaN・ゲートを確認 | 合格 |
| M1 | A/B/C/D × 複数seedの比較実験 | 次の正式タスク |
| M2 | 効果量を含めて仮説を採否判定 | 未実施 |
| M3以降 | 追加機構を一つずつ検証 | M2成功まで保留 |

`component_lora.py`と`newmark_lora.py`は、M2で効果が確認されるまで意図的にstubのままにしています。

---

## Runovaとの関係

LLLMと[Runova](https://github.com/lowebbebb777/Runova)は、長期的には一つの循環を構成します。

```text
Runova
  実タスク、失敗、修復、テスト結果、採用diffを生成
          ↓
LLLM
  実行経験を学習可能なデータへ変換し、7Bモデルへ内在化
          ↓
改良された7Bモデル
  次のRunova開発と実タスク処理を改善
```

役割は次の通りです。

- **Runova**: 外部記憶、観測、ツール実行、検証、失敗修復を担当する実行環境
- **LLLM**: 小型モデル自身の内部能力を育てる研究

ただし、Runovaの軌跡収集から再学習・評価までを自動化した閉ループは、現在まだ構想・設計段階です。実現済みの機能としては扱いません。

---

## 実行例

```bash
cd varlora

# 依存関係
pip install -r requirements.txt

# CPU単体テスト
python3 tests/test_variational_lora.py
python3 tests/test_inject.py
python3 tests/test_evaluate.py
python3 tests/test_train_config.py

# パラメータ数・条件配線の確認
python3 src/train.py --config configs/cond_C.yaml --report-only

# M0
bash scripts/run_m0.sh

# M1アブレーション
bash scripts/run_ablation.sh
```

CUDA版PyTorchは環境に合わせて別途導入してください。

---

## 現在地の評価

| 項目 | 状態 |
|---|---|
| 独自アイデア | あり |
| 基本実装 | 完了 |
| CPU単体テスト | 35件成功と記録 |
| RTX 3060での学習 | M0合格 |
| 初期学習安定性 | 確認済み |
| 標準LoRAとの性能比較 | 未完了 |
| 動的ゲートの有効性 | 未証明 |
| 高度推論7Bモデル | 未完成 |
| Runovaとの学習循環 | 構想・設計段階 |

このリポジトリは単なる構想だけではなく、独自アダプタを実装し、RTX 3060で実際に学習できるところまで到達しています。

一方で、現在地はまだ次の段階です。

> **新方式が動くことを確認した。次は比較実験で、本当に意味がある方式かを判定する。**

新しい壮大な機構を増やす前に、M1を再現可能な形で実行し、VariationalLoRAが標準LoRAを上回るか決着させることを最優先とします。

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

He is currently applying this engineering background to the development of compact local language models capable of advanced reasoning under limited computational resources.

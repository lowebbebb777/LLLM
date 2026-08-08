# DeltaFEM-LLM D1 アーキテクチャ比較計画

最終更新: 2026-08-09

## 1. 決定事項

D1は、単一Transformerモデルのactivation差分を観測する実験ではなく、**同一のD1 recorderと同一指標をTransformer / SSM / SSM-Transformer Hybridの3系統へ適用し、LLMアーキテクチャごとのincremental computabilityを比較する実験**として扱う。

研究系統は次で固定する。

```text
D0: 線形・増分参照
        |
        v
D1: 共通 Activation / State Recorder
        |
        +-- D1-T: Transformer control
        |       `-- Qwen 0.5B
        |
        +-- D1-S: SSM
        |
        `-- D1-H: SSM / Transformer Hybrid
                |
                v
        共通指標による比較
                |
                v
D2: 最も差分局所性が強かった方式・領域だけを実装対象にする
```

Qwen2.5-Coder-0.5B-Instructは本命モデルではない。**D1-TのTransformer対照群**として使用する。

SSMおよびHybridは、DeltaFEM-LLMの「前状態 + 増分 + 再整合」という考え方と構造的に相性がよい可能性があるため、Transformerと並列に比較する。

---

## 2. 研究問い

D1で調べる中心的な問いは次である。

> モデル精度やパラメータ数だけではなく、状態変化の局所性・低ランク性・ブロック性、すなわち **incremental computability** がLLMアーキテクチャによって本質的に異なるか。

単に「差分値が小さいか」は見ない。

次を満たすかを測る。

- 少数のactive自由度で差分エネルギーの大部分を表現できるか
- 差分が層を進むにつれて密化するか、局所性を維持するか
- 差分行列が低ランク構造を持つか
- 局所prompt editと逐次token stepで性質が変わるか
- Transformer / SSM / Hybridで差が再現するか

---

## 3. なぜ3系統を比較するのか

### D1-T: Transformer control

Transformerでは局所的な入力変化がAttentionと正規化を通じて広範囲へ伝播しやすい。

概念的には、

```text
local Delta h
  -> Q / K / V
  -> attention score
  -> softmax normalization
  -> many channels / tokens
```

となるため、初期の局所差分が深い層で急速に密化する可能性がある。

したがってTransformerは、DeltaFEM方式にとって**悲観側の対照群**として重要である。

Qwen2.5-Coder-0.5B-Instructは、小さくRTX 3060上で扱いやすく、residual / attention / MLP / KVを観測しやすいためD1-Tの最初の対照モデルとする。

### D1-S: SSM

SSMでは状態更新が概念的に

```text
s_t = A s_(t-1) + B x_t
```

と表せる。

基準軌道との差を取れば、線形化または線形部分では

```text
Delta s_t = A Delta s_(t-1) + B Delta x_t
```

となり、DeltaFEM-LLMの

```text
previous state + increment -> new state
```

という考え方に近い。

特に状態遷移が対角、ブロック構造、低次元表現などを持つ場合、局所差分を追跡しやすい可能性がある。

### D1-H: SSM / Transformer Hybrid

Hybridは、

- SSMの状態更新性
- Transformer Attentionの柔軟な長距離相互作用

の中間に位置する。

D1-Hでは、実用的な表現力を維持しながら、Transformerより差分局所性が残るかを調べる。

研究上の初期仮説は、あくまで検証対象として

```text
rho90(SSM) < rho90(Hybrid) < rho90(Transformer)
```

という順序が現れる可能性がある、というものである。

この順序を前提に結果を解釈してはならない。再現しなければ仮説を棄却する。

---

## 4. 共通主指標 rho_l^90

層 `l` の状態差分を `Delta h_l` とする。

差分エネルギーの90%を保持するために必要な最小active自由度数を `k_l^90`、全自由度数を `d_l` として、

```text
rho_l^90 = k_l^90 / d_l
```

をD1の主指標とする。

解釈:

- `rho_l^90 -> 0`: 少数自由度に差分が集中している
- `rho_l^90 -> 1`: 差分がほぼ全面に広がっている

同様に95%、99%も補助指標として保存する。

### block版

実GPU実装では要素単位の不規則sparseより、規則的block sparseの方が重要なので、hidden dimensionまたはstate dimensionをblock分割し、90%エネルギー保持に必要なblock率も測る。

```text
rho_block,l^90 = active blocks / all blocks
```

---

## 5. 共通計測項目

全アーキテクチャで、可能な範囲で以下を共通化する。

### 差分局所性

- `rho^90`, `rho^95`, `rho^99`
- block active率
- relative threshold後のchanged fraction
- top-k energy curve

### 低ランク性

複数token、複数入力、またはsequence位置を並べた差分行列について、

- stable rank
- entropy effective rank
- 90% / 95% / 99% SVD energy rank

を測る。

### 深層方向の差分密化

層深度に対する `rho_l^90`、changed fraction、block active率を記録する。

単純な補助量として、

```text
densification = (rho_last^90 - rho_first^90) / max(number_of_layers - 1, 1)
```

を記録する。

値が正なら深層化で密化、負なら局所化傾向を示す。

### 状態種別

可能な範囲で以下を観測する。

| 系統 | 観測候補 |
|---|---|
| Transformer | residual hidden / attention output / MLP output / KV cache |
| SSM | residual hidden / recurrent state / mixer output / state-space internal state |
| Hybrid | residual hidden / SSM state / attention output / KV cache / mixer output |

モデル固有内部状態を直接取得できない場合は、その旨を結果metadataへ記録し、共通hidden state比較を最低ラインとする。

---

## 6. 共通入力regime

### A. Local prompt edit

意味の大部分を保持したまま、一部分だけを編集する。

例:

- コードの境界条件だけ変更
- null処理だけ追加
- 型制約だけ追加
- 数値定数だけ変更
- 日本語指示の一節だけ変更

これはRunovaの反復開発とも相性がよい重要regimeである。

### B. Token step

同じコンテキストで逐次生成し、隣接step間の内部状態差分を測る。

### C. 将来追加候補

- KV block update
- 長文部分編集
- tool-result追記
- Runovaの同一taskに対する反復修正

D1初期ではA/Bを優先する。

---

## 7. 比較の公平性

アーキテクチャ差を測るため、可能な限り以下を揃える。

- 同一prompt pair
- 同一tokenization可能範囲で意味的に同一入力
- batch size = 1
- 同一Windows 11 / RTX 3060 12GB環境
- 同一dtypeまたはdtypeをmetadataへ明記
- 同一max length
- 同一activation view (`last_token`, `aligned_sequence`)
- 同一block size候補
- 同一relative threshold候補
- warm-up後に計測
- seedを保存
- モデルparameter count、hidden size、layer countを保存

モデルサイズを完全一致させることは必須としない。代わりに、モデル規模を必ず記録し、**差分構造の比較**と**実時間性能の比較**を分離する。

D1ではまず差分構造を比較する。速度比較はD4で行う。

---

## 8. D1実験行列

| 軸 | 初期条件 |
|---|---|
| architecture | D1-T / D1-S / D1-H |
| regime | prompt edit / token step |
| activation view | last token / aligned sequence |
| max length | 64 / 128 / 256 |
| block size | 32 / 64 / 128 |
| relative threshold | 1e-4 / 1e-3 / 1e-2 |
| energy target | 90% / 95% / 99% |
| input family | code / numeric constraint / Japanese instruction |

モデルは各系統で、Windows + RTX 3060 12GB上で現実的に実行でき、内部状態またはhidden stateを取得できるものを選ぶ。

D1-SとD1-Hの具体モデルは、実装時に以下を優先して選定する。

1. 公開・再現可能
2. Windows/CUDAで扱いやすい
3. Hugging Face等から通常forward可能
4. 0.1B〜数B級でRTX 3060に収まる
5. state / mixer / hidden hookを実装可能
6. 特殊kernel必須で測定器自体が不安定にならない

---

## 9. Go / No-Go判定

### 局所Go

少なくとも1アーキテクチャの1つ以上の実用regimeで、

- `rho^90 <= 0.10` または `rho_block^90 <= 0.10`
- 複数層で成立
- 複数入力で再現
- zero deltaではない
- toy modelだけではない

を満たせば、その領域をD2候補とする。

### アーキテクチャ比較Go

次のような差が再現すれば、アーキテクチャによってincremental computabilityが異なるという仮説を支持する。

```text
Transformer: rho^90 high / rapid densification
Hybrid:      intermediate
SSM:         rho^90 low / slow densification
```

ただし順序は事前仮説であり、実測が異なれば実測を優先する。

### No-Go / 範囲限定

全系統で、

- `rho^90`が高い
- block局所性がない
- effective rankが高い
- 深い層で急速にdense化する

場合、activation差分そのものを使う汎用incremental inferenceは停止または縮小する。

その場合でも、

- KV限定
- 局所prompt edit限定
- SSM state限定
- CPU/GPU協調
- speculative verification

へ対象を絞る。

否定結果も正式な研究結果として保存する。

---

## 10. D2への接続

D2では3系統すべてにpredictor/correctorを実装しない。

D1の結果から、

1. 最も低い `rho^90`
2. 最も低いblock active率
3. 最も低いeffective rank
4. 最も遅いdensification
5. 実用regimeとの相性

を総合して、**最も差分局所性が強かったアーキテクチャと状態領域だけ**を選ぶ。

その領域に対して、

- top-k predictor
- block sparse predictor
- low-rank predictor
- weighted residual assembly
- local corrector
- periodic reanchor

を実装する。

これにより、研究が3系統の実装競争へ膨張することを防ぐ。

---

## 11. 現時点の位置づけ

```text
VariationalLoRA
  -> 学習可能性を確認した探索済み研究資産として凍結

DeltaFEM D0
  -> 線形増分・重み付き残差台帳・損益式を検証済み

DeltaFEM D1
  -> 共通recorder実装済み
  -> Transformer / SSM / Hybridの横断比較へ研究設計を拡張
  -> Windows RTX 3060で実測する段階
```

D1-TのQwen 0.5Bは研究本命ではなく、Transformer対照群である。

本研究の次の重要な問いは、

> **どのアーキテクチャが最も高性能かではなく、どのアーキテクチャが「差分だけを計算する」という数値計算的構造を最も許容するか。**

である。

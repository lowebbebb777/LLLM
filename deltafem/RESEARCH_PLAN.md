# DeltaFEM-LLM 研究計画

最終更新: 2026-08-05

## 1. 研究問い

LLM推論で毎回すべての状態を密に再計算せず、主にGPUでは変化した局所領域の増分だけを処理し、アンカー状態・差分和・全量状態の整合性を予測子–修正子法で管理することで、精度を制御しながら計算資源を削減できるか。

## 2. 中心仮説

### H1: 差分圧縮可能性

層・ヘッド・チャネル・KVブロックの少なくとも一部では、連続する推論状態または小規模な入力編集に対する差分エネルギーが、少数のactive自由度へ集中する。

### H2: 局所更新の損益

active自由度の割合を`rho`、管理費用比を`h`、完全再計算間隔を`M`とすると、概算条件

```text
rho + h + 1/M < 1
```

を満たす領域では密計算より小さい演算量を実現できる。

### H3: 誤差制御

差分近似で生じる誤差は、低次元残差監視と局所corrector、必要時の全量reanchorを組み合わせることで所定の許容誤差内に制御できる。

### H4: 適用領域の非一様性

通常の逐次1-token生成よりも、長文の局所編集、反復エージェント、KVキャッシュ更新、SSMまたはTransformer–SSMハイブリッドの方が高い差分局所性を示す可能性がある。

## 3. FEM・数値計算法との対応

| 数値計算法 | DeltaFEM-LLM |
|---|---|
| 節点自由度 | hidden channel / token / KV block |
| 要素・領域 | attention head / FFN block / tensor tile |
| 要素増分 | 局所activation差分 |
| assembly | residual streamへの重み付き加算 |
| 全体残差 | 差分和と検証済み全状態の不整合 |
| predictor | 低ランク・疎・局所モデルによる増分予測 |
| corrector | active領域の再計算または完全forward |
| 再メッシュ | active領域・ブロック分割の再選択 |
| checkpoint/restart | anchorの再確定 |

これはFEMの支配方程式をLLMへ直接導入する主張ではなく、状態管理・領域分割・誤差制御の計算原理を移植する研究です。

## 4. 研究原則

- FLOP削減と実時間高速化を分けて評価する
- 差分値が小さいだけでは高速化と見なさない。疎性・低ランク性・ブロック局所性を測る
- 近似結果とexact verificationを分離する
- 毎回のCPU全量再計算は採用しない。CPUは台帳、assembly、低次元監視を中心とする
- GPU専用実装へ進む前に、棄却可能なCPU実験で必要条件を確認する
- ベースラインは密計算、KV再利用、speculative decoding、層skip等とする

## 5. マイルストーン

### D0: 参照実装と損益式 — 実装済み

- アンカー＋重み付き差分台帳
- 疎な線形増分の厳密一致
- 補償和とreanchor
- 管理費用を含む理論FLOPモデル
- 合成データ試験

合格条件:

- 線形参照で相対誤差`1e-12`未満
- dense changeでは速度利得なしと判定される
- 低いchanged fractionでは理論的余地が現れる

### D1: 実モデルactivation差分の観測

対象候補: TinyLlama、OPT-125M、Qwen 0.5B級。

各層で測るもの:

- `Delta h`のtop-k energy比
- block sparsity
- 有効ランク
- 層深度に対する差分の密化率
- attention / MLP / residual / KVの比較
- 逐次生成と局所プロンプト編集の比較

Go条件の初期案:

- 10%以下のactive成分またはブロックで90%以上の差分エネルギーを保持する領域が複数層で再現する

### D2: 近似predictorとcorrector

- top-k、block sparse、低ランクJacobian近似を比較
- exact residual oracleで誤差伝播を測定
- 局所correctorと全量reanchorの頻度を測定
- logit順位、KL、perplexityへの影響を評価

### D3: 安価な残差監視

- ランダム射影スケッチ
- 監視チャネル
- norm統計
- logit margin
- 巡回検査ブロック

false negativeを最重要指標とし、危険な誤差を見逃す監視法は棄却する。

### D4: GPUプロトタイプ

- 規則的なblock sparse形式へ限定
- kernel launch数とメモリ転送を測定
- PyTorch dense baselineと比較
- CPU–GPUを非同期化

Go条件:

- 同一精度帯でend-to-end token latencyまたは編集再評価時間が改善
- VRAM、RAM、転送量を含めて総合的に利得がある

### D5: LLM推論統合

候補:

- 局所編集の再評価
- KVブロック更新
- self-speculativeな中間activation draft
- SSM状態の増分更新
- Runovaの反復状態更新

## 6. Phase 0実験行列

| 変数 | 候補 |
|---|---|
| changed fraction | 1%, 5%, 10%, 25%, 50%, 100% |
| reanchor interval | 4, 8, 16, 32 |
| management cost | dense計算の0%, 2%, 5%, 10% |
| delta scale | 1e-4, 1e-3, 1e-2, 1e-1 |
| dtype | float64参照、後にfloat32/float16 |

記録:

- max absolute / relative error
- theoretical FLOPs
- changed fraction
- reanchor頻度
- 後続段階ではwall timeと転送量

## 7. 棄却・停止条件

次のいずれかが継続して成立する場合、汎用Transformer推論高速化としては停止または適用範囲を限定する。

- 差分が初期層から急速に密化し、active率が常に50%以上
- 残差監視費用が密計算に近い
- corrector/reanchorが高頻度で必要
- GPU上で不規則アクセスとkernel launch費用が削減FLOPを上回る
- 同一出力品質で既存のKV reuseやspeculative decodingを上回れない

否定結果も研究成果として保存し、局所編集・SSM・CPU/GPU協調など成功可能性の高い領域へ絞る。

## 8. 直近の実装ステップ

1. D0参照実装を固定し、Phase-0 JSONを保存
2. activation recorderのモデル非依存インターフェースを設計
3. 最小モデルで連続token間と局所編集前後のactivationを収集
4. layer別のtop-k energy、block sparsity、有効ランクをCSV/JSON化
5. D1 Go/No-Go判定を行う

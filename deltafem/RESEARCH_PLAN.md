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
- GPU専用実装へ進む前に、棄却可能な実験で必要条件を確認する
- ベースラインは密計算、KV再利用、speculative decoding、層skip等とする
- toyモデルは計測配線の検証にだけ使い、仮説採否には使わない

## 5. 標準実験環境

2026-08-05から、主な検証環境をWindowsへ移行する。

```text
Repository: C:\Users\soich\PycharmProjects\LLLM
OS: Windows 11
GPU: NVIDIA GeForce RTX 3060 12GB
Python environment: C:\Users\soich\PycharmProjects\LLLM\.venv
Model cache: C:\Users\soich\PycharmProjects\LLLM\.cache\huggingface
```

環境生成は`deltafem/scripts/setup_windows.ps1`を正本とし、実験結果にはPython、PyTorch、CUDA、GPU名、モデル名、入力条件を記録する。

## 6. マイルストーン

### D0: 参照実装と損益式 — 合格

- アンカー＋重み付き差分台帳
- 疎な線形増分の厳密一致
- 補償和とreanchor
- 管理費用を含む理論FLOPモデル
- 合成データ試験

合格条件:

- 線形参照で相対誤差`1e-12`未満
- dense changeでは速度利得なしと判定される
- 低いchanged fractionでは理論的余地が現れる

### D1: 実モデルactivation差分の観測 — 計測実装完了、Windows実機測定待ち

実装済み:

- Hugging Face causal LMのモデル非依存forward recorder
- residual hidden statesの採取
- attention / MLP module hookの自動選択
- legacy cacheとDynamicCacheを想定したKV採取
- 局所プロンプト編集前後の比較
- greedy逐次token間の比較
- last-token比較と整列sequence比較
- top-k energy、block energy、changed fraction、有効ランク
- 層別densification指標
- JSON / UTF-8 BOM付きCSV出力
- D1 Go候補の自動スクリーニング
- 外部モデル不要のtoy smokeと単体テスト

初回実モデル:

```text
Qwen/Qwen2.5-Coder-0.5B-Instruct
```

比較候補:

- Qwen2.5-Coder-1.5B-Instruct
- TinyLlama級
- OPT-125M級

各層で測るもの:

- `Delta h`のtop-k energy比
- block sparsity
- stable rank / entropy effective rank / energy rank
- 層深度に対する差分の密化率
- attention / MLP / residual / KVの比較
- 逐次生成と局所プロンプト編集の比較

Go条件の初期案:

- 10%以下のactive成分またはブロックで90%以上の差分エネルギーを保持する領域が複数層・複数入力で再現する
- zero deltaやtoy smokeだけではGoにしない
- 少なくとも2種類の入力群または2モデルで再現性を確認してからD2へ進む

D1実験行列:

| 軸 | 初期条件 |
|---|---|
| regime | prompt edit / token step |
| activation view | last token / aligned sequence |
| model | Qwen 0.5B、次に1.5Bまたは別系列 |
| max length | 64 / 128 / 256 |
| block size | 32 / 64 / 128 |
| threshold | relative `1e-4` / `1e-3` / `1e-2` |
| data | コード境界条件、null処理、制約追加、日本語編集 |

### D2: 近似predictorとcorrector

D1 Go条件を満たした領域だけを対象にする。

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

## 7. 棄却・停止条件

次のいずれかが継続して成立する場合、汎用Transformer推論高速化としては停止または適用範囲を限定する。

- 差分が初期層から急速に密化し、active率が常に50%以上
- 残差監視費用が密計算に近い
- corrector/reanchorが高頻度で必要
- GPU上で不規則アクセスとkernel launch費用が削減FLOPを上回る
- 同一出力品質で既存のKV reuseやspeculative decodingを上回れない

否定結果も研究成果として保存し、局所編集・SSM・CPU/GPU協調など成功可能性の高い領域へ絞る。

## 8. 直近の実行ステップ

1. Windowsで`setup_windows.ps1`を実行し`.venv`を生成
2. `verify_windows.ps1`で15テストとtoy smokeを確認
3. Qwen2.5-Coder-0.5BでD1初回測定
4. `last_token`と`aligned_sequence`を比較
5. max lengthとblock sizeを振り、局所性が測定定義に依存しないか確認
6. 実結果の要約だけを`results/`へ固定コミット
7. D1 Go/No-Goを判定し、Go領域だけD2へ進める

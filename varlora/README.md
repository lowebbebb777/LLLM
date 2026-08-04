# VariationalLoRA

> **研究状態: 探索的M0検証を完了し、2026-08-05から当面凍結。**  
> RTX 3060上で学習可能であることは確認済みですが、標準LoRAに対する性能優位と動的ゲートの寄与は未証明です。実装と記録は研究資産として保存し、主研究は[`../deltafem/`](../deltafem/)へ移行します。

有限要素法の変分原理における「連続場・離散自由度・形状関数・重み付き統合」を構造的アナロジーとして借用し、二つの低ランク経路を入力依存ゲートで統合する独自LoRAアダプタです。

詳細な設計と元の実験計画は[`SPEC.md`](./SPEC.md)を参照してください。

## 結論として確定したこと

RTX 3060 12GB、条件C、約500件、1 epochの単回実機学習でM0を通過しました。

| 確認項目 | 結果 |
|---|---|
| VRAM内で実行 | OOMなし。4bit、PagedAdamW8bit、gradient checkpointingを使用 |
| loss低下 | `1.91 -> 約0.5`、63更新 |
| NaN / Inf | 発生なし |
| ゲート崩壊 | `qw_mean`が約0.33で、0/1への張り付きなし |
| ゲート勾配経路 | `gate_lr`倍率に応じた変化を確認 |

したがって、次だけを主張します。

> VariationalLoRAはRTX 3060級の環境で学習可能であり、初期の短時間学習では即座に発散・崩壊しなかった。

## 確定していないこと

- 標準LoRAより高性能か
- 同じ学習可能パラメータ数のLoRAより高性能か
- 入力依存ゲートが固定ゲートより有効か
- 数値整合性やコード生成能力が安定して改善するか
- 複数seedで効果が再現するか

元のM1はA/B/C/Dを複数seedで比較する計画でしたが、2026-08-05時点では未実施です。VariationalLoRAを棄却したわけではなく、限られた研究資源を「重み付残差法＋動的差分推論」の検証へ移すため凍結します。

## 構造

```text
入力 x
  |-- cont_A -> cont_B --|
  |                      |-- 入力依存ゲート -> 出力
  |-- disc_A -> disc_B --|
```

概略:

```python
cont = cont_B(cont_A(x))
disc = disc_B(disc_A(x))
N = softmax(shape_fn(x) / tau)
qw = sum(N * quad_weights)
out = qw * cont + (1 - qw) * disc
```

## 保存されている実装

- VariationalLoRA本体
- Qwen2.5-Coder-7B FFN層への注入
- 4bit QLoRA学習
- A/B/C/D条件切り替え
- 条件BとCのパラメータ数近似一致
- 自前学習ループ
- gradient clipping、NaN/Inf検出、状態ダンプ
- ゲート統計
- pass@k、数値整合性、経路独立性、効果量評価
- CPU単体テスト

`component_lora.py`と`newmark_lora.py`は、追加仮説を未検証のまま積み重ねないためstubとして残しています。

## 実行

```bash
cd varlora
python3 -m pip install -r requirements.txt

python3 tests/test_variational_lora.py
python3 tests/test_inject.py
python3 tests/test_evaluate.py
python3 tests/test_train_config.py

# 設定とパラメータ数の確認
python3 src/train.py --config configs/cond_C.yaml --report-only

# 保存済みM0スクリプト
bash scripts/run_m0.sh

# 将来研究を再開する場合のA/B/C/D比較
bash scripts/run_ablation.sh
```

## 再開条件

次のいずれかが明確になった場合にM1を再開します。

- DeltaFEM-LLMで二経路ゲートを比較対照として必要とする
- 再現可能な評価セットと複数seed実行時間を確保できる
- VariationalLoRA固有の仮説を、標準LoRA・固定ゲートと交絡なく判定できる

再開時も成功を前提とせず、`C > B`かつ`C > D`を最低条件として比較します。

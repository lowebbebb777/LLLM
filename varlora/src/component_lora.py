"""ComponentSeparatedLoRA (M3, 成分分離法のアナロジー) — ★実装は M2 成功まで保留★

SPEC §7'-M3 / §9 の指示により、本機構は **意図的に未実装** とする。

理由 (SPEC より):
    - 「M2 で VariationalLoRA の効果が確認できなければ、M3 以降には進まない」(§7-M2)
    - 「2成分で効果が出ないなら多成分・時間積分に拡張しても無駄」(§7-M2)
    - 計算効率/疎性アナロジー (ハードルーティング, Gumbel-softmax/STE) は
      「M0手応え前には入れない。複雑性が勝つため」(§7'-M3 第二の動機)

M2 成功後に着手する際の設計メモ (SPEC §7'-M3):
    J_total = J_volume + J_surface + J_interface + J_crack   (加法的分解)
    出力 = Σ_c route_c(x) · component_c(x)
    - VariationalLoRA の 2 成分 (連続/離散) を N 成分に一般化
    - 各成分に意味を割当 (構文/数値/制御フロー) ＝ MoE に対する解釈可能性の差別化点
    - n_components は RTX 3060 では 2〜3 が現実的上限
    - 各 comp_B はゼロ初期化
    - mode collapse 対策: auxiliary loss (負荷分散) 必須
    - 検証は M3-A (標準LoRA, パラメータ一致) / M3-C / M3-D (ルーティング固定) で交絡を断つ
"""

raise NotImplementedError(
    "ComponentSeparatedLoRA (M3) は SPEC §7-M2/§9 により M2 成功確認まで実装保留。"
    " VariationalLoRA (M0→M2) の効果が確認されてから本ファイルを実装すること。"
)

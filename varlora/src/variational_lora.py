"""VariationalLoRA — 有限要素法の変分原理を構造的アナロジーとして借用した LoRA アダプタ.

SPEC §2 に対応する本体実装。設計意図 (変分原理アナロジー) を失わないよう、
FEM 側の概念と NN 側の構成要素の対応をクラス/変数名とコメントで明示する。

    FEM 側                              NN 側 (本モジュール)
    ------------------------------     -----------------------------------------
    連続場 σ(x,y), ε(x,y)               cont 経路  (cont_A → cont_B)
    離散場 節点自由度 u_I               disc 経路  (disc_A → disc_B)
    形状関数 N_I(ξ), ΣN_I = 1           shape_fn → softmax (分割の単位性 = ΣN=1)
    ガウス求積 Σ w_g·f(ξ_g)             quad_weights による求積点での重み付き統合
    変分原理 内部仕事 + 境界仕事         out = qw·cont + (1-qw)·disc

標準 LoRA との違い (SPEC §2.1):
    標準 LoRA : W' = W + (α/r)·B·A          (単一の低ランク経路)
    Variational: 連続場経路 + 離散場経路 を入力依存ゲートで統合する 2 系統。

実装制約 (SPEC §2.3):
    - ゼロ初期化: cont_B / disc_B は zeros。学習開始時にアダプタ寄与 = 0 で
      ベースモデルを壊さない (標準 LoRA 同様)。
    - 求積重み初期値: quad_weights は 1/n_quad で初期化 (一様)。default n_quad=3
      は元コードの 3 点求積 (WR3=0.1667) に対応。
    - ゲートの数値安定性: softmax 前の logit を温度 τ でスケール (softmax(logit/τ))。
      NaN 対策 (SPEC §6.3) の主要ポイント。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1 つの注入対象 module (gate_proj / up_proj / down_proj) ごとの VariationalLoRA 寄与。
# ガウス求積点 (n_quad) は「連続場 → 離散場」へ射影する分割の単位 (ΣN_I=1) を構成する。
DEFAULT_N_QUAD = 3  # 元コードの 3 点求積に対応 (WR3=0.1667)


@dataclass
class VariationalLoRAConfig:
    """VariationalLoRA 1 層分の構成。

    gate_mode:
        "dynamic" → SPEC 条件 C。shape_fn + quad_weights を学習し、求積重み qw を
                     入力依存で動的に決める (本仮説)。
        "fixed"   → SPEC 条件 D。qw を学習させず定数 fixed_gate (default 0.5) に固定。
                     cont と disc を等 weight で合成し、ゲート動的性の寄与を分離する。
    """

    in_features: int
    out_features: int
    r: int = 16
    alpha: int = 32
    n_quad: int = DEFAULT_N_QUAD
    gate_mode: str = "dynamic"  # "dynamic" (条件C) | "fixed" (条件D)
    fixed_gate: float = 0.5  # gate_mode="fixed" のときの定数 qw
    tau: float = 1.0  # softmax 温度 τ (SPEC §2.3 / §6.3, NaN 対策)
    clamp_quad_weight: bool = False  # qw を [0,1] にクランプ (partition-of-unity 保険)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.gate_mode not in ("dynamic", "fixed"):
            raise ValueError(f"gate_mode must be 'dynamic' or 'fixed', got {self.gate_mode!r}")
        if self.r <= 0:
            raise ValueError(f"r must be positive, got {self.r}")
        if self.n_quad <= 0:
            raise ValueError(f"n_quad must be positive, got {self.n_quad}")


class VariationalLoRA(nn.Module):
    """有限要素法の変分原理アナロジーに基づく 2 系統 LoRA アダプタ (SPEC §2.2)。

    forward は base 層の出力に「加算」する寄与 (delta) を返す。注入は inject.py が
    base linear をラップして `base(x) + VariationalLoRA(x)` を実現する。
    """

    def __init__(self, config: VariationalLoRAConfig) -> None:
        super().__init__()
        self.cfg = config
        d_in, d_out, r, q = (
            config.in_features,
            config.out_features,
            config.r,
            config.n_quad,
        )

        # --- 2 系統の低ランクアダプタ ---------------------------------------
        # cont 経路 = 連続場の寄与 (体積積分項アナロジー)
        self.cont_A = nn.Linear(d_in, r, bias=False)
        self.cont_B = nn.Linear(r, d_out, bias=False)
        # disc 経路 = 離散場の寄与 (節点和アナロジー)
        self.disc_A = nn.Linear(d_in, r, bias=False)
        self.disc_B = nn.Linear(r, d_out, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # --- 形状関数ゲート + 求積重み (gate_mode="dynamic" のみ) -----------
        if config.gate_mode == "dynamic":
            # 形状関数 N_I(ξ): 入力 x を n_quad 個の求積点 logit へ写す。
            # softmax により ΣN_I = 1 (分割の単位性) を保証する。
            self.shape_fn = nn.Linear(d_in, q, bias=True)
            # ガウス求積重み w_g。初期値 1/n_quad で一様 (SPEC §2.3)。学習可能 ——
            # 一様固定だと qw = Σ N_q·(1/q) = (1/q)·ΣN_q = 1/q となり入力非依存に
            # なってしまう (softmax の ΣN=1 のため)。動的ゲートが「動的」であるためには
            # quad_weights が非一様化する必要があるので requires_grad=True とする。
            self.quad_weights = nn.Parameter(torch.full((q,), 1.0 / q))
        else:
            # 条件 D: ゲート機構を持たない。qw は定数 fixed_gate。
            self.shape_fn = None
            self.register_parameter("quad_weights", None)

        # (α/r) スケール (標準 LoRA と同じ)
        self.scaling = config.alpha / config.r

        # M0 観察用: 直近 forward の求積重み qw 平均 (ゲートの 0/1 張り付き検出)
        self._last_qw_mean = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # cont_A / disc_A / shape_fn は通常の線形層初期化 (kaiming)。SPEC §2.3。
        nn.init.kaiming_uniform_(self.cont_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.disc_A.weight, a=math.sqrt(5))
        if self.shape_fn is not None:
            nn.init.kaiming_uniform_(self.shape_fn.weight, a=math.sqrt(5))
            nn.init.zeros_(self.shape_fn.bias)
        # ゼロ初期化: cont_B / disc_B は zeros → 学習開始時アダプタ寄与 = 0。
        nn.init.zeros_(self.cont_B.weight)
        nn.init.zeros_(self.disc_B.weight)
        if self.quad_weights is not None:
            with torch.no_grad():
                self.quad_weights.fill_(1.0 / self.cfg.n_quad)

    # ------------------------------------------------------------------ gate
    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """求積重み qw を返す。形状 [*, 1] (cont/disc をブレンドするスカラー場)。

        dynamic: qw = Σ_g N_g(x)·w_g   (N=softmax(shape_fn(x)/τ), ΣN=1)
        fixed  : qw = fixed_gate        (定数, 学習しない)
        """
        if self.cfg.gate_mode == "fixed":
            # 条件 D: cont と disc を等 weight (default 0.5) で合成。
            return x.new_full((*x.shape[:-1], 1), self.cfg.fixed_gate)

        # 形状関数ゲート: ΣN_I = 1 を softmax で保証 (SPEC §2.2)。
        # 温度 τ で logit をスケール (数値安定性 / NaN 対策)。
        logits = self.shape_fn(x) / self.cfg.tau  # [*, n_quad]
        N = F.softmax(logits, dim=-1)  # 分割の単位性 ΣN_I = 1
        # ガウス求積による統合: qw = Σ_g N_g·w_g → スカラー化。
        qw = (N * self.quad_weights).sum(dim=-1, keepdim=True)  # [*, 1]
        if self.cfg.clamp_quad_weight:
            # partition-of-unity (qw∈[0,1]) を強制したいときの保険。default off。
            qw = qw.clamp(0.0, 1.0)
        return qw

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)

        # 2 系統の低ランクアダプタ
        cont = self.cont_B(self.cont_A(x))  # 連続場の寄与 (体積積分項)
        disc = self.disc_B(self.disc_A(x))  # 離散場の寄与 (節点和)

        # 求積重み (形状関数ゲート + ガウス求積)
        qw = self.gate(x)  # [*, 1]
        if self.cfg.gate_mode == "dynamic":
            # 観察用に直近 qw 平均を記録 (M0: ゲートが 0/1 に張り付いてないか)
            self._last_qw_mean = qw.detach().mean()

        # 変分原理的結合: 内部仕事 (連続) + 境界仕事 (離散)
        out = qw * cont + (1.0 - qw) * disc
        return self.scaling * out

    # ------------------------------------------------------------ utilities
    def num_adapter_parameters(self) -> int:
        """この層の VariationalLoRA 追加パラメータ数 (SPEC §2.4 の実測)。

        cont_A: d_in·r, cont_B: r·d_out, disc_A: d_in·r, disc_B: r·d_out,
        shape_fn: d_in·q (+bias q), quad_weights: q。
        bias を無視すると ≈ 2r(d_in+d_out) + d_in·q。
        """
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"in={c.in_features}, out={c.out_features}, r={c.r}, alpha={c.alpha}, "
            f"n_quad={c.n_quad}, gate_mode={c.gate_mode}, tau={c.tau}"
        )


def variational_lora_param_count(
    in_features: int, out_features: int, r: int, n_quad: int, *, count_bias: bool = True
) -> int:
    """注入 1 module あたりの VariationalLoRA 追加パラメータ数 (解析式, SPEC §2.4)。

    交絡対照 (条件 B) の rank 算出 (inject.compute_matched_rank) と整合させるため、
    モジュールを実体化せずに数えられる解析式として独立に提供する。
    """
    p = 2 * r * (in_features + out_features)  # cont/disc の A,B 2 系統
    p += in_features * n_quad  # shape_fn weight
    if count_bias:
        p += n_quad  # shape_fn bias
    p += n_quad  # quad_weights
    return p

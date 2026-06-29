"""FFN 層への注入ロジック + 交絡対照 (条件 B) の rank 一致計算 (SPEC §3.2, §2.4)。

適用対象 (SPEC §3.2):
    - 対象 module: gate_proj, up_proj, down_proj  (FFN 層のみ)
    - Attention 層 (q/k/v/o_proj) には適用しない
      理由: FFN はトークン独立の非線形変換であり、要素ごとのガウス求積アナロジーと
      構造的に最も整合する。Attention は勾配が不安定になりやすく、RTX 3060 の
      デバッグ反復回数が限られる中ではリスクが高い。

交絡対照 (SPEC §2.4, §4.2 条件 B, §9-3):
    条件 C (VariationalLoRA) の追加パラメータ数に、条件 B (標準 LoRA) の rank を
    一致させる。これを怠ると「性能差がアナロジー由来か単なるパラメータ増加由来か」を
    分離できない。compute_matched_rank() がその rank を厳密に算出する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from variational_lora import (
    VariationalLoRA,
    VariationalLoRAConfig,
    variational_lora_param_count,
)

# SPEC §3.2: FFN 層のみ。Attention には適用しない。
FFN_TARGET_MODULES: List[str] = ["gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# 交絡対照: 条件 B の rank 一致計算 (SPEC §2.4)
# ---------------------------------------------------------------------------
def compute_matched_rank(
    in_features: int,
    out_features: int,
    r: int,
    n_quad: int,
    *,
    count_bias: bool = True,
) -> int:
    """標準 LoRA の rank を VariationalLoRA(r, n_quad) の追加パラメータ数に一致させる。

    標準 LoRA 1 module の追加パラメータ数 = r_B·(in + out)。
    VariationalLoRA 1 module = 2r(in+out) + in·q (+ q bias) + q (quad_weights)。
    両者を等しく置いて r_B を解く:

        r_B = [2r(in+out) + in·q (+ q) + q] / (in + out)
            = 2r + q·(in + 2) / (in + out)     (bias を数える場合)

    四捨五入して最小 1 を返す。FFN の各 module は形状 (in, out) が異なるため、
    module ごとに rank が変わりうる (build_matched_rank_pattern で吸収する)。
    """
    var_params = variational_lora_param_count(
        in_features, out_features, r, n_quad, count_bias=count_bias
    )
    denom = in_features + out_features
    r_b = round(var_params / denom)
    return max(1, int(r_b))


def build_matched_rank_pattern(
    module_shapes: Dict[str, tuple[int, int]],
    r: int,
    n_quad: int,
    *,
    count_bias: bool = True,
) -> Dict[str, int]:
    """各対象 module 名 → 交絡一致 rank の辞書 (peft の rank_pattern 用)。

    module_shapes: {"gate_proj": (in, out), "up_proj": (in, out), "down_proj": (in, out)}
    """
    return {
        name: compute_matched_rank(in_f, out_f, r, n_quad, count_bias=count_bias)
        for name, (in_f, out_f) in module_shapes.items()
    }


def collect_ffn_module_shapes(
    model: nn.Module, target_modules: List[str] = FFN_TARGET_MODULES
) -> Dict[str, tuple[int, int]]:
    """対象 module 名 → 代表的な (in_features, out_features) を 1 つ収集する。

    同名 module (各層の gate_proj 等) は形状が同一なので最初の 1 つで代表させる。
    """
    shapes: Dict[str, tuple[int, int]] = {}
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in target_modules and leaf not in shapes:
            in_f, out_f = _linear_in_out(module)
            if in_f is not None:
                shapes[leaf] = (in_f, out_f)
    return shapes


def _linear_in_out(module: nn.Module) -> tuple[Optional[int], Optional[int]]:
    """nn.Linear / bitsandbytes Linear4bit から (in_features, out_features) を得る。"""
    in_f = getattr(module, "in_features", None)
    out_f = getattr(module, "out_features", None)
    if in_f is not None and out_f is not None:
        return int(in_f), int(out_f)
    weight = getattr(module, "weight", None)
    if weight is not None and weight.dim() == 2:
        return int(weight.shape[1]), int(weight.shape[0])
    return None, None


# ---------------------------------------------------------------------------
# VariationalLoRA の注入 (条件 C / D)
# ---------------------------------------------------------------------------
class VariationalLoRALinear(nn.Module):
    """base linear (凍結/4bit) を VariationalLoRA でラップする。

    forward: base(x) + VariationalLoRA(x)
    base の重みは学習させず、追加されるのは VariationalLoRA のパラメータのみ。
    """

    def __init__(self, base: nn.Module, var_lora: VariationalLoRA) -> None:
        super().__init__()
        self.base = base
        self.var_lora = var_lora
        # base は凍結 (QLoRA: 4bit 量子化済み重みは学習しない)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # アダプタはそれ自身の dtype で計算し、base 出力 dtype に合わせて加算する。
        adapter_dtype = self.var_lora.cont_A.weight.dtype
        delta = self.var_lora(x.to(adapter_dtype))
        return base_out + delta.to(base_out.dtype)


@dataclass
class VariationalInjectionSummary:
    n_injected: int
    target_modules: List[str]
    trainable_params: int
    per_module_param_count: Dict[str, int]


def inject_variational_lora(
    model: nn.Module,
    *,
    r: int = 16,
    alpha: int = 32,
    n_quad: int = 3,
    gate_mode: str = "dynamic",
    fixed_gate: float = 0.5,
    tau: float = 1.0,
    dropout: float = 0.0,
    clamp_quad_weight: bool = False,
    target_modules: List[str] = FFN_TARGET_MODULES,
) -> VariationalInjectionSummary:
    """model の FFN 対象 module を VariationalLoRALinear に置き換える (in-place)。

    条件 C: gate_mode="dynamic"、条件 D: gate_mode="fixed", fixed_gate=0.5。
    """
    replacements: List[tuple[nn.Module, str, nn.Module]] = []
    per_module_param_count: Dict[str, int] = {}

    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf not in target_modules:
            continue
        in_f, out_f = _linear_in_out(module)
        if in_f is None:
            continue
        cfg = VariationalLoRAConfig(
            in_features=in_f,
            out_features=out_f,
            r=r,
            alpha=alpha,
            n_quad=n_quad,
            gate_mode=gate_mode,
            fixed_gate=fixed_gate,
            tau=tau,
            dropout=dropout,
            clamp_quad_weight=clamp_quad_weight,
        )
        var = VariationalLoRA(cfg).to(_module_param_device(module))
        wrapped = VariationalLoRALinear(module, var)
        parent, attr = _find_parent(model, name)
        replacements.append((parent, attr, wrapped))
        per_module_param_count.setdefault(leaf, var.num_adapter_parameters())

    for parent, attr, wrapped in replacements:
        setattr(parent, attr, wrapped)

    trainable = count_trainable_parameters(model)
    return VariationalInjectionSummary(
        n_injected=len(replacements),
        target_modules=list(target_modules),
        trainable_params=trainable,
        per_module_param_count=per_module_param_count,
    )


def _find_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _module_param_device(module: nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


def freeze_base_train_adapters(model: nn.Module) -> None:
    """base を全凍結し、VariationalLoRA 配下のみ学習可能にする。"""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, VariationalLoRA):
            for p in module.parameters():
                p.requires_grad_(True)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 4bit 量子化設定 (SPEC §3.3) — bitsandbytes は CUDA 必須なので遅延 import
# ---------------------------------------------------------------------------
def build_bnb_config():
    """QLoRA 構成の BitsAndBytesConfig (SPEC §3.3)。

    load_in_4bit=True, nf4, compute_dtype=float16, double_quant=True (VRAM 節約)。
    """
    from transformers import BitsAndBytesConfig  # 遅延 import (CPU sandbox では不要)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ---------------------------------------------------------------------------
# 標準 LoRA 設定 (条件 A / B) — peft.LoraConfig を遅延 import で組む
# ---------------------------------------------------------------------------
def build_standard_lora_peft_config(
    model: Optional[nn.Module],
    *,
    r: int,
    alpha: int,
    dropout: float = 0.0,
    n_quad_for_match: Optional[int] = None,
    target_modules: List[str] = FFN_TARGET_MODULES,
):
    """条件 A / B 用の peft LoraConfig を返す。

    n_quad_for_match を渡すと条件 B として動作する: 各対象 module の rank を
    VariationalLoRA(r, n_quad) の追加パラメータ数に一致させる rank_pattern を
    生成する (SPEC §2.4)。FFN module は形状が異なるため module ごとに rank が変わる。

    n_quad_for_match=None なら条件 A (一様 rank=r) として動作する。
    """
    from peft import LoraConfig  # 遅延 import

    rank_pattern: Dict[str, int] = {}
    base_rank = r
    if n_quad_for_match is not None:
        if model is None:
            raise ValueError("条件 B (rank 一致) には model が必要 (module 形状を読むため)")
        shapes = collect_ffn_module_shapes(model, target_modules)
        rank_pattern = build_matched_rank_pattern(shapes, r, n_quad_for_match)
        # rank_pattern は個別 rank を上書きする。デフォルト r は代表値にしておく。
        if rank_pattern:
            base_rank = max(rank_pattern.values())

    return LoraConfig(
        r=base_rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        rank_pattern=rank_pattern,
        bias="none",
        task_type="CAUSAL_LM",
    )

"""FFN 注入 + 交絡対照 rank 一致の単体テスト (CPU, GPU 不要)。

SPEC §3.2 (FFN のみ注入, Attention は不可侵) と §2.4 (条件 B の rank 一致) を検証。
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inject import (  # noqa: E402
    VariationalLoRALinear,
    collect_ffn_module_shapes,
    compute_matched_rank,
    build_matched_rank_pattern,
    count_trainable_parameters,
    freeze_base_train_adapters,
    inject_variational_lora,
)
from variational_lora import VariationalLoRA, variational_lora_param_count  # noqa: E402


class ToyMLP(nn.Module):
    def __init__(self, d=32, dff=64):
        super().__init__()
        self.gate_proj = nn.Linear(d, dff, bias=False)
        self.up_proj = nn.Linear(d, dff, bias=False)
        self.down_proj = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyModel(nn.Module):
    def __init__(self, n_layers=2, d=32, dff=64):
        super().__init__()
        self.layers = nn.ModuleList([ToyMLP(d, dff) for _ in range(n_layers)])
        # Attention 相当 — 注入されてはならない (SPEC §3.2)
        self.q_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def test_inject_replaces_only_ffn():
    model = ToyModel(n_layers=2)
    summary = inject_variational_lora(model, r=4, alpha=8, n_quad=3)
    # 2 layers * 3 FFN modules = 6
    assert summary.n_injected == 6, summary.n_injected
    # FFN は VariationalLoRALinear に置換
    for layer in model.layers:
        assert isinstance(layer.gate_proj, VariationalLoRALinear)
        assert isinstance(layer.up_proj, VariationalLoRALinear)
        assert isinstance(layer.down_proj, VariationalLoRALinear)
    # Attention は不可侵
    assert isinstance(model.q_proj, nn.Linear)
    assert not isinstance(model.q_proj, VariationalLoRALinear)


def test_inject_preserves_output_at_init():
    # ゼロ初期化により base+delta == base (注入直後はモデル出力不変)
    model = ToyModel(n_layers=2)
    x = torch.randn(2, 5, 32)
    before = model(x).clone()
    inject_variational_lora(model, r=4, alpha=8, n_quad=3)
    after = model(x)
    assert torch.allclose(before, after, atol=1e-6), (before - after).abs().max()


def test_freeze_base_train_adapters():
    model = ToyModel(n_layers=2)
    inject_variational_lora(model, r=4, alpha=8, n_quad=3)
    freeze_base_train_adapters(model)
    # base linear は凍結、VariationalLoRA のみ学習可能
    for module in model.modules():
        if isinstance(module, VariationalLoRALinear):
            for p in module.base.parameters():
                assert not p.requires_grad
    trainable = count_trainable_parameters(model)
    # 6 modules 分の VariationalLoRA パラメータ
    expected = sum(
        m.num_adapter_parameters() for m in model.modules() if isinstance(m, VariationalLoRA)
    )
    assert trainable == expected, (trainable, expected)


def test_collect_ffn_module_shapes():
    model = ToyModel(d=32, dff=64)
    shapes = collect_ffn_module_shapes(model)
    assert shapes["gate_proj"] == (32, 64)
    assert shapes["up_proj"] == (32, 64)
    assert shapes["down_proj"] == (64, 32)


def test_matched_rank_param_counts_close():
    # 条件 B の rank で標準 LoRA を組んだとき、パラメータ数が条件 C に十分近いこと
    for (in_f, out_f) in [(3584, 18944), (18944, 3584), (32, 64), (64, 32)]:
        r, q = 16, 3
        r_b = compute_matched_rank(in_f, out_f, r, q)
        standard = r_b * (in_f + out_f)  # 標準 LoRA 1 module の追加パラメータ
        variational = variational_lora_param_count(in_f, out_f, r, q)
        rel = abs(standard - variational) / variational
        # rank は整数丸めのため完全一致はしないが、数% 以内に収まるべき
        assert rel < 0.05, (in_f, out_f, r_b, standard, variational, rel)


def test_build_matched_rank_pattern():
    model = ToyModel(d=32, dff=64)
    shapes = collect_ffn_module_shapes(model)
    pattern = build_matched_rank_pattern(shapes, r=16, n_quad=3)
    assert set(pattern.keys()) == {"gate_proj", "up_proj", "down_proj"}
    for v in pattern.values():
        assert isinstance(v, int) and v >= 1
    # rank はおおむね 2*r 前後 (SPEC §2.4: 標準 LoRA の約2倍)
    assert all(2 * 16 - 5 <= v <= 2 * 16 + 10 for v in pattern.values()), pattern


def test_fixed_gate_injection():
    model = ToyModel(n_layers=1)
    inject_variational_lora(model, r=4, alpha=8, n_quad=3, gate_mode="fixed", fixed_gate=0.5)
    for m in model.modules():
        if isinstance(m, VariationalLoRA):
            assert m.cfg.gate_mode == "fixed"
            assert m.shape_fn is None


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"OK: {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

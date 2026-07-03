"""カスタム学習ループのヘルパーの単体テスト (CPU, transformers/GPU 不要)。

train_loop 本体は HF モデル (.loss) と get_scheduler を要するため実機側で回すが、
NaN ダンプ・ゲート観察・optimizer 選択・アダプタ保存の各ヘルパーはここで検証する。
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inject import inject_variational_lora, freeze_base_train_adapters  # noqa: E402
from train import (  # noqa: E402
    TrainConfig,
    build_optimizer,
    build_param_groups,
    gate_statistics,
    save_adapter,
)
from variational_lora import VariationalLoRA  # noqa: E402


class ToyMLP(nn.Module):
    def __init__(self, d=16, dff=32):
        super().__init__()
        self.gate_proj = nn.Linear(d, dff, bias=False)
        self.up_proj = nn.Linear(d, dff, bias=False)
        self.down_proj = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyMLP() for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def _injected(gate_mode="dynamic"):
    m = ToyModel()
    inject_variational_lora(m, r=4, alpha=8, n_quad=3, gate_mode=gate_mode)
    freeze_base_train_adapters(m)
    return m


def test_last_qw_mean_recorded_dynamic():
    m = _injected("dynamic")
    # forward 前は None
    for sub in m.modules():
        if isinstance(sub, VariationalLoRA):
            assert sub._last_qw_mean is None
    m(torch.randn(2, 4, 16))
    found = [s for s in m.modules() if isinstance(s, VariationalLoRA)]
    assert all(s._last_qw_mean is not None for s in found)
    # qw は (0,1) 付近のスカラー
    v = float(found[0]._last_qw_mean)
    assert 0.0 <= v <= 1.0, v


def test_gate_statistics_string():
    m = _injected("dynamic")
    m(torch.randn(2, 4, 16))
    s = gate_statistics(m)
    assert "qw_mean=" in s and "quad_w=" in s, s


def test_gate_statistics_fixed_is_na():
    m = _injected("fixed")
    s = gate_statistics(m)
    assert s == "qw=n/a", s  # fixed は dynamic ゲートを持たない


def test_build_param_groups_separates_gate():
    m = _injected("dynamic")
    cfg = TrainConfig(condition="C", learning_rate=1e-4, gate_lr_multiplier=10.0)
    groups, n_gate = build_param_groups(m, cfg)
    # 2 group: base(cont/disc) と gate(shape_fn/quad_weights)
    assert len(groups) == 2, groups
    base_g, gate_g = groups[0], groups[1]
    assert abs(base_g["lr"] - 1e-4) < 1e-12
    assert abs(gate_g["lr"] - 1e-3) < 1e-12  # 10x
    # gate group の要素数 = (shape_fn.weight + shape_fn.bias + quad_weights) * 注入数
    # ToyModel(2層) * 3 FFN = 6 modules → 6 * 3 = 18
    assert n_gate == 18, n_gate
    # gate と base は重複しない
    gate_ids = {id(p) for p in gate_g["params"]}
    base_ids = {id(p) for p in base_g["params"]}
    assert gate_ids.isdisjoint(base_ids)


def test_build_param_groups_separates_energy_gate_for_E():
    m = _injected("equilibrium")  # 条件 E
    cfg = TrainConfig(condition="E", learning_rate=1e-4, gate_lr_multiplier=10.0)
    groups, n_gate = build_param_groups(m, cfg)
    assert len(groups) == 2
    assert abs(groups[1]["lr"] - 1e-3) < 1e-12  # energy 頭は 10x LR
    # energy_cont(w,b) + energy_disc(w,b) = 4 tensors × 6 modules(2層×3) = 24
    assert n_gate == 24, n_gate


def test_gate_statistics_equilibrium():
    m = _injected("equilibrium")
    m(torch.randn(2, 4, 16))
    s = gate_statistics(m)
    assert "qw_mean=" in s and "equilibrium" in s, s


def test_fixed_gate_has_no_gate_group():
    m = _injected("fixed")  # 条件 D 相当: ゲートパラメータ無し
    cfg = TrainConfig(condition="D")
    groups, n_gate = build_param_groups(m, cfg)
    assert n_gate == 0
    assert len(groups) == 1  # base group のみ


def test_build_optimizer_falls_back_to_adamw():
    cfg = TrainConfig(condition="C", optim_8bit=True)
    p = [torch.nn.Parameter(torch.randn(3, 3))]
    opt = build_optimizer(p, cfg)
    # bitsandbytes が無い/CUDA 不整合の環境では AdamW にフォールバックする
    assert isinstance(opt, torch.optim.AdamW)


def test_save_adapter_variational_only_trainable():
    m = _injected("dynamic")
    with tempfile.TemporaryDirectory() as d:
        cfg = TrainConfig(condition="C", output_dir=d)
        out = save_adapter(m, cfg)
        path = os.path.join(out, "variational_lora.pt")
        assert os.path.exists(path)
        sd = torch.load(path)
        # 保存されたのは学習可能パラメータ (VariationalLoRA) のみ
        trainable_names = {n for n, p in m.named_parameters() if p.requires_grad}
        assert set(sd.keys()) == trainable_names
        # base (gate_proj.base.weight 等) は含まれない
        assert not any(".base." in k for k in sd.keys())


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"OK: {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

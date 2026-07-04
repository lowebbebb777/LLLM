"""VariationalLoRA 本体の単体テスト (CPU, GPU 不要)。

SPEC §2 の必須制約を検証する:
  - forward 形状
  - ゼロ初期化 → 学習開始時アダプタ寄与 = 0 (ベースモデルを壊さない)
  - 形状関数ゲート ΣN_I = 1 (分割の単位性)
  - 条件 D: ゲート固定 0.5
  - パラメータ数が解析式 (§2.4) と一致
  - 動的ゲートが入力依存になりうる
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from variational_lora import (  # noqa: E402
    VariationalLoRA,
    VariationalLoRAConfig,
    variational_lora_param_count,
)


def _make(gate_mode="dynamic", d_in=32, d_out=48, r=8, n_quad=3, **kw):
    cfg = VariationalLoRAConfig(
        in_features=d_in, out_features=d_out, r=r, alpha=16, n_quad=n_quad,
        gate_mode=gate_mode, **kw
    )
    return VariationalLoRA(cfg)


def test_forward_shape():
    m = _make()
    x = torch.randn(4, 7, 32)
    y = m(x)
    assert y.shape == (4, 7, 48), y.shape


def test_zero_init_zero_output():
    # cont_B / disc_B が zeros → 出力は厳密に 0 (ベースモデル不変)
    for gm in ("dynamic", "fixed"):
        m = _make(gate_mode=gm)
        x = torch.randn(3, 5, 32)
        y = m(x)
        assert torch.allclose(y, torch.zeros_like(y)), f"{gm}: not zero at init"


def test_gate_partition_of_unity():
    # 形状関数 N = softmax(shape_fn(x)/τ) は ΣN_I = 1
    m = _make(gate_mode="dynamic", n_quad=3)
    x = torch.randn(2, 6, 32)
    logits = m.shape_fn(x) / m.cfg.tau
    N = torch.softmax(logits, dim=-1)
    s = N.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), s


def test_fixed_gate_is_half():
    m = _make(gate_mode="fixed", fixed_gate=0.5)
    x = torch.randn(2, 4, 32)
    qw = m.gate(x)
    assert qw.shape == (2, 4, 1)
    assert torch.allclose(qw, torch.full_like(qw, 0.5))
    # 条件 D はゲート用パラメータを持たない
    assert m.shape_fn is None
    assert m.quad_weights is None


def test_param_count_matches_formula():
    d_in, d_out, r, q = 32, 48, 8, 3
    m = _make(d_in=d_in, d_out=d_out, r=r, n_quad=q)
    analytic = variational_lora_param_count(d_in, d_out, r, q, count_bias=True)
    assert m.num_adapter_parameters() == analytic, (
        m.num_adapter_parameters(), analytic
    )


def test_adapter_contributes_after_perturbation():
    m = _make(gate_mode="dynamic")
    with torch.no_grad():
        m.cont_B.weight.normal_()
        m.disc_B.weight.normal_()
    x = torch.randn(2, 3, 32)
    y = m(x)
    assert not torch.allclose(y, torch.zeros_like(y))


def test_dynamic_gate_is_input_dependent():
    # quad_weights を非一様にすると qw が入力に依存する (動的性の核心)
    m = _make(gate_mode="dynamic", n_quad=3)
    with torch.no_grad():
        m.quad_weights.copy_(torch.tensor([0.0, 0.5, 1.0]))
    x1 = torch.randn(1, 1, 32) * 5
    x2 = torch.randn(1, 1, 32) * 5
    qw1, qw2 = m.gate(x1), m.gate(x2)
    assert not torch.allclose(qw1, qw2), "gate should vary with input"


def test_gradients_flow_to_adapters():
    m = _make(gate_mode="dynamic")
    with torch.no_grad():
        m.cont_B.weight.normal_()
        m.disc_B.weight.normal_()
    x = torch.randn(2, 3, 32, requires_grad=False)
    loss = m(x).pow(2).mean()
    loss.backward()
    assert m.cont_A.weight.grad is not None
    assert m.quad_weights.grad is not None
    assert torch.isfinite(loss)


def test_uniform_quad_weights_make_gate_constant():
    # 説明的テスト: quad_weights が一様だと softmax の ΣN=1 により qw=1/q (入力非依存)
    m = _make(gate_mode="dynamic", n_quad=3)  # init は一様 1/3
    x = torch.randn(5, 4, 32) * 10
    qw = m.gate(x)
    assert torch.allclose(qw, torch.full_like(qw, 1.0 / 3.0), atol=1e-5)


def test_equilibrium_forward_shape_and_zero_init():
    m = _make(gate_mode="equilibrium")
    x = torch.randn(2, 4, 32)
    y = m(x)
    assert y.shape == (2, 4, 48)
    # cont_B / disc_B が zeros → 出力厳密に 0
    assert torch.allclose(y, torch.zeros_like(y))


def test_equilibrium_gate_starts_at_half():
    # energy 頭ゼロ初期化 → E_cont=E_disc=0 → qw=σ(0)=0.5
    m = _make(gate_mode="equilibrium")
    x = torch.randn(2, 4, 32)
    qw = m.gate(x, m.cont_A(x), m.disc_A(x))
    assert qw.shape == (2, 4, 1)
    assert torch.allclose(qw, torch.full_like(qw, 0.5), atol=1e-6)
    assert m.shape_fn is None and m.quad_weights is None
    assert m.energy_cont is not None and m.energy_disc is not None


def test_equilibrium_gate_in_unit_interval_and_input_dependent():
    m = _make(gate_mode="equilibrium")
    with torch.no_grad():  # エネルギー頭を非ゼロに
        m.energy_cont.weight.normal_()
        m.energy_disc.weight.normal_()
        m.energy_cont.bias.normal_()
        m.energy_disc.bias.normal_()
    x1, x2 = torch.randn(1, 1, 32) * 5, torch.randn(1, 1, 32) * 5
    qw1 = m.gate(x1, m.cont_A(x1), m.disc_A(x1))
    qw2 = m.gate(x2, m.cont_A(x2), m.disc_A(x2))
    assert (qw1 > 0).all() and (qw1 < 1).all()  # sigmoid ∈ (0,1)
    assert not torch.allclose(qw1, qw2)  # 入力依存 (平衡が動く)


def test_equilibrium_param_count_matches_formula():
    d_in, d_out, r, q = 32, 48, 8, 3
    m = _make(gate_mode="equilibrium", d_in=d_in, d_out=d_out, r=r, n_quad=q)
    analytic = variational_lora_param_count(d_in, d_out, r, q, gate_mode="equilibrium")
    assert m.num_adapter_parameters() == analytic, (m.num_adapter_parameters(), analytic)


def test_equilibrium_gradients_flow_to_energy_heads():
    m = _make(gate_mode="equilibrium")
    with torch.no_grad():
        m.cont_B.weight.normal_()
        m.disc_B.weight.normal_()
    loss = m(torch.randn(2, 3, 32)).pow(2).mean()
    loss.backward()
    assert m.energy_cont.weight.grad is not None
    assert torch.isfinite(loss)


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"OK: {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

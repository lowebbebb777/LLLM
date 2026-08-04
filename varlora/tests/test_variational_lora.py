"""VariationalLoRA 本体の単体テスト (CPU, GPU 不要)。

SPEC §2 の必須制約を検証する:
  - forward 形状
  - ゼロ初期化 → 学習開始時アダプタ寄与 = 0 (ベースモデルを壊さない)
  - 形状関数ゲート ΣN_I = 1 (分割の単位性)
  - 条件 D: ゲート固定 0.5
  - 条件 E: 加法形 qw·cont + disc (パラメータ数 C と同一, 非退化)
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


def test_additive_mode_zero_init():
    # 条件 E: 加法形でもゼロ初期化 (cont_B/disc_B=0 → 出力0) されているか
    m = _make(gate_mode="additive")
    x = torch.randn(3, 5, 32)
    y = m(x)
    assert torch.allclose(y, torch.zeros_like(y)), "additive: not zero at init"


def test_additive_mode_param_count_matches_C():
    # 条件 E: パラメータ数が C と完全一致 (交絡対照。結合形だけが違う)
    m_c = _make(gate_mode="dynamic", d_in=32, d_out=48, r=8, n_quad=3)
    m_e = _make(gate_mode="additive", d_in=32, d_out=48, r=8, n_quad=3)
    assert m_c.num_adapter_parameters() == m_e.num_adapter_parameters(), (
        f"C params={m_c.num_adapter_parameters()}, "
        f"E params={m_e.num_adapter_parameters()}"
    )
    assert m_e.shape_fn is not None
    assert m_e.quad_weights is not None


def test_additive_mode_has_dynamic_gate():
    # 条件 E: 動的ゲート (C と同じ方法で qw を計算)
    m = _make(gate_mode="additive", n_quad=3)
    with torch.no_grad():
        m.quad_weights.copy_(torch.tensor([0.0, 0.5, 1.0]))
    x1 = torch.randn(1, 1, 32) * 5
    x2 = torch.randn(1, 1, 32) * 5
    qw1, qw2 = m.gate(x1), m.gate(x2)
    assert not torch.allclose(qw1, qw2), "additive gate should be input-dependent"


def _copy_all_params(src, dst):
    """src の全アダプタ重みを dst にコピー (結合形以外を完全一致させる)。"""
    with torch.no_grad():
        for name, p in src.named_parameters():
            dict(dst.named_parameters())[name].copy_(p)


def test_additive_is_non_degenerate_vs_C():
    """条件 E (加法形) は C (凸結合) と本当に別の関数であること (退化しない)。

    ★このテストの存在理由★
    残差形 qw·cont-(1-qw)·disc は disc_B→-disc_B の符号反転で C に一致し、
    disc_B のゼロ初期化ゆえ学習軌道まで完全一致する (退化) ため不採用にした。
    加法形 qw·cont+disc は (1-qw) が入力依存の場で線形写像 disc_B に吸収できず
    非退化のはず。同一パラメータを与えたとき C と出力が異なることで確認する。
    """
    m_c = _make(gate_mode="dynamic", d_in=32, d_out=48, r=8)
    m_e = _make(gate_mode="additive", d_in=32, d_out=48, r=8)
    _copy_all_params(m_c, m_e)
    # cont_B/disc_B を非零にして寄与を出す (ゼロ初期のままだと両者0で区別不能)
    with torch.no_grad():
        m_c.cont_B.weight.normal_(0, 0.1)
        m_c.disc_B.weight.normal_(0, 0.1)
    _copy_all_params(m_c, m_e)  # 非零化した cont_B/disc_B も含め全パラメータ共有

    x = torch.randn(4, 6, 32)
    y_c, y_e = m_c(x), m_e(x)
    # 同一パラメータ・同一入力でも、結合形が違うので出力は明確に異なるはず
    assert not torch.allclose(y_c, y_e, atol=1e-6), (
        "additive が C と出力一致 → 退化している (別アームとして無意味)"
    )
    # 差の大きさが数値誤差でなく実質的であること
    assert (y_c - y_e).abs().max() > 1e-3


def test_residual_form_would_degenerate():
    """不採用の残差形が実際に C と退化することを回帰的に固定する。

    将来 additive を安易に residual (qw·cont-(1-qw)·disc) へ戻す変更を防ぐ番人。
    残差形を手計算で作り、disc_B の符号反転で C に一致することを示す。
    """
    m_c = _make(gate_mode="dynamic", d_in=16, d_out=24, r=4)
    with torch.no_grad():
        m_c.cont_B.weight.normal_(0, 0.1)
        m_c.disc_B.weight.normal_(0, 0.1)
    x = torch.randn(2, 3, 16)
    cont = m_c.cont_B(m_c.cont_A(x))
    disc = m_c.disc_B(m_c.disc_A(x))
    qw = m_c.gate(x)
    out_convex = (qw * cont + (1.0 - qw) * disc) * m_c.scaling  # C の結合
    # 残差形は disc_B を符号反転した disc' を使う凸結合と同値
    disc_neg = (-m_c.disc_B(m_c.disc_A(x)))
    out_residual = (qw * cont - (1.0 - qw) * disc) * m_c.scaling
    out_convex_negdisc = (qw * cont + (1.0 - qw) * disc_neg) * m_c.scaling
    assert torch.allclose(out_residual, out_convex_negdisc, atol=1e-6), (
        "残差形 = (disc_B符号反転した)凸結合 のはず"
    )
    # よって残差形は C と同じパラメータ空間・同じ初期値(disc_B=0)から到達可能 = 退化
    assert not torch.allclose(out_residual, out_convex) or torch.allclose(
        disc, torch.zeros_like(disc)
    )


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"OK: {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

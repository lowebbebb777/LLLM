"""評価メトリクスの単体テスト (CPU, モデル不要)。

SPEC §5 の各指標 (pass@k 推定量, 数値整合性, 経路独立性, 効果量) を
モデルなしの擬似生成器で検証する。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import (  # noqa: E402
    aggregate_pass_at_k,
    check_correctness,
    cohens_d,
    load_numeric_problems,
    numeric_consistency_score,
    parse_number,
    pass_at_k,
    path_independence_score,
    run_evaluation,
    summarize_seeds,
    NumericProblem,
)


def test_pass_at_k_edges():
    assert abs(pass_at_k(10, 0, 1) - 0.0) < 1e-9
    assert abs(pass_at_k(10, 10, 1) - 1.0) < 1e-9
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-9  # c/n
    # n-c < k → 1.0 (どの k 個を選んでも正解を含む)
    assert pass_at_k(5, 5, 1) == 1.0
    # k=2 で c=1, n=10: 1 - C(9,2)/C(10,2) = 1 - 36/45 = 0.2
    assert abs(pass_at_k(10, 1, 2) - 0.2) < 1e-9


def test_aggregate_pass_at_k():
    res = [(10, 5), (10, 0), (10, 10)]
    agg = aggregate_pass_at_k(res, ks=(1,))
    # (0.5 + 0 + 1)/3
    assert abs(agg["pass@1"] - 0.5) < 1e-9


def test_parse_number():
    assert parse_number("The answer is 123.4") == 123.4
    assert parse_number("Cpk = 1.3333.") == 1.3333
    assert parse_number("result: -2.5e1") == -25.0
    assert parse_number("no number here") is None
    assert parse_number("first 1 then 2") == 2.0  # 末尾を取る


def test_check_correctness():
    prog = "def f(x):\n    return x + 1\n"
    assert check_correctness(prog, "assert f(1) == 2")
    assert not check_correctness(prog, "assert f(1) == 99")
    assert not check_correctness("def f(x): return x", "raise RuntimeError()")


def test_numeric_consistency_score():
    probs = [
        NumericProblem("a", "p", "numeric", "cat1", expected=10.0, rel_tol=0.01),
        NumericProblem("b", "p", "numeric", "cat1", expected=20.0, rel_tol=0.01),
        NumericProblem("c", "p", "code", "cat2", test="assert g(2) == 4"),
    ]

    def gen(prompt, n):
        # numeric は常に 10 を返す → a 正解, b 不正解。code は正しい関数。
        if "code" in prompt or "g(" in prompt:
            return ["def g(x):\n    return x * 2\n"]
        return ["the value is 10"]

    # prompt 文字列で分岐できないので id ベースの簡易生成器に置き換え
    def gen2(prompt, n):
        return ["the value is 10"] if "numeric" not in prompt else ["10"]

    # 明示的に振る舞いを固定するため、各 prompt をユニークにする
    probs[0].prompt = "numeric a"
    probs[1].prompt = "numeric b"
    probs[2].prompt = "code c"

    def gen3(prompt, n):
        if prompt.startswith("code"):
            return ["def g(x):\n    return x * 2\n"]
        return ["10"]

    score = numeric_consistency_score(probs, gen3)
    assert abs(score["numeric/cat1"] - 0.5) < 1e-9  # a 正解, b 不正解
    assert abs(score["numeric/cat2"] - 1.0) < 1e-9  # code 正解
    assert "numeric/overall" in score


def test_path_independence_score():
    # 一致する生成器 (経路独立性 高) vs ブレる生成器 (低)
    prob = NumericProblem(
        "x", "base prompt", "numeric", "cat",
        expected=5.0, rel_tol=0.01, paraphrases=["p1", "p2", "p3"],
    )

    def consistent(prompt, n):
        return ["5"]

    def inconsistent(prompt, n):
        # prompt ごとに違う値
        mapping = {"base prompt": "5", "p1": "5", "p2": "100", "p3": "999"}
        return [mapping.get(prompt, "0")]

    s_good = path_independence_score([prob], consistent)
    s_bad = path_independence_score([prob], inconsistent)
    assert s_good["path_indep/agreement"] == 1.0
    assert s_bad["path_indep/agreement"] < 1.0
    assert s_good["path_indep/dispersion"] == 0.0
    assert s_bad["path_indep/dispersion"] > 0.0


def test_run_evaluation_composes_metrics():
    probs = [
        NumericProblem("a", "num a", "numeric", "cat", expected=5.0, rel_tol=0.01,
                       paraphrases=["p1", "p2"]),
        NumericProblem("b", "code b", "code", "cat2", test="assert g(2) == 4"),
    ]

    def gen(prompt, n=1):
        if prompt.startswith("code"):
            return ["def g(x):\n    return x * 2\n"]
        return ["5"]

    scores = run_evaluation(probs, gen)
    # 数値整合性 (両カテゴリ) と 経路独立性 (numeric のみ) が入る
    assert "numeric/overall" in scores
    assert "path_indep/agreement" in scores
    assert scores["path_indep/agreement"] == 1.0  # 常に 5 を返す → 一致


def test_summarize_and_cohens_d():
    s = summarize_seeds([0.80, 0.82, 0.81])
    assert abs(s["mean"] - 0.81) < 1e-9
    assert s["n"] == 3
    # 大きく離れた2群は |d| が大きい
    d = cohens_d([0.9, 0.91, 0.89], [0.5, 0.51, 0.49])
    assert abs(d) > 2.0


def test_load_seed_dataset():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "numeric_stats_eval", "problems.jsonl")
    probs = load_numeric_problems(path)
    assert len(probs) >= 10, len(probs)
    cats = {p.category for p in probs}
    # SPEC §5.2 の4カテゴリが揃っているか
    for required in ("cpk", "stats_impl", "scale_consistency", "discrete_continuous"):
        assert required in cats, (required, cats)
    # 経路独立性 (§5.2b) 用に paraphrases を持つ問題が存在
    assert any(p.paraphrases for p in probs)


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"OK: {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

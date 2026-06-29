"""評価 — HumanEval/MBPP / 自作数値統計セット / 経路独立性 (SPEC §5)。

検証基準 (SPEC §5):
    1. コード生成ベンチ: HumanEval (pass@1, pass@10), MBPP
    2. 自作評価セット (数値・統計の内部表現): numeric-consistency score
    3. 経路独立性スコア (J積分の経路独立性アナロジー, §5.2b): 言い換え耐性

判定の作法 (SPEC §5.3):
    - 各条件×3seed を平均±標準偏差で報告
    - 条件間の差が標準偏差の範囲内ならノイズ
    - 効果量で語る (1%上下で結論しない)

モデル生成そのもの (重い) はインターフェイス `Generator` として切り出し、
メトリクス計算 (pass@k 推定量・整合性・経路独立性) は GPU/モデルなしで単体テスト可能。
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# 生成関数の型: (prompt, n_samples) -> List[str] (n 個の補完)
Generator = Callable[[str, int], List[str]]


# ---------------------------------------------------------------------------
# pass@k 推定量 (Chen et al. 2021, HumanEval 論文の不偏推定量)
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """n サンプル中 c 個正解のとき pass@k の不偏推定値。

    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def aggregate_pass_at_k(
    results: Sequence[tuple[int, int]], ks: Sequence[int] = (1, 10)
) -> Dict[str, float]:
    """各問題の (n_samples, n_correct) 列から pass@k を平均する。"""
    out: Dict[str, float] = {}
    for k in ks:
        vals = [pass_at_k(n, c, k) for (n, c) in results if n >= k]
        out[f"pass@{k}"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# コード実行による正誤判定 (HumanEval/MBPP 共通)
# ---------------------------------------------------------------------------
def check_correctness(
    program: str, test_code: str, *, timeout: float = 10.0
) -> bool:
    """program + test_code をサブプロセスで実行し、例外なく通れば True。

    注意: 任意コード実行になるため、本番では必ずサンドボックス内で実行すること。
    HumanEval 公式の `execution.py` に倣い、ここでは最小実装を提供する。
    """
    import multiprocessing as mp

    def _target(q):
        try:
            ns: Dict[str, object] = {}
            exec(program + "\n" + test_code, ns)
            q.put(True)
        except BaseException:
            q.put(False)

    q: "mp.Queue[bool]" = mp.Queue()
    p = mp.Process(target=_target, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    return not q.empty() and q.get()


def evaluate_humaneval(
    problems: Sequence[Dict],
    generate: Generator,
    *,
    n_samples: int = 10,
    ks: Sequence[int] = (1, 10),
    extract_code: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """HumanEval / MBPP 形式の問題群を評価する。

    problems の各要素: {"prompt": str, "test": str, "entry_point": str (任意)}
    generate(prompt, n_samples) -> n 個の補完。
    """
    extract = extract_code or extract_python_code
    results: List[tuple[int, int]] = []
    for prob in problems:
        completions = generate(prob["prompt"], n_samples)
        correct = 0
        for comp in completions:
            program = prob.get("prompt", "") + extract(comp)
            if check_correctness(program, prob["test"]):
                correct += 1
        results.append((len(completions), correct))
    return aggregate_pass_at_k(results, ks)


def extract_python_code(text: str) -> str:
    """生成テキストから python コードを抽出 (```python ... ``` or そのまま)。"""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


# ---------------------------------------------------------------------------
# 自作評価セット: 数値整合性スコア (SPEC §5.2)
# ---------------------------------------------------------------------------
@dataclass
class NumericProblem:
    """数値・統計の内部表現を測る問題。

    kind:
        "code"    → コード生成 (test で検証, pass/fail)
        "numeric" → 数値解答 (expected と相対誤差で照合 → 数値整合性スコア)
    """

    id: str
    prompt: str
    kind: str
    category: str  # cpk | stats_impl | scale_consistency | discrete_continuous ...
    test: Optional[str] = None  # kind=="code"
    expected: Optional[float] = None  # kind=="numeric"
    rel_tol: float = 1e-3
    paraphrases: List[str] = field(default_factory=list)  # 経路独立性 (§5.2b) 用


def parse_number(text: str) -> Optional[float]:
    """生成テキスト末尾付近から数値を 1 つ取り出す (桁・スケールの一貫性を測る)。"""
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def numeric_consistency_score(
    problems: Sequence[NumericProblem],
    generate: Generator,
    *,
    n_samples: int = 1,
) -> Dict[str, float]:
    """数値整合性スコア: pass@1 とは別軸で「数として正しく扱えているか」を測る。

    - kind=="code": test を実行して pass 率
    - kind=="numeric": 生成数値が expected と相対誤差 rel_tol 以内なら正
    カテゴリ別と全体のスコアを返す。
    """
    per_category: Dict[str, List[float]] = {}
    for prob in problems:
        comp = generate(prob.prompt, n_samples)[0]
        if prob.kind == "code":
            program = extract_python_code(comp)
            ok = check_correctness(program, prob.test or "")
        elif prob.kind == "numeric":
            val = parse_number(comp)
            ok = (
                val is not None
                and prob.expected is not None
                and _rel_close(val, prob.expected, prob.rel_tol)
            )
        else:
            raise ValueError(f"unknown kind: {prob.kind}")
        per_category.setdefault(prob.category, []).append(1.0 if ok else 0.0)

    out = {f"numeric/{cat}": float(np.mean(v)) for cat, v in per_category.items()}
    allv = [x for v in per_category.values() for x in v]
    out["numeric/overall"] = float(np.mean(allv)) if allv else float("nan")
    return out


def _rel_close(a: float, b: float, rel_tol: float) -> bool:
    if b == 0:
        return abs(a) <= rel_tol
    return abs(a - b) / abs(b) <= rel_tol


# ---------------------------------------------------------------------------
# 経路独立性スコア (J積分の経路独立性アナロジー, SPEC §5.2b)
# ---------------------------------------------------------------------------
def path_independence_score(
    problems: Sequence[NumericProblem],
    generate: Generator,
    *,
    answer_extractor: Optional[Callable[[str], Optional[float]]] = None,
) -> Dict[str, float]:
    """同じ問題を異なる「経路」(言い換え/順序変更) で解かせ、答えの一致度を測る。

    FEM の J積分は積分経路 Γ1,Γ2,... によらず一致すべき (経路独立)。LLM も内部表現が
    頑健なら言い換えに対し答えがブレないはず (§5.2b の反証可能な予測)。

    各問題について prompt + paraphrases の全経路で生成し:
        - agreement: 数値答えが互いに一致する割合
        - dispersion: 答えの (正規化) 標準偏差の小ささ
    を集計する。値が高い (agreement) / 低い (dispersion) ほど経路独立性が高い。
    """
    extract = answer_extractor or parse_number
    agreements: List[float] = []
    dispersions: List[float] = []

    for prob in problems:
        paths = [prob.prompt, *prob.paraphrases]
        if len(paths) < 2:
            continue
        answers: List[Optional[float]] = [extract(generate(p, 1)[0]) for p in paths]
        valid = [a for a in answers if a is not None]
        if len(valid) < 2:
            agreements.append(0.0)
            continue
        # 一致割合: 最頻値 (相対誤差 tol 以内) を共有する経路の割合
        agreements.append(_modal_agreement(valid, prob.rel_tol))
        # 分散の小ささ: 平均で正規化した標準偏差 (CV)
        mean = statistics.fmean(valid)
        if mean != 0:
            dispersions.append(statistics.pstdev(valid) / abs(mean))

    return {
        "path_indep/agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "path_indep/dispersion": float(np.mean(dispersions)) if dispersions else float("nan"),
        "path_indep/n_problems": float(len(agreements)),
    }


def _modal_agreement(values: Sequence[float], rel_tol: float) -> float:
    """最大のクラスタ (相対誤差 rel_tol 以内で一致する値の集合) のサイズ比率。"""
    best = 0
    for anchor in values:
        cluster = sum(1 for v in values if _rel_close(v, anchor, rel_tol))
        best = max(best, cluster)
    return best / len(values)


# ---------------------------------------------------------------------------
# 統計集計 (SPEC §5.3: 平均±標準偏差, 効果量)
# ---------------------------------------------------------------------------
def summarize_seeds(per_seed: Sequence[float]) -> Dict[str, float]:
    """3 seed 等の結果列を平均±標準偏差で集計する。"""
    arr = list(per_seed)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": float(len(arr)),
    }


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """効果量 Cohen's d。条件間の差をノイズと区別するため (SPEC §5.3)。"""
    a, b = np.asarray(group_a, float), np.asarray(group_b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = math.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------
def load_numeric_problems(path: str) -> List[NumericProblem]:
    problems: List[NumericProblem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            problems.append(
                NumericProblem(
                    id=d["id"],
                    prompt=d["prompt"],
                    kind=d["kind"],
                    category=d["category"],
                    test=d.get("test"),
                    expected=d.get("expected"),
                    rel_tol=d.get("rel_tol", 1e-3),
                    paraphrases=d.get("paraphrases", []),
                )
            )
    return problems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="評価メトリクスの自己テスト")
    parser.add_argument("--numeric-set", default="data/numeric_stats_eval/problems.jsonl")
    args = parser.parse_args()
    probs = load_numeric_problems(args.numeric_set)
    print(f"loaded {len(probs)} numeric problems")
    cats: Dict[str, int] = {}
    for p in probs:
        cats[p.category] = cats.get(p.category, 0) + 1
    print("categories:", json.dumps(cats, indent=2, ensure_ascii=False))

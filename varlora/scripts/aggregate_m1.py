"""M1 の条件別評価 JSON を集計し、SPEC §4.3 の判定材料を表示する。

各条件の eval.json (evaluate.py --out が出力) を読み、指標ごとに A/B/C/D を
並べて表示。1 seed のときは素の値、複数 seed のときは平均±標準偏差 (SPEC §5.3)。

    python3 scripts/aggregate_m1.py outputs/cond_A/eval.json outputs/cond_B/eval.json ...
    python3 scripts/aggregate_m1.py --glob 'outputs/cond_*_seed*/eval.json'
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import statistics
from collections import defaultdict
from typing import Dict, List


def load(paths: List[str]):
    # condition -> metric -> [values across seeds]
    data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        cond = d["condition"]
        for k, v in d["scores"].items():
            if isinstance(v, (int, float)):
                data[cond][k].append(float(v))
    return data


def fmt(vals: List[float]) -> str:
    if not vals:
        return "   -   "
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.fmean(vals):.3f}±{statistics.pstdev(vals):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="eval.json のパス群")
    parser.add_argument("--glob", default=None, help="glob パターンで eval.json を収集")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.glob:
        paths += globmod.glob(args.glob)
    if not paths:
        parser.error("eval.json を 1 つ以上指定してください")

    data = load(paths)
    conds = [c for c in ["A", "B", "C", "D"] if c in data]
    metrics = sorted({m for c in data for m in data[c]})

    # 表示
    w = 22
    header = f"{'metric':32}" + "".join(f"{c:>{w}}" for c in conds)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:32}" + "".join(f"{fmt(data[c].get(m, [])):>{w}}" for c in conds)
        print(row)

    # SPEC §4.3 判定: 仮説の中核指標について C vs {A,B,D} を平均差 + 効果量で見る
    print("\n=== SPEC §4.3 判定 (効果量 Cohen's d 込み) ===")
    d_fn = _load_cohens_d()
    verdict_keys = [k for k in ("numeric/overall", "path_indep/agreement") if any(k in data[c] for c in conds)]
    for key in verdict_keys:
        _report_key(data, key, d_fn)

    print("\n判定ルール (SPEC §4.3):")
    print("  C>B かつ C>D (かつ効果量が無視できない) → 『変分原理アナロジーに効果あり』")
    print("  C>A だが C≈B → 効果はパラメータ増加由来 / C≈D → ゲート動的性は不要")
    print("  差が標準偏差の範囲内・|d|が小さいならノイズ (SPEC §5.3)")


def _report_key(data, key, d_fn) -> None:
    def vals(c):
        return data.get(c, {}).get(key, [])

    def mean(c):
        v = vals(c)
        return statistics.fmean(v) if v else float("nan")

    present = [c for c in ("A", "B", "C", "D") if vals(c)]
    line = "  ".join(f"{c}={mean(c):.3f}" for c in present)
    print(f"\n[{key}]  {line}")
    if not vals("C"):
        print("  C が無いため判定不可")
        return
    for other in ("B", "D", "A"):
        if not vals(other):
            continue
        gt = mean("C") > mean(other)
        d = d_fn(vals("C"), vals(other)) if len(vals("C")) > 1 and len(vals(other)) > 1 else float("nan")
        d_str = f"d={d:+.2f}" if d == d else "d=n/a(1seed)"
        print(f"  C vs {other}: C>{other}={gt}  Δ={mean('C') - mean(other):+.3f}  {d_str}")


def _load_cohens_d():
    """evaluate.cohens_d を借りる (src を path に追加)。無ければ簡易実装。"""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    try:
        from evaluate import cohens_d
        return cohens_d
    except Exception:
        import math

        def cohens_d(a, b):
            if len(a) < 2 or len(b) < 2:
                return float("nan")
            va, vb = statistics.variance(a), statistics.variance(b)
            pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
            return (statistics.fmean(a) - statistics.fmean(b)) / pooled if pooled else float("nan")

        return cohens_d


if __name__ == "__main__":
    main()

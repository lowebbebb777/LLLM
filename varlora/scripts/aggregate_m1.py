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

    # SPEC §4.3 の判定ヒント (overall 系指標で C vs B, C vs D を機械的にチェック)
    print("\n=== SPEC §4.3 判定ヒント (overall 指標) ===")
    key = "numeric/overall"
    if all(key in data[c] for c in ("B", "C", "D") if c in data):
        def mean(c):
            return statistics.fmean(data[c][key]) if data.get(c, {}).get(key) else float("nan")
        c, b, dd = mean("C"), mean("B"), mean("D")
        print(f"{key}:  A={mean('A'):.3f}  B={b:.3f}  C={c:.3f}  D={dd:.3f}")
        print(f"  C>B: {c > b}  C>D: {c > dd}  → 両方 True なら『変分原理アナロジーに効果あり』の候補")
        print("  ※ 1 seed では確定不可。3 seed で平均±標準偏差・効果量 (evaluate.cohens_d) を見ること")
    else:
        print("numeric/overall が全条件に揃っていません")


if __name__ == "__main__":
    main()

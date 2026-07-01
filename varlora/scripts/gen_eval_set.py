"""自作評価セットを拡張生成する (SPEC §5.2: 50〜100問)。

答えは Python で計算して検証済みの値を埋め込む (手計算ミスを排除)。numeric 問題には
経路独立性 (§5.2b) 用の言い換え paraphrases を付ける。4 カテゴリを均等に生成。

    python3 scripts/gen_eval_set.py --n-per-cat 14 --out data/numeric_stats_eval/problems.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List


def num(id_, cat, prompt, expected, paraphrases, rel_tol=0.01) -> Dict:
    return {
        "id": id_, "category": cat, "kind": "numeric", "prompt": prompt,
        "expected": round(float(expected), 6), "rel_tol": rel_tol, "paraphrases": paraphrases,
    }


def code(id_, cat, prompt, test) -> Dict:
    return {"id": id_, "category": cat, "kind": "code", "prompt": prompt, "test": test}


def gen_scale(rng, i) -> Dict:
    kind = rng.choice(["mul", "round", "div", "frac", "pct"])
    if kind == "mul":
        f = round(rng.uniform(1.0, 99.0), 2)
        return num(f"scale_{i}", "scale_consistency", f"What is {f} multiplied by 10?",
                   f * 10, [f"Compute {f} * 10. Number only.", f"Ten times {f} equals what?"])
    if kind == "round":
        f, d = round(rng.uniform(1.0, 9.0), 4), rng.randint(1, 3)
        return num(f"scale_{i}", "scale_consistency", f"Round {f} to {d} decimal places.",
                   round(f, d), [f"Express {f} with {d} digits after the decimal point.",
                                 f"{f} rounded to {d} dp is?"])
    if kind == "div":
        f = round(rng.uniform(10.0, 99.0), 2)
        return num(f"scale_{i}", "scale_consistency", f"What is {f} divided by 10?",
                   f / 10, [f"Compute {f} / 10.", f"One tenth of {f} is?"])
    if kind == "frac":
        b = rng.randint(2, 16)
        return num(f"scale_{i}", "scale_consistency", f"Express 1/{b} as a decimal to 4 dp.",
                   round(1 / b, 4), [f"What is one over {b} as a decimal (4 dp)?",
                                     f"Divide 1 by {b}, 4 decimals."], rel_tol=0.02)
    p, n = rng.choice([5, 10, 20, 25, 50]), rng.randint(20, 200)
    return num(f"scale_{i}", "scale_consistency", f"What is {p}% of {n}?",
               n * p / 100, [f"Compute {p} percent of {n}.", f"{p}% times {n} equals?"])


def gen_cpk(rng, i) -> Dict:
    usl, lsl = rng.randint(10, 22), rng.randint(1, 5)
    mu, s = rng.randint(lsl + 2, usl - 2), rng.choice([1, 2, 3])
    if rng.random() < 0.5:
        cpk = min((usl - mu) / (3 * s), (mu - lsl) / (3 * s))
        return num(f"cpk_{i}", "cpk",
                   f"A process has USL={usl}, LSL={lsl}, mean={mu}, sigma={s}. Compute Cpk.",
                   cpk, [f"With upper limit {usl}, lower limit {lsl}, mean {mu}, std {s}, what is Cpk?",
                         f"Cpk for mu={mu}, sigma={s}, spec [{lsl},{usl}]?"])
    cp = (usl - lsl) / (6 * s)
    return num(f"cpk_{i}", "cpk",
               f"A process has USL={usl}, LSL={lsl}, sigma={s}. Compute Cp.",
               cp, [f"Process potential Cp for limits [{lsl},{usl}], sigma {s}?",
                    f"Cp = (USL-LSL)/(6 sigma) with USL={usl}, LSL={lsl}, sigma={s}?"])


def gen_stats(rng, i) -> Dict:
    xs = [rng.randint(1, 40) for _ in range(rng.randint(4, 7))]
    kind = rng.choice(["mean_code", "var_code", "std_num", "zscore_code", "ols_code"])
    if kind == "mean_code":
        return code(f"stats_{i}", "stats_impl",
                    "Write a Python function `mean(values)` returning the arithmetic mean.",
                    f"assert abs(mean({xs}) - {sum(xs)/len(xs)!r}) < 1e-6")
    if kind == "var_code":
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return code(f"stats_{i}", "stats_impl",
                    "Write a Python function `variance(xs)` returning the population variance.",
                    f"assert abs(variance({xs}) - {v!r}) < 1e-6")
    if kind == "std_num":
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return num(f"stats_{i}", "stats_impl",
                   f"Compute the population standard deviation of {xs}.",
                   sd, [f"What is sigma (population) of the data {xs}?",
                        f"Population std of {xs}?"], rel_tol=0.02)
    if kind == "zscore_code":
        return code(f"stats_{i}", "stats_impl",
                    "Write `zscore(xs)` returning population z-scores (mean 0, pop std 1).",
                    "r = zscore([2,4,4,4,5,5,7,9])\nassert abs(sum(r)) < 1e-9\nassert abs(r[0] + 1.5) < 1e-6")
    a = rng.randint(1, 5)
    ys = [a * x for x in xs]
    return code(f"stats_{i}", "stats_impl",
                "Write `ols_slope(xs, ys)` returning the least-squares slope of ys on xs.",
                f"assert abs(ols_slope({xs}, {ys}) - {a}) < 1e-6")


def gen_disc(rng, i) -> Dict:
    vals = [rng.randint(1, 20) for _ in range(rng.randint(4, 6))]
    flags = [rng.random() < 0.5 for _ in vals]
    if rng.random() < 0.5:
        total = sum(v for v, f in zip(vals, flags) if f)
        fl = [int(f) for f in flags]
        return num(f"disc_{i}", "discrete_continuous",
                   f"Given values {vals} and flags {fl} (1=keep), sum the kept values.",
                   total, [f"Sum entries of {vals} where mask {fl} is 1.",
                           f"Add {vals} items selected by {fl}."], rel_tol=0.001)
    return code(f"disc_{i}", "discrete_continuous",
                "Write `mean_of_flagged(values, flags)`: mean of values where flag is True, else 0.0.",
                "assert abs(mean_of_flagged([1,2,3,4],[True,False,True,False]) - 2.0) < 1e-9\n"
                "assert mean_of_flagged([1,2],[False,False]) == 0.0")


def generate(n_per_cat: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    gens = {
        "scale_consistency": gen_scale, "cpk": gen_cpk,
        "stats_impl": gen_stats, "discrete_continuous": gen_disc,
    }
    rows: List[Dict] = []
    for cat, fn in gens.items():
        seen = set()
        i = 0
        while sum(1 for r in rows if r["category"] == cat) < n_per_cat and i < n_per_cat * 40:
            i += 1
            r = fn(rng, len(rows))
            key = r["prompt"]
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-cat", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "numeric_stats_eval", "problems.jsonl"))
    args = parser.parse_args()

    rows = generate(args.n_per_cat, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cats: Dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"wrote {len(rows)} problems → {args.out}")
    print("per category:", json.dumps(cats, ensure_ascii=False))


if __name__ == "__main__":
    main()

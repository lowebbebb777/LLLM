"""M0 手応え確認用の小データを生成する (SPEC §7-M0: 100〜500件で1エポック)。

コード生成 + 数値・統計テーマ (本プロジェクトの狙い) に沿った短いスニペットを
テンプレートから多様に生成する。学習が回る/lossが下がる/NaNが出ない/ゲートが
0-1に張り付かない、を観察するための M0 専用データ。本評価データではない。

    python3 scripts/gen_m0_data.py            # → data/m0_smoke.jsonl (約500件)
    python3 scripts/gen_m0_data.py --n 300 --out data/m0_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random

VAR_NAMES = ["a", "b", "x", "y", "n", "val", "total", "acc", "data", "xs"]


def code_samples(rng: random.Random) -> list[str]:
    a, b = rng.randint(1, 99), rng.randint(1, 99)
    v1, v2 = rng.sample(VAR_NAMES, 2)
    nums = [rng.randint(1, 50) for _ in range(rng.randint(3, 6))]
    usl, lsl = rng.randint(8, 20), rng.randint(1, 5)
    mean, sigma = rng.randint(lsl + 1, usl - 1), rng.choice([1, 2, 3])
    return [
        f"def add_{a}_{b}({v1}, {v2}):\n    \"\"\"Return the sum.\"\"\"\n    return {v1} + {v2}\n",
        f"def scale({v1}):\n    return {v1} * {a}\n",
        f"def mean(values):\n    return sum(values) / len(values)\n# mean({nums}) == {sum(nums)/len(nums):.4f}\n",
        f"def cpk(usl, lsl, mu, sigma):\n    return min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))\n"
        f"# cpk({usl}, {lsl}, {mean}, {sigma}) -> capability index\n",
        f"def variance(xs):\n    m = sum(xs) / len(xs)\n    return sum((v - m) ** 2 for v in xs) / len(xs)\n",
        f"def clamp({v1}, lo={lsl}, hi={usl}):\n    return max(lo, min(hi, {v1}))\n",
        f"def count_flagged(values, flags):\n    return sum(1 for v, f in zip(values, flags) if f)\n",
        f"def zscore(xs):\n    m = sum(xs) / len(xs)\n    sd = (sum((v - m) ** 2 for v in xs) / len(xs)) ** 0.5\n"
        f"    return [(v - m) / sd for v in xs]\n",
    ]


def qa_samples(rng: random.Random) -> list[str]:
    a, b = rng.randint(2, 40), rng.randint(2, 40)
    f = round(rng.uniform(0.1, 9.9), 2)
    return [
        f"Q: What is {a} + {b}?\nA: {a + b}\n",
        f"Q: What is {a} multiplied by {b}?\nA: {a * b}\n",
        f"Q: Round {f} to 1 decimal place.\nA: {round(f, 1)}\n",
        f"Q: What is the mean of {a} and {b}?\nA: {(a + b) / 2}\n",
        f"Q: Is {a} greater than {b}?\nA: {'yes' if a > b else 'no'}\n",
    ]


def generate(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    pool: list[str] = []
    while len(pool) < n:
        pool.extend(code_samples(rng))
        pool.extend(qa_samples(rng))
    rng.shuffle(pool)
    return [{"text": t} for t in pool[:n]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "m0_smoke.jsonl"
    )
    rows = generate(args.n, args.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples → {out}")


if __name__ == "__main__":
    main()

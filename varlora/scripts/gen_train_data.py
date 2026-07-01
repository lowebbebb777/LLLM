"""M1 用の合成学習データ生成 (拡張版, SPEC §4 / ユーザー選択: 合成データ拡張)。

コード生成 + 数値・統計テーマの多様なスニペットをテンプレートから大量生成する。
全条件 (A/B/C/D) がこの同一データで学習する (SPEC §4.4: データセットは全条件で固定)。

★リーク対策★: 評価セット (data/numeric_stats_eval/problems.jsonl) の prompt と
完全一致/包含する生成文は除外する。全条件が同一データで学習するため相対比較は
リークに頑健だが、絶対スコアの汚染も避けるため念のため除去する。

    python3 scripts/gen_train_data.py --n 4000 --out data/m1_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Callable, List

VARS = ["a", "b", "x", "y", "n", "val", "total", "acc", "data", "xs", "arr", "nums"]


def _rints(rng, k, lo=1, hi=50):
    return [rng.randint(lo, hi) for _ in range(k)]


# --- コードテンプレート ------------------------------------------------------
def code_templates(rng: random.Random) -> List[str]:
    v1, v2 = rng.sample(VARS, 2)
    a, b = rng.randint(1, 99), rng.randint(1, 99)
    nums = _rints(rng, rng.randint(3, 6))
    usl, lsl = rng.randint(8, 20), rng.randint(1, 5)
    mu, sigma = rng.randint(lsl + 1, usl - 1), rng.choice([1, 2, 3])
    mult = rng.randint(2, 99)
    return [
        f"def add({v1}, {v2}):\n    \"\"\"Return the sum.\"\"\"\n    return {v1} + {v2}\n",
        f"def sub({v1}, {v2}):\n    return {v1} - {v2}\n",
        f"def scale({v1}, k={mult}):\n    return {v1} * k\n",
        f"def mean(values):\n    return sum(values) / len(values)\n",
        f"def variance(xs):\n    m = sum(xs) / len(xs)\n    return sum((v - m) ** 2 for v in xs) / len(xs)\n",
        f"def stddev(xs):\n    m = sum(xs) / len(xs)\n    return (sum((v - m) ** 2 for v in xs) / len(xs)) ** 0.5\n",
        f"def zscore(xs):\n    m = sum(xs) / len(xs)\n    sd = (sum((v - m) ** 2 for v in xs) / len(xs)) ** 0.5\n    return [(v - m) / sd for v in xs]\n",
        f"def cpk(usl, lsl, mu, sigma):\n    return min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))\n",
        f"def cp(usl, lsl, sigma):\n    return (usl - lsl) / (6 * sigma)\n",
        f"def normalize(xs):\n    lo, hi = min(xs), max(xs)\n    return [(v - lo) / (hi - lo) for v in xs] if hi != lo else [0.0 for _ in xs]\n",
        f"def clamp(v, lo={lsl}, hi={usl}):\n    return max(lo, min(hi, v))\n",
        f"def count_flagged(values, flags):\n    return sum(1 for v, f in zip(values, flags) if f)\n",
        f"def sum_flagged(values, flags):\n    return sum(v for v, f in zip(values, flags) if f)\n",
        f"def category_means(labels, values):\n    s, c = {{}}, {{}}\n    for k, v in zip(labels, values):\n        s[k] = s.get(k, 0.0) + v\n        c[k] = c.get(k, 0) + 1\n    return {{k: s[k] / c[k] for k in s}}\n",
        f"def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
        f"def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
        f"def moving_average(xs, w):\n    return [sum(xs[i:i+w]) / w for i in range(len(xs) - w + 1)]\n",
        f"def percentile(xs, p):\n    s = sorted(xs)\n    k = int(round((len(s) - 1) * p / 100))\n    return s[k]\n",
        f"def ols_slope(xs, ys):\n    n = len(xs)\n    mx, my = sum(xs) / n, sum(ys) / n\n    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))\n    den = sum((x - mx) ** 2 for x in xs)\n    return num / den\n",
        f"def accuracy(pred, true):\n    return sum(1 for p, t in zip(pred, true) if p == t) / len(true)\n",
        f"# mean({nums}) == {sum(nums)/len(nums):.4f}\ndef mean(values):\n    return sum(values) / len(values)\n",
        f"# cpk({usl}, {lsl}, {mu}, {sigma}) demonstrates process capability\ndef cpk(usl, lsl, mu, sigma):\n    return min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))\n",
    ]


# --- 数値 Q&A テンプレート ---------------------------------------------------
def qa_templates(rng: random.Random) -> List[str]:
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    f = round(rng.uniform(0.1, 99.9), 2)
    d = rng.randint(2, 12)
    pct = rng.choice([5, 10, 15, 20, 25, 50])
    return [
        f"Q: What is {a} + {b}?\nA: {a + b}\n",
        f"Q: What is {a} - {b}?\nA: {a - b}\n",
        f"Q: What is {a} multiplied by {b}?\nA: {a * b}\n",
        f"Q: What is {a} divided by {b}, rounded to 2 decimals?\nA: {round(a / b, 2)}\n",
        f"Q: Round {f} to 1 decimal place.\nA: {round(f, 1)}\n",
        f"Q: What is the mean of {a} and {b}?\nA: {(a + b) / 2}\n",
        f"Q: What is {pct}% of {a}?\nA: {round(a * pct / 100, 2)}\n",
        f"Q: Is {a} greater than {b}?\nA: {'yes' if a > b else 'no'}\n",
        f"Q: What is {a} modulo {d}?\nA: {a % d}\n",
        f"Q: Express 1/{d} as a decimal (4 dp).\nA: {round(1 / d, 4)}\n",
        f"Q: What is {a} squared?\nA: {a * a}\n",
        f"Q: Convert {f} to an integer by truncation.\nA: {int(f)}\n",
    ]


def generate(n: int, seed: int, exclude: set[str]) -> List[dict]:
    rng = random.Random(seed)
    makers: List[Callable[[random.Random], List[str]]] = [code_templates, qa_templates]
    rows: List[dict] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < n and guard < n * 50:
        guard += 1
        text = rng.choice(rng.choice(makers)(rng))
        if any(ex and ex in text for ex in exclude):
            continue  # 評価セットとのリーク除去
        if text in seen:
            # 重複は 30% だけ許容 (多様性を保ちつつ自然な反復も残す)
            if rng.random() > 0.3:
                continue
        seen.add(text)
        rows.append({"text": text})
    return rows


def load_exclusions(eval_path: str) -> set[str]:
    ex: set[str] = set()
    if eval_path and os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("prompt"):
                    ex.add(d["prompt"])
                for p in d.get("paraphrases", []):
                    ex.add(p)
    return ex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--eval-set",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "numeric_stats_eval", "problems.jsonl",
        ),
    )
    args = parser.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "m1_train.jsonl"
    )
    exclude = load_exclusions(args.eval_set)
    rows = generate(args.n, args.seed, exclude)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples → {out} (excluded {len(exclude)} eval prompts)")


if __name__ == "__main__":
    main()

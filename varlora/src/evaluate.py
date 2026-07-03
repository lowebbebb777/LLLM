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
import os
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


# few-shot 例。ベース (非 Instruct) モデルに「答えの形」を示し、区切り ### で止める。
FEWSHOT_NUMERIC = (
    "Q: What is 5 plus 3?\nA: 8\n###\n"
    "Q: What is 6 multiplied by 4?\nA: 24\n###\n"
    "Q: Round 2.718 to 2 decimal places.\nA: 2.72\n###\n"
    # Cp/Cpk の手本 (出力形式を示す。数値は評価問題と別。全条件共通で公平)
    "Q: A process has USL=12, LSL=0, sigma=2. Compute Cp.\nA: 1.0\n###\n"
    "Q: A process has USL=10, LSL=0, mean=6, sigma=1. Compute Cpk.\nA: 1.3333\n###\n"
)
FEWSHOT_CODE = (
    "Write a Python function `inc(x)` that returns x plus 1.\n"
    "def inc(x):\n    return x + 1\n###\n"
)
EVAL_STOPS = ("###", "\nQ:")


def build_eval_prompt(problem: "NumericProblem", text: str) -> str:
    """few-shot + 形式合わせのプロンプトを組む (train の Q:/A: 形式に整合)。"""
    if problem.kind == "numeric":
        return FEWSHOT_NUMERIC + f"Q: {text}\nA:"
    return FEWSHOT_CODE + f"{text}\n"


def truncate_at_stops(text: str, stops: Sequence[str] = EVAL_STOPS) -> str:
    """最初の停止文字列で切る (few-shot の続き生成を除去)。"""
    cut = len(text)
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def numeric_consistency_score(
    problems: Sequence[NumericProblem],
    generate: Generator,
    *,
    n_samples: int = 1,
    prompt_fn=None,
) -> Dict[str, float]:
    """数値整合性スコア: pass@1 とは別軸で「数として正しく扱えているか」を測る。

    - kind=="code": test を実行して pass 率
    - kind=="numeric": 生成数値が expected と相対誤差 rel_tol 以内なら正
    カテゴリ別と全体のスコアを返す。
    """
    per_category: Dict[str, List[float]] = {}
    for prob in problems:
        prompt = prompt_fn(prob, prob.prompt) if prompt_fn else prob.prompt
        comp = truncate_at_stops(generate(prompt, n_samples)[0])
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
    prompt_fn=None,
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
        answers: List[Optional[float]] = [
            extract(truncate_at_stops(generate(prompt_fn(prob, p) if prompt_fn else p, 1)[0]))
            for p in paths
        ]
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


# ---------------------------------------------------------------------------
# M1 評価ドライバ: 学習済みアダプタを読み込み → 生成 → メトリクス
# (重い依存は関数内で遅延 import。メトリクス本体は上部の純関数で単体テスト済み)
# ---------------------------------------------------------------------------
def load_trained_model(
    model_name: str,
    condition: str,
    adapter_dir: str,
    *,
    use_4bit: bool = True,
    r0: int = 16,
    alpha: int = 32,
    n_quad: int = 3,
    tau: float = 1.0,
    fixed_gate: float = 0.5,
    target_modules=None,
):
    """条件に応じて base + アダプタを復元する。A/B=PEFT, C/D=VariationalLoRA 再注入。"""
    import os
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inject import FFN_TARGET_MODULES, build_bnb_config, inject_variational_lora

    target_modules = target_modules or FFN_TARGET_MODULES
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = {"trust_remote_code": True}
    if use_4bit:
        kw["quantization_config"] = build_bnb_config()
        kw["device_map"] = "auto"
    base = AutoModelForCausalLM.from_pretrained(model_name, **kw)

    if condition in ("A", "B"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        gate_mode = {"C": "dynamic", "D": "fixed", "E": "equilibrium"}[condition]
        inject_variational_lora(
            base, r=r0, alpha=alpha, n_quad=n_quad, gate_mode=gate_mode,
            fixed_gate=fixed_gate, tau=tau, target_modules=target_modules,
        )
        sd = torch.load(os.path.join(adapter_dir, "variational_lora.pt"), map_location="cpu")
        missing, unexpected = base.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[eval] warning: {len(unexpected)} unexpected keys in adapter state")
        model = base

    model.eval()
    return model, tok


def make_hf_generator(
    model, tokenizer, *, max_new_tokens: int = 100, temperature: float = 0.0,
    stops: Sequence[str] = EVAL_STOPS,
):
    """(prompt, n) -> List[str] の生成器。temperature=0 で greedy (経路独立性は決定的に)。

    n==1 のときは停止文字列 (### 等) で早期停止して高速化する。
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnText(StoppingCriteria):
        def __init__(self, prompt_len: int):
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kw) -> bool:
            tail = tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
            return any(s in tail for s in stops)

    @torch.no_grad()
    def gen(prompt: str, n: int = 1) -> List[str]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temperature > 0
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            kwargs.update(temperature=temperature, top_p=0.95)
        if n == 1:  # 早期停止 (バッチ 1 のときのみ安全)
            kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnText(prompt_len)])
        outs = model.generate(**inputs, **kwargs)
        return [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outs]

    return gen


def run_evaluation(problems, generate, *, n_samples: int = 1, few_shot: bool = True) -> Dict[str, float]:
    """自作数値整合性 + 経路独立性 (numeric のみ) をまとめて算出 (SPEC §5.2, §5.2b)。

    few_shot=True: ベース (非 Instruct) モデル用に few-shot + 停止トークンで綺麗な
    答えを引き出す (床張り付き対策)。全条件で同一プロンプトなので比較は公平。
    """
    pf = build_eval_prompt if few_shot else None
    scores: Dict[str, float] = {}
    scores.update(numeric_consistency_score(problems, generate, n_samples=n_samples, prompt_fn=pf))
    numeric_only = [p for p in problems if p.kind == "numeric" and p.paraphrases]
    scores.update(path_independence_score(numeric_only, generate, prompt_fn=pf))
    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="評価: メトリクス自己テスト or 学習済みモデル評価")
    parser.add_argument("--numeric-set", default="data/numeric_stats_eval/problems.jsonl")
    parser.add_argument("--adapter", default=None, help="指定すると学習済みモデルを評価")
    parser.add_argument("--condition", default="C", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--out", default=None, help="評価結果 JSON の出力先")
    parser.add_argument("--r0", type=int, default=16)
    parser.add_argument("--n-quad", type=int, default=3)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--dump", type=int, default=0,
                        help="各カテゴリ先頭N問の生プロンプト/生成/抽出を表示して終了 (デバッグ)")
    args = parser.parse_args()

    probs = load_numeric_problems(args.numeric_set)

    if args.adapter is not None and args.dump > 0:
        # デバッグ: 生成テキストと抽出結果を目視する (cpk=0 等の原因調査)
        model, tok = load_trained_model(
            args.model, args.condition, args.adapter,
            r0=args.r0, n_quad=args.n_quad, tau=args.tau,
        )
        gen = make_hf_generator(model, tok, max_new_tokens=args.max_new_tokens)
        seen: Dict[str, int] = {}
        for prob in probs:
            if seen.get(prob.category, 0) >= args.dump:
                continue
            seen[prob.category] = seen.get(prob.category, 0) + 1
            prompt = build_eval_prompt(prob, prob.prompt)
            raw = gen(prompt, 1)[0]
            trunc = truncate_at_stops(raw)
            got = parse_number(trunc) if prob.kind == "numeric" else "(code)"
            print(f"\n--- [{prob.category}] {prob.id} ({prob.kind}) ---")
            print(f"Q: {prob.prompt}")
            print(f"RAW: {raw[:160]!r}")
            print(f"TRUNC: {trunc[:120]!r}")
            print(f"PARSED: {got}   EXPECTED: {prob.expected}")
        raise SystemExit(0)

    if args.adapter is None:
        # 自己テスト: データの内訳のみ表示
        cats: Dict[str, int] = {}
        for p in probs:
            cats[p.category] = cats.get(p.category, 0) + 1
        print(f"loaded {len(probs)} numeric problems")
        print("categories:", json.dumps(cats, indent=2, ensure_ascii=False))
    else:
        # 学習済みモデル評価
        model, tok = load_trained_model(
            args.model, args.condition, args.adapter,
            r0=args.r0, n_quad=args.n_quad, tau=args.tau,
        )
        gen = make_hf_generator(model, tok, max_new_tokens=args.max_new_tokens)
        scores = run_evaluation(probs, gen)
        result = {"condition": args.condition, "adapter": args.adapter, "scores": scores}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[eval] saved → {args.out}")

"""Run DeltaFEM-LLM Phase D1 activation-delta observation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deltafem.d1_metrics import DeltaMetricConfig  # noqa: E402
from deltafem.d1_probe import (  # noqa: E402
    ToyCausalLM,
    ToyTokenizer,
    load_huggingface_model,
    load_prompt_pairs,
    package_metadata,
    run_prompt_edit_probe,
    run_token_step_probe,
    write_probe_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "hf"), default="toy")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--pairs", type=Path, default=PROJECT_ROOT / "data" / "d1_prompt_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--regimes", nargs="+", choices=("prompt_edit", "token_step"), default=("prompt_edit", "token_step"))
    parser.add_argument("--generation-steps", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--view", choices=("last_token", "aligned_sequence"), default="last_token")
    parser.add_argument("--alignment", choices=("prefix", "suffix"), default="suffix")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--absolute-threshold", type=float, default=1e-8)
    parser.add_argument("--relative-threshold", type=float, default=1e-3)
    parser.add_argument("--no-modules", action="store_true")
    parser.add_argument("--no-kv", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    config = DeltaMetricConfig(
        block_size=args.block_size,
        absolute_threshold=args.absolute_threshold,
        relative_threshold=args.relative_threshold,
    )
    pairs = load_prompt_pairs(args.pairs)

    if args.mode == "toy":
        model = ToyCausalLM().to(device)
        tokenizer = ToyTokenizer()
        model_name = "deltafem-toy-causal-lm"
    else:
        model, tokenizer = load_huggingface_model(args.model, device=device)
        model_name = args.model

    rows = []
    common = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "config": config,
        "max_length": args.max_length,
        "view": args.view,
        "include_modules": not args.no_modules,
        "include_kv": not args.no_kv,
    }
    if "prompt_edit" in args.regimes:
        rows.extend(run_prompt_edit_probe(pairs=pairs, alignment=args.alignment, **common))
    if "token_step" in args.regimes:
        rows.extend(
            run_token_step_probe(
                prompts=pairs,
                generation_steps=args.generation_steps,
                **common,
            )
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or PROJECT_ROOT / "results" / f"d1_{args.mode}_{timestamp}"
    metadata = {
        **package_metadata(config),
        "phase": "D1",
        "mode": args.mode,
        "model": model_name,
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pairs": str(args.pairs),
        "regimes": list(args.regimes),
        "view": args.view,
        "alignment": args.alignment,
        "max_length": args.max_length,
    }
    json_path, csv_path = write_probe_results(rows=rows, output_dir=output, metadata=metadata)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

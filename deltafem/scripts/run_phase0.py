#!/usr/bin/env python3
"""Run the DeltaFEM-LLM Phase-0 synthetic experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deltafem import run_phase0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input-features", type=int, default=256)
    parser.add_argument("--output-features", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--reanchor-interval", type=int, default=16)
    parser.add_argument(
        "--changed-fractions",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0],
    )
    parser.add_argument("--management-fraction", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_phase0(
        seed=args.seed,
        input_features=args.input_features,
        output_features=args.output_features,
        steps=args.steps,
        changed_fractions=args.changed_fractions,
        reanchor_interval=args.reanchor_interval,
        management_fraction_of_dense=args.management_fraction,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

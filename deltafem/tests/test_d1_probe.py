from __future__ import annotations

import json
from pathlib import Path

import torch

from deltafem.d1_metrics import DeltaMetricConfig
from deltafem.d1_probe import (
    PromptPair,
    ToyCausalLM,
    ToyTokenizer,
    run_prompt_edit_probe,
    run_token_step_probe,
    summarize_rows,
    write_probe_results,
)


def test_toy_prompt_edit_collects_all_components() -> None:
    model = ToyCausalLM(hidden_size=12, layers=2)
    rows = run_prompt_edit_probe(
        model=model,
        tokenizer=ToyTokenizer(),
        pairs=[PromptPair("edit", "abc", "abd")],
        device=torch.device("cpu"),
        config=DeltaMetricConfig(block_size=4),
        max_length=16,
    )
    components = {row["component"] for row in rows}
    assert {"residual", "attention", "mlp", "kv"}.issubset(components)
    assert all("active_fraction_90" in row for row in rows)


def test_toy_token_step_and_result_files(tmp_path: Path) -> None:
    rows = run_token_step_probe(
        model=ToyCausalLM(hidden_size=8, layers=2),
        tokenizer=ToyTokenizer(),
        prompts=[PromptPair("step", "hello", "unused")],
        device=torch.device("cpu"),
        config=DeltaMetricConfig(block_size=4),
        generation_steps=2,
        max_length=16,
    )
    assert rows
    assert {row["regime"] for row in rows} == {"token_step"}
    summary = summarize_rows(rows)
    assert summary["rows"] == len(rows)

    json_path, csv_path = write_probe_results(rows=rows, output_dir=tmp_path, metadata={"phase": "D1"})
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["phase"] == "D1"
    assert payload["summary"]["rows"] == len(rows)
    assert csv_path.stat().st_size > 0

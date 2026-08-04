"""Phase D1 activation-delta recorder for toy and Hugging Face causal LMs."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .d1_metrics import DeltaMetricConfig, analyze_activation_delta

_LAYER_PATTERN = re.compile(r"(?:layers|layer|h|blocks)\.(\d+)")


@dataclass(frozen=True)
class PromptPair:
    pair_id: str
    base: str
    edited: str
    label: str = "prompt_edit"


@dataclass
class ActivationSnapshot:
    activations: dict[str, torch.Tensor]
    next_token_logits: torch.Tensor | None = None


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _component(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("residual"):
        return "residual"
    if lowered.startswith("kv"):
        return "kv"
    if "self_attn" in lowered or ".attn" in lowered or "attention" in lowered:
        return "attention"
    if ".mlp" in lowered or "feed_forward" in lowered or ".ffn" in lowered:
        return "mlp"
    return "module"


def _layer_index(name: str) -> int | None:
    match = _LAYER_PATTERN.search(name)
    if match:
        return int(match.group(1))
    trailing = re.search(r"(?:layer_|layer\.)(\d+)", name)
    return int(trailing.group(1)) if trailing else None


def _module_is_probe_target(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.endswith(".self_attn")
        or lowered.endswith(".attn")
        or lowered.endswith(".attention")
        or lowered.endswith(".mlp")
        or lowered.endswith(".ffn")
        or lowered.endswith(".feed_forward")
    )


def _extract_kv(cache: Any) -> dict[str, torch.Tensor]:
    records: dict[str, torch.Tensor] = {}
    if cache is None:
        return records
    if hasattr(cache, "to_legacy_cache"):
        try:
            cache = cache.to_legacy_cache()
        except (AttributeError, TypeError, RuntimeError):
            pass
    if isinstance(cache, (tuple, list)):
        for index, layer in enumerate(cache):
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                key = _first_tensor(layer[0])
                value = _first_tensor(layer[1])
                if key is not None:
                    records[f"kv.layers.{index}.key"] = key.detach().cpu()
                if value is not None:
                    records[f"kv.layers.{index}.value"] = value.detach().cpu()
        return records
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for index, layer in enumerate(layers):
            key = getattr(layer, "keys", None)
            if key is None:
                key = getattr(layer, "key_cache", None)
            value = getattr(layer, "values", None)
            if value is None:
                value = getattr(layer, "value_cache", None)
            if isinstance(key, torch.Tensor):
                records[f"kv.layers.{index}.key"] = key.detach().cpu()
            if isinstance(value, torch.Tensor):
                records[f"kv.layers.{index}.value"] = value.detach().cpu()
    return records


def capture_activations(
    model: nn.Module,
    model_inputs: Mapping[str, torch.Tensor],
    *,
    include_modules: bool = True,
    include_kv: bool = True,
) -> ActivationSnapshot:
    module_records: dict[str, torch.Tensor] = {}
    handles = []
    if include_modules:
        for name, module in model.named_modules():
            if not _module_is_probe_target(name):
                continue

            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, module_name: str = name) -> None:
                tensor = _first_tensor(output)
                if tensor is not None:
                    module_records[f"module.{module_name}"] = tensor.detach().cpu()

            handles.append(module.register_forward_hook(hook))

    try:
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=include_kv,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    records = dict(module_records)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        for index, tensor in enumerate(hidden_states):
            label = "embedding" if index == 0 else f"layer_{index - 1}"
            records[f"residual.{label}"] = tensor.detach().cpu()
    if include_kv:
        records.update(_extract_kv(getattr(outputs, "past_key_values", None)))
    logits = getattr(outputs, "logits", None)
    next_logits = logits[:, -1, :].detach().cpu() if isinstance(logits, torch.Tensor) else None
    return ActivationSnapshot(records, next_logits)


def compare_snapshots(
    base: ActivationSnapshot,
    edited: ActivationSnapshot,
    *,
    pair_id: str,
    regime: str,
    config: DeltaMetricConfig,
    view: str,
    alignment: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(base.activations).intersection(edited.activations)):
        try:
            metrics = analyze_activation_delta(
                base.activations[name],
                edited.activations[name],
                config=config,
                view=view,
                alignment=alignment,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        rows.append(
            {
                "pair_id": pair_id,
                "regime": regime,
                "activation": name,
                "component": _component(name),
                "layer_index": _layer_index(name),
                **metrics,
            }
        )
    _append_densification_metrics(rows)
    return rows


def _append_densification_metrics(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("layer_index") is not None:
            grouped[str(row["component"])].append(row)
    for component_rows in grouped.values():
        component_rows.sort(key=lambda item: (int(item["layer_index"]), str(item["activation"])))
        previous: dict[str, Any] | None = None
        for row in component_rows:
            if previous is None:
                row["changed_fraction_growth"] = None
                row["active_fraction_90_growth"] = None
            else:
                row["changed_fraction_growth"] = float(row["changed_fraction"]) - float(previous["changed_fraction"])
                row["active_fraction_90_growth"] = float(row["active_fraction_90"]) - float(previous["active_fraction_90"])
            previous = row


def load_prompt_pairs(path: str | Path) -> list[PromptPair]:
    pairs: list[PromptPair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                pairs.append(
                    PromptPair(
                        pair_id=str(payload.get("id", f"pair_{line_number}")),
                        base=str(payload["base"]),
                        edited=str(payload["edited"]),
                        label=str(payload.get("label", "prompt_edit")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"missing {exc.args[0]!r} at line {line_number}") from exc
    if not pairs:
        raise ValueError("prompt pair file is empty")
    return pairs


def _move_inputs(inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in inputs.items()}


def run_prompt_edit_probe(
    *,
    model: nn.Module,
    tokenizer: Any,
    pairs: Iterable[PromptPair],
    device: torch.device,
    config: DeltaMetricConfig,
    max_length: int = 128,
    view: str = "last_token",
    alignment: str = "suffix",
    include_modules: bool = True,
    include_kv: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for pair in pairs:
        base_inputs = _move_inputs(tokenizer(pair.base, return_tensors="pt", truncation=True, max_length=max_length), device)
        edited_inputs = _move_inputs(tokenizer(pair.edited, return_tensors="pt", truncation=True, max_length=max_length), device)
        base_snapshot = capture_activations(model, base_inputs, include_modules=include_modules, include_kv=include_kv)
        edited_snapshot = capture_activations(model, edited_inputs, include_modules=include_modules, include_kv=include_kv)
        pair_rows = compare_snapshots(
            base_snapshot,
            edited_snapshot,
            pair_id=pair.pair_id,
            regime=pair.label,
            config=config,
            view=view,
            alignment=alignment,
        )
        base_tokens = int(base_inputs["input_ids"].shape[-1])
        edited_tokens = int(edited_inputs["input_ids"].shape[-1])
        for row in pair_rows:
            row["base_tokens"] = base_tokens
            row["edited_tokens"] = edited_tokens
        rows.extend(pair_rows)
    return rows


def run_token_step_probe(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompts: Iterable[PromptPair],
    device: torch.device,
    config: DeltaMetricConfig,
    generation_steps: int = 3,
    max_length: int = 128,
    view: str = "last_token",
    include_modules: bool = True,
    include_kv: bool = True,
) -> list[dict[str, Any]]:
    if generation_steps <= 0:
        raise ValueError("generation_steps must be positive")
    rows: list[dict[str, Any]] = []
    model.eval()
    for prompt_pair in prompts:
        inputs = _move_inputs(tokenizer(prompt_pair.base, return_tensors="pt", truncation=True, max_length=max_length), device)
        current = capture_activations(model, inputs, include_modules=include_modules, include_kv=include_kv)
        for step in range(generation_steps):
            if current.next_token_logits is None:
                raise RuntimeError("model output does not expose logits")
            next_token = current.next_token_logits.argmax(dim=-1, keepdim=True).to(device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(next_token, device=device)], dim=-1
                )
            following = capture_activations(model, inputs, include_modules=include_modules, include_kv=include_kv)
            step_rows = compare_snapshots(
                current,
                following,
                pair_id=f"{prompt_pair.pair_id}_step_{step + 1}",
                regime="token_step",
                config=config,
                view=view,
                alignment="suffix",
            )
            for row in step_rows:
                row["base_tokens"] = int(inputs["input_ids"].shape[-1] - 1)
                row["edited_tokens"] = int(inputs["input_ids"].shape[-1])
            rows.extend(step_rows)
            current = following
    return rows


def summarize_rows(rows: list[dict[str, Any]], *, go_fraction: float = 0.10) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["component"]), row.get("layer_index"))].append(row)

    layer_summaries = []
    candidate_layers: set[tuple[str, str, int]] = set()
    for (regime, component, layer), items in sorted(grouped.items(), key=lambda item: str(item[0])):
        active = float(np.median([float(item["active_fraction_90"]) for item in items]))
        block = float(np.median([float(item["active_block_fraction_90"]) for item in items]))
        changed = float(np.median([float(item["changed_fraction"]) for item in items]))
        delta_l2 = float(np.median([float(item["delta_l2"]) for item in items]))
        summary = {
            "regime": regime,
            "component": component,
            "layer_index": layer,
            "samples": len(items),
            "median_active_fraction_90": active,
            "median_active_block_fraction_90": block,
            "median_changed_fraction": changed,
            "median_delta_l2": delta_l2,
            "candidate": delta_l2 > 0.0 and min(active, block) <= go_fraction,
        }
        layer_summaries.append(summary)
        if summary["candidate"] and layer is not None:
            candidate_layers.add((regime, component, int(layer)))

    return {
        "rows": len(rows),
        "go_fraction": go_fraction,
        "candidate_layer_count": len(candidate_layers),
        "d1_go_candidate": len(candidate_layers) >= 2,
        "decision_note": (
            "Screening signal only. D2 requires reproduction across multiple prompts and an actual latency model."
        ),
        "layer_summaries": layer_summaries,
    }


def write_probe_results(
    *,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    json_path = destination / "d1_results.json"
    json_path.write_text(
        json.dumps({"metadata": dict(metadata), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = destination / "d1_metrics.csv"
    if rows:
        fields = sorted({key for row in rows for key in row if key != "shape"}) + ["shape"]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                payload = dict(row)
                payload["shape"] = json.dumps(payload.get("shape", []))
                writer.writerow(payload)
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


class ToyTokenizer:
    """Deterministic byte tokenizer for D1 pipeline tests; not a language model tokenizer."""

    def __init__(self, vocab_size: int = 257):
        self.vocab_size = vocab_size

    def __call__(self, text: str, *, return_tensors: str, truncation: bool, max_length: int) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("ToyTokenizer only supports return_tensors='pt'")
        values = [byte % (self.vocab_size - 1) + 1 for byte in text.encode("utf-8")]
        values = values[-max_length:] if truncation else values
        if not values:
            values = [0]
        input_ids = torch.tensor([values], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class ToySelfAttention(nn.Module):
    """Tiny causal context mixer used only to validate the D1 recorder."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        cumulative = hidden.cumsum(dim=-2)
        denominator = torch.arange(
            1,
            hidden.shape[-2] + 1,
            device=hidden.device,
            dtype=hidden.dtype,
        ).view(*([1] * (hidden.ndim - 2)), -1, 1)
        return self.projection(cumulative / denominator)


class ToyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = ToySelfAttention(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + torch.tanh(self.self_attn(hidden))
        return hidden + self.mlp(hidden)


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 257, hidden_size: int = 24, layers: int = 3):
        super().__init__()
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(11)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([ToyBlock(hidden_size) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        torch.random.set_rng_state(generator_state)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_hidden_states: bool = True,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, return_dict
        hidden = self.embedding(input_ids)
        hidden_states = [hidden]
        caches = []
        for block in self.blocks:
            hidden = block(hidden)
            hidden_states.append(hidden)
            if use_cache:
                key = hidden.unsqueeze(1)
                value = (0.5 * hidden).unsqueeze(1)
                caches.append((key, value))
        logits = self.lm_head(hidden)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            past_key_values=tuple(caches) if use_cache else None,
        )


def load_huggingface_model(model_name: str, *, device: torch.device) -> tuple[nn.Module, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for --mode hf; run setup_windows.ps1") from exc

    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    return model, tokenizer


def package_metadata(config: DeltaMetricConfig) -> dict[str, Any]:
    return {"metric_config": asdict(config), "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}

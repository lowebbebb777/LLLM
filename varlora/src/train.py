"""学習ループ — 全条件 (A/B/C/D) を切り替えて QLoRA で学習する (SPEC §4, §6)。

条件 (SPEC §4.2):
    A : 標準 LoRA (rank=r0)                         — ベースライン (対照群)
    B : 標準 LoRA (rank 拡大で C とパラメータ数一致)  — 交絡対照
    C : VariationalLoRA (ゲート動的)                 — 実験群 (本仮説)
    D : VariationalLoRA (ゲート固定0.5)              — 機構分離

VRAM / 安定性対策 (SPEC §6):
    - gradient_checkpointing=True 必須
    - per_device_train_batch_size=1, gradient_accumulation_steps=8〜16
    - max_grad_norm=1.0 (gradient clipping) 必須
    - softmax 温度 τ, learning rate 小さめ (1e-4)
    - loss NaN 検知 → そのstep の state/input をダンプ (NaNDetectionCallback)

重い依存 (transformers/peft/trl/bitsandbytes) は関数内で遅延 import する。
config パースや条件分岐のテストを GPU/重依存なしで行えるようにするため。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

# inject はこのファイルの隣 (src/) にある。
from inject import (
    FFN_TARGET_MODULES,
    build_bnb_config,
    build_standard_lora_peft_config,
    collect_ffn_module_shapes,
    build_matched_rank_pattern,
    count_trainable_parameters,
    freeze_base_train_adapters,
    inject_variational_lora,
)
from variational_lora import variational_lora_param_count

VALID_CONDITIONS = ("A", "B", "C", "D")


@dataclass
class TrainConfig:
    # --- 条件 ---
    condition: str = "C"  # A|B|C|D

    # --- モデル / データ ---
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    dataset: str = "data/m0_smoke.jsonl"  # jsonl パス or HF dataset 名
    text_field: str = "text"
    max_seq_length: int = 1024  # SPEC §6.2: 1024〜2048 から開始
    output_dir: str = "outputs/run"

    # --- アダプタ共通 ---
    r0: int = 16  # 手応え確認フェーズで決めた値 (SPEC §4.2)
    alpha: int = 32
    lora_dropout: float = 0.0

    # --- VariationalLoRA (条件 C/D) ---
    n_quad: int = 3
    tau: float = 1.0
    fixed_gate: float = 0.5
    clamp_quad_weight: bool = False

    # --- 量子化 ---
    use_4bit: bool = True

    # --- 統制変数 (SPEC §4.4: 全条件で固定) ---
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4  # SPEC §6.3: 学習初期は小さめ
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 1  # SPEC §6.2
    gradient_accumulation_steps: int = 8  # SPEC §6.2: 8〜16
    max_grad_norm: float = 1.0  # SPEC §6.3: gradient clipping 必須
    gradient_checkpointing: bool = True  # SPEC §6.2: 必須
    seed: int = 0
    logging_steps: int = 5
    save_steps: int = 200
    bf16: bool = False
    fp16: bool = True
    target_modules: list = field(default_factory=lambda: list(FFN_TARGET_MODULES))

    def __post_init__(self) -> None:
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(f"condition must be one of {VALID_CONDITIONS}, got {self.condition}")


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"未知の config キー: {sorted(unknown)}")
    return TrainConfig(**raw)


# ---------------------------------------------------------------------------
# パラメータ数レポート (SPEC §2.4 / §4.1 交絡対照の検証)
# ---------------------------------------------------------------------------
def parameter_report(model, cfg: TrainConfig) -> Dict[str, Any]:
    """条件 C 相当の解析パラメータ数と条件 B 一致 rank を計算して表示する。

    交絡対照 (B≈C) が実際に成立しているかを学習前に確認するための報告。
    """
    shapes = collect_ffn_module_shapes(model, cfg.target_modules)
    n_layers_per_module: Dict[str, int] = {}
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in cfg.target_modules:
            n_layers_per_module[leaf] = n_layers_per_module.get(leaf, 0) + 1

    report: Dict[str, Any] = {"per_module": {}, "condition": cfg.condition}
    total_var = 0
    for leaf, (in_f, out_f) in shapes.items():
        per = variational_lora_param_count(in_f, out_f, cfg.r0, cfg.n_quad)
        n = n_layers_per_module.get(leaf, 0)
        report["per_module"][leaf] = {
            "in": in_f,
            "out": out_f,
            "n_layers": n,
            "variational_params_each": per,
            "variational_params_total": per * n,
        }
        total_var += per * n
    report["variational_total_C"] = total_var

    matched = build_matched_rank_pattern(shapes, cfg.r0, cfg.n_quad)
    report["matched_rank_pattern_B"] = matched
    # 条件 B の総数 (一致 rank で計算)
    total_b = 0
    for leaf, (in_f, out_f) in shapes.items():
        r_b = matched[leaf]
        total_b += r_b * (in_f + out_f) * n_layers_per_module.get(leaf, 0)
    report["standard_total_B"] = total_b
    report["standard_total_A"] = sum(
        cfg.r0 * (in_f + out_f) * n_layers_per_module.get(leaf, 0)
        for leaf, (in_f, out_f) in shapes.items()
    )
    if total_var:
        report["B_over_C_ratio"] = round(total_b / total_var, 4)
    return report


# ---------------------------------------------------------------------------
# モデル構築とアダプタ適用
# ---------------------------------------------------------------------------
def build_model_and_tokenizer(cfg: TrainConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if cfg.use_4bit:
        model_kwargs["quantization_config"] = build_bnb_config()
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, tokenizer


def apply_adapter(model, cfg: TrainConfig):
    """条件に応じてアダプタを適用し、(model, kind) を返す。

    kind: "peft" (条件 A/B) | "variational" (条件 C/D)。
    """
    if cfg.condition in ("A", "B"):
        from peft import get_peft_model

        n_quad_for_match = cfg.n_quad if cfg.condition == "B" else None
        lora_cfg = build_standard_lora_peft_config(
            model,
            r=cfg.r0,
            alpha=cfg.alpha,
            dropout=cfg.lora_dropout,
            n_quad_for_match=n_quad_for_match,
            target_modules=cfg.target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        return model, "peft"

    # 条件 C/D: VariationalLoRA を注入
    gate_mode = "dynamic" if cfg.condition == "C" else "fixed"
    summary = inject_variational_lora(
        model,
        r=cfg.r0,
        alpha=cfg.alpha,
        n_quad=cfg.n_quad,
        gate_mode=gate_mode,
        fixed_gate=cfg.fixed_gate,
        tau=cfg.tau,
        dropout=cfg.lora_dropout,
        clamp_quad_weight=cfg.clamp_quad_weight,
        target_modules=cfg.target_modules,
    )
    freeze_base_train_adapters(model)
    print(f"[inject] {summary.n_injected} modules, trainable={summary.trainable_params:,}")
    return model, "variational"


# ---------------------------------------------------------------------------
# データセット
# ---------------------------------------------------------------------------
def build_dataset(cfg: TrainConfig, tokenizer):
    from datasets import load_dataset

    if cfg.dataset.endswith(".jsonl") or cfg.dataset.endswith(".json"):
        ds = load_dataset("json", data_files=cfg.dataset, split="train")
    else:
        ds = load_dataset(cfg.dataset, split="train")

    def tokenize(batch):
        out = tokenizer(
            batch[cfg.text_field],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# NaN 検知コールバック (SPEC §6.3)
# ---------------------------------------------------------------------------
def make_nan_detection_callback(dump_dir: str):
    from transformers import TrainerCallback

    class NaNDetectionCallback(TrainerCallback):
        """loss が NaN/Inf になった step の state を JSON にダンプして停止する。"""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss")
            if loss is not None and (loss != loss or loss in (float("inf"), float("-inf"))):
                os.makedirs(dump_dir, exist_ok=True)
                path = os.path.join(dump_dir, f"nan_step_{state.global_step}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"global_step": state.global_step, "logs": logs, "log_history": state.log_history},
                        f,
                        indent=2,
                    )
                print(f"[NaN] loss={loss} at step {state.global_step}. dumped → {path}")
                control.should_training_stop = True

    return NaNDetectionCallback()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_training_arguments(cfg: TrainConfig):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,  # SPEC §6.3
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        data_seed=cfg.seed,  # SPEC §4.4: 評価seed/seed を固定
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        report_to=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VariationalLoRA 学習 (条件 A/B/C/D)")
    parser.add_argument("--config", required=True, help="configs/cond_*.yaml")
    parser.add_argument("--seed", type=int, default=None, help="config の seed を上書き")
    parser.add_argument("--report-only", action="store_true", help="パラメータ数レポートのみ")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    from transformers import Trainer, DataCollatorForLanguageModeling, set_seed

    set_seed(cfg.seed)  # SPEC §4.4: seed 固定

    model, tokenizer = build_model_and_tokenizer(cfg)

    report = parameter_report(model, cfg)
    print("[param-report]", json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_only:
        return

    model, kind = apply_adapter(model, cfg)
    print(f"[adapter] condition={cfg.condition} kind={kind} "
          f"trainable={count_trainable_parameters(model):,}")

    train_ds = build_dataset(cfg, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = build_training_arguments(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[make_nan_detection_callback(os.path.join(cfg.output_dir, "nan_dumps"))],
    )
    trainer.train()
    trainer.save_model(os.path.join(cfg.output_dir, "adapter"))
    print(f"[done] saved → {cfg.output_dir}/adapter")


if __name__ == "__main__":
    main()

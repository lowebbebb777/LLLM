"""学習ループ — 全条件 (A/B/C/D/E) を切り替えて QLoRA で学習する (SPEC §4, §6)。

条件 (SPEC §4.2):
    A : 標準 LoRA (rank=r0)                         — ベースライン (対照群)
    B : 標準 LoRA (rank 拡大で C とパラメータ数一致)  — 交絡対照
    C : VariationalLoRA (ゲート動的, 凸結合)         — 実験群 (本仮説)
    D : VariationalLoRA (ゲート固定0.5, 凸結合)      — 機構分離
    E : VariationalLoRA (ゲート動的, 加法形)         — 結合形の対照 (総仮想仕事形)

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

VALID_CONDITIONS = ("A", "B", "C", "D", "E")


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
    optim_8bit: bool = True  # SPEC §6.2: VRAM 節約。bitsandbytes PagedAdamW8bit を優先
    gate_lr_multiplier: float = 10.0  # ゲート(shape_fn/quad_weights)の LR 倍率。
    # ゲートは僅少パラメータで 62M の経路パラメータと同 LR だと学習が進まず qw が
    # 一様初期値(1/n_quad)に張り付く → C が D に縮退して交絡。別 group で高 LR にする。
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

    kind: "peft" (条件 A/B) | "variational" (条件 C/D/E)。
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

    # 条件 C/D/E: VariationalLoRA を注入
    gate_mode = {"C": "dynamic", "D": "fixed", "E": "additive"}[cfg.condition]
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
        # labels はパディングのみ -100 でマスクする collate 側で作る (正規の eos を
        # マスクしないため)。ここでは input_ids / attention_mask のみ返す。
        return tokenizer(
            batch[cfg.text_field],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# カスタム学習ループ (SPEC §6)
# ---------------------------------------------------------------------------
# transformers.Trainer は 4bit 量子化モデルの学習を「PEFT 経由でアダプタが付いて
# いる」場合のみ許可する (validate_quantization_for_training)。条件 C/D/E は PEFT を
# 介さず VariationalLoRA を直接注入するためその検査に弾かれる。よって Trainer を
# 使わず自前ループを回す。これにより SPEC §6.3 の NaN ダンプ・勾配 clip と、
# M0 チェックリストの「ゲート値が 0/1 に張り付いてないか」観察を直接実装できる。


def make_collate_fn(tokenizer):
    """input_ids/attention_mask を動的パディングし、パディング位置のみ labels=-100。

    pad_token==eos_token のとき正規の eos までマスクしてしまう DataCollator の罠を
    避けるため、attention_mask==0 の位置だけを -100 にする。
    """
    import torch

    def collate(features):
        batch = tokenizer.pad(features, return_tensors="pt", pad_to_multiple_of=8)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    return collate


def build_param_groups(model, cfg: TrainConfig):
    """ゲートパラメータ (shape_fn / quad_weights) を高 LR の別 group に分ける。

    ゲートは数百パラメータしかなく、62M の経路パラメータ (cont/disc) と同 LR では
    学習が進まず qw が一様初期値に張り付く。別 group に gate_lr_multiplier 倍の LR を
    与えて、ゲートの動的性が出るか否かを公平に検証できるようにする。
    返り値: (param_groups, n_gate_param_tensors)
    """
    gate, base = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "shape_fn" in n or n.endswith("quad_weights"):
            gate.append(p)
        else:
            base.append(p)
    groups = [{"params": base, "lr": cfg.learning_rate}]
    if gate:
        groups.append({"params": gate, "lr": cfg.learning_rate * cfg.gate_lr_multiplier})
    return groups, len(gate)


def build_optimizer(params, cfg: TrainConfig):
    """VRAM 節約のため 8bit paged optimizer を優先 (SPEC §6.2)。無ければ AdamW。

    params は素のパラメータ列でも param_groups (dict のリスト) でも可。
    """
    import torch

    if cfg.optim_8bit:
        try:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit(params, lr=cfg.learning_rate)
        except Exception as e:  # CUDA 不整合等
            print(f"[optim] 8bit optimizer 使用不可 ({e}); AdamW にフォールバック")
    return torch.optim.AdamW(params, lr=cfg.learning_rate)


def gate_statistics(model) -> str:
    """動的ゲートの観察 (M0 チェックリスト: 0/1 張り付き検出)。

    最初の dynamic な VariationalLoRA から直近 forward の qw 平均と、学習で
    quad_weights が一様 (1/n_quad) からどれだけ動いたかを返す。
    """
    from variational_lora import VariationalLoRA

    for m in model.modules():
        if isinstance(m, VariationalLoRA) and m.cfg.gate_mode in ("dynamic", "additive"):
            qw = getattr(m, "_last_qw_mean", None)
            qw_v = float(qw) if qw is not None else float("nan")
            qwt = m.quad_weights.detach()
            return (
                f"qw_mean={qw_v:.3f} "
                f"quad_w=[{qwt.min().item():.3f},{qwt.max().item():.3f}]"
            )
    return "qw=n/a"


def save_adapter(model, cfg: TrainConfig) -> str:
    import torch

    out = os.path.join(cfg.output_dir, "adapter")
    os.makedirs(out, exist_ok=True)
    if cfg.condition in ("A", "B"):
        model.save_pretrained(out)  # PEFT がアダプタのみ保存
    else:  # C/D/E
        sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(sd, os.path.join(out, "variational_lora.pt"))
    return out


def train_loop(model, tokenizer, train_ds, cfg: TrainConfig) -> None:
    import math

    import torch
    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    device = next(p.device for p in model.parameters())
    loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    param_groups, n_gate = build_param_groups(model, cfg)
    optimizer = build_optimizer(param_groups, cfg)

    accum = cfg.gradient_accumulation_steps
    updates_per_epoch = math.ceil(len(loader) / accum)
    total_updates = max(1, int(updates_per_epoch * cfg.num_train_epochs))
    warmup = int(total_updates * cfg.warmup_ratio)
    scheduler = get_scheduler(
        cfg.lr_scheduler_type, optimizer,
        num_warmup_steps=warmup, num_training_steps=total_updates,
    )

    dump_dir = os.path.join(cfg.output_dir, "nan_dumps")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step, running, last_gnorm, micro_since_log = 0, 0.0, 0.0, 0
    n_epochs = math.ceil(cfg.num_train_epochs)
    print(f"[train] total_updates={total_updates} warmup={warmup} "
          f"updates/epoch={updates_per_epoch} optimizer={type(optimizer).__name__} "
          f"gate_params={n_gate} gate_lr={cfg.learning_rate * cfg.gate_lr_multiplier:.1e}")

    for epoch in range(n_epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss

            # NaN/Inf 検知 → そのstepの input/state をダンプして停止 (SPEC §6.3)
            if not torch.isfinite(loss):
                os.makedirs(dump_dir, exist_ok=True)
                path = os.path.join(dump_dir, f"nan_step_{global_step}.pt")
                torch.save(
                    {"global_step": global_step, "loss": float(loss),
                     "input_ids": batch["input_ids"].detach().cpu(),
                     "gate": gate_statistics(model)},
                    path,
                )
                print(f"[NaN] non-finite loss at step {global_step}; dumped → {path}")
                return

            (loss / accum).backward()
            running += loss.item()
            micro_since_log += 1

            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                # gradient clipping 必須 (SPEC §6.3)
                last_gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # 最初・logging_steps毎・最終 を必ずログ (短い run でも loss/ゲートが見える)
                if (global_step == 1 or global_step % cfg.logging_steps == 0
                        or global_step >= total_updates):
                    print(f"step {global_step}/{total_updates} "
                          f"loss={running / max(1, micro_since_log):.4f} "
                          f"grad_norm={last_gnorm:.3f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} {gate_statistics(model)}")
                    running, micro_since_log = 0.0, 0
                if global_step >= total_updates:
                    break
        if global_step >= total_updates:
            break

    out = save_adapter(model, cfg)
    print(f"[done] saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VariationalLoRA 学習 (条件 A/B/C/D)")
    parser.add_argument("--config", required=True, help="configs/cond_*.yaml")
    parser.add_argument("--seed", type=int, default=None, help="config の seed を上書き")
    parser.add_argument("--report-only", action="store_true", help="パラメータ数レポートのみ")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    from transformers import set_seed

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
    train_loop(model, tokenizer, train_ds, cfg)


if __name__ == "__main__":
    main()

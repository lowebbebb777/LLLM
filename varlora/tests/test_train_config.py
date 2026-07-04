"""train.py の config パースと条件分岐の単体テスト (重依存なし)。

transformers/peft/bitsandbytes は train.py 内で遅延 import されるため、
config 読み込みと条件検証は GPU/重依存なしでテストできる。
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import TrainConfig, load_config, VALID_CONDITIONS  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_all_configs_load():
    files = sorted(glob.glob(os.path.join(CONFIG_DIR, "cond_*.yaml")))
    assert len(files) == 5, files  # A/B/C/D/E
    seen = set()
    for f in files:
        cfg = load_config(f)
        assert cfg.condition in VALID_CONDITIONS
        seen.add(cfg.condition)
        # 統制変数が全 config で揃っているか (SPEC §4.4) の一部チェック
        assert cfg.max_grad_norm == 1.0  # gradient clipping 必須
        assert cfg.gradient_checkpointing is True
        assert cfg.r0 == 16
        assert cfg.alpha == 32
    assert seen == set(VALID_CONDITIONS), seen


def test_controlled_variables_identical_across_conditions():
    # SPEC §4.4: epoch/lr/warmup/seed/batch 構成は全条件で固定されているべき
    cfgs = {load_config(os.path.join(CONFIG_DIR, f"cond_{c}.yaml")).condition:
            load_config(os.path.join(CONFIG_DIR, f"cond_{c}.yaml")) for c in "ABCDE"}
    ref = cfgs["A"]
    for c, cfg in cfgs.items():
        assert cfg.num_train_epochs == ref.num_train_epochs, c
        assert cfg.learning_rate == ref.learning_rate, c
        assert cfg.warmup_ratio == ref.warmup_ratio, c
        assert cfg.per_device_train_batch_size == ref.per_device_train_batch_size, c
        assert cfg.gradient_accumulation_steps == ref.gradient_accumulation_steps, c
        assert cfg.max_seq_length == ref.max_seq_length, c


def test_unknown_key_rejected(tmp_path=None):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("condition: C\nbogus_key: 123\n")
        path = f.name
    try:
        raised = False
        try:
            load_config(path)
        except ValueError:
            raised = True
        assert raised, "未知キーは ValueError を投げるべき"
    finally:
        os.unlink(path)


def test_invalid_condition_rejected():
    raised = False
    try:
        TrainConfig(condition="Z")
    except ValueError:
        raised = True
    assert raised


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"OK: {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

"""aggregate_m1.py の集計ロジックの単体テスト (純 python)。"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import aggregate_m1  # noqa: E402


def _write(path, condition, scores):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"condition": condition, "adapter": "x", "scores": scores}, f)


def test_load_groups_by_condition_and_metric():
    with tempfile.TemporaryDirectory() as d:
        pa = os.path.join(d, "a.json")
        pc1 = os.path.join(d, "c1.json")
        pc2 = os.path.join(d, "c2.json")
        _write(pa, "A", {"numeric/overall": 0.50})
        _write(pc1, "C", {"numeric/overall": 0.70})
        _write(pc2, "C", {"numeric/overall": 0.80})  # 2 seed 目
        data = aggregate_m1.load([pa, pc1, pc2])
        assert data["A"]["numeric/overall"] == [0.50]
        assert sorted(data["C"]["numeric/overall"]) == [0.70, 0.80]


def test_fmt_single_and_multi():
    assert aggregate_m1.fmt([0.5]) == "0.500"
    out = aggregate_m1.fmt([0.7, 0.8])
    assert "0.750" in out and "±" in out  # 平均±標準偏差
    assert aggregate_m1.fmt([]) == "   -   "


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"OK: {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

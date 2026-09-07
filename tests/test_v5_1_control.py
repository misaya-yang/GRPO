import importlib.util
from pathlib import Path

import pytest


def control():
    spec = importlib.util.spec_from_file_location(
        "v51_control", Path(__file__).parents[1] / "scripts/v5_1_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_gpu_cannot_launch(monkeypatch):
    torch = pytest.importorskip("torch")
    c = control()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected(*args, **kwargs):
        raise AssertionError("Must not spawn any experiment without GPU")

    monkeypatch.setattr(c.subprocess, "Popen", unexpected)
    with pytest.raises(RuntimeError, match="No GPU"):
        c.start(7200, False)


def test_carried_ledger_has_no_duplicate_cycle_charge():
    import json

    c = control()
    record = c.prior_budget()
    source = c.ROOT / record["source"]
    events = [json.loads(s) for s in source.read_text().splitlines()]
    assert sum(e["event"] == "end" for e in events) == 18
    assert record["remaining_pilot_seconds_before_v5_1"] + sum(
        e.get("elapsed_seconds", 0) for e in events
    ) == pytest.approx(36000)


def test_bounded_stage_timeout_records_stage(tmp_path):
    import sys
    import time

    c = control()
    with pytest.raises(TimeoutError, match="exceeded"):
        c.execute_stage(
            tmp_path,
            "sleeping",
            [sys.executable, "-c", "import time; time.sleep(5)"],
            time.time() + 3,
            0.05,
        )
    assert (tmp_path / "sleeping.log").exists()
    assert (tmp_path / "state.json").exists()

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The load phase split, and the ways a timer like this reports a lie.

Three of these guards exist because the obvious implementation gets them wrong
and the wrong answer looks exactly like a real result in the report:

* a timer that never attached reporting **0.0 seconds**, which reads as "no
  download happened" rather than "nothing was measured";
* `snapshot_download` calling `hf_hub_download` per file, so a naive sum counts
  the same seconds twice and can report more download time than the phase it
  sat inside;
* a raising download leaving the timer installed, so every later call in the
  process is still wrapped.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(PAYLOAD))

from phase_timers import FetchTimer  # noqa: E402


@pytest.fixture
def hub(monkeypatch):
    """A stand-in `huggingface_hub` with the two entry points the timer wraps."""
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = lambda *a, **k: ""
    module.snapshot_download = lambda *a, **k: ""
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


def test_an_unattached_timer_reports_none_not_zero(monkeypatch):
    """The finding this whole file exists for. With nothing patched, `seconds`
    must be None: a report showing 0.0 is indistinguishable from a warm cache,
    and one of those is a measurement while the other is a broken instrument."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    monkeypatch.delitem(sys.modules, "transformers.utils.hub", raising = False)
    timer = FetchTimer().install()
    try:
        assert timer.patched == []
        assert timer.seconds is None
        record = timer.record(12.0)
        assert record["fetch_seconds"] is None
        assert record["weight_load_seconds"] is None
        assert "do not read the absence" in record["note"]
    finally:
        timer.uninstall()


def test_a_real_download_is_timed_and_sized(hub, tmp_path):
    blob = tmp_path / "model.safetensors"
    blob.write_bytes(b"x" * 4096)

    def slow_download(*_a, **_k):
        time.sleep(0.05)
        return str(blob)

    hub.hf_hub_download = slow_download
    with FetchTimer() as timer:
        import huggingface_hub
        huggingface_hub.hf_hub_download(repo_id = "org/model")
    record = timer.record(1.0)
    assert record["calls"] == 1
    assert record["fetch_seconds"] >= 0.0
    assert timer.bytes == 4096
    assert record["weight_load_seconds"] == round(1.0 - timer._seconds, 1)


def test_nested_calls_are_not_counted_twice(hub, tmp_path):
    """`snapshot_download` calls `hf_hub_download` per file. Without the depth
    counter the inner calls add their own seconds again, and the phase can
    report more download time than it lasted."""
    blob = tmp_path / "f.bin"
    blob.write_bytes(b"y" * 10)

    def inner(*_a, **_k):
        time.sleep(0.03)
        return str(blob)

    def outer(*_a, **_k):
        import huggingface_hub
        for _ in range(3):
            huggingface_hub.hf_hub_download()
        return str(tmp_path)

    hub.hf_hub_download = inner
    hub.snapshot_download = outer

    with FetchTimer() as timer:
        import huggingface_hub
        huggingface_hub.snapshot_download(repo_id = "org/model")

    assert timer.calls == 4, "every call is counted"
    # Three 0.03s inner sleeps happen INSIDE the outer call, so the outer span
    # is about 0.09s. Double counting would land near 0.18s.
    assert timer._seconds < 0.15, f"nested seconds counted twice: {timer._seconds}"


def test_a_raising_download_still_restores_the_module(hub):
    def boom(*_a, **_k):
        raise RuntimeError("hub is down")

    hub.hf_hub_download = boom
    original = hub.hf_hub_download
    timer = FetchTimer().install()
    import huggingface_hub

    with pytest.raises(RuntimeError):
        huggingface_hub.hf_hub_download()
    timer.uninstall()
    assert huggingface_hub.hf_hub_download is original, "the wrapper outlived the timer"
    # The failed attempt is still time spent trying, so it counts.
    assert timer.calls == 1


def test_the_split_never_reports_a_negative_weight_load(hub, tmp_path):
    """The two clocks are the same clock, but rounding can still put the fetch a
    tenth past the phase, and a negative duration reads as a broken report."""
    blob = tmp_path / "f.bin"
    blob.write_bytes(b"z")

    def slow(*_a, **_k):
        time.sleep(0.2)
        return str(blob)

    hub.hf_hub_download = slow
    with FetchTimer() as timer:
        import huggingface_hub
        huggingface_hub.hf_hub_download()
    record = timer.record(0.05)
    assert record["weight_load_seconds"] == 0.0


def test_every_leg_ships_the_module():
    """A timer the payload cannot import measures nothing. Asserted against the
    registry rather than a hardcoded list, so a new leg cannot miss it."""
    sys.path.insert(0, str(ROOT / ".github" / "scripts"))
    from kaggle_t4_ci.legs import LEGS

    missing = sorted(n for n, leg in LEGS.items() if "phase_timers.py" not in leg.files)
    assert missing == [], f"legs that cannot import phase_timers: {missing}"


def test_the_payload_actually_calls_the_timer():
    """A module shipped and never used is coverage that does nothing."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert "from phase_timers import FetchTimer" in src
    assert "with FetchTimer() as fetch_timer:" in src
    assert '"load_phases": load_phases' in src

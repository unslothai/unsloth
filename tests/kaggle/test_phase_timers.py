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

        started = time.time()
        huggingface_hub.snapshot_download(repo_id = "org/model")
        elapsed = time.time() - started

    assert timer.calls == 4, "every call is counted"
    # Against the outer call's OWN elapsed time, not a fixed ceiling: the three
    # inner sleeps run inside that call, so double counting lands near 2x it
    # while a correct sum lands at 1x. A host that deschedules the runner
    # stretches both sides equally, where a constant bound would go red on
    # scheduling and report it as double counting.
    assert (
        timer._seconds <= elapsed * 1.5
    ), f"nested seconds counted twice: {timer._seconds} against {elapsed} elapsed"


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


def test_no_alias_of_a_hub_download_is_left_unwrapped(monkeypatch):
    """`transformers.utils.hub` does `from huggingface_hub import ...` at import
    time, so it holds its OWN reference and rebinding the public name leaves it
    untouched. `cached_files` calls that alias for a multi-file (sharded)
    checkpoint, which is the biggest download any leg does, so missing it moves
    the dominant fetch into `weight_load_seconds` while `patched` stays
    non-empty and the record still looks valid.

    Derived, not listed: the aliases are DISCOVERED by comparing each module's
    attributes against the originals before patching, so a module that starts
    holding one of these names is covered without editing this test, and
    dropping a target fails here rather than on hardware.
    """
    real_hub = types.ModuleType("huggingface_hub")
    real_hub.hf_hub_download = lambda *a, **k: ""
    real_hub.snapshot_download = lambda *a, **k: ""
    # Exactly what `from huggingface_hub import snapshot_download` produces.
    alias = types.ModuleType("transformers.utils.hub")
    alias.hf_hub_download = real_hub.hf_hub_download
    alias.snapshot_download = real_hub.snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", real_hub)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", alias)

    originals = {
        (name, attr): getattr(module, attr)
        for name, module in (("huggingface_hub", real_hub), ("transformers.utils.hub", alias))
        for attr in ("hf_hub_download", "snapshot_download")
        if callable(getattr(module, attr, None))
    }
    assert len(originals) == 4, "the fixture no longer models the alias"

    timer = FetchTimer().install()
    try:
        unwrapped = sorted(
            f"{name}.{attr}"
            for (name, attr), original in originals.items()
            if getattr(sys.modules[name], attr) is original
        )
    finally:
        timer.uninstall()
    assert unwrapped == [], f"these download entry points are never timed: {unwrapped}"

    for (name, attr), original in originals.items():
        assert getattr(sys.modules[name], attr) is original, f"{name}.{attr} outlived the timer"


def _render(report: dict) -> str:
    """The real renderer, loaded by path: `.github/scripts` is not a package."""
    import importlib.util

    path = ROOT / ".github" / "scripts" / "kaggle_t4_ci" / "report.py"
    spec = importlib.util.spec_from_file_location("_t4_report_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return "\n".join(module.render(report))


def test_the_job_summary_shows_the_split_it_was_added_to_answer():
    """A number that only reaches `launch_result.json` answers nobody: reading it
    means downloading the evidence artifact, which is not where anyone looks."""
    rendered = _render(
        {
            "label": "control",
            "load_phases": {
                "patched": ["huggingface_hub.hf_hub_download"],
                "fetch_seconds": 61.7,
                "fetch_mb": 12550.0,
                "fetch_mb_s": 203.4,
                "weight_load_seconds": 40.9,
                "total_seconds": 102.6,
            },
        }
    )
    assert "61.7" in rendered, "the fetch half is missing from the summary"
    assert "40.9" in rendered, "the weight-load half is missing from the summary"
    assert "203.4" in rendered, "the achieved rate is missing from the summary"


def test_an_unattached_timer_is_not_rendered_as_a_zero_second_fetch():
    """The failure the instrument was built around, at the reporting layer: a
    timer that never attached must not read as 'the download took no time'."""
    rendered = _render(
        {
            "label": "control",
            "load_phases": {
                "patched": [],
                "fetch_seconds": None,
                "fetch_mb": None,
                "weight_load_seconds": None,
                "total_seconds": 102.6,
            },
        }
    )
    assert "never attached" in rendered
    assert "= fetch" not in rendered, "a split was rendered from a timer that measured nothing"


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

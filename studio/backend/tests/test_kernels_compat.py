# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""kernels >= 0.15 next to transformers < 5.13 breaks `import transformers.activations` at import time."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types

import pytest

import utils.kernels_compat as kc

_MARKER_ERR = b"ValueError: Either a revision or a version must be specified.\n"


class _Logger:
    def __init__(self):
        self.warnings: list = []
        self.infos: list = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def debug(self, msg, *args):
        pass


class _Run:
    """Recorded subprocess.run: pip calls succeed; the import check answers from ``import_ok``."""

    def __init__(self, import_ok):
        self.calls: list = []
        self.import_ok = list(import_ok)

    def __call__(self, cmd, **kwargs):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[1:3] == ["-c", kc._VERIFY_CODE]:
            ok = self.import_ok.pop(0) if len(self.import_ok) > 1 else self.import_ok[0]
            return types.SimpleNamespace(
                returncode = 0 if ok else 1, stderr = b"" if ok else _MARKER_ERR
            )
        return types.SimpleNamespace(returncode = 0, stderr = b"")

    def pip(self):
        return [c for c in self.calls if "pip" in c]


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.setattr(kc, "_healed", False)
    monkeypatch.setattr(kc, "_hidden_reason", None)
    monkeypatch.setattr(kc.importlib, "invalidate_caches", lambda: None)
    monkeypatch.delitem(sys.modules, "kernels", raising = False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    yield
    if sys.modules.get("kernels", object()) is None:
        del sys.modules["kernels"]


def _dists(monkeypatch, **versions):
    table = {k.replace("_", "-") if k != "huggingface_hub" else k: v for k, v in versions.items()}
    monkeypatch.setattr(kc, "_dist_version", lambda name: table.get(name))
    return table


@pytest.mark.parametrize(
    "tf, hub, spec",
    [
        ("5.5.0", "1.33.0", "kernels==0.12.3"),
        ("5.8.0", None, "kernels==0.12.3"),
        ("5.12.1", "1.33.0", "kernels==0.12.3"),
        ("5.13.0rc0", "1.33.0", "kernels==0.12.3"),
        ("5.13.0.dev0", "1.33.0", "kernels==0.12.3"),
        ("5.13.0", "1.33.0", "kernels>=0.12.3"),
        ("5.17.0", "1.33.0", "kernels>=0.12.3"),
        ("6.0.0", None, "kernels>=0.12.3"),
        (None, None, "kernels==0.12.3"),
        ("garbage", None, "kernels==0.12.3"),
        ("5.17.0", "1.5.0", "kernels==0.12.3"),
    ],
)
def test_install_spec_follows_transformers(tf, hub, spec):
    assert kc.kernels_install_spec(tf, hub) == spec


@pytest.mark.parametrize(
    "kernels, tf, conflict",
    [
        ("0.18.2", "5.5.0", True),
        ("0.15.1", "5.12.1", True),
        ("0.14.1", "5.5.0", False),
        ("0.12.3", "5.5.0", False),
        ("0.18.2", "5.13.0", False),
        ("0.18.2", "5.17.0", False),
        (None, "5.5.0", False),
        ("0.18.2", None, False),
    ],
)
def test_conflict_prediction_matches_the_measured_matrix(kernels, tf, conflict):
    assert kc.predicted_conflict(kernels, tf) is conflict


def test_on_demand_install_pins_kernels_for_old_transformers(monkeypatch):
    import core.inference.diffusion_attention as att
    import importlib.util

    _dists(monkeypatch, transformers = "5.5.0", huggingface_hub = "1.33.0")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    monkeypatch.setattr(att, "_INSTALL_ATTEMPTED", set())
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    refreshed = []
    monkeypatch.setattr(
        att, "_refresh_diffusers_kernels_version", lambda log = None: refreshed.append(1)
    )
    run = _Run([True])
    monkeypatch.setattr(subprocess, "run", run)
    assert att._ensure_backend_package("sage", _Logger()) is None
    assert run.pip()[0][-1] == "kernels==0.12.3"
    assert run.calls[1][1:] == ["-c", "import transformers.activations"]
    assert refreshed == [1]
    assert "kernels" not in sys.modules


def test_failed_import_check_rolls_the_install_back_and_hides_kernels(monkeypatch):
    import core.inference.diffusion_attention as att
    import importlib.util

    table = _dists(monkeypatch, transformers = "5.5.0")
    monkeypatch.setattr(att, "_dist_version", lambda name: table.get(name))
    monkeypatch.setattr(att, "_INSTALL_ATTEMPTED", set())
    monkeypatch.setattr(att, "_kernels_hub_compatible", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    refreshed = []
    monkeypatch.setattr(
        att, "_refresh_diffusers_kernels_version", lambda log = None: refreshed.append(1)
    )

    def run(cmd, **kw):
        out = _Run([False])(cmd, **kw)
        calls.append(list(cmd))
        if "install" in cmd:
            table["kernels"] = "0.18.2"
        return out

    calls: list = []
    monkeypatch.setattr(subprocess, "run", run)
    log = _Logger()
    assert att._ensure_backend_package("sage", log) is None
    assert calls[-1][-3:] == ["uninstall", "-y", "kernels"]
    assert sys.modules["kernels"] is None
    assert refreshed == []
    assert any("removed it again" in w and "default attention backend" in w for w in log.warnings)
    # The hidden package is reported with the real reason on the next request.
    assert "breaks `import transformers.activations`" in att._ensure_backend_package("sage", log)
    with pytest.raises(ImportError):
        import kernels  # noqa: F401


def test_rollback_reinstalls_the_previous_version(monkeypatch):
    run = _Run([False])
    monkeypatch.setattr(subprocess, "run", run)
    _dists(monkeypatch, transformers = "5.5.0", kernels = "0.18.2")
    assert kc.check_kernels_install("0.11.7", _Logger()) is False
    assert run.calls[-1][-1] == "kernels==0.11.7"


def test_unconfirmed_import_check_keeps_the_install(monkeypatch):
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", run)
    _dists(monkeypatch, transformers = "5.5.0", kernels = "0.14.1")
    log = _Logger()
    assert kc.check_kernels_install(None, log) is True
    assert "kernels" not in sys.modules and log.warnings


def test_heal_is_metadata_only_without_a_conflict(monkeypatch):
    run = _Run([True])
    monkeypatch.setattr(subprocess, "run", run)
    _dists(monkeypatch, transformers = "5.5.0", kernels = "0.12.3")
    assert kc.heal_kernels_conflict(_Logger()) is None
    assert run.calls == []


def test_heal_repairs_a_broken_install(monkeypatch):
    run = _Run([False, True])
    monkeypatch.setattr(subprocess, "run", run)
    _dists(monkeypatch, transformers = "5.5.0", kernels = "0.18.2", huggingface_hub = "1.33.0")
    log = _Logger()
    assert kc.heal_kernels_conflict(log) == "repaired: kernels==0.12.3"
    assert run.pip()[0][-1] == "kernels==0.12.3" and "--no-deps" in run.pip()[0]
    assert "kernels" not in sys.modules
    assert any("reinstalled" in w for w in log.warnings)
    # Once per process.
    assert kc.heal_kernels_conflict(log) is None and len(run.calls) == 3


def test_heal_hides_kernels_when_installs_are_off(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "0")
    run = _Run([False])
    monkeypatch.setattr(subprocess, "run", run)
    _dists(monkeypatch, transformers = "5.5.0", kernels = "0.18.2")
    log = _Logger()
    assert kc.heal_kernels_conflict(log).startswith("hidden:")
    assert run.pip() == []
    assert sys.modules["kernels"] is None
    assert any(
        "hub kernels are disabled" in w and "pip install --no-deps" in w for w in log.warnings
    )


def test_heal_hides_kernels_already_imported_even_after_a_repair(monkeypatch):
    monkeypatch.setitem(sys.modules, "kernels", types.ModuleType("kernels"))
    run = _Run([False, True])
    monkeypatch.setattr(subprocess, "run", run)
    _dists(monkeypatch, transformers = "5.5.0", kernels = "0.18.2")
    log = _Logger()
    assert kc.heal_kernels_conflict(log).startswith("hidden:")
    assert run.pip() and sys.modules["kernels"] is None
    assert any("after Unsloth restarts" in w for w in log.warnings)


def test_heal_leaves_a_working_pair_alone(monkeypatch):
    run = _Run([True])
    monkeypatch.setattr(subprocess, "run", run)
    _dists(monkeypatch, transformers = "5.5.0", kernels = "0.18.2")
    assert kc.heal_kernels_conflict(_Logger()) is None
    assert run.pip() == [] and "kernels" not in sys.modules


def test_heal_runs_before_every_diffusers_import_window(monkeypatch):
    import utils.torch_warmup as tw

    seen = []
    monkeypatch.setattr(kc, "heal_kernels_conflict", lambda log = None: seen.append(log))
    with tw.background_media_import():
        pass
    monkeypatch.setattr(tw, "ensure_dynamo_imported", lambda **kw: True)
    monkeypatch.setattr(tw, "_prime_cuda_driver", lambda: None)
    monkeypatch.setattr(tw, "claim_media_import_window", lambda log = None: True)
    tw.close_dynamo_import_window(_Logger())
    assert len(seen) == 2


def test_hidden_kernels_lets_real_transformers_import(tmp_path):
    """A kernels whose LayerRepository raises like 0.15+ does breaks the real transformers import; hidden, it imports."""
    pytest.importorskip("transformers")
    fake = tmp_path / "kernels"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        textwrap.dedent(
            """
            class _Any:
                def __init__(self, *a, **k):
                    raise ValueError("Either a revision or a version must be specified.")
            def __getattr__(name):
                return _Any
            """
        )
    )
    from transformers.utils import import_utils

    # A version this transformers treats as available, so hub_kernels really imports it.
    ver = getattr(import_utils, "KERNELS_MIN_VERSION", "0.18.2")
    meta = tmp_path / f"kernels-{ver}.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text(f"Metadata-Version: 2.1\nName: kernels\nVersion: {ver}\n")
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH = os.pathsep.join([str(tmp_path), backend]))
    code = "import transformers.activations"
    broken = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, timeout = 300)
    assert broken.returncode != 0 and kc.KERNELS_CONFLICT_MARKER.encode() in broken.stderr
    hidden = subprocess.run(
        [
            sys.executable,
            "-c",
            "import utils.kernels_compat as kc; kc.hide_kernels('test'); " + code,
        ],
        env = env,
        capture_output = True,
        timeout = 300,
    )
    assert hidden.returncode == 0, hidden.stderr.decode()[-800:]


def test_a_concurrent_caller_waits_for_the_repair_in_flight(monkeypatch):
    import threading

    monkeypatch.setattr(kc, "_healed", False)
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow_heal(logger):
        calls.append(1)
        started.set()
        release.wait(5)
        return "repaired"

    monkeypatch.setattr(kc, "_heal", slow_heal)
    first = threading.Thread(target = kc.heal_kernels_conflict)
    first.start()
    assert started.wait(5)
    second_done = threading.Event()
    second = threading.Thread(target = lambda: (kc.heal_kernels_conflict(), second_done.set()))
    second.start()
    # The second caller must not return (and import diffusers) while the repair is still running.
    assert not second_done.wait(0.3)
    release.set()
    first.join(5)
    second.join(5)
    assert second_done.is_set() and calls == [1]

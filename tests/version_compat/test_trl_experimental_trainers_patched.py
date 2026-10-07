# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL 0.29+ moved ORPO, CPO, GKD, PPO, Online DPO, Nash-MD, XPO, BCO and PRM from trl.trainer to trl.experimental.

Each check imports unsloth in a fresh CPU-spoofed interpreter, since trainer patching is process-wide and import-order
sensitive.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


if importlib.util.find_spec("torch") is None or importlib.util.find_spec("trl") is None:
    pytest.skip("torch and trl are needed to patch the trainers", allow_module_level = True)

_TESTS_DIR = Path(__file__).resolve().parents[1]

_MOVED = [
    ("bco", "BCO"),
    ("cpo", "CPO"),
    ("gkd", "GKD"),
    ("nash_md", "NashMD"),
    ("online_dpo", "OnlineDPO"),
    ("orpo", "ORPO"),
    ("ppo", "PPO"),
    ("prm", "PRM"),
    ("xpo", "XPO"),
]

_CHILD = r"""
import importlib, importlib.util, json, os, sys, warnings
sys.path.insert(0, TESTS_DIR)
for k, v in (("UNSLOTH_COMPILE_DISABLE", "1"), ("TORCHDYNAMO_DISABLE", "1"), ("TORCH_COMPILE_DISABLE", "1"),
             ("TRL_EXPERIMENTAL_SILENCE", "1")):
    os.environ.setdefault(k, v)
for name in PRE_IMPORT:
    importlib.import_module(name)
# The same CPU-runner harness pytest loads for every test, then the aggressive spoof, as the fake-train tests do.
spec = importlib.util.spec_from_file_location("_unsloth_tests_conftest", os.path.join(TESTS_DIR, "conftest.py"))
spec.loader.exec_module(importlib.util.module_from_spec(spec))
import _zoo_aggressive_cuda_spoof as spoof
spoof.apply()
import unsloth, trl, trl.trainer
in_trainer = set(dir(trl.trainer))
out = {"unsloth": unsloth.__file__, "trl": trl.__version__, "moved": {}}
for pkg, prefix in MOVED:
    if f"{pkg}_trainer" in in_trainer or importlib.util.find_spec("trl.experimental") is None:
        continue
    try:
        if importlib.util.find_spec(f"trl.experimental.{pkg}") is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            package = importlib.import_module(f"trl.experimental.{pkg}")
            module = importlib.import_module(f"trl.experimental.{pkg}.{pkg}_trainer")
    except Exception as e:
        out["moved"][pkg] = {"import_error": f"{type(e).__name__}: {e}"}
        continue
    trainer = getattr(package, f"{prefix}Trainer")
    config = getattr(package, f"{prefix}Config")
    out["moved"][pkg] = {
        "trainer": trainer.__name__,
        "module_trainer": getattr(module, f"{prefix}Trainer").__name__,
        "config_has_unsloth_fields": "unsloth_num_chunks" in getattr(config, "__dataclass_fields__", {}),
        "mro": [f"{c.__module__}.{c.__name__}" for c in trainer.__mro__],
        "leaked": [x for x in (f"{prefix}Trainer", f"{prefix}Config") if x in vars(trl) or x in vars(trl.trainer)]
        + ([f"trl.trainer.{pkg}_trainer"] if f"{pkg}_trainer" in vars(trl.trainer) else []),
    }
print("RESULT " + json.dumps(out))
"""


def _run(pre_import = ()):
    code = (
        f"TESTS_DIR = {str(_TESTS_DIR)!r}\nMOVED = {_MOVED!r}\nPRE_IMPORT = {list(pre_import)!r}\n"
        + _CHILD
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(lines[-1][len("RESULT ") :])
    moved = {k: v for k, v in result["moved"].items() if "import_error" not in v}
    if not moved:
        pytest.skip(f"TRL {result['trl']} keeps these trainers in trl.trainer: {result['moved']}")
    return moved


@pytest.fixture(scope = "module")
def after_unsloth():
    return _run()


def test_experimental_trainers_and_configs_are_patched(after_unsloth):
    for pkg, got in after_unsloth.items():
        prefix = dict(_MOVED)[pkg]
        assert got["trainer"] == f"Unsloth{prefix}Trainer", (pkg, got)
        assert got["module_trainer"] == f"Unsloth{prefix}Trainer", (pkg, got)
        assert got["config_has_unsloth_fields"], (pkg, got)


def test_patching_leaves_no_trl_trainer_aliases(after_unsloth):
    for pkg, got in after_unsloth.items():
        assert got["leaked"] == [], (pkg, got)


def test_gkd_keeps_trls_own_sft_base_whatever_the_import_order(after_unsloth):
    """GKDTrainer subclasses SFTTrainer; importing it after unsloth must not pick up UnslothSFTTrainer as its base."""
    if "gkd" not in after_unsloth:
        pytest.skip("no trl.experimental.gkd on this TRL")
    before = _run(pre_import = ["trl.experimental.gkd"])["gkd"]
    for got in (after_unsloth["gkd"], before):
        assert got["trainer"] == "UnslothGKDTrainer", got
        assert "trl.trainer.sft_trainer.SFTTrainer" in got["mro"], got
        assert not any(c.endswith("SFTTrainer") and "Unsloth" in c for c in got["mro"]), got
    assert after_unsloth["gkd"]["mro"][2:] == before["mro"][2:]

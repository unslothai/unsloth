# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL 0.29+ moved ORPO, CPO, GKD, PPO, Online DPO, Nash-MD, XPO, BCO and PRM from trl.trainer to trl.experimental, and the
served-teacher, GOLD and SDFT distillation trainers only ever lived there.

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

# (trl.experimental subpackage, class prefix) for every trainer that used to live in trl.trainer.
_MOVED = [
    ("bco", "BCO"),
    ("cpo", "CPO"),
    ("gkd", "GKD"),
    ("gold", "GOLD"),
    ("nash_md", "NashMD"),
    ("online_dpo", "OnlineDPO"),
    ("orpo", "ORPO"),
    ("ppo", "PPO"),
    ("prm", "PRM"),
    ("sdft", "SDFT"),
    ("server_distillation", "ServerDistillation"),
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
import inspect, re
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
        "num_generations": [
            getattr(inspect.signature(config.__init__).parameters.get("num_generations"), "default", None),
            getattr(getattr(config.__mro__[1], "__dataclass_fields__", {}).get("num_generations"), "default", None),
        ],
        "relative_imports": re.findall(r"^\s*from \.+[\w.]* import .*$", inspect.getsource(sys.modules[trainer.__module__]), re.M),
        "leaked": [x for x in (f"{prefix}Trainer", f"{prefix}Config") if x in vars(trl) or x in vars(trl.trainer)]
        + ([f"trl.trainer.{pkg}_trainer"] if f"{pkg}_trainer" in vars(trl.trainer) else []),
    }
def _config_probe(pkg, prefix, **kwargs):
    try:
        cfg = getattr(importlib.import_module(f"trl.experimental.{pkg}"), f"{prefix}Config")(
            output_dir = "unused", report_to = "none", **kwargs
        )
        return {"per_device_train_batch_size": cfg.per_device_train_batch_size}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
if "gold" in out["moved"]:
    out["gold_config"] = _config_probe(
        "gold", "GOLD", num_generations = 8, per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2, lmbda = 1.0,
    )
if "sdft" in out["moved"]:
    out["sdft_eval_config"] = _config_probe("sdft", "SDFT", eval_strategy = "steps", eval_steps = 1)
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
    for pkg, key in (("gold", "gold_config"), ("sdft", "sdft_eval_config")):
        if pkg in moved:
            moved[pkg]["config_probe"] = result.get(key)
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


def test_generated_trainers_have_no_relative_imports(after_unsloth):
    """The generated module has no package, so `from ...generation.vllm_client import VLLMClient` would fail in __init__."""
    for pkg, got in after_unsloth.items():
        assert got["relative_imports"] == [], (pkg, got["relative_imports"])


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


@pytest.mark.parametrize(
    "pkg, parent", [("server_distillation", "DistillationTrainer"), ("gold", "SFTTrainer")]
)
def test_distillation_trainers_keep_trls_own_base_whatever_the_import_order(
    after_unsloth, pkg, parent
):
    """ServerDistillationTrainer and GOLDTrainer subclass trainers Unsloth also patches; the import order must not decide their base."""
    if pkg not in after_unsloth:
        pytest.skip(f"no trl.experimental.{pkg} on this TRL")
    before = _run(pre_import = [f"trl.experimental.{pkg}"])[pkg]
    for got in (after_unsloth[pkg], before):
        assert any(
            c.startswith("trl.trainer.") and c.endswith(f".{parent}") for c in got["mro"]
        ), got
        assert not any(c.endswith(parent) and "Unsloth" in c for c in got["mro"][2:]), got
    assert after_unsloth[pkg]["mro"][2:] == before["mro"][2:]


def test_gold_keeps_trls_num_generations_default(after_unsloth):
    """GRPO's num_generations=8 default must not reach GOLD, where it is samples per prompt and would also lift the batch size to 8."""
    if "gold" not in after_unsloth:
        pytest.skip("no trl.experimental.gold on this TRL")
    patched, pristine = after_unsloth["gold"]["num_generations"]
    assert patched == pristine, after_unsloth["gold"]["num_generations"]


def test_gold_keeps_a_valid_batch_size(after_unsloth):
    """TRL accepts num_generations=8 with batch 4 x grad accum 2; GRPO's multiple-of-num_generations rewrite must not double it."""
    got = after_unsloth.get("gold")
    if got is None:
        pytest.skip("no trl.experimental.gold on this TRL")
    assert got["config_probe"] == {"per_device_train_batch_size": 4}, got["config_probe"]


def test_sdft_config_with_evaluation_still_builds(after_unsloth):
    """SDFT needs the eval batch divisible by num_generations (8); Unsloth's eval batch default of 4 made this raise."""
    got = after_unsloth.get("sdft")
    if got is None:
        pytest.skip("no trl.experimental.sdft on this TRL")
    assert "error" not in got["config_probe"], got["config_probe"]

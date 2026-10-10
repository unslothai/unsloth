# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth#3551: `fast_inference = True` (vLLM) is refused under FSDP2, before vLLM loads."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from real_accelerator import has_real_cuda
from unsloth.models import loader_utils
from unsloth.models.loader_utils import fsdp2_requested, raise_if_fast_inference_under_fsdp2


@pytest.fixture
def no_launcher(monkeypatch):
    monkeypatch.delenv("FSDP_VERSION", raising = False)
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setattr(state.AcceleratorState, "_shared_state", {})
    return monkeypatch


@pytest.mark.parametrize("value, fsdp2", [("2", True), ("1", False), ("0", False)])
def test_the_launcher_env_names_fsdp2(no_launcher, value, fsdp2):
    no_launcher.setenv("FSDP_VERSION", value)
    assert fsdp2_requested() is fsdp2


def test_an_accelerator_with_an_fsdp2_plugin(no_launcher):
    from accelerate.state import AcceleratorState
    no_launcher.setattr(
        AcceleratorState, "_shared_state", {"fsdp_plugin": SimpleNamespace(fsdp_version = 2)}
    )
    assert fsdp2_requested()


def test_vllm_under_fsdp2_is_refused_with_the_way_out(no_launcher):
    no_launcher.setenv("FSDP_VERSION", "2")
    with pytest.raises(NotImplementedError, match = "fast_inference = False"):
        raise_if_fast_inference_under_fsdp2(True)


@pytest.mark.parametrize("fsdp_version, fast_inference", [("2", False), ("1", True), (None, True)])
def test_everything_else_loads(no_launcher, fsdp_version, fast_inference):
    if fsdp_version is not None:
        no_launcher.setenv("FSDP_VERSION", fsdp_version)
    raise_if_fast_inference_under_fsdp2(fast_inference)


def test_both_loaders_refuse_after_the_dgx_spark_fallback():
    """FastLanguageModel and FastModel: GB10 still falls back to native inference before the FSDP2 check."""
    source = Path(loader_utils.__file__).with_name("loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    checked = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
        ):
            guards = [
                n.lineno
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "raise_if_fast_inference_under_fsdp2"
            ]
            spark = [
                n.lineno
                for n in ast.walk(fn)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and "DGX Spark detected" in n.value
            ]
            if spark:
                assert len(guards) == 1 and guards[0] > max(spark), (cls.name, guards, spark)
                checked.append(cls.name)
    assert {"FastLanguageModel", "FastModel"} <= set(checked), checked


_TRAINER_PROBE = textwrap.dedent(
    """
    import os
    from types import SimpleNamespace
    import torch
    from unsloth import FastLanguageModel  # noqa: F401, patches the TRL trainers
    from trl import GRPOConfig, GRPOTrainer

    class FastInferenceModel(torch.nn.Module):
        vllm_engine = object()
        max_seq_length = 64

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(dtype = torch.bfloat16, torch_dtype = torch.bfloat16)
            self.warnings_issued = {}

        def for_training(self, *args, **kwargs):
            pass

    def build(**kwargs):
        try:
            args = GRPOConfig(output_dir = os.environ["PROBE_OUT"], **kwargs)
            GRPOTrainer(model = FastInferenceModel(), reward_funcs = [], args = args)
        except NotImplementedError as e:
            return "REFUSED" if "FSDP2" in str(e) else "OTHER"
        except Exception:
            return "OTHER"
        return "BUILT"

    print("PROBE", build(fsdp = "full_shard", fsdp_config = {"fsdp_version": 2}),
          build(fsdp = "full_shard", fsdp_config = {"fsdp_version": 1}), build())
    """
)


@pytest.mark.skipif(
    not has_real_cuda(), reason = "the probe imports unsloth outside the conftest CPU spoof"
)
def test_trainer_args_fsdp2_is_refused_when_the_trainer_is_built(tmp_path):
    """torchrun + TrainingArguments(fsdp = ..., fsdp_config = {"fsdp_version": 2}): nothing said FSDP2 at load time."""
    env = {k: v for k, v in os.environ.items() if k not in ("FSDP_VERSION", "ACCELERATE_USE_FSDP")}
    env.update(
        UNSLOTH_COMPILE_LOCATION = str(tmp_path / "cache"),
        PROBE_OUT = str(tmp_path / "out"),
        HF_HUB_OFFLINE = "1",
    )
    run = subprocess.run(
        [sys.executable, "-c", _TRAINER_PROBE], env = env, capture_output = True, text = True, timeout = 900
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith("PROBE ")]
    assert lines, f"probe failed:\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}"
    fsdp2, fsdp1, plain = lines[-1].split()[1:]
    assert fsdp2 == "REFUSED"
    assert fsdp1 != "REFUSED" and plain != "REFUSED"

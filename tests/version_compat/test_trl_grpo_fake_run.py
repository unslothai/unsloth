# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Fake-CUDA GRPO patch run against the *installed* TRL (CPU-only, no training).

The static symbol/source-string canaries (test_trl_grpo_pinned_symbols.py)
grep raw TRL source; they never execute unsloth's transforms. This test drives
the real pipeline: under the aggressive CUDA spoof it imports unsloth and calls
`_patch_trl_rl_trainers_impl`, which reads the installed GRPOTrainer via
inspect.getsource, applies every rl.py/rl_replacements.py rewrite, and compiles
the result into an UnslothGRPOTrainer. A structural TRL change that slips past
the greps (e.g. TRL 1.7.0's 2->3-tuple return arity, or a restructured PEFT
ref-adapter block) surfaces here as a transform error, a broken generated
source, or a violated contract -- with no GPU and no training run.

Meant to run in CI against `trl==latest` and `trl @ main` (see
version-compat-ci.yml). The tests/conftest.py harness pre-loads device_type
with DEVICE_COUNT=0 so unsloth's kernel init takes the CPU-safe path.
"""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest


# daily-fresh-fetch collects tests/version_compat/ with only pytest installed;
# the spoof and the rest of this module need the real torch runtime. Skip the
# whole module cleanly when torch is absent rather than crashing collection.
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed; fake-run needs the real runtime", allow_module_level = True)

# Apply the spoof BEFORE any unsloth-touching import (mirrors
# tests/vllm_compat/test_extended_module_imports.py).
_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


def _stub_module(name: str, attrs: dict | None = None) -> None:
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name = name, loader = None, origin = "<test stub>")
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m


_stub_module("torchcodec")


def _trl_version():
    import trl
    from packaging.version import Version
    return Version(trl.__version__.split("+")[0])


def _patch_grpo_and_get_source() -> str:
    """Run the GRPO patcher against the installed TRL and return the generated
    UnslothGRPOTrainer source. Calls the impl (not the try/except wrapper) so a
    transform/compile regression surfaces as a hard error instead of a silent
    no-op."""
    import trl.trainer.grpo_trainer as _g

    from unsloth.models import rl as _rl

    _rl._patch_trl_rl_trainers_impl("grpo_trainer")
    patched = _g.GRPOTrainer
    assert patched.__name__ == "UnslothGRPOTrainer", (
        f"GRPO patch silently no-oped: trl.trainer.grpo_trainer.GRPOTrainer is "
        f"{patched.__name__!r}, expected 'UnslothGRPOTrainer' (transform failed "
        f"or dispatch key drifted on this TRL)"
    )
    # The transformed body (__init__ rewrites, injected per-token-logps) lives in
    # the generated module's `_UnslothGRPOTrainer` base + module-level funcs, not
    # the thin UnslothGRPOTrainer subclass -- read the whole generated module.
    mod = inspect.getmodule(patched)
    return inspect.getsource(mod) if mod is not None else inspect.getsource(patched)


@pytest.fixture(scope = "module")
def generated_grpo_source():
    if importlib.util.find_spec("unsloth") is None:
        pytest.skip("unsloth not installed")
    if importlib.util.find_spec("trl") is None:
        pytest.skip("trl not installed")
    # Do NOT swallow import errors: unsloth is installed here, so a failing
    # `import unsloth` is exactly the import-time TRL/transformers drift this
    # canary must surface as a failure, not a skip.
    import unsloth  # noqa: F401  -- _gpu_init bootstrap under spoof

    return _patch_grpo_and_get_source()


def test_grpo_patch_generates_valid_source(generated_grpo_source):
    """The generated UnslothGRPOTrainer must be syntactically valid Python."""
    ast.parse(generated_grpo_source)


def test_grpo_patch_aux_fail_fast_injected(generated_grpo_source):
    """TRL >= 1.7.0: rl.py injects a fail-fast for the unsupported MoE router
    aux-loss opt-in right after `self.aux_loss_enabled = ...`."""
    from packaging.version import Version

    if _trl_version() < Version("1.7.0"):
        pytest.skip("aux_loss_enabled / router_aux_loss_coef are TRL >= 1.7.0")
    assert "does not compute the MoE router auxiliary loss" in generated_grpo_source, (
        "aux fail-fast raise missing from generated trainer; rl.py's "
        "aux_loss_enabled .replace() anchor did not match this TRL"
    )


def test_grpo_patch_three_tuple_return(generated_grpo_source):
    """TRL >= 1.7.0 call sites unpack a 3-tuple from
    _get_per_token_logps_and_entropies; the injected replacement must return
    (logps, entropies, aux_loss)."""
    from packaging.version import Version
    if _trl_version() >= Version("1.7.0"):
        assert "return logprobs.detach(), entropies, aux_loss" in generated_grpo_source, (
            "3-tuple per-token-logps return missing; the arity version-gate in "
            "rl_replacements.py did not emit the >=1.7.0 form"
        )
    else:
        assert (
            "return logprobs.detach(), entropies, aux_loss" not in generated_grpo_source
        ), "2-tuple TRL got the 3-tuple return; arity gate mis-fired"


def test_grpo_patch_preserves_grad_checkpointing_block(generated_grpo_source):
    """The tightened PR #6904 PEFT regex must remove only the ref-adapter init,
    not the following enable_input_require_grads gradient-checkpointing block."""
    from packaging.version import Version

    if _trl_version() < Version("1.7.0"):
        pytest.skip("ref-adapter elif block is the TRL >= 1.7.0 shape")
    assert "enable_input_require_grads" in generated_grpo_source, (
        "gradient-checkpointing enable_input_require_grads() block was swallowed "
        "by the PEFT-removal regex (over-reach regression)"
    )


def test_grpo_patch_neutralizes_ref_adapter_and_qlora_cast(generated_grpo_source):
    """TRL >= 1.7.0: the ref-adapter copy and the hardcoded QLoRA bf16 cast must
    both be gone from the generated trainer."""
    from packaging.version import Version

    if _trl_version() < Version("1.7.0"):
        pytest.skip("targets the TRL >= 1.7.0 PEFT / _is_quantized_model shapes")
    assert (
        "ref_param.data.copy_(param.data)" not in generated_grpo_source
    ), "TRL's PEFT ref-adapter init survived; rl.py peft_pattern re.sub no-oped"
    assert (
        "if _is_quantized_model:" not in generated_grpo_source
    ), "TRL's hardcoded QLoRA bf16 cast survived; rl.py neutralization no-oped"


# SFT / DPO: the same source-transform patcher runs on them (a fake patch run,
# no training), so a structural TRL change can break generation. Assert the patch
# produces a valid, importable Unsloth trainer AND that the shared QLoRA
# `_is_quantized_model` bf16 cast is neutralized (TRL 1.7's spelling), which the
# patcher applies to every trainer. Catches "and or others" beyond GRPO.


def _patch_and_get_source(trainer_file: str, trainer_cls: str) -> str:
    if importlib.util.find_spec("unsloth") is None or importlib.util.find_spec("trl") is None:
        pytest.skip("unsloth or trl not installed")
    # Let a real import failure fail the test (import-time drift is the target).
    import unsloth  # noqa: F401
    import trl.trainer  # noqa: F401

    from unsloth.models import rl as _rl

    _rl._patch_trl_rl_trainers_impl(trainer_file)
    mod = importlib.import_module(f"trl.trainer.{trainer_file}")
    patched = getattr(mod, trainer_cls)
    assert patched.__name__ == f"Unsloth{trainer_cls}", (
        f"{trainer_cls} patch silently no-oped on this TRL "
        f"(got {patched.__name__!r}); source-transform dispatch drifted"
    )
    gen = inspect.getmodule(patched)
    src = inspect.getsource(gen) if gen is not None else inspect.getsource(patched)
    ast.parse(src)
    return src


def _assert_quantized_cast_neutralized(src: str, trainer_cls: str) -> None:
    from packaging.version import Version
    if _trl_version() < Version("1.7.0"):
        pytest.skip("pre-1.7.0 spells the QLoRA cast differently (is_loaded_in_4bit)")
    assert "if _is_quantized_model:" not in src, (
        f"{trainer_cls}: TRL's hardcoded QLoRA bf16 cast survived; the shared "
        f"rl.py `if _is_quantized_model:` -> `if False:` neutralization no-oped"
    )


def test_sft_patch_generates_valid_source():
    src = _patch_and_get_source("sft_trainer", "SFTTrainer")
    _assert_quantized_cast_neutralized(src, "SFTTrainer")


def test_dpo_patch_generates_valid_source():
    src = _patch_and_get_source("dpo_trainer", "DPOTrainer")
    _assert_quantized_cast_neutralized(src, "DPOTrainer")


# The installed TRL in CI is always >= 1.7.0, so the < 1.7.0 return-arity
# downgrade is never exercised by the fake-run above. Lock both arities by
# monkeypatching rl_replacements.trl_version and re-generating the injected
# _get_per_token_logps_and_entropies source directly (no TRL install needed).
def test_per_token_logps_arity_gate_both_directions(monkeypatch):
    if importlib.util.find_spec("unsloth") is None:
        pytest.skip("unsloth not installed")
    import unsloth  # noqa: F401
    from packaging.version import Version

    from unsloth.models import rl_replacements as _rlr

    gate = _rlr.grpo_trainer__get_per_token_logps_and_entropies

    # >= 1.7.0: 3-tuple return kept.
    monkeypatch.setattr(_rlr, "trl_version", Version("1.7.0"), raising = False)
    src_new = gate("_get_per_token_logps_and_entropies", None)
    assert (
        "return logprobs.detach(), entropies, aux_loss" in src_new
    ), "3-tuple return missing for TRL >= 1.7.0"

    # < 1.7.0: aux_loss element dropped -> 2-tuple. A no-op downgrade must raise
    # (fail loud), never silently ship a 3-tuple to older TRL.
    monkeypatch.setattr(_rlr, "trl_version", Version("1.6.0"), raising = False)
    src_old = gate("_get_per_token_logps_and_entropies", None)
    assert (
        "return logprobs.detach(), entropies  # logps, entropies" in src_old
    ), "2-tuple return missing for TRL < 1.7.0"
    assert (
        "entropies, aux_loss" not in src_old
    ), "aux_loss element still present in the TRL < 1.7.0 downgrade"

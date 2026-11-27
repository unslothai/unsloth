# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused LoRA kernels cannot read FSDP shard views, so patch_peft_model must decline them."""

from __future__ import annotations

import ast
import os
import types
from pathlib import Path

import pytest

from unsloth.models.loader_utils import fsdp_will_wrap

_LLAMA = Path(__file__).resolve().parent.parent / "unsloth" / "models" / "llama.py"

_FSDP_ENV = (
    "ACCELERATE_USE_FSDP",
    "FSDP_VERSION",
    "UNSLOTH_FORCE_FUSED_LORA",
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for key in _FSDP_ENV:
        monkeypatch.delenv(key, raising = False)


def test_a_plain_launch_keeps_the_fused_kernels(monkeypatch):
    """DDP and single-GPU launches export neither variable and keep the fast path."""
    assert fsdp_will_wrap() is False
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    assert fsdp_will_wrap() is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "ON"])
def test_accelerate_use_fsdp_declines(monkeypatch, value):
    """`accelerate launch` with an FSDP config exports the literal "true"."""
    monkeypatch.setenv("ACCELERATE_USE_FSDP", value)
    assert fsdp_will_wrap() is True


@pytest.mark.parametrize("value", ["false", "0", "", "no"])
def test_a_falsy_accelerate_use_fsdp_keeps_the_fused_kernels(monkeypatch, value):
    monkeypatch.setenv("ACCELERATE_USE_FSDP", value)
    assert fsdp_will_wrap() is False


@pytest.mark.parametrize("value,expected", [("1", True), ("2", True), ("0", False), ("", False)])
def test_fsdp_version_alone_is_enough(monkeypatch, value, expected):
    """torchrun users set it by hand; "0" is how a config says "not FSDP"."""
    monkeypatch.setenv("FSDP_VERSION", value)
    assert fsdp_will_wrap() is expected


def test_the_override_wins_over_every_signal(monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    monkeypatch.setenv("FSDP_VERSION", "2")
    monkeypatch.setenv("UNSLOTH_FORCE_FUSED_LORA", "1")
    assert fsdp_will_wrap() is False


def test_a_live_accelerator_state_is_consulted_first(monkeypatch):
    """A caller-built Accelerator carries no env, so its live state must be read."""
    accelerate_state = pytest.importorskip("accelerate.state")
    shared = accelerate_state.AcceleratorState._shared_state
    monkeypatch.setitem(shared, "distributed_type", "DistributedType.FSDP")
    assert fsdp_will_wrap() is True


def test_the_fused_lora_install_is_gated_on_the_probe():
    """The three fused installs share one `if`; splitting them must carry the gate."""
    source = _LLAMA.read_text(encoding = "utf-8")
    assert "fsdp_will_wrap" in source, "llama.py no longer probes for FSDP"

    tree = ast.parse(source)
    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "fused_lora_declined_for_fsdp" not in test:
            continue
        body = ast.unparse(node)
        if "apply_qkv" in body or "apply_o" in body or "_apply_lora_mlp" in body:
            guarded.append(test)
    assert guarded, (
        "no `if` guarding apply_qkv / apply_o / apply_lora_mlp consults "
        "fused_lora_declined_for_fsdp; under FSDP the fused kernels will read "
        "shard views again"
    )


def test_the_declined_message_names_the_override():
    from unsloth.models.llama import _fused_lora_skip_reason
    assert _fused_lora_skip_reason(0, "none") == ""
    assert "UNSLOTH_FORCE_FUSED_LORA" in _fused_lora_skip_reason(0, "none", fsdp = True)


def _fused_model(llama):
    import types

    import torch

    attn = torch.nn.Module()
    attn.apply_qkv = llama.apply_lora_qkv
    attn.apply_o = llama.apply_lora_o
    mlp = torch.nn.Linear(2, 2)
    mlp.forward = types.MethodType(llama.apply_lora_mlp_swiglu, mlp)
    tiled = torch.nn.Linear(2, 2)
    tiled._unsloth_forward = types.MethodType(llama.apply_lora_mlp_geglu_approx, tiled)
    root = torch.nn.Module()
    root.attn, root.mlp, root.tiled = attn, mlp, tiled
    return root


def test_a_trainer_side_fsdp_gets_peft_forwards_back():
    """`SFTConfig(fsdp=...)` exports no env: the Trainer undoes the fused installs once it knows."""
    import torch
    from unsloth.models import llama

    root = _fused_model(llama)
    assert llama._decline_fused_lora_for_fsdp(root) == 4
    assert root.attn.apply_qkv is llama.original_apply_qkv
    assert root.attn.apply_o is llama.original_apply_o
    assert "forward" not in root.mlp.__dict__
    assert root.tiled._unsloth_forward is torch.nn.Linear.forward
    assert llama._decline_fused_lora_for_fsdp(root) == 0


def test_the_trainer_side_decline_honours_the_override(monkeypatch):
    from unsloth.models import llama

    monkeypatch.setenv("UNSLOTH_FORCE_FUSED_LORA", "1")
    root = _fused_model(llama)
    assert llama._decline_fused_lora_for_fsdp(root) == 0
    assert root.attn.apply_qkv is llama.apply_lora_qkv


def test_the_trainer_init_wrapper_calls_the_decline():
    import inspect

    from unsloth.models import _utils

    source = inspect.getsource(_utils.patch_gradient_accumulation_fix)
    assert "is_fsdp_enabled" in source and "_decline_fused_lora_for_fsdp" in source


class _GroupedMLP(__import__("torch").nn.Module):
    def __init__(self):
        import torch
        super().__init__()
        self.up = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.up(x)


def _adopted_forward(self, x):
    return self.up(x) * 3


@pytest.mark.parametrize("tiled", [False, True, "after"])
def test_declining_restores_the_forward_before_the_fused_kernel(tiled):
    """Decline restores the prior instance forward, not the class forward."""
    import types

    import torch
    from unsloth.models import llama

    mlp = _GroupedMLP()
    mlp.forward = types.MethodType(_adopted_forward, mlp)
    registry = pytest.importorskip("unsloth_zoo.tiling_registry") if tiled else None
    if tiled is True:
        registry.install_module(mlp, "rows:2")
        assert mlp.__dict__["_unsloth_forward"].__func__ is _adopted_forward
    llama._install_fused_lora_mlp(mlp, llama.apply_lora_mlp_swiglu)
    if tiled == "after":
        registry.install_module(mlp, "rows:2")
    assert mlp in llama._fused_lora_modules(mlp)
    assert llama._decline_fused_lora_for_fsdp(mlp) == 1
    slot = "_unsloth_forward" if tiled else "forward"
    assert registry is None or registry.is_tiled(mlp)
    assert mlp.__dict__[slot].__func__ is _adopted_forward
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(mlp(x), mlp.up(x) * 3)
    assert llama._decline_fused_lora_for_fsdp(mlp) == 0


@pytest.mark.parametrize("tiled", [False, True])
def test_declining_a_plain_module_still_restores_the_class_forward(tiled):
    import types

    import torch
    from unsloth.models import llama

    mlp = _GroupedMLP()
    if tiled:
        registry = pytest.importorskip("unsloth_zoo.tiling_registry")
        registry.install_module(mlp, "rows:2")
    llama._install_fused_lora_mlp(mlp, llama.apply_lora_mlp_swiglu)
    llama._install_fused_lora_mlp(
        mlp, llama.apply_lora_mlp_swiglu
    )  # a second patch pass keeps the first record
    assert llama._decline_fused_lora_for_fsdp(mlp) == 1
    if tiled:
        assert mlp.__dict__["_unsloth_forward"] is _GroupedMLP.forward
    else:
        assert "forward" not in mlp.__dict__
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(mlp(x), mlp.up(x))


def test_tile_fuse_untile_decline_leaves_a_callable_module():
    """Decline after untile must not put the unbound class function in `forward`."""
    import torch
    from unsloth.models import llama

    registry = pytest.importorskip("unsloth_zoo.tiling_registry")
    mlp = _GroupedMLP()
    registry.install_module(mlp, "rows:2")
    llama._install_fused_lora_mlp(mlp, llama.apply_lora_mlp_swiglu)
    registry.uninstall(mlp)
    assert llama._decline_fused_lora_for_fsdp(mlp) == 1
    assert "forward" not in mlp.__dict__
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(mlp(x), mlp.up(x))


def _counting_hook():
    hooks = pytest.importorskip("accelerate.hooks")

    class _Count(hooks.ModelHook):
        calls = 0

        def pre_forward(self, module, *args, **kwargs):
            type(self).calls += 1
            return args, kwargs

    return hooks, _Count


def test_a_kernel_behind_an_adopted_accelerate_hook_is_found_and_declined():
    """Kernel moved into an adopted hook's `_old_forward` is still detected and declined; the hook keeps running."""
    import torch
    from unsloth.models import llama

    registry = pytest.importorskip("unsloth_zoo.tiling_registry")
    hooks, count = _counting_hook()
    mlp = _GroupedMLP()
    registry.install_module(mlp, "rows:2")
    llama._install_fused_lora_mlp(mlp, llama.apply_lora_mlp_swiglu)
    hooks.add_hook_to_module(mlp, count())
    registry.install_module(mlp, "rows:2")  # Trainer revalidation
    assert mlp in llama._fused_lora_modules(mlp)
    assert llama._decline_fused_lora_for_fsdp(mlp) == 1
    assert llama._fused_lora_modules(mlp) == []
    x = torch.randn(2, 3, 4)
    before = count.calls
    torch.testing.assert_close(mlp(x), mlp.up(x))
    assert count.calls > before


def test_a_kernel_behind_an_accelerate_hook_after_untile_is_declined():
    """After untile the kernel sits in the hook's `_old_forward` and is still detected and declined."""
    import torch
    from unsloth.models import llama
    from unsloth.tiling import untile_mlp

    registry = pytest.importorskip("unsloth_zoo.tiling_registry")
    hooks, count = _counting_hook()
    mlp = _GroupedMLP()
    registry.install_module(mlp, "rows:2")
    llama._install_fused_lora_mlp(mlp, llama.apply_lora_mlp_swiglu)
    hooks.add_hook_to_module(mlp, count())
    untile_mlp(mlp)
    assert mlp in llama._fused_lora_modules(mlp)
    assert llama._decline_fused_lora_for_fsdp(mlp) == 1
    assert llama._fused_lora_modules(mlp) == []
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(mlp(x), mlp.up(x))


@pytest.mark.parametrize("tiled", [False, True])
def test_installing_the_kernel_keeps_an_accelerate_hook_running(tiled):
    import torch
    from unsloth.models import llama

    hooks, count = _counting_hook()
    mlp = _GroupedMLP()
    registry = pytest.importorskip("unsloth_zoo.tiling_registry") if tiled else None
    if tiled:
        registry.install_module(mlp, "rows:2")
    hooks.add_hook_to_module(mlp, count())
    if tiled:
        registry.install_module(mlp, "rows:2")  # the hook is adopted as the inner forward
    seen = []
    kernel = llama.apply_lora_mlp_swiglu
    llama._install_fused_lora_mlp(mlp, kernel)
    assert mlp in llama._fused_lora_modules(mlp)
    slot = llama._fused_lora_mlp_slot(mlp)
    mlp.__dict__[slot] = types.MethodType(lambda self, x: (seen.append(1), self.up(x))[1], mlp)
    x = torch.randn(2, 3, 4)
    before = count.calls
    mlp(x)
    assert count.calls > before and seen


@pytest.mark.parametrize("wrap", ["lambda", "compiler_disable"])
def test_instance_forward_fuse_tile_decline_still_calls(wrap):
    """A restored instance function must not get `self` bound twice in a tiled inner slot."""
    import torch
    from unsloth.models import llama

    registry = pytest.importorskip("unsloth_zoo.tiling_registry")
    mlp = _GroupedMLP()
    if wrap == "lambda":
        mlp.forward = lambda x: mlp.up(x) * 3
    else:
        mlp.forward = torch.compiler.disable(types.MethodType(_adopted_forward, mlp))
    llama._install_fused_lora_mlp(mlp, llama.apply_lora_mlp_swiglu)
    registry.install_module(mlp, "rows:2")
    assert llama._decline_fused_lora_for_fsdp(mlp) == 1
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(mlp(x), mlp.up(x) * 3)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AUTO_SKIP_BASES: with the switch unset, INT8 attention stays off Qwen-Image-2.1 (25 steps, further from bf16 with
the kernel) and keeps racing on Qwen-Image-2.1-Turbo; an explicit on, the name or the env switch still routes base
2.1, and the status says why Auto skipped it. CPU, fakes as in test_diffusion_int8_attention_native_blocks."""

from __future__ import annotations

import ast
import pathlib
import types

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_dispatch as D

pytest.importorskip("diffusers.models.attention_dispatch")
# imported before the fixture fakes a CUDA card: diffusers' lazy import reads the device list torch really has
from diffusers import WanTransformer3DModel  # noqa: E402

BASE = "Qwen/Qwen-Image-2.1"
TURBO = "Qwen/Qwen-Image-2.1-Turbo"
# what the load stamps on the pipeline (diffusion.py, before apply_attention_backend)
_ATTR = "_unsloth_base_repo"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    from core.inference import rocm_bf16

    for env in (
        I.INT8_ATTENTION_ENV,
        I.IMPL_ENV,
        D.DISPATCH_ENV,
        "UNSLOTH_DIFFUSION_INT8_ATTENTION_NATIVE_BLOCKS",
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(D, "MIN_TOKENS", 0)
    monkeypatch.setattr(D, "_on_card", lambda x, route: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "cuda")
    monkeypatch.setattr(D, "_race_wins", lambda *a, **k: False)
    monkeypatch.setattr(I, "verdict", lambda *a, **k: {"impl": "cuda", "ok": True, "min_tokens": 0})
    D._DECISIONS.clear()
    yield
    D._DECISIONS.clear()


def _dit(n = 4):
    """A ``QwenImage21Transformer2DModel`` whose ``transformer_blocks.<i>.attn`` call diffusers' attention dispatch."""
    torch.manual_seed(0)
    wan = WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 64,
        in_channels = 4,
        out_channels = 4,
        text_dim = 32,
        freq_dim = 32,
        ffn_dim = 64,
        num_layers = n,
        rope_max_seq_len = 32,
    ).to(torch.bfloat16)
    wan.set_attention_backend("native")
    dit = type("QwenImage21Transformer2DModel", (torch.nn.Module,), {})()
    dit.transformer_blocks = torch.nn.ModuleList()
    for b in wan.blocks:
        blk = torch.nn.Module()
        blk.attn = b.attn1
        dit.transformer_blocks.append(blk)
    return dit


class _Pipe:
    def __init__(
        self,
        transformer,
        base = None,
    ):
        self.transformer = transformer
        if base is not None:
            setattr(self, _ATTR, base)


_TARGET = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def test_auto_skips_base_qwen_image_21_with_a_reason():
    pipe = _Pipe(_dit(), BASE)
    assert D.install(pipe, _TARGET) == 0
    assert D.routed_count(pipe) == 0
    why = getattr(pipe, D.AUTO_SKIP_ATTR)
    assert "Qwen-Image-2.1" in why and "Turbo keeps it" in why
    # the status entry carries it instead of the generic "not measured faster" line
    entry = I.resolved_entry(None, pipe)
    assert entry[1] == "off" and entry[2] == why
    # the deferred speed profile re-applies on the same pipeline at the 3rd image: still skipped
    assert D.install(pipe, _TARGET) == 0 and getattr(pipe, D.AUTO_SKIP_ATTR) == why


def test_auto_keeps_turbo_racing():
    pipe = _Pipe(_dit(), TURBO)
    # block 0 stays native (NATIVE_BLOCKS), as on main
    assert D.install(pipe, _TARGET) == 3
    assert getattr(pipe, D.AUTO_SKIP_ATTR) is None
    assert I.resolved_entry(None, pipe)[1] == "on"


@pytest.mark.parametrize(
    "base",
    [
        None,  # no base stamped (another loader): unchanged
        "someone/Qwen-Image-2.1",  # same name, another owner: the full id decides
        "Qwen/Qwen-Image-2.1-Turbo-Finetune",
    ],
)
def test_only_the_full_base_id_matches(base):
    assert D.install(_Pipe(_dit(), base), _TARGET) == 3


def test_base_id_match_ignores_case_and_padding():
    assert D.install(_Pipe(_dit(), " qwen/QWEN-image-2.1 "), _TARGET) == 0


@pytest.mark.parametrize("how", ["policy", "name", "env"])
def test_explicit_on_still_routes_base_21(monkeypatch, how):
    pipe = _Pipe(_dit(), BASE)
    kwargs = {}
    if how == "policy":
        kwargs["policy"] = "on"
    elif how == "name":
        kwargs["requested"] = I.BACKEND
    else:
        monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    # forced: every block, block 0 included
    assert D.install(pipe, _TARGET, **kwargs) == 4
    assert getattr(pipe, D.AUTO_SKIP_ATTR) is None


def test_off_clears_a_stale_reason():
    pipe = _Pipe(_dit(), BASE)
    assert D.install(pipe, _TARGET) == 0 and getattr(pipe, D.AUTO_SKIP_ATTR)
    assert D.install(pipe, _TARGET, policy = "off") == 0
    assert getattr(pipe, D.AUTO_SKIP_ATTR) is None
    assert I.resolved_entry("off", pipe)[2] == "off for this load"


def test_strided_auto_skips_base_21_and_races_turbo(monkeypatch):
    """``install_auto`` (strided processors) applies the same rule before any race."""
    from core.inference import diffusion_flux3_attn

    monkeypatch.setattr(diffusion_flux3_attn, "ensure_strided_processors", lambda *a, **k: 0)
    raced = []
    monkeypatch.setattr(I, "install_int8_attention", lambda t, **k: raced.append(t) or 2)
    monkeypatch.setattr(I, "_give_back_unengaged", lambda t: None)
    base = _Pipe(_dit(), BASE)
    assert I.install_auto(base, _TARGET) == 0 and not raced
    assert "Qwen-Image-2.1" in getattr(base, D.AUTO_SKIP_ATTR)
    turbo = _Pipe(_dit(), TURBO)
    assert I.install_auto(turbo, _TARGET) == 2 and raced == [turbo.transformer]


def test_the_load_stamps_the_resolved_base_before_the_attention_pick():
    """diffusion.py sets BASE_REPO_ATTR (canonical base) on the pipeline before the load's apply_attention_backend;
    the deferred re-apply reuses that pipeline, so render 3 reads the same base."""
    src = (
        pathlib.Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py"
    ).read_text()
    tree = ast.parse(src)
    stamp = apply = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "setattr"
            and len(node.args) == 3
            and getattr(node.args[1], "id", None) == "BASE_REPO_ATTR"
        ):
            stamp = node.lineno
            assert "canonical_base" in ast.unparse(node.args[2])
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "apply_attention_backend"
        ):
            apply = node.lineno if apply is None else min(apply, node.lineno)
    assert D.BASE_REPO_ATTR == _ATTR
    assert stamp is not None and apply is not None and stamp < apply

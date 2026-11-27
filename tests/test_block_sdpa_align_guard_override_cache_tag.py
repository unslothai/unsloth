# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""UNSLOTH_BLOCK_SDPA_ALIGN_GUARD=1 keeps the stock SDPA lowering, so its compile cache key equals that of
a run without the efficient-backward guard from before the cache tag existed (the lowering patch is not
in the key). On CUDA torch < 2.14 the override must add the tag too, or a warm cache keeps serving that
run's faulting efficient backward (`attn_bias is not correctly aligned`). CPU only, no compile."""

import importlib.util
import os
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_block_compile.py"


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location("_u_block_compile_override_tag", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lowering_untouched():
    import torch._inductor.lowering as lowering

    table = dict(getattr(lowering, "_maybe_layout_constraints", {}))  # absent on torch 2.4
    stock = lowering.sdpa_constraint
    yield lowering
    assert lowering.sdpa_constraint is stock
    assert dict(getattr(lowering, "_maybe_layout_constraints", {})) == table


def test_block_compile_module_is_this_tree():
    assert pathlib.Path(_load().__file__).resolve() == _PATH


@pytest.mark.skipif(
    getattr(getattr(torch, "compiler", None), "config", None) is None
    or not hasattr(torch.compiler.config, "cache_key_tag"),
    reason = "this torch has no compile cache key tag",
)
@pytest.mark.parametrize("keeps", [True, False], ids = ["torch_lt_2_14", "torch_2_14"])
def test_override_tags_the_cache_key(monkeypatch, lowering_untouched, keeps):
    M = _load()
    monkeypatch.setenv("UNSLOTH_BLOCK_SDPA_ALIGN_GUARD", "1")
    monkeypatch.setattr(M, "_efficient_backward_keeps_guard", lambda: keeps)
    monkeypatch.setattr(torch.compiler.config, "cache_key_tag", "user-tag")
    assert M._sdpa_alignment_without_guard() is False  # every guard kept, nothing patched
    tagged = "user-tag," + M._EFFICIENT_BACKWARD_CACHE_TAG
    assert torch.compiler.config.cache_key_tag == (tagged if keeps else "user-tag")
    M._sdpa_alignment_without_guard()
    assert torch.compiler.config.cache_key_tag == (tagged if keeps else "user-tag")
    assert os.environ["UNSLOTH_BLOCK_SDPA_ALIGN_GUARD"] == "1"

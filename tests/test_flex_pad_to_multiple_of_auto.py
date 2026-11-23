# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_ENABLE_FLEX_ATTENTION pads to the flex block size when pad_to_multiple_of is left at TRL's default.

TRL <= 1.15 defaults it to None, TRL >= 1.16 to "auto"; both mean the user did not choose one.
"""

from __future__ import annotations

import dataclasses

import pytest


pytest.importorskip("trl")

BLOCK = 128


@pytest.fixture(scope = "module")
def patched():
    import unsloth  # noqa: F401
    import trl.trainer.sft_config as config_module
    return config_module.SFTConfig


@pytest.fixture
def flex_on(monkeypatch):
    import unsloth_zoo.flex_attention as flex

    monkeypatch.setenv("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")
    monkeypatch.setattr(flex, "HAS_FLEX_ATTENTION", True, raising = False)
    monkeypatch.setattr(flex, "FLEX_ATTENTION_BLOCK_SIZE", BLOCK, raising = False)


def _make(config_class, output_dir, **extra):
    """Pins CPU, since TRL turns bf16=None into True and transformers refuses without an accelerator."""
    kwargs = {"output_dir": str(output_dir)}
    declared = {field.name for field in dataclasses.fields(config_class)}
    for name, value in (("bf16", False), ("fp16", False), ("use_cpu", True)):
        if name in declared:
            kwargs[name] = value
    kwargs.update(extra)
    return config_class(**kwargs)


def test_the_patched_config_is_unsloths(patched):
    assert patched.__mro__[1] is not patched
    assert "pad_to_multiple_of" in {field.name for field in dataclasses.fields(patched)}


def test_trl_default_pads_to_the_flex_block(patched, flex_on, tmp_path):
    assert _make(patched, tmp_path).pad_to_multiple_of == BLOCK


def test_explicit_auto_pads_to_the_flex_block(patched, flex_on, tmp_path):
    assert _make(patched, tmp_path, pad_to_multiple_of = "auto").pad_to_multiple_of == BLOCK


def test_a_user_multiple_is_kept(patched, flex_on, tmp_path):
    assert _make(patched, tmp_path, pad_to_multiple_of = 64).pad_to_multiple_of == 64


def test_flex_off_leaves_trls_resolution_alone(patched, monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_ENABLE_FLEX_ATTENTION", raising = False)
    pristine = patched.__mro__[1]
    assert (
        _make(patched, tmp_path).pad_to_multiple_of == _make(pristine, tmp_path).pad_to_multiple_of
    )

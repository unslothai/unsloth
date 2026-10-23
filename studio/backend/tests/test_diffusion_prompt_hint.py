# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``prompt_hint``: the per-family prompt tip, from the family row to every image engine's status.

Only vetted families carry a tip. Each engine's loaded status copies the family's value, and
``DiffusionStatusResponse`` has to declare the key, since the route builds it with ``**status``
and Pydantic drops undeclared keys.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

from core.inference.diffusion_families import _FAMILIES, DiffusionFamily, detect_family
from models.inference import DiffusionStatusResponse

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Reuse each engine's own fixtures so the status read here is the real one.
from test_diffusion_backend import (  # noqa: E402,F401
    _loaded_backend as _loaded_diffusers_backend,
    _unmeasured_torchao,
    fake_runtime,
)
from test_mlx_diffusion_backend import _load as _load_mlx, env  # noqa: E402,F401
from test_sd_cpp_backend import _loaded_backend as _loaded_sd_cpp_backend  # noqa: E402

_TIPPED = {"flux.1-kontext", "qwen-image-edit", "qwen-image-2.1", "ideogram-4"}
_HINT = "Describe the scene first."


def test_prompt_hint_defaults_to_none():
    assert DiffusionFamily.__dataclass_fields__["prompt_hint"].default is None


def test_only_the_vetted_families_carry_a_tip():
    assert {f.name for f in _FAMILIES if f.prompt_hint is not None} == _TIPPED


@pytest.mark.parametrize("name", sorted(_TIPPED))
def test_each_tip_is_short_and_names_unsloth_studio_in_full(name):
    hint = detect_family(name).prompt_hint
    assert hint and len(hint) <= 240
    # One or two sentences.
    assert 1 <= len(re.findall(r"[.!?](?:\s|\"?$)", hint)) <= 2
    assert not re.search(r"(?<!Unsloth )\bStudio\b", hint)


def _tipped_state(state, hint = _HINT):
    return dataclasses.replace(state, family = dataclasses.replace(state.family, prompt_hint = hint))


def test_diffusers_status_carries_the_family_tip(fake_runtime, tmp_path):
    backend = _loaded_diffusers_backend(tmp_path)
    assert backend.status()["prompt_hint"] is None
    assert DiffusionStatusResponse(**backend.status()).prompt_hint is None

    backend._state = _tipped_state(backend._state)
    assert backend.status()["prompt_hint"] == _HINT
    assert DiffusionStatusResponse(**backend.status()).prompt_hint == _HINT

    backend.unload()
    assert DiffusionStatusResponse(**backend.status()).prompt_hint is None


def test_sd_cpp_status_carries_the_family_tip():
    assert _loaded_sd_cpp_backend("z-image").status()["prompt_hint"] is None

    backend = _loaded_sd_cpp_backend("flux.1-kontext")
    hint = detect_family("flux.1-kontext").prompt_hint
    assert backend.status()["prompt_hint"] == hint
    assert DiffusionStatusResponse(**backend.status()).prompt_hint == hint

    backend._state = None
    assert DiffusionStatusResponse(**backend.status()).prompt_hint is None


def test_mlx_status_carries_the_family_tip(env):
    backend, _runtime, _calls = env
    assert DiffusionStatusResponse(**backend.status()).prompt_hint is None
    assert _load_mlx(backend)["phase"] == "ready"
    assert backend.status()["prompt_hint"] is None

    backend._state = _tipped_state(backend._state)
    assert backend.status()["prompt_hint"] == _HINT
    assert DiffusionStatusResponse(**backend.status()).prompt_hint == _HINT

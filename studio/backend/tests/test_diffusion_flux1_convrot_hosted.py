# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.1-schnell int8 asks for the hosted rotated file first and keeps the plain one behind it; nothing else moves."""

import importlib.util
import sys
from pathlib import Path

import pytest

import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_families import detect_family
from core.inference.diffusion_prequant import (
    PREQUANT_MIRROR_ENV,
    is_comfy_prequant_filename,
    resolve_prequant_source,
)

SCHNELL = "black-forest-labs/FLUX.1-schnell"
ROTATED = "FLUX.1-schnell-INT8-ConvRot.safetensors"
REPO = "unsloth/FLUX.1-schnell-FP8"
PLAIN = ("FLUX.1-schnell-INT8.safetensors", "FLUX.1-schnell-INT8.pt", "transformer_int8.pt")


def _names(source):
    return (source.filename,) + tuple(source.fallback_filenames)


def _own(names):
    return tuple(n for n in names if not is_comfy_prequant_filename(n))


@pytest.fixture(autouse = True)
def _default_env(monkeypatch):
    monkeypatch.delenv(tq.INT8_CONVROT_ENV, raising = False)
    monkeypatch.delenv(PREQUANT_MIRROR_ENV, raising = False)


def test_schnell_int8_asks_for_the_rotated_file_first_with_the_plain_chain_behind_it():
    src = resolve_prequant_source(detect_family(SCHNELL), "int8", base_repo = SCHNELL)
    assert src.location == REPO
    own = _own(_names(src))
    assert own[0] == ROTATED
    # Until the file is on the repo the plain hosted file still loads, so the change is inert.
    assert own[1:] == PLAIN
    assert src.declared_filenames == (ROTATED,)


@pytest.mark.parametrize("off", ["0", "off", "false", "no"])
def test_the_convrot_kill_switch_resolves_what_main_resolved(monkeypatch, off):
    monkeypatch.setenv(tq.INT8_CONVROT_ENV, off)
    src = resolve_prequant_source(detect_family(SCHNELL), "int8", base_repo = SCHNELL)
    assert ROTATED not in _names(src)
    assert _own(_names(src)) == PLAIN


@pytest.mark.parametrize(
    "base",
    ["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-Krea-dev"],
)
def test_dev_and_krea_dev_never_ask_for_the_schnell_file(base):
    src = resolve_prequant_source(detect_family(base), "int8", base_repo = base)
    assert src.location != REPO
    assert not any("ConvRot" in n for n in _names(src))


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4", "mxfp8"])
def test_other_schemes_are_unchanged(scheme):
    src = resolve_prequant_source(detect_family(SCHNELL), scheme, base_repo = SCHNELL)
    if src is not None:
        assert not any("ConvRot" in n for n in _names(src))


def test_quantizing_at_load_stays_plain_int8():
    # group 0: the rotation lives in the file's metadata only, so no at-load path rotates.
    assert tq.int8_convrot_enabled("flux.1")
    assert tq.convrot_spec_for_scheme(tq.TQ_INT8, "flux.1") == (0, ())
    assert tq.convrot_prequant_filename(tq.TQ_INT8, "flux.1") == ROTATED
    assert tq.convrot_prequant_repo(tq.TQ_INT8, "flux.1") == REPO
    assert not tq.convrot_drops_plain_hosted(tq.TQ_INT8, "flux.1")


def test_an_operator_mirror_serves_the_rotated_file_before_the_hub(monkeypatch, tmp_path):
    from core.inference.diffusion_prequant import _first_mirrored

    target = tmp_path / "unsloth" / "FLUX.1-schnell-FP8" / ROTATED
    target.parent.mkdir(parents = True)
    target.write_bytes(b"x")
    monkeypatch.setenv(PREQUANT_MIRROR_ENV, str(tmp_path))
    src = resolve_prequant_source(detect_family(SCHNELL), "int8", base_repo = SCHNELL)
    assert _first_mirrored(src.location, _names(src), None) == str(target.resolve())


def test_the_build_script_publishes_a_group_0_family_under_its_declared_name():
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_prequant_checkpoint.py"
    spec = importlib.util.spec_from_file_location("build_prequant_checkpoint", script)
    build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build
    spec.loader.exec_module(build)
    fam = detect_family(SCHNELL)
    assert (
        build.upload_destination(
            fam,
            "int8",
            rotated = True,
            convrot_group = 256,
            safetensors = True,
            upload_repo = REPO,
        )
        == ROTATED
    )

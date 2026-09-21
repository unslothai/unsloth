# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two decisions ``scripts/build_prequant_checkpoint.py`` makes that a bad answer to is
silent: whether a ConvRot build is buildable at all, and what filename it publishes under.

Both end in an artifact that costs GPU-hours and tens of gigabytes and then cannot be resolved
or cannot be loaded, with nothing at build time saying so, which is why they are pulled out as
pure functions and asserted here rather than left inline in ``main``."""

import importlib.util
import sys
from pathlib import Path

import pytest

from core.inference.diffusion_convrot import rotation_metadata, rotation_metadata_error
from core.inference.diffusion_families import detect_family
from core.inference.video_families import detect_video_family

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_prequant_checkpoint.py"


def _script():
    """The build script as a module. Imported by path: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("build_prequant_checkpoint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_convrot_group_that_rotates_nothing_is_refused_before_anything_is_built():
    build = _script()
    # The group divides no quantized input axis, so the rotation would be empty.
    refusal = build.convrot_refusal(4096, (), ("blocks.0.ff.net.0", "blocks.0.attn.to_q"))
    assert refusal is not None
    assert "4096" in refusal and "2" in refusal
    # And the reason it has to be refused: the artifact it would have written is unloadable.
    assert rotation_metadata_error(rotation_metadata(4096, ())) is not None
    # A non-empty set is built normally.
    assert build.convrot_refusal(256, ("blocks.0.attn.to_q",), ()) is None


def test_a_rotated_upload_goes_to_the_name_the_loader_asks_for_not_the_legacy_fallback():
    build = _script()
    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert h3 is not None
    # The rotated INT8 denoiser is published under the family's declared name, which is the one
    # resolve_prequant_source asks for first. transformer_int8.pt is never asked for on this
    # family, so an upload landing there would be invisible.
    assert build.upload_destination(h3, "int8", rotated = True) == "MiniMax-H3-INT8-ConvRot.pt"
    # A plain build keeps the legacy name it has always used, so nothing else moves.
    assert build.upload_destination(h3, "int8", rotated = False) == "transformer_int8.pt"


def test_a_rotated_upload_with_no_declared_name_is_refused_rather_than_published_over_the_fallback():
    build = _script()
    zimage = detect_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image")
    assert zimage is not None
    # Publishing a v2 artifact under transformer_<scheme>.pt hands it to every OLDER build as the
    # fallback, which refuses the tag and drops to the dense download. Refuse instead.
    with pytest.raises(ValueError, match = "prequant_filenames"):
        build.upload_destination(zimage, "int8", rotated = True)
    # An explicit name is the operator's escape hatch, rotated or not.
    assert (
        build.upload_destination(
            zimage, "int8", rotated = True, override = "Z-Image-Turbo-INT8-ConvRot.pt"
        )
        == "Z-Image-Turbo-INT8-ConvRot.pt"
    )
    assert build.upload_destination(zimage, "fp8", rotated = False) == "transformer_fp8.pt"


def test_a_safetensors_upload_needs_a_name_the_loader_would_ask_for():
    build = _script()
    zimage = detect_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image")
    assert zimage is not None
    # Every DERIVED name ends in .pt, so no build ever asks the Hub for a safetensors artifact
    # unless the family declares one. Publishing under a derived name produces a file nothing can
    # reach, on a repo that looks like it has a checkpoint.
    with pytest.raises(ValueError, match = "prequant_filenames"):
        build.upload_destination(zimage, "fp8", rotated = False, safetensors = True)
    assert (
        build.upload_destination(
            zimage,
            "fp8",
            rotated = False,
            safetensors = True,
            override = "Z-Image-Turbo-FP8.safetensors",
        )
        == "Z-Image-Turbo-FP8.safetensors"
    )


def test_a_safetensors_build_refuses_a_declared_name_that_reads_as_a_pickle():
    build = _script()
    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert h3 is not None
    # H3 declares a .pt name for int8. Uploading a safetensors artifact there gives every loader a
    # file whose extension says pickle and whose bytes are not one, so the load fails for a reason
    # that has nothing to do with the real mistake. Refuse at build time and say which name to fix.
    with pytest.raises(ValueError, match = "not a safetensors name"):
        build.upload_destination(h3, "int8", rotated = False, safetensors = True)


def test_an_override_still_has_to_match_the_container_it_is_naming():
    """The escape hatch skips the family table, not the extension.

    Every loader dispatches on the extension alone, so a safetensors build published as ``.pt`` is
    read as a pickle and a pickle published as ``.safetensors`` is read from a header it does not
    have. Both upload cleanly and neither can ever be opened, after the hours the quantization
    took, which is why this is refused before the upload rather than reported after it.
    """
    build = _script()
    zimage = detect_family("Tongyi-MAI/Z-Image-Turbo", override = "z-image")
    assert zimage is not None

    with pytest.raises(ValueError, match = r"does not end in '\.safetensors'"):
        build.upload_destination(
            zimage, "fp8", rotated = False, safetensors = True, override = "Z-Image-Turbo-FP8.pt"
        )
    with pytest.raises(ValueError, match = r"does not end in '\.pt'"):
        build.upload_destination(
            zimage,
            "fp8",
            rotated = True,
            safetensors = False,
            override = "Z-Image-Turbo-FP8.safetensors",
        )
    # The matching pairs are untouched, including the rotated escape hatch above.
    assert (
        build.upload_destination(
            zimage, "fp8", rotated = True, override = "Z-Image-Turbo-FP8-ConvRot.pt"
        )
        == "Z-Image-Turbo-FP8-ConvRot.pt"
    )
    assert (
        build.upload_destination(
            zimage,
            "fp8",
            rotated = False,
            safetensors = True,
            override = "Z-Image-Turbo-FP8.safetensors",
        )
        == "Z-Image-Turbo-FP8.safetensors"
    )

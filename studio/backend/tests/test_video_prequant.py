# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted pre-quantized DENOISER route for video families.

Covers the family table + resolver, the repo-root checkpoint naming, and the
``validate_load_request`` refusals that must fire BEFORE anything downloads. All network-free and
torch-free: the refusals are deliberately placed ahead of the diffusers availability probe so they
still run (and still get tested) in an environment where diffusers cannot be imported at all.
"""

from __future__ import annotations

import pytest

from core.inference.diffusion_prequant import (
    cached_checkpoint_path,
    resolve_prequant_source,
)
from core.inference.video import VideoBackend
from core.inference.video_families import (
    VideoFamily,
    detect_video_family,
    video_family_prequant_repo,
    video_family_prequant_schemes,
)


def _fam(**kwargs) -> VideoFamily:
    base = dict(
        name = "test-video",
        pipeline_class = "TestPipeline",
        transformer_class = "TestTransformer3DModel",
        base_repo = "org/test-video",
    )
    base.update(kwargs)
    return VideoFamily(**base)


# ── the resolver ─────────────────────────────────────────────────────────────────
def test_resolves_the_hosted_repo_for_a_scheme():
    fam = _fam(prequant_repos = (("int8", "org/test-INT8"), ("fp8", "org/test-FP8")))
    assert video_family_prequant_repo(fam, "int8") == "org/test-INT8"
    assert video_family_prequant_repo(fam, "fp8") == "org/test-FP8"
    assert video_family_prequant_repo(fam, "nvfp4") is None


def test_a_variant_checkpoint_wins_over_the_family_default():
    # A checkpoint is baked from ONE base's weights, so a variant base with its own artifact must
    # take it rather than the family default, which the base_model_id check would then reject.
    fam = _fam(
        prequant_repos = (("int8", "org/test-INT8"),),
        prequant_variant_repos = (("org/test-video-v2", "int8", "org/test-v2-INT8"),),
    )
    assert video_family_prequant_repo(fam, "int8", "org/test-video-v2") == "org/test-v2-INT8"
    # Case and surrounding whitespace must not change the answer.
    assert video_family_prequant_repo(fam, "int8", "  ORG/Test-Video-V2 ") == "org/test-v2-INT8"
    # A base without its own entry falls back to the family default.
    assert video_family_prequant_repo(fam, "int8", "org/other") == "org/test-INT8"


def test_a_malformed_table_row_is_skipped_rather_than_raised():
    # This runs on the refusal path of a load request: a table typo must not turn a legitimate
    # pick into a 500.
    fam = _fam(prequant_repos = (("int8",), ("int8", ""), ("int8", "org/good")))
    assert video_family_prequant_repo(fam, "int8") == "org/good"


def test_a_family_without_the_fields_simply_has_no_checkpoint():
    import types
    assert video_family_prequant_repo(types.SimpleNamespace(), "int8") is None
    assert video_family_prequant_schemes(types.SimpleNamespace()) == ()


def test_schemes_are_listed_in_table_order():
    # The refusal message names these, so the order is what the user is told to try first.
    fam = _fam(prequant_repos = (("int8", "org/a"), ("fp8", "org/b")))
    assert video_family_prequant_schemes(fam) == ("int8", "fp8")


# ── the H3 table ─────────────────────────────────────────────────────────────────
def test_minimax_h3_declares_hosted_denoiser_checkpoints():
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert fam is not None and fam.name == "minimax-h3"
    assert set(video_family_prequant_schemes(fam)) == {"int8", "fp8"}
    for scheme in ("int8", "fp8"):
        repo = video_family_prequant_repo(fam, scheme)
        # Curated hosted artifacts only: a third-party repo here would be unpickled by the loader.
        assert repo and repo.startswith("unsloth/")


def test_both_h3_schemes_resolve_to_one_repo():
    # Both schemes live in the SAME hosted repo. Two repos meant one of them had to be named for a
    # scheme it did not carry, and it is the pair (repo, scheme) that names the file.
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    repos = {s: video_family_prequant_repo(fam, s) for s in ("int8", "fp8")}
    assert repos["int8"] == repos["fp8"], repos


# ── repo-root naming ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "scheme, expected",
    [("int8", "MiniMax-H3-INT8.pt"), ("fp8", "MiniMax-H3-FP8.pt")],
)
def test_h3_resolves_the_primary_name_at_the_repo_root(scheme, expected):
    # The name the hosted repo actually publishes. It has to be the PRIMARY, not the fallback:
    # cached_checkpoint_path deliberately credits only the primary, so landing on the fallback
    # would report a cached checkpoint as "this would have to download" and hand the pick to GGUF.
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    src = resolve_prequant_source(fam, scheme)
    assert src.filename == expected
    # Root-level: no directory component at all, on any platform.
    assert "/" not in src.filename and "\\" not in src.filename


def test_the_h3_primary_name_is_what_memory_planning_credits():
    # The under-crediting bug in full: seed the cache under the primary name and
    # cached_checkpoint_path must find it. A nested (or otherwise non-primary) name would not.
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    src = resolve_prequant_source(fam, "int8")
    seen = {}

    def _fake_try_to_load_from_cache(repo_id, filename, cache_dir = None):
        seen["filename"] = filename
        return "/cache/blobs/h3" if filename == "MiniMax-H3-INT8.pt" else None

    import huggingface_hub
    import os
    real_hub = huggingface_hub.try_to_load_from_cache
    real_isfile = os.path.isfile
    huggingface_hub.try_to_load_from_cache = _fake_try_to_load_from_cache
    os.path.isfile = lambda p: p == "/cache/blobs/h3" or real_isfile(p)
    try:
        assert cached_checkpoint_path(src) == "/cache/blobs/h3"
    finally:
        huggingface_hub.try_to_load_from_cache = real_hub
        os.path.isfile = real_isfile
    assert seen["filename"] == "MiniMax-H3-INT8.pt"


def test_the_names_are_built_from_the_repo_and_the_scheme():
    # One repo serves both schemes, so the -FP8 suffix on the repo must be stripped and REPLACED by
    # the requested scheme rather than carried through.
    fam = _fam(prequant_repos = (("int8", "unsloth/Test-FP8"), ("fp8", "unsloth/Test-FP8")))
    assert resolve_prequant_source(fam, "int8").filename == "Test-INT8.pt"
    assert resolve_prequant_source(fam, "fp8").filename == "Test-FP8.pt"
    # The legacy per-scheme name stays available for repos that have not been renamed.
    assert resolve_prequant_source(fam, "int8").fallback_filename == "transformer_int8.pt"


# ── validate_load_request: refuse BEFORE the download ────────────────────────────
def test_a_modular_family_refuses_a_single_file_load_before_anything_downloads():
    # Previously this reached the loader only after ~98.7 GB had downloaded AND after the resident
    # pipeline had been evicted to make room for it.
    backend = VideoBackend()
    with pytest.raises(ValueError) as excinfo:
        backend.validate_load_request(
            "MiniMaxAI/MiniMax-H3",
            gguf_filename = "minimax_h3_fl2va_pruned_int8_rowwise.safetensors",
            model_kind = "single_file",
        )
    message = str(excinfo.value)
    # The refusal is only useful if it says what to pick instead.
    assert "MiniMaxAI/MiniMax-H3" in message
    assert "unsloth/MiniMax-H3-GGUF" in message


def test_an_unavailable_transformer_quant_is_refused_with_the_workable_schemes():
    backend = VideoBackend()
    with pytest.raises(ValueError) as excinfo:
        backend.validate_load_request("MiniMaxAI/MiniMax-H3", transformer_quant = "nvfp4")
    message = str(excinfo.value)
    assert "nvfp4" in message
    # Naming the schemes that DO work is the whole point: a bare refusal leaves the user guessing.
    assert "int8" in message and "fp8" in message


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_a_scheme_with_a_hosted_checkpoint_is_not_refused(scheme):
    # The mirror image of the test above: the refusal must not swallow the picks it exists to enable.
    backend = VideoBackend()
    try:
        backend.validate_load_request("MiniMaxAI/MiniMax-H3", transformer_quant = scheme)
    except ValueError as exc:  # pragma: no cover - only on a regression
        pytest.fail(f"{scheme} should be loadable but was refused: {exc}")
    except Exception:
        # Anything past the quant check (a diffusers import probe in a skewed env) is not this
        # test's business; reaching it already proves the refusal did not fire.
        pass


def test_auto_is_never_refused():
    # "auto" asks the backend to choose, not for a specific scheme, so it stays on the released
    # components rather than being rejected as unavailable.
    backend = VideoBackend()
    try:
        backend.validate_load_request("MiniMaxAI/MiniMax-H3", transformer_quant = "auto")
    except ValueError as exc:  # pragma: no cover - only on a regression
        pytest.fail(f"auto should never be refused: {exc}")
    except Exception:
        pass


def test_the_refusals_run_before_the_diffusers_availability_probe():
    # Placement matters: the probe below imports diffusers, and on an environment where that raises
    # the user would get an unrelated error instead of the actionable refusal. Asserting the
    # message identifies the refusal proves it ran first.
    backend = VideoBackend()
    with pytest.raises(ValueError, match = "cannot load from a single .safetensors checkpoint"):
        backend.validate_load_request(
            "MiniMaxAI/MiniMax-H3",
            gguf_filename = "anything.safetensors",
            model_kind = "single_file",
        )


def test_a_non_modular_video_family_is_unaffected():
    # LTX-2.3 legitimately loads a single-file DiT; the new refusal must be scoped to modular
    # workflows or it would break the artifact the picker routes to by default. Asserted on the
    # MESSAGE rather than on success: the diffusers availability probe further down is not
    # importable in every environment, and that unrelated failure must not mask a real regression
    # here (nor make this test pass for the wrong reason).
    backend = VideoBackend()
    try:
        fam = backend.validate_load_request(
            "Lightricks/LTX-2.3",
            gguf_filename = "ltx-2.3-22b-distilled.safetensors",
            model_kind = "single_file",
        )
    except ValueError as exc:
        pytest.fail(f"a single-file LTX-2.3 load must not be refused: {exc}")
    except RuntimeError:
        # Reaching the diffusers probe already proves the modular refusal did not fire.
        return
    assert fam.name == "ltx-2.3"

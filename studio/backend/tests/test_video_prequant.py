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
    video_family_prequant_available,
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
    # int8 is the ConvRot-rotated denoiser, which the family names explicitly; fp8 keeps the
    # derived <Model>-<SCHEME>.pt.
    [("int8", "MiniMax-H3-INT8-ConvRot.pt"), ("fp8", "MiniMax-H3-FP8.pt")],
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


def test_h3_int8_keeps_the_plain_denoiser_as_its_fallback():
    # The rotated artifact carries the v2 format tag, which a Studio predating the online rotation
    # refuses. Naming it explicitly and demoting the derived name to the fallback is what stops
    # that refusal from reaching anyone: an older install still resolves MiniMax-H3-INT8.pt, and
    # this one takes the rotated file when the repo has it.
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    src = resolve_prequant_source(fam, "int8")
    assert src.fallback_filename == "MiniMax-H3-INT8.pt"
    assert "/" not in src.fallback_filename and "\\" not in src.fallback_filename


def test_the_h3_primary_name_is_what_memory_planning_credits():
    # The under-crediting bug in full: seed the cache under the primary name and
    # cached_checkpoint_path must find it. A nested (or otherwise non-primary) name would not.
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    src = resolve_prequant_source(fam, "int8")
    seen = {}

    def _fake_try_to_load_from_cache(
        repo_id,
        filename,
        cache_dir = None,
    ):
        seen["filename"] = filename
        return "/cache/blobs/h3" if filename == "MiniMax-H3-INT8-ConvRot.pt" else None

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
    assert seen["filename"] == "MiniMax-H3-INT8-ConvRot.pt"


def test_the_names_are_built_from_the_repo_and_the_scheme():
    # One repo serves both schemes, so the -FP8 suffix on the repo must be stripped and REPLACED by
    # the requested scheme rather than carried through.
    fam = _fam(prequant_repos = (("int8", "unsloth/Test-FP8"), ("fp8", "unsloth/Test-FP8")))
    assert resolve_prequant_source(fam, "int8").filename == "Test-INT8.pt"
    assert resolve_prequant_source(fam, "fp8").filename == "Test-FP8.pt"
    # The legacy per-scheme name stays available for repos that have not been renamed.
    assert resolve_prequant_source(fam, "int8").fallback_filename == "transformer_int8.pt"


# ── task-keyed artifacts: one repo, one scheme, two denoiser partitions ──────────
#
# MiniMax-H3 hosts a keyframe (fl2va, which also covers text-only) and a reference (ref2va)
# denoiser. They share a class, a config, a 635-key state dict and a base_model_id, so no check
# downstream can tell them apart: the TASK is the only thing standing between a reference load and
# the keyframe weights, and getting it wrong renders plausibly rather than failing.


def test_a_task_specific_row_beats_the_task_agnostic_one():
    fam = _fam(
        prequant_repos = (("int8", "unsloth/Test-FP8"),),
        prequant_filenames = (
            ("int8", "Test-INT8-ConvRot.pt"),
            ("int8", "ref2va", "Test-Ref2VA-INT8-ConvRot.pt"),
        ),
        prequant_partition_tasks = ("ref2va",),
    )
    assert resolve_prequant_source(fam, "int8", task = "ref2va").filename == (
        "Test-Ref2VA-INT8-ConvRot.pt"
    )
    # Case and whitespace must not change which partition is picked.
    assert resolve_prequant_source(fam, "int8", task = " Ref2VA ").filename == (
        "Test-Ref2VA-INT8-ConvRot.pt"
    )


def test_a_task_specific_artifact_gets_no_filename_fallback():
    # The fallback exists so an older name still resolves when the preferred one is absent. Here
    # every other file in the repo is the same family, scheme and base, so a fallback would install
    # ANOTHER PARTITION's denoiser -- it would pass every check and generate the wrong thing. No
    # artifact is the correct outcome: the load keeps the released bfloat16 denoiser.
    fam = _fam(
        prequant_repos = (("int8", "unsloth/Test-FP8"),),
        prequant_filenames = (
            ("int8", "Test-INT8-ConvRot.pt"),
            ("int8", "ref2va", "Test-Ref2VA-INT8-ConvRot.pt"),
        ),
        prequant_partition_tasks = ("ref2va",),
    )
    assert resolve_prequant_source(fam, "int8", task = "ref2va").fallback_filename is None
    # The task-agnostic pick keeps its fallback, unchanged.
    assert resolve_prequant_source(fam, "int8").fallback_filename == "Test-INT8.pt"


def test_a_scheme_without_a_task_row_resolves_exactly_what_it_did_before():
    # Back-compat, stated as an equality rather than a literal: whatever the task-agnostic lookup
    # gives, a task the table says nothing about must give the same thing.
    fam = _fam(
        prequant_repos = (("int8", "unsloth/Test-FP8"), ("fp8", "unsloth/Test-FP8")),
        prequant_filenames = (("int8", "Test-INT8-ConvRot.pt"),),
    )
    for scheme in ("int8", "fp8"):
        plain = resolve_prequant_source(fam, scheme)
        for task in ("fl2va", "t2va", "anything-at-all"):
            assert resolve_prequant_source(fam, scheme, task = task) == plain


def test_a_family_predating_the_task_shape_is_unaffected_by_a_task():
    # A table written entirely as 2-tuples, asked with a task. It must not raise and must not
    # change its answer -- the field is free to ignore for every family with one denoiser.
    import types

    fam = _fam(prequant_repos = (("fp8", "unsloth/Test-FP8"),))
    assert resolve_prequant_source(fam, "fp8", task = "ref2va") == resolve_prequant_source(fam, "fp8")
    assert resolve_prequant_source(types.SimpleNamespace(), "fp8", task = "ref2va") is None


def test_a_partition_task_with_no_artifact_of_its_own_is_unavailable():
    # The refusal's whole condition. The scheme HAS a hosted repo, so the old per-scheme question
    # answers yes; the pair (scheme, task) has nothing, and serving the keyframe file instead is
    # the failure mode this replaced.
    fam = _fam(
        prequant_repos = (("int8", "unsloth/Test-FP8"), ("fp8", "unsloth/Test-FP8")),
        prequant_filenames = (("fp8", "ref2va", "Test-Ref2VA-FP8.pt"),),
        prequant_partition_tasks = ("ref2va",),
    )
    assert video_family_prequant_repo(fam, "int8") == "unsloth/Test-FP8"
    assert video_family_prequant_available(fam, "int8", task = "ref2va") is False
    assert video_family_prequant_available(fam, "fp8", task = "ref2va") is True
    # Not a partition task, so the artifact-per-task rule does not apply.
    assert video_family_prequant_available(fam, "int8", task = "fl2va") is True
    assert video_family_prequant_available(fam, "int8") is True
    # And the refusal message names only what actually works for that task.
    assert video_family_prequant_schemes(fam, task = "ref2va") == ("fp8",)
    assert video_family_prequant_schemes(fam) == ("int8", "fp8")


@pytest.mark.parametrize(
    "scheme, expected",
    [("int8", "MiniMax-H3-Ref2VA-INT8-ConvRot.pt"), ("fp8", "MiniMax-H3-Ref2VA-FP8.pt")],
)
def test_h3_reference_video_resolves_its_own_hosted_denoiser(scheme, expected):
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    src = resolve_prequant_source(fam, scheme, task = "ref2va")
    assert src.filename == expected
    assert src.fallback_filename is None
    assert "/" not in src.filename and "\\" not in src.filename


@pytest.mark.parametrize("task", [None, "fl2va", "t2va"])
def test_h3_keyframe_and_text_only_resolve_exactly_what_they_resolved_before(task):
    # The published fl2va artifacts must not move: the rotated INT8 by name (with the plain one
    # still its fallback for older installs) and FP8 by the derived repo-root name.
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    int8 = resolve_prequant_source(fam, "int8", task = task)
    assert int8.filename == "MiniMax-H3-INT8-ConvRot.pt"
    assert int8.fallback_filename == "MiniMax-H3-INT8.pt"
    fp8 = resolve_prequant_source(fam, "fp8", task = task)
    assert fp8.filename == "MiniMax-H3-FP8.pt"


def test_the_h3_partition_task_matches_the_reference_workflow_name():
    # The registry spells the task as a literal to stay import-free; pin it to the constant the
    # loader and the download planner branch on, so the two cannot drift apart silently.
    from core.inference.video_minimax_h3 import H3_TASK_REFERENCES
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert fam.prequant_partition_tasks == (H3_TASK_REFERENCES,)


def test_a_reference_load_is_refused_when_its_scheme_has_no_reference_artifact(monkeypatch):
    # The refusal is now conditional, not blanket, so it needs a family where the pair genuinely
    # does not exist. int8 here has the repo but only a keyframe artifact.
    fam = _fam(
        name = "partitioned",
        modular_workflow = "fl2va",
        prequant_repos = (("int8", "unsloth/Test-FP8"), ("fp8", "unsloth/Test-FP8")),
        prequant_filenames = (("fp8", "ref2va", "Test-Ref2VA-FP8.pt"),),
        prequant_partition_tasks = ("ref2va",),
    )
    monkeypatch.setattr("core.inference.video._detect_load_family", lambda *a, **k: fam)
    monkeypatch.setattr("core.inference.video._is_trusted_video_repo", lambda repo: True)
    backend = VideoBackend()
    with pytest.raises(ValueError) as excinfo:
        backend.validate_load_request(
            "org/test-video",
            model_kind = "pipeline",
            transformer_quant = "int8",
            h3_task = "ref2va",
        )
    message = str(excinfo.value)
    assert "int8" in message and "ref2va" in message
    # It says what to use instead, and does NOT advertise the scheme that only covers the other
    # partition.
    assert "fp8" in message
    # The pair that DOES exist is not refused. Asserted on the message rather than on success:
    # this synthetic family names a pipeline class the diffusers probe further down cannot find,
    # and that unrelated failure must neither mask a regression nor pass this test for the wrong
    # reason.
    try:
        backend.validate_load_request(
            "org/test-video",
            model_kind = "pipeline",
            transformer_quant = "fp8",
            h3_task = "ref2va",
        )
    except Exception as exc:  # noqa: BLE001
        assert "transformer_quant" not in str(
            exc
        ), f"fp8 ref2va should be loadable but was refused: {exc}"


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
    # "ltx-2.3" is how the repo spells it; the family's canonical name is "ltx-2" and carries the
    # repo spelling as an alias. Pin the resolved family, not the alias that reached it.
    assert fam.name == "ltx-2" and "ltx-2.3" in fam.aliases
    assert fam.modular_workflow is None


# ── keeping the pre-quantized denoiser out of the offload rotation ───────────────
#
# ComponentsManager.enable_auto_cpu_offload parks every component on the CPU and moves each one
# onto the accelerator inside its own pre_forward, i.e. from within the block already executing.
# A torchao pre-quantized denoiser does not survive that mid-block move: the device change reaches
# return_and_correct_aliasing, which tries to alias a CPU storage to an accelerator tensor and
# raises "Attempted to set the storage of a tensor on device cuda:0 to a storage on different
# device cpu", killing MiniMax-H3's denoise loop on its first step. Placing it once at load time
# and unhooking it is the fix, so both halves are asserted: hook removed AND module placed.


class _FakeInnerHook:
    def __init__(self) -> None:
        self.other_hooks: list = []


class _FakeUserHook:
    def __init__(self, model) -> None:
        self.model = model
        self.hook = _FakeInnerHook()
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _FakeModule:
    def __init__(self) -> None:
        self.moved_to: list = []

    def to(self, device):
        self.moved_to.append(device)
        return self


class _FakeOffloadManager:
    def __init__(self, models) -> None:
        self.model_hooks = [_FakeUserHook(model) for model in models]
        for hook in self.model_hooks:
            hook.hook.other_hooks = [other for other in self.model_hooks if other is not hook]


def test_pinning_unhooks_the_denoiser_and_places_it():
    from core.inference.diffusion_prequant import pin_prequantized_module

    transformer, encoder, vae = _FakeModule(), _FakeModule(), _FakeModule()
    manager = _FakeOffloadManager([transformer, encoder, vae])
    denoiser_hook = manager.model_hooks[0]

    assert pin_prequantized_module(manager, transformer, "cuda") is True

    # The hook is gone, so nothing moves the denoiser per forward ...
    assert denoiser_hook.removed is True
    assert denoiser_hook not in manager.model_hooks
    assert [hook.model for hook in manager.model_hooks] == [encoder, vae]
    # ... and no surviving component can pick it as the thing to evict, which would strand it on
    # the CPU with no hook left to bring it back.
    for hook in manager.model_hooks:
        assert denoiser_hook not in hook.hook.other_hooks
    # It is placed exactly once, here, outside any executing block.
    assert transformer.moved_to == ["cuda"]
    # The components that CAN be moved safely keep their hooks and their rotation.
    assert encoder.moved_to == [] and vae.moved_to == []


def test_pinning_still_places_the_module_when_the_manager_is_unrecognisable():
    """Best-effort on the hook surgery: an unmanaged module must still end up on the device."""
    from core.inference.diffusion_prequant import pin_prequantized_module

    transformer = _FakeModule()
    manager = _FakeOffloadManager([_FakeModule()])
    before = list(manager.model_hooks)

    assert pin_prequantized_module(manager, transformer, "cuda") is False
    assert transformer.moved_to == ["cuda"]
    assert manager.model_hooks == before

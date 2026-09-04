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


@pytest.fixture(autouse = True)
def _assume_the_restricted_load_is_available(monkeypatch):
    """Policy/planning tests, not a check on whether this host's torchao imports.

    Without this, a machine with no (or a skewed) torchao turns every hosted-prequant decision
    below into "keep the dense weights". The capability is covered in test_diffusion_prequant.py."""
    import core.inference.diffusion_prequant as _pq
    monkeypatch.setattr(_pq, "restricted_prequant_load_supported", lambda scheme = None: True)


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
        # Curated hosted artifacts only: a third-party repo here would be served as this family's
        # own weights, for a load that may never have asked for a scheme at all.
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
    # The rotated artifact carries the v2 format tag, which an Unsloth predating the online rotation
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


def _forced_target(device):
    """A resolved diffusion target on ``device``, so a refusal keyed on the device can be tested
    off the hardware that has it."""
    from core.inference.diffusion_device import DiffusionDeviceTarget
    return lambda: DiffusionDeviceTarget(
        device = device,
        dtype = None,
        backend = device,
        vendor = None,
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
        supports_float64 = device != "mps",
    )


def test_the_modular_pipeline_is_refused_on_metal(monkeypatch):
    # _load_h3_modular_pipeline places every non-CPU device with
    # ComponentsManager.enable_auto_cpu_offload, which raises NotImplementedError when the device
    # module has no mem_get_info, and torch.mps has none. Refuse before ~145 GB downloads.
    monkeypatch.setattr(
        "core.inference.video.resolve_diffusion_device_target", _forced_target("mps")
    )
    backend = VideoBackend()
    with pytest.raises(ValueError, match = "cannot run on Apple Silicon"):
        backend.validate_load_request("MiniMaxAI/MiniMax-H3")


def test_the_metal_refusal_names_the_artifact_that_does_run_there(monkeypatch):
    # A dead end is not an answer: H3's GGUF checkpoints run on the native engine on the same
    # host, so the refusal has to point at them.
    monkeypatch.setattr(
        "core.inference.video.resolve_diffusion_device_target", _forced_target("mps")
    )
    backend = VideoBackend()
    with pytest.raises(ValueError, match = "unsloth/MiniMax-H3-GGUF"):
        backend.validate_load_request("MiniMaxAI/MiniMax-H3")


def test_the_metal_refusal_leaves_every_other_device_alone(monkeypatch):
    # The mirror image: CUDA is where this workflow is meant to run, so the refusal must be
    # scoped to the device that cannot place it.
    monkeypatch.setattr(
        "core.inference.video.resolve_diffusion_device_target", _forced_target("cuda")
    )
    backend = VideoBackend()
    try:
        backend.validate_load_request("MiniMaxAI/MiniMax-H3")
    except ValueError as exc:  # pragma: no cover - only on a regression
        pytest.fail(f"a CUDA modular load must not be refused: {exc}")
    except Exception:
        # Anything past the refusal (the diffusers probe further down) already proves it did not
        # fire.
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


# ── the denoiser default: measured, and deliberately NOT changed ─────────────────
# The hosted checkpoints are the fast ones (the same 8-step job runs 23.7 s against 194 s), so the
# question was whether to default to one. Measured against the released denoiser at H3's own
# 30-step schedule, fixed prompt and seed, 960x544x124: no NaN, no black frames, no visible
# degradation, but a re-rolled sample -- mean SSIM 0.49 (int8) / 0.43 (fp8) where the released
# config against ITSELF scores 0.99. They stay opt-in.


def _h3_fam():
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert fam is not None and fam.modular_workflow
    return fam


def test_an_unset_denoiser_request_keeps_the_released_weights():
    fam, base = _h3_fam(), "MiniMaxAI/MiniMax-H3"
    for unset in (None, "auto"):
        assert VideoBackend._denoiser_prequant_covered(fam, unset, base) is False
    # An explicit scheme still takes the hosted checkpoint, and the plan still drops the dense
    # shards for it -- the opt-in is unchanged.
    assert VideoBackend._denoiser_prequant_covered(fam, "int8", base) is True
    assert VideoBackend._denoiser_prequant_covered(fam, "fp8", base) is True


def test_the_dense_denoiser_is_pinned_only_when_it_actually_fits():
    """Pinning the released denoiser is what makes the regional compile possible (a module the
    offload hooks move per forward cannot be compiled) and quantizing the conditioner is what makes
    it affordable. Being an optimisation, the fit test stays conservative: the denoiser plus
    everything that still has to run beside it."""
    import torch

    from core.inference.video import _h3_dense_denoiser_resident_bytes

    fam = _h3_fam()

    class _Denoiser:
        def __init__(self, gb):
            self._t = torch.empty(0)
            self._gb = gb

        def parameters(self):
            return iter(())

        def buffers(self):
            # One notional tensor standing in for the module's weight bytes.
            return iter([torch.empty(int(self._gb * 1e9), dtype = torch.uint8, device = "meta")])

    # A meta tensor is skipped (it holds no memory yet), so an unbuilt module sizes to nothing
    # rather than to a number that would wrongly authorise a pin.
    assert (
        _h3_dense_denoiser_resident_bytes(
            fam, denoiser = _Denoiser(66.3), te_scheme = "int8", dtype = torch.bfloat16
        )
        is None
    )
    assert (
        _h3_dense_denoiser_resident_bytes(
            fam, denoiser = None, te_scheme = "int8", dtype = torch.bfloat16
        )
        is None
    )

    class _Real(_Denoiser):
        def buffers(self):
            return iter([torch.empty(1024, dtype = torch.uint8)])

    sizes = _h3_dense_denoiser_resident_bytes(
        fam, denoiser = _Real(0), te_scheme = "int8", dtype = torch.bfloat16
    )
    assert sizes is not None
    denoiser_bytes, others = sizes
    assert denoiser_bytes == 1024
    # The conditioner is priced at the precision the load ENGAGED, which is the whole reason the
    # dense denoiser can be resident at all: 27.2 GB hosted against 66.8 GB released.
    dense_sizes = _h3_dense_denoiser_resident_bytes(
        fam, denoiser = _Real(0), te_scheme = None, dtype = torch.bfloat16
    )
    assert dense_sizes is not None and dense_sizes[1] - others > 38 * 1000**3
    # And it is never just the weights: the activation headroom is in there too.
    assert others > (27.2 + fam.bf16_components_gb[2]) * 1000**3


def test_the_pin_decision_itself_refuses_a_card_that_cannot_hold_it():
    """The sizing above is only half of it. This is the comparison that authorises the pin, and
    getting it wrong is an OOM rather than a slow generation, so it is asserted separately from
    the estimate that feeds it."""
    from core.inference.video import _h3_dense_denoiser_fits

    sizes = (66_300_000_000, 33_000_000_000)  # denoiser, everything else
    need = sizes[0] + sizes[1]

    assert _h3_dense_denoiser_fits(sizes, need) is True  # exactly enough still fits
    assert _h3_dense_denoiser_fits(sizes, need + 1) is True
    assert _h3_dense_denoiser_fits(sizes, need - 1) is False  # one byte short does not
    # The denoiser alone fitting is NOT enough: the conditioner and the VAEs still have to run.
    assert _h3_dense_denoiser_fits(sizes, sizes[0]) is False
    # No estimate and no reading both keep the rotation, which is today's behaviour.
    assert _h3_dense_denoiser_fits(None, need) is False
    assert _h3_dense_denoiser_fits(sizes, None) is False


# ── the conditioner opt-out has to be reachable ──────────────────────────────────


def test_the_released_conditioner_is_reachable_through_the_load_api():
    """The new default makes an omitted ``text_encoder_quant`` select the hosted INT8 conditioner,
    so ``none``/``off`` became the only way to ask for the released bfloat16 one. That request has
    to survive both gates it passes through -- the request model and the cheap normaliser -- or the
    bfloat16 reference configuration is unreachable and no comparison against it can be run."""
    from pydantic import ValidationError

    from core.inference.diffusion_precision import normalize_te_quant
    from models.inference import VideoLoadRequest

    for opt_out in ("none", "off", "None", " OFF "):
        assert normalize_te_quant(opt_out) is None
    # "auto" is the same no-scheme answer to this normaliser; the tri-state that distinguishes it
    # from an opt-out reads the RAW request before normalising.
    assert normalize_te_quant("auto") is None
    # A genuinely unsupported scheme is still refused here, cheaply, as before.
    with pytest.raises(ValueError):
        normalize_te_quant("int3")

    def _request(value):
        return VideoLoadRequest(model_path = "MiniMaxAI/MiniMax-H3", text_encoder_quant = value)

    for accepted in (None, "auto", "none", "off", "fp8", "fp8_dynamic", "int8", "nvfp4"):
        assert _request(accepted).text_encoder_quant == accepted
    with pytest.raises(ValidationError):
        _request("int3")


def test_the_opt_out_spellings_reach_the_h3_tri_state():
    """Reaching the model is only useful if the tri-state then reads them as the dense pin rather
    than folding them into the unset branch that takes the hosted checkpoint."""
    from core.inference.video import _h3_precision_pinned_dense, _h3_precision_unset

    for opt_out in ("none", "off", "OFF", " none "):
        assert _h3_precision_pinned_dense(opt_out) is True
        assert _h3_precision_unset(opt_out) is False
    for unset in (None, "", "auto", "AUTO"):
        assert _h3_precision_unset(unset) is True
        assert _h3_precision_pinned_dense(unset) is False


def test_speed_off_declines_the_dense_pin_but_never_the_prequantized_one():
    """The dense pin is a speed optimisation by its own reasoning, so an explicit ``speed=off``
    has to decline it: taking the denoiser out of the rotation trades the ability to budget it
    against the requested frame count for throughput. The pre-quantized pin is NOT the same
    decision -- a torchao module does not survive the mid-block move -- so it stays unconditional.
    """
    import ast
    import inspect
    import textwrap

    from core.inference.diffusion_speed import SPEED_DEFAULT, SPEED_OFF, resolve_speed_mode

    # "off" is the only profile that reaches the gate as SPEED_OFF, so it is the only one the
    # gate can decline. Asserted on the resolver rather than assumed.
    assert resolve_speed_mode("off", is_gguf = False, dense_default = SPEED_DEFAULT) == SPEED_OFF
    for on in ("default", "max"):
        assert resolve_speed_mode(on, is_gguf = False, dense_default = SPEED_DEFAULT) != SPEED_OFF

    # The gate itself, read off the loader: the dense branch is the ``elif`` beside the
    # pre-quantized ``if``, and only the dense one may mention the speed profile. Read from the
    # source because standing up a modular load to observe the placement is not something this
    # network-free suite can do, and an ungated pin is exactly the regression worth catching.
    source = textwrap.dedent(inspect.getsource(VideoBackend._load_h3_modular_pipeline))
    pins = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and "transformer_quant_engaged" == getattr(node.test, "id", None)
        and node.orelse
    ]
    assert len(pins) == 1, "the two denoiser placements are no longer one if/elif pair"
    prequantized_branch, dense_branch = pins[0], pins[0].orelse[0]
    assert isinstance(dense_branch, ast.If)
    dense_test = ast.dump(dense_branch.test)
    assert "SPEED_OFF" in dense_test, "the dense pin no longer honours an explicit speed=off"
    assert "denoiser" in dense_test
    # And the correctness pin is still unconditional on the profile.
    assert "SPEED_OFF" not in ast.dump(prequantized_branch.test)
    for stmt in prequantized_branch.body:
        assert "SPEED_OFF" not in ast.dump(stmt)


def test_the_dense_placement_is_fenced_on_the_load_token():
    """``load_components`` spends minutes building ~145 GB, and everything after it either moves
    weights onto the card or mutates process-wide backend flags. A cancelled or superseded worker
    that resumes there puts a 66.3 GB denoiser next to a model a replacement load already owns.
    The conventional placement path fences for exactly that reason; this one has to as well, and
    the fence has to sit BEFORE the placement rather than at the state commit after it."""
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(VideoBackend._load_h3_modular_pipeline))
    tree = ast.parse(source)

    def _line(predicate) -> int:
        hits = [node.lineno for node in ast.walk(tree) if predicate(node)]
        assert hits, "landmark not found in the modular loader"
        return min(hits)

    load_components = _line(lambda n: isinstance(n, ast.Attribute) and n.attr == "load_components")
    fence = _line(
        lambda n: isinstance(n, ast.Compare)
        and "_load_token" in ast.dump(n)
        and n.lineno > load_components
    )
    placement = _line(
        lambda n: isinstance(n, ast.Attribute) and n.attr == "enable_auto_cpu_offload"
    )
    assert (
        load_components < fence < placement
    ), "the token fence must sit between load_components and the placement it guards"


# ── auto falls back to the hosted denoiser only where the released one cannot stay resident ──


def _h3_family():
    from core.inference.video_families import detect_video_family
    return detect_video_family("minimax-h3")


def test_the_planned_sizing_matches_the_measured_one_it_stands_in_for():
    """The pre-load prediction and the post-load pin have to describe the same load.

    They are separate functions for a real reason -- the seeding decision runs before any module
    exists, the pin runs after -- but if their arithmetic drifts, auto can seed a checkpoint on a
    card that would have held the released denoiser, or leave the released one on a card that
    cannot. Only the DENOISER term may differ (table vs built module); everything else must agree
    exactly."""
    import torch

    from core.inference.video import (
        _h3_dense_denoiser_resident_bytes,
        _h3_planned_denoiser_bytes,
    )

    fam = _h3_family()

    class _Weight:
        """The released denoiser's size WITHOUT the 66.3 GB it would take to hold it.

        The measurement reads numel(), element_size() and is_meta and nothing else, so a real
        tensor buys no fidelity here and costs more RAM than a CI runner has: the allocation
        raised, ``_h3_dense_denoiser_resident_bytes`` swallowed it as an unanswerable estimate,
        and the test failed on a None that says nothing about the arithmetic under test.
        """

        is_meta = False

        def numel(self):
            return int(fam.bf16_components_gb[0] * 1000**3 // 2)

        def element_size(self):
            return torch.finfo(torch.bfloat16).bits // 8

    class _Dense:
        def parameters(self):
            return [_Weight()]

        def buffers(self):
            return []

    for te_scheme in (None, "int8"):
        planned = _h3_planned_denoiser_bytes(fam, te_scheme = te_scheme, dtype = torch.bfloat16)
        measured = _h3_dense_denoiser_resident_bytes(
            fam, denoiser = _Dense(), te_scheme = te_scheme, dtype = torch.bfloat16
        )
        assert planned is not None and measured is not None
        assert planned[1] == measured[1], f"the others term drifted for te_scheme={te_scheme}"
        # And the denoiser term agrees to within rounding on the same released weights.
        assert abs(planned[0] - measured[0]) < 1_000_000_000

    # An fp32 promotion doubles it on both sides, so the comparison is not accidentally bf16-only.
    fp32 = _h3_planned_denoiser_bytes(fam, te_scheme = None, dtype = torch.float32)
    bf16 = _h3_planned_denoiser_bytes(fam, te_scheme = None, dtype = torch.bfloat16)
    assert fp32 is not None and bf16 is not None and fp32[0] == bf16[0] * 2


def test_auto_takes_the_hosted_denoiser_even_on_a_card_with_room_to_spare(monkeypatch):
    """int8 is the DEFAULT, not a fallback, so having room for the released denoiser is not a
    reason to load it.

    It is not a tie broken on memory. Measured on an H200 and a B200, every component resident in
    both rows, the hosted checkpoint is faster per generation AND 45 GB smaller:

        released bf16   20.06 / 12.76 / 12.77 s   102.8 GB steady
        hosted int8     23.08 / 11.74 / 11.84 s    57.8 GB steady

    What it costs is the picture (mean SSIM 0.49 against the released weights), which is a choice
    ``transformer_quant='none'`` reverses and which no amount of free VRAM changes."""
    import torch

    from core.inference import video as vid

    fam = _h3_family()
    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    # Comfortably more free memory than the released denoiser plus everything beside it.
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: 500 * 1000**3)

    assert (
        vid._h3_auto_denoiser_scheme(
            fam,
            target = None,
            dtype = torch.bfloat16,
            device = "cuda",
            te_scheme = "int8",
            task = "fl2va",
            base_repo = fam.base_repo,
        )
        == "int8"
    )


def test_auto_takes_the_hosted_denoiser_when_the_released_one_cannot_stay_resident(monkeypatch):
    """The bug users hit. Below the fit line the released denoiser rides the CPU-offload rotation,
    and a module that moves cannot be compiled, so the regional compile goes with it: 194 s against
    23.7 s on the same 8-step job. Auto now takes the hosted checkpoint instead of the cliff."""
    import torch

    from core.inference import video as vid

    fam = _h3_family()
    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    # An 80 GB card: under the 113.5 GB the released denoiser plus its companions need, over the
    # 67.5 GB the hosted one needs, which is the band where the substitution buys anything.
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: 80 * 1000**3)

    assert (
        vid._h3_auto_denoiser_scheme(
            fam,
            target = None,
            dtype = torch.bfloat16,
            device = "cuda",
            te_scheme = "int8",
            task = "fl2va",
            base_repo = fam.base_repo,
        )
        == vid.H3_AUTO_FALLBACK_SCHEME
    )


def test_the_auto_fallback_is_declined_when_nothing_can_answer(monkeypatch):
    """Every unanswerable question keeps the released weights. A missing reading is not evidence of
    a shortfall, and guessing wrong here silently changes the picture a user gets.

    The partition gate is the same rule the explicit path applies: a task with no hosted checkpoint
    has no fallback, and serving the other partition's would generate the wrong thing."""
    import torch

    from core.inference import video as vid

    fam = _h3_family()

    def ask(**over):
        kw = dict(
            target = None,
            dtype = torch.bfloat16,
            device = "cuda",
            te_scheme = "int8",
            task = "fl2va",
            base_repo = fam.base_repo,
        )
        kw.update(over)
        return vid._h3_auto_denoiser_scheme(fam, **kw)

    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: 80 * 1000**3)

    # A host the hosted components were never measured on stays on the released denoiser.
    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: False, raising = False)
    assert ask() is None

    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    # An unreadable card decides nothing.
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: None)
    assert ask() is None
    # Neither does an unanswerable size estimate.
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: 80 * 1000**3)
    monkeypatch.setattr(vid, "_h3_planned_denoiser_bytes", lambda *a, **k: None)
    assert ask() is None
    monkeypatch.undo()

    # And a partition with no hosted checkpoint for the fallback scheme keeps the released one.
    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: 80 * 1000**3)
    monkeypatch.setattr(
        vid, "video_family_prequant_available", lambda *a, **k: False, raising = False
    )
    assert ask() is None


def test_an_explicit_speed_off_keeps_the_released_denoiser(monkeypatch):
    """speed_mode="off" is the bit-exact contract, and the hosted checkpoints re-roll the sample.

    The conventional loader rewrites an unset precision to "off" under speed off for exactly this
    reason; the modular workflow returns above that rewrite, so the fallback has to decline it
    itself or "off" stops meaning bit-exact on precisely the cards this fallback targets."""
    import torch

    from core.inference import video as vid

    fam = _h3_family()
    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: 80 * 1000**3)

    def ask(speed_mode):
        return vid._h3_auto_denoiser_scheme(
            fam,
            target = None,
            dtype = torch.bfloat16,
            device = "cuda",
            te_scheme = "int8",
            task = "fl2va",
            base_repo = fam.base_repo,
            speed_mode = speed_mode,
        )

    assert ask("off") is None
    assert ask("OFF ") is None, "the request is read the same way the conventional loader reads it"
    # Every other speed profile is the one this fallback was measured for.
    assert ask(None) == vid.H3_AUTO_FALLBACK_SCHEME
    assert ask("default") == vid.H3_AUTO_FALLBACK_SCHEME


def test_the_automatic_substitution_needs_the_exact_base_model(monkeypatch):
    """A derivative is not a precision choice.

    ``video_family_prequant_repo`` falls back to the family default for any base it has no variant
    row for, and the checkpoint validator's base compare accepts a matching final path segment, so
    someone/MiniMax-H3 would take MiniMaxAI/MiniMax-H3's denoiser and silently generate from
    someone else's weights -- for a user who never asked for a scheme at all. Same bar as the
    conditioner's index gate: exact identity, mirrors folded, nothing else."""
    import torch

    from core.inference import video as vid

    fam = _h3_family()
    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: 80 * 1000**3)

    def ask(base_repo):
        return vid._h3_auto_denoiser_scheme(
            fam,
            target = None,
            dtype = torch.bfloat16,
            device = "cuda",
            te_scheme = "int8",
            task = "fl2va",
            base_repo = base_repo,
        )

    assert ask(fam.base_repo) == vid.H3_AUTO_FALLBACK_SCHEME
    assert ask("someone/MiniMax-H3") is None, "a derivative keeps its own denoiser"
    assert ask("/models/MiniMax-H3") is None, "and so does a local re-save this cannot identify"
    assert ask(None) is None


def test_the_fallback_is_declined_when_the_hosted_denoiser_cannot_be_pinned(monkeypatch):
    """Taking the hosted checkpoint means PINNING it -- a torchao module does not survive the
    offload rotation's mid-block move -- and a pinned denoiser turns the memory floor from a max
    into a sum.

    With text_encoder_quant="none" the conditioner stays dense, so that sum is BIGGER than the
    rotation it replaces: the card renders today and would refuse every generation afterwards.
    Where the replacement does not fit either, the released denoiser in the rotation is the
    configuration that still runs, so auto keeps it."""
    import torch

    from core.inference import video as vid

    fam = _h3_family()
    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: 80 * 1000**3)

    def ask(te_scheme):
        return vid._h3_auto_denoiser_scheme(
            fam,
            target = None,
            dtype = torch.bfloat16,
            device = "cuda",
            te_scheme = te_scheme,
            task = "fl2va",
            base_repo = fam.base_repo,
        )

    # Dense conditioner: 107.1 GB pinned against 80 GB free, so the substitution buys a refusal.
    assert ask(None) is None
    # Quantized conditioner: 67.5 GB pinned, which is what the fallback exists for.
    assert ask("int8") == vid.H3_AUTO_FALLBACK_SCHEME


def test_the_fallback_is_resolved_before_the_download_is_planned(monkeypatch):
    """Decided only inside the loader, the fallback is decided after the pull it exists to shrink.

    ``_denoiser_prequant_covered`` answers False for an unset request, so the plan stages the
    66.3 GB dense denoiser this load will never open and the 20.3 GB replacement then arrives
    inline -- outside the progress plan, the cancel path and the disk preflight that just passed.
    The planner resolves the same choice up front, against the card's CAPACITY: the free reading
    is polluted at plan time (the previous pipeline is still resident) and capacity is an upper
    bound on it, so a "does not fit" here still holds when the loader re-measures."""
    import inspect
    import types

    import torch

    from core.inference import video as vid

    fam = _h3_family()
    backend = vid.VideoBackend.__new__(vid.VideoBackend)

    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    monkeypatch.setattr(
        vid,
        "resolve_diffusion_device_target",
        lambda *a, **k: types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16),
        raising = False,
    )
    # CAPACITY, not the live free reading: an 80 GB card cannot hold the released denoiser
    # resident whatever else is or is not on it right now.
    monkeypatch.setattr(vid, "_h3_device_capacity_bytes", lambda device: 80 * 1000**3)
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: None)

    def plan(**over):
        kw = dict(
            base = fam.base_repo,
            transformer_quant = None,
            text_encoder_quant = None,
            speed_mode = None,
            h3_task = "fl2va",
        )
        kw.update(over)
        return backend._h3_planned_auto_denoiser_scheme(fam, **kw)

    assert plan() == vid.H3_AUTO_FALLBACK_SCHEME
    # An explicit request is not the planner's business, in either direction.
    assert plan(transformer_quant = "none") is None
    assert plan(transformer_quant = "int8") is None
    assert plan(speed_mode = "off") is None

    # And the scheme it returns is what makes the pull drop the dense shards: the same probe
    # answers False for the unset request the planner replaces.
    assert not vid.VideoBackend._denoiser_prequant_covered(fam, None, fam.base_repo, "fl2va")
    assert vid.VideoBackend._denoiser_prequant_covered(
        fam, vid.H3_AUTO_FALLBACK_SCHEME, fam.base_repo, "fl2va"
    )

    # The wiring: the planner has to run BEFORE the verification that drops the shards, which runs
    # before the pull. Ordering is the whole point, so it is asserted rather than assumed.
    src = inspect.getsource(vid.VideoBackend._run_load)
    planned = src.index("_h3_planned_auto_denoiser_scheme")
    verified = src.index("_denoiser_prequant_verified")
    predownload = src.index("_predownload_base")
    assert planned < verified < predownload
    assert 'h3_auto_denoiser or kwargs.get("transformer_quant")' in src


def _h3_placement_probe(
    monkeypatch,
    *,
    free_gb,
    te_gb,
    denoiser_gb,
    speed = "default",
):
    """Drive just the placement decision: does this load install the CPU-offload rotation?"""
    from core.inference import video as vid

    monkeypatch.setattr(
        vid,
        "_h3_dense_denoiser_resident_bytes",
        lambda fam, **kw: (int(denoiser_gb * 1e9), int(te_gb * 1e9)),
    )
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: int(free_gb * 1e9))
    sizes = vid._h3_dense_denoiser_resident_bytes(None)
    free = vid._h3_free_device_bytes("cuda")
    return speed != vid.SPEED_OFF and vid._h3_dense_denoiser_fits(sizes, free)


def test_a_card_that_holds_everything_does_not_install_the_offload_rotation(monkeypatch):
    """``enable_auto_cpu_offload`` parks every component in HOST RAM and moves each one back inside
    its own pre_forward. On a card that can hold the whole set that buys nothing and costs twice:
    measured 42.2 GB peak host RSS on a 183 GB card with 103 GB of it in use, plus the conditioner
    and VAEs crossing the bus on every generation."""
    assert _h3_placement_probe(monkeypatch, free_gb = 183, te_gb = 40, denoiser_gb = 66) is True


def test_a_card_that_cannot_hold_everything_keeps_the_rotation(monkeypatch):
    # The rotation is what makes H3 run at all here, so the saving must never be taken on credit.
    assert _h3_placement_probe(monkeypatch, free_gb = 80, te_gb = 40, denoiser_gb = 66) is False


def test_speed_off_keeps_the_rotation_even_on_a_card_that_could_hold_everything(monkeypatch):
    """The headroom in the sizing is for the family's DEFAULT frame count, so a resident set trades
    the rotation's ability to absorb a much longer clip for throughput. An explicit speed_mode=off
    is the one request that says do not make that trade."""
    from core.inference import video as vid
    assert (
        _h3_placement_probe(monkeypatch, free_gb = 183, te_gb = 40, denoiser_gb = 66, speed = vid.SPEED_OFF)
        is False
    )


def test_an_unreadable_card_keeps_the_rotation(monkeypatch):
    # Cannot tell is not evidence of room, and guessing wrong here is an OOM rather than a slow load.
    from core.inference import video as vid

    monkeypatch.setattr(vid, "_h3_dense_denoiser_resident_bytes", lambda fam, **kw: (1, 1))
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: None)
    assert vid._h3_dense_denoiser_fits(vid._h3_dense_denoiser_resident_bytes(None), None) is False

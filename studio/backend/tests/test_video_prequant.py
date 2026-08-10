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


# ── the denoiser default: measured, and deliberately NOT changed ─────────────────
# The hosted checkpoints are the fast ones (the same 8-step job runs 23.7 s against 194 s), so the
# question was whether to make one of them the default. Measured against the released denoiser at
# MiniMax-H3's own 30-step schedule, fixed prompt and seed, 960x544x124: no NaN, no black frames
# and no visible degradation, but a re-rolled sample -- mean SSIM 0.49 (int8) and 0.43 (fp8) where
# the released config compared against ITSELF scores 0.99. They stay opt-in.


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
    offload hooks move per forward cannot be compiled), and quantizing the conditioner is what
    makes it affordable. It is an optimisation, so the fit test has to be conservative: the
    denoiser plus everything that still has to run beside it."""
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

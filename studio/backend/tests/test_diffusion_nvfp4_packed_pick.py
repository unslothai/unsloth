# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit nvfp4 pick of FLUX.3 Klein loads BFL's NVFP4 denoiser as a single-file pick with its layers kept packed,
while the family's NVFP4 deny (no gate record) keeps auto and the W4A4 runtime off."""

import pytest

from core.inference import diffusion_families as df
from core.inference import diffusion_transformer_quant as tq

BASE = "black-forest-labs/flux-3-image-klein"
FILE_REPO = "black-forest-labs/flux-3-image-klein-comfy"
FILE = "diffusion_models/flux3-image-klein-nvfp4.safetensors"


@pytest.fixture(autouse = True)
def _nvfp4_on(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    monkeypatch.delenv(df.NVFP4_PACKED_PICK_ENV, raising = False)


def _pick(repo = BASE, **kw):
    kw.setdefault("transformer_quant", "nvfp4")
    return df.nvfp4_packed_pick(repo, **kw)


def test_explicit_nvfp4_pick_of_the_base_becomes_the_packed_single_file_load():
    assert _pick() == {
        "repo_id": FILE_REPO,
        "gguf_filename": FILE,
        "base_repo": BASE,
        "family_override": "flux.3-klein",
        "model_kind": "single_file",
        "transformer_quant": "none",
    }
    # spelling and the ungated mirror of the base are the same pick
    assert _pick(" Black-Forest-Labs/FLUX-3-Image-Klein ", transformer_quant = "NVFP4") is not None
    assert _pick(df.mirror_repo(BASE), model_kind = "pipeline", base_repo = BASE) is not None


def test_the_deny_still_rules_nvfp4_out_for_auto_and_the_runtime():
    # nothing lifts the deny: auto never offers nvfp4 and the W4A4 runtime stays off for the file's layers
    assert tq.family_denies_scheme("flux.3-klein", "nvfp4", BASE)
    from core.inference.diffusion_comfy_block import comfy_block_backend

    runtime, reason = comfy_block_backend("nvfp4", None, "flux.3-klein")
    assert runtime is None and "denied" in reason


@pytest.mark.parametrize(
    "kw",
    [
        {"transformer_quant": "auto"},
        {"transformer_quant": None},
        {"transformer_quant": "int8"},
        {"gguf_filename": "flux-3-image-klein-Q4_K_M.gguf"},
        {"model_kind": "gguf"},
        {"model_kind": "single_file"},
        {"transformer_prequant_path": "/models/mine.safetensors"},
        {"base_repo": "someone/flux-3-image-klein-finetune"},
    ],
)
def test_other_requests_are_left_alone(kw):
    assert _pick(**kw) is None


@pytest.mark.parametrize(
    "repo",
    [
        "black-forest-labs/flux-3-image-klein-base",  # no file named for the base
        "Qwen/Qwen-Image-2.1",  # another family
        "someone/flux-3-image-klein-finetune",  # same family, other weights
        "/models/flux-3-image-klein",  # a local folder
    ],
)
def test_only_the_family_base_itself_is_redirected(repo):
    assert _pick(repo) is None


def test_nvfp4_switch_off_keeps_todays_refusal(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "0")
    assert _pick() is None


def test_kill_switch_restores_the_refusal(monkeypatch):
    monkeypatch.setenv(df.NVFP4_PACKED_PICK_ENV, "0")
    assert _pick() is None
    fam = df.detect_family(BASE)
    assert not df.nvfp4_packed_file_matches(fam, FILE)


def test_a_gate_record_lifting_the_deny_keeps_the_regular_nvfp4_path(monkeypatch):
    monkeypatch.setattr(tq, "family_denies_scheme", lambda *a, **k: False)
    assert _pick() is None


def test_the_packed_file_matches_by_repo_relative_name():
    fam = df.detect_family(BASE)
    snap = f"/hf/hub/models--black-forest-labs--flux-3-image-klein-comfy/snapshots/abc/{FILE}"
    assert df.nvfp4_packed_file_matches(fam, snap)
    assert df.nvfp4_packed_file_matches(fam, FILE)
    assert df.nvfp4_packed_file_matches(fam, snap.replace("/", "\\"))
    assert not df.nvfp4_packed_file_matches(fam, snap.replace("nvfp4", "int8-convrot"))
    assert not df.nvfp4_packed_file_matches(fam, "/x/other-flux3-image-klein-nvfp4.safetensors")
    assert not df.nvfp4_packed_file_matches(df.detect_family("Qwen/Qwen-Image-2.1"), snap)
    assert not df.nvfp4_packed_file_matches(fam, None)


def test_the_load_request_is_rewritten_before_the_route_preflight():
    from models.inference import DiffusionLoadRequest

    req = DiffusionLoadRequest(model_path = BASE, transformer_quant = "nvfp4")
    assert (req.model_path, req.gguf_filename, req.model_kind) == (FILE_REPO, FILE, "single_file")
    assert (req.base_repo, req.family_override, req.transformer_quant) == (
        BASE,
        "flux.3-klein",
        "none",
    )
    # the retired full copy resolves to the base first, then becomes the same pick
    req = DiffusionLoadRequest(model_path = "unsloth/flux-3-image-klein", transformer_quant = "nvfp4")
    assert (req.model_path, req.gguf_filename) == (FILE_REPO, FILE)
    plain = DiffusionLoadRequest(model_path = BASE, transformer_quant = "int8")
    assert (plain.model_path, plain.gguf_filename, plain.transformer_quant) == (BASE, None, "int8")


def test_the_load_request_is_unchanged_with_the_kill_switch(monkeypatch):
    from models.inference import DiffusionLoadRequest

    monkeypatch.setenv(df.NVFP4_PACKED_PICK_ENV, "0")
    req = DiffusionLoadRequest(model_path = BASE, transformer_quant = "nvfp4")
    assert (req.model_path, req.gguf_filename, req.transformer_quant) == (BASE, None, "nvfp4")


class _Validated(Exception):
    pass


@pytest.mark.parametrize("entry", ["begin_load", "load_pipeline"])
def test_backend_entry_points_rewrite_a_direct_call(monkeypatch, entry):
    from core.inference import diffusion as D

    seen = {}

    def _validate(self, repo_id, **kw):
        seen.update(repo_id = repo_id, **kw)
        raise _Validated

    monkeypatch.setattr(D.DiffusionBackend, "validate_load_request", _validate)
    backend = D.DiffusionBackend()
    with pytest.raises(_Validated):
        getattr(backend, entry)(BASE, transformer_quant = "nvfp4")
    assert seen["repo_id"] == FILE_REPO
    assert seen["gguf_filename"] == FILE
    assert seen["family_override"] == "flux.3-klein"
    assert seen["model_kind"] == "single_file"


def test_a_packed_block_denoiser_has_a_measured_streaming_headroom(monkeypatch):
    from core.inference import diffusion_memory as dm

    whole = dm.measured_image_runtime_mib(
        "flux.3-klein", "default", packed_block = True, dense_tier_seeds = True
    )
    streamed = dm.measured_image_runtime_mib(
        "flux.3-klein", "default", packed_block = True, dense_tier_seeds = True, streamed = True
    )
    assert whole == 3072 and streamed == 2304  # seed x 1.15, rounded up to 256 MiB
    assert streamed < whole
    # deferred / eager tiers alike (the seed was measured on both), larger renders scale it
    assert (
        dm.measured_image_runtime_mib(
            "flux.3-klein", "deferred", packed_block = True, dense_tier_seeds = True
        )
        == whole
    )
    assert (
        dm.measured_image_runtime_mib(
            "flux.3-klein",
            "default",
            packed_block = True,
            dense_tier_seeds = True,
            width = 2048,
            height = 1024,
        )
        > whole
    )
    # an fp32 pipeline (FLUX.3 on an fp16-only card) was not measured: learned peak / flat fallback only
    assert (
        dm.measured_image_runtime_mib(
            "flux.3-klein", "default", packed_block = True, dense_tier_seeds = True, compute_bytes = 4
        )
        is None
    )
    assert (
        dm.measured_image_runtime_mib(
            "flux.3-klein", "default", packed_block = True, dense_tier_seeds = True, compute_bytes = 2
        )
        == whole
    )
    # math-only / unprobed attention or not NVIDIA (``_measured_target`` False): learned peak / flat fallback only
    assert dm.measured_image_runtime_mib("flux.3-klein", "default", packed_block = True) is None
    # unmeasured family, a dense / torchao denoiser, or the switch off: no packed seed
    assert (
        dm.measured_image_runtime_mib(
            "qwen-image-2.1-turbo", "off", packed_block = True, dense_tier_seeds = True
        )
        is None
    )
    assert dm.measured_image_runtime_mib("flux.3-klein", "default", static_seed = False) is None
    monkeypatch.setenv(dm.PACKED_BLOCK_RESIDENCY_ENV, "0")
    assert (
        dm.measured_image_runtime_mib(
            "flux.3-klein", "default", packed_block = True, dense_tier_seeds = True
        )
        is None
    )


def test_packed_block_holders_are_read_from_the_denoisers():
    import types

    from core.inference import diffusion_memory as dm
    from core.inference.diffusion_packed_block import PACKED_BLOCK_ATTR

    held = types.SimpleNamespace(**{PACKED_BLOCK_ATTR: 198})
    assert dm._pipe_denoisers_hold_packed_blocks(types.SimpleNamespace(transformer = held))
    assert not dm._pipe_denoisers_hold_packed_blocks(
        types.SimpleNamespace(transformer = types.SimpleNamespace())
    )
    assert not dm._pipe_denoisers_hold_packed_blocks(types.SimpleNamespace())


def test_request_sizing_reads_the_packed_block_seed_the_placement_reserved():
    import types

    from core.inference import diffusion_memory as dm

    # the reserve diffusion.py records for a streamed packed-block placement (static_seed False, streamed, packed)
    reserve = (
        2304,
        "flux.3-klein",
        "default",
        5038,
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        False,
        True,
        True,
    )
    pipe = types.SimpleNamespace(_unsloth_measured_reserve = reserve, _unsloth_dense_tier_seeds = True)
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == 0
    bigger = dm.measured_request_extra_mib(pipe, width = 2048, height = 2048)
    assert (
        bigger
        == dm.measured_image_runtime_mib(
            "flux.3-klein",
            "default",
            width = 2048,
            height = 2048,
            packed_block = True,
            dense_tier_seeds = True,
            streamed = True,
        )
        - 2304
    )
    assert bigger > 0
    # off the measured target (math-only attention) the packed seed is never read: only a learned peak sizes it
    off = types.SimpleNamespace(_unsloth_measured_reserve = reserve, _unsloth_dense_tier_seeds = False)
    assert dm.measured_request_extra_mib(off, width = 2048, height = 2048) == 0

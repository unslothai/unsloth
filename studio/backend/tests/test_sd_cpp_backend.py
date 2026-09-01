# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the native sd.cpp diffusion backend (the no-GPU engine)."""

from __future__ import annotations

import threading
import types

import pytest
from PIL import Image

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import (
    _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS,
    detect_family,
)
from core.inference.sd_cpp_args import SdCppGenParams, SdCppModelFiles
from core.inference.sd_cpp_backend import (
    SdCppDiffusionBackend,
    _map_guidance,
    ensure_sd_cpp_binary,
)
from core.inference.sd_cpp_engine import SdCppCancelled


class _FakeEngine:
    """Stands in for SdCppEngine: writes a 1x1 PNG and records the args."""

    def __init__(
        self,
        *,
        fail = None,
        cancel_on_call = False,
    ):
        self.calls = []
        self.fail = fail
        self.cancel_on_call = cancel_on_call

    def is_available(self):
        return True

    def version(self, **_):
        return "fake sd-cli"

    def generate(
        self,
        files,
        params,
        *,
        output_path,
        cancel_event = None,
        **kw,
    ):
        self.calls.append((files, params, output_path, kw))
        if self.cancel_on_call and cancel_event is not None:
            cancel_event.set()
        if self.fail is not None:
            raise self.fail
        if cancel_event is not None and cancel_event.is_set():
            raise SdCppCancelled("cancelled")
        Image.new("RGB", (1, 1), (10, 20, 30)).save(output_path)
        from pathlib import Path

        return Path(output_path)


def _loaded_backend(fam_name = "z-image", engine = None):
    b = SdCppDiffusionBackend(engine = engine or _FakeEngine())
    fam = detect_family(fam_name)
    b._state = bk._SdState(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        base_repo = fam.base_repo,
        family = fam,
        device = "cpu",
        files = SdCppModelFiles(
            diffusion_model = "/m/z.gguf", vae = "/m/vae.safetensors", llm = "/m/llm.safetensors"
        ),
        vae_format = fam.sd_cpp_vae_format,
        sampling_method = fam.sd_cpp_sampling_method,
        flow_shift = fam.sd_cpp_flow_shift,
        mode = "oneshot",
    )
    return b


def test_loaded_repo_ids_includes_native_companions():
    # The one-shot engine re-reads its companion VAE / encoder files per generation, so surface those repos.
    b = _loaded_backend("flux.1")
    ids = set(b.loaded_repo_ids())
    fam = detect_family("flux.1")
    assert "unsloth/Z-Image-Turbo-GGUF" in ids
    assert fam.base_repo in ids
    assert fam.sd_cpp_vae[0] in ids
    for terepo, _f, _k in fam.sd_cpp_text_encoders:
        assert terepo in ids
    b._state = None
    assert b.loaded_repo_ids() == ()


def test_loaded_repo_ids_tracks_variant_encoder_by_gguf_filename():
    # A local *klein-9B*.gguf carries the variant only in the basename, so the filename must be included.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family("flux.2-klein")
    b._state = bk._SdState(
        repo_id = "local/my-klein-checkpoints",
        base_repo = fam.base_repo,
        family = fam,
        device = "cpu",
        files = SdCppModelFiles(diffusion_model = "/m/FLUX.2-klein-9B-Q4_K_M.gguf"),
        vae_format = fam.sd_cpp_vae_format,
        sampling_method = fam.sd_cpp_sampling_method,
        flow_shift = fam.sd_cpp_flow_shift,
        mode = "oneshot",
        gguf_filename = "FLUX.2-klein-9B-Q4_K_M.gguf",
    )
    ids = set(b.loaded_repo_ids())
    assert "unsloth/FLUX.2-klein-9B-ComfyUI" in ids
    assert "unsloth/Z-Image-Turbo-ComfyUI" not in ids


class _FakeServer:
    """Stands in for SdCppServer: records the spawn + one img_gen per whole batch."""

    def __init__(self, binary):
        self.binary = binary
        self.started = None
        self.stopped = False
        self.payloads = []
        self.timeouts = []
        self.alive = True
        self.lora_dir = None
        self.img_gen_error = None

    def is_alive(self):
        return self.alive and not self.stopped

    def start(
        self,
        files,
        *,
        vae_format = None,
        offload = None,
        native_speed = None,
        threads = None,
        extra_args = None,
    ):
        self.started = dict(
            files = files,
            vae_format = vae_format,
            offload = offload,
            native_speed = native_speed,
            threads = threads,
            extra_args = list(extra_args or []),
        )

    def img_gen(
        self,
        payload,
        *,
        on_step = None,
        cancel_event = None,
        total_timeout = None,
    ):
        import io as _io

        self.payloads.append(payload)
        self.timeouts.append(total_timeout)
        if self.img_gen_error is not None:
            err, self.img_gen_error = self.img_gen_error, None
            self.alive = False
            raise err
        if on_step is not None:
            steps = payload.get("sample_params", {}).get("sample_steps", 0)
            on_step(f"  {steps}/{steps}")
        n = int(payload.get("batch_count", 1))
        blobs = []
        for i in range(n):
            buf = _io.BytesIO()
            Image.new("RGB", (1, 1), (i, i, i)).save(buf, format = "PNG")
            blobs.append(buf.getvalue())
        return blobs

    def stop(self):
        self.stopped = True




@pytest.mark.parametrize(
    "fam_name,expect_kinds",
    [
        ("flux.1", {"diffusion_model", "vae", "clip_l", "t5xxl"}),
        ("z-image", {"diffusion_model", "vae", "llm"}),
        ("qwen-image", {"diffusion_model", "vae", "qwen2vl"}),
        ("flux.2-klein", {"diffusion_model", "vae", "llm"}),
    ],
)
def test_asset_specs_cover_required_files(fam_name, expect_kinds):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family(fam_name)
    specs = b._asset_specs("unsloth/x-GGUF", "x-Q4_K_M.gguf", fam)
    kinds = {kind for _, _, kind in specs}
    assert kinds == expect_kinds
    assert all(repo and fn for repo, fn, _ in specs)
    tr = [s for s in specs if s[2] == "diffusion_model"][0]
    assert tr[0] == "unsloth/x-GGUF" and tr[1] == "x-Q4_K_M.gguf"


@pytest.fixture(autouse = True)
def _plan_sees_an_empty_cache(monkeypatch):
    """Plan tests describe their cache state, so a developer's real one cannot drop an entry."""
    from core.inference.diffusion import DiffusionBackend
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kwargs: False),
    )


def test_download_plan_skips_assets_already_in_the_cache(monkeypatch):
    # _fetch_assets resolves either cache root, so staging a file it can open re-downloads and fails offline.
    from core.inference.diffusion import DiffusionBackend

    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(
        SdCppDiffusionBackend,
        "_plan_file_sizes",
        staticmethod(
            lambda by_repo, token: {
                ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"): 4_000,
                ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/vae/ae.safetensors"): 300,
                (
                    "unsloth/Z-Image-Turbo-ComfyUI",
                    "split_files/text_encoders/qwen_3_4b.safetensors",
                ): 8_000,
            }
        ),
    )
    cached = {"z-image-turbo-Q4_K_M.gguf"}

    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(
            lambda repo_id, filename, revision = None, expected_size = None, **kwargs: (
                filename in cached
            )
        ),
    )

    plan = b.download_plan(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
    )
    cached.add("split_files/vae/ae.safetensors")
    warming = b.download_plan(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert [e["repo_id"] for e in plan["entries"]] == ["unsloth/Z-Image-Turbo-ComfyUI"]
    assert plan["entries"][0]["files"] == warming["entries"][0]["files"]
    assert plan["total_bytes"] == 8_300
    assert warming["total_bytes"] == 8_000
    assert plan["required_bytes"] == 12_300
    assert plan["checkpoint_bytes"] == 4_000


def test_download_plan_does_not_label_same_repo_companions_as_checkpoint(monkeypatch):
    import core.inference.sd_cpp_backend as module
    from core.inference.diffusion import DiffusionBackend

    repo = "unsloth/Z-Image-Turbo-GGUF"
    checkpoint = "model-Q4_K_M.gguf"
    companion = "vae/ae.safetensors"
    backend = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(
        backend,
        "_asset_specs",
        lambda *args, **kwargs: [
            (repo, checkpoint, "diffusion_model"),
            (repo, companion, "vae"),
        ],
    )
    monkeypatch.setattr(module, "_fetch_repo_map", lambda specs, token: {repo: repo})
    monkeypatch.setattr(backend, "_preflight_companion_repos", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_plan_file_sizes",
        lambda by_repo, token: {(repo, checkpoint): 4_000, (repo, companion): 300},
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_loadable",
        staticmethod(lambda _repo, filename, *_args, **_kwargs: filename == checkpoint),
    )

    plan = backend.download_plan(repo, gguf_filename = checkpoint, model_kind = "gguf")

    assert plan["entries"][0]["bytes"] == 300
    assert plan["entries"][0]["checkpoint"] is False


def test_download_plan_restages_a_native_asset_a_stale_live_copy_shadows(monkeypatch):
    # reuse_other_cache_root only switches roots when the LIVE lookup finds nothing, so a stale
    # same-named copy in the live root shadows the good one and crediting the other root stages nothing.
    from core.inference.diffusion import DiffusionBackend

    shadowed = "split_files/vae/ae.safetensors"

    def probe(
        repo_id,
        filename,
        revision = None,
        expected_size = None,
        roots = None,
        **kwargs,
    ):
        asks_live = roots is not None and roots != (None,)
        if filename == shadowed:
            return expected_size is None if asks_live else True
        return True

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(probe))

    assert not DiffusionBackend._hub_file_is_loadable("r", shadowed, None, 300)
    assert DiffusionBackend._hub_file_is_loadable("r", "other.safetensors", None, 300)


def test_download_plan_restages_a_native_asset_that_changed_size(monkeypatch):
    # A same-named republish is invisible to the cache probe from the ref alone, so pass the declared
    # size or the stale copy reads as complete and the load fetches it inline, outside the preflight.
    from core.inference.diffusion import DiffusionBackend

    b = SdCppDiffusionBackend(engine = _FakeEngine())
    seen = {}
    monkeypatch.setattr(
        SdCppDiffusionBackend,
        "_plan_file_sizes",
        staticmethod(
            lambda by_repo, token: {
                ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"): 4_000,
                ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/vae/ae.safetensors"): 300,
                (
                    "unsloth/Z-Image-Turbo-ComfyUI",
                    "split_files/text_encoders/qwen_3_4b.safetensors",
                ): 8_000,
            }
        ),
    )

    def probe(
        repo_id,
        filename,
        revision = None,
        expected_size = None,
        **kwargs,
    ):
        seen[filename] = expected_size
        return True

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(probe))

    b.download_plan(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert seen["z-image-turbo-Q4_K_M.gguf"] == 4_000
    assert seen["split_files/vae/ae.safetensors"] == 300
    assert all(size for size in seen.values()), "an unsized probe trusts the local ref alone"


def test_download_plan_is_empty_when_every_native_asset_is_cached(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kwargs: True),
    )

    plan = b.download_plan(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert plan["entries"] == [] and plan["total_bytes"] == 0


def test_download_plan_stages_exactly_what_sd_cli_opens(monkeypatch):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    sizes = {
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"): 4_000,
        ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/vae/ae.safetensors"): 300,
        ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors"): 8_000,
    }
    monkeypatch.setattr(
        SdCppDiffusionBackend,
        "_plan_file_sizes",
        staticmethod(lambda by_repo, token: sizes),
    )

    plan = b.download_plan(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
        transformer_quant = "int8",
        memory_mode = "low_vram",
    )

    fam = detect_family("z-image")
    expected = {
        (r, f)
        for r, f, _k in b._asset_specs(
            "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf", fam
        )
    }
    listed = {(e["repo_id"], f) for e in plan["entries"] for f in e["files"]}
    assert listed == expected
    assert plan["total_bytes"] == 12_300
    assert plan["required_bytes"] == 12_300
    assert plan["checkpoint_bytes"] == 4_000
    tr = [e for e in plan["entries"] if e["gguf_filename"]]
    assert len(tr) == 1 and tr[0]["repo_id"] == "unsloth/Z-Image-Turbo-GGUF"
    assert len([e for e in plan["entries"] if e["repo_id"] == "unsloth/Z-Image-Turbo-ComfyUI"]) == 1


def test_download_plan_merges_asset_repos_that_share_one_fetch_repo(monkeypatch):
    # Two upstream repos can resolve to ONE fetch repo, so keying by fetch repo drops whichever landed first.
    import core.inference.sd_cpp_backend as S

    b = SdCppDiffusionBackend(engine = _FakeEngine())
    specs = [
        ("unsloth/FLUX.2-dev-GGUF", "flux2-dev-Q4_K_M.gguf", "transformer"),
        ("unsloth/FLUX.2-VAE", "vae/ae.safetensors", "vae"),
        ("unsloth/FLUX.2-dev-ComfyUI", "text_encoders/mistral.safetensors", "text_encoder"),
    ]
    monkeypatch.setattr(SdCppDiffusionBackend, "_asset_specs", lambda *a, **k: specs)
    monkeypatch.setattr(
        S,
        "_fetch_repo_map",
        lambda assets, token: {
            "unsloth/FLUX.2-dev-GGUF": "unsloth/FLUX.2-dev-GGUF",
            "unsloth/FLUX.2-VAE": "Comfy-Org/flux2-dev",
            "unsloth/FLUX.2-dev-ComfyUI": "Comfy-Org/flux2-dev",
        },
    )
    monkeypatch.setattr(SdCppDiffusionBackend, "_preflight_companion_repos", lambda *a, **k: None)
    sizes = {
        ("unsloth/FLUX.2-dev-GGUF", "flux2-dev-Q4_K_M.gguf"): 4_000,
        ("Comfy-Org/flux2-dev", "vae/ae.safetensors"): 300,
        ("Comfy-Org/flux2-dev", "text_encoders/mistral.safetensors"): 8_000,
    }
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: sizes)
    )
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_loadable",
        staticmethod(lambda *a, **k: False),
    )

    plan = b.download_plan(
        "unsloth/FLUX.2-dev-GGUF", gguf_filename = "flux2-dev-Q4_K_M.gguf", model_kind = "gguf"
    )

    shared = next(e for e in plan["entries"] if e["repo_id"] == "Comfy-Org/flux2-dev")
    assert sorted(shared["files"]) == [
        "text_encoders/mistral.safetensors",
        "vae/ae.safetensors",
    ], "neither collapsed repo's files may be dropped"
    assert plan["required_bytes"] == 12_300


def test_download_plan_skips_a_local_transformer_but_still_stages_the_assets(monkeypatch, tmp_path):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    (tmp_path / "z-image-turbo-Q4_K_M.gguf").write_bytes(b"gguf")
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )

    plan = b.download_plan(
        str(tmp_path), gguf_filename = "z-image-turbo-Q4_K_M.gguf", model_kind = "gguf"
    )

    assert str(tmp_path) not in {e["repo_id"] for e in plan["entries"]}
    assert {e["repo_id"] for e in plan["entries"]} == {"unsloth/Z-Image-Turbo-ComfyUI"}
    assert plan["total_bytes"] == 0


def _no_cache(monkeypatch):
    """Report every upstream as uncached, so a local cache cannot mask the mirror decision."""
    monkeypatch.setattr(
        "core.inference.diffusion_families._upstream_is_cached",
        lambda repo_id, files = None, **kwargs: False,
    )
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)


def test_download_plan_stages_the_mirrored_asset_repo(monkeypatch):
    """STAGED before the load runs, so a gated asset repo left here 401s an anonymous user and
    _fetch_assets' swap is never reached. FLUX.1's VAE lives in the gated FLUX.1-schnell."""
    _no_cache(monkeypatch)
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )

    plan = b.download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf", model_kind = "gguf"
    )

    staged = {e["repo_id"] for e in plan["entries"]}
    assert "unsloth/FLUX.1-schnell" in staged
    assert "black-forest-labs/FLUX.1-schnell" not in staged
    tr = [e for e in plan["entries"] if e["gguf_filename"]]
    assert len(tr) == 1 and tr[0]["repo_id"] == "unsloth/FLUX.1-dev-GGUF"


def test_download_plan_and_fetch_assets_pick_the_same_repo(monkeypatch):
    """Staging one repo and then downloading from the other is the failure this feature removes,
    so both sides take the decision from the same per-repo file list."""
    _no_cache(monkeypatch)
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )
    pulled: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: (pulled.append((repo, fn)), f"/cache/{fn}")[1],
    )

    fam = detect_family("flux.1")
    specs = b._asset_specs("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", fam)
    b._fetch_assets(specs, None)
    plan = b.download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf", model_kind = "gguf"
    )

    assert {(e["repo_id"], f) for e in plan["entries"] for f in e["files"]} == set(pulled)


def test_delete_guard_covers_the_mirrored_asset_repos(monkeypatch):
    """The bytes land under whichever of the pair was fetched, so guarding only the upstream leaves
    the mirror cache deletable under a one-shot sd-cli that re-reads it."""
    b = _loaded_backend("flux.1")
    ids = set(b.loaded_repo_ids())
    assert "black-forest-labs/FLUX.1-schnell" in ids
    assert "unsloth/FLUX.1-schnell" in ids

    b._state = None
    b._loading = bk._SdLoading(
        repo_id = "unsloth/FLUX.1-dev-GGUF",
        base_repo = "black-forest-labs/FLUX.1-dev",
        asset_repos = ("black-forest-labs/FLUX.1-schnell",),
    )
    loading = set(b.loading_repo_ids())
    assert {"black-forest-labs/FLUX.1-schnell", "unsloth/FLUX.1-schnell"} <= loading
    assert {"black-forest-labs/FLUX.1-dev", "unsloth/FLUX.1-dev"} <= loading


def test_download_plan_refuses_a_pick_native_cannot_serve():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "gguf_filename is required"):
        b.download_plan("unsloth/Z-Image-Turbo-GGUF", model_kind = "gguf")
    with pytest.raises(ValueError, match = "native sd.cpp asset mapping"):
        b.download_plan(
            "stabilityai/sdxl-turbo", gguf_filename = "sdxl-Q4_K_M.gguf", model_kind = "gguf"
        )


def test_asset_specs_flux2_klein_selects_encoder_by_variant():
    # FLUX.2-klein 4B pairs with Qwen3-4B and 9B with Qwen3-8B: the encoder comes from the load identity.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family("flux.2-klein")

    specs_4b = b._asset_specs("unsloth/FLUX.2-klein-4B-GGUF", "FLUX.2-klein-4B-Q4_K_M.gguf", fam)
    te_4b = [(r, f) for r, f, k in specs_4b if k == "llm"]
    assert te_4b == [
        ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors")
    ]

    specs_9b = b._asset_specs("unsloth/FLUX.2-klein-9B-GGUF", "FLUX.2-klein-9B-Q4_K_M.gguf", fam)
    te_9b = [(r, f) for r, f, k in specs_9b if k == "llm"]
    assert te_9b == [
        (
            "unsloth/FLUX.2-klein-9B-ComfyUI",
            "split_files/text_encoders/qwen_3_8b.safetensors",
        )
    ]


# A rename in a community repack breaks every no-GPU load, so each file has a byte-identical unsloth mirror.
_REPACKER_ORGS = frozenset(
    {"comfy-org", "comfyanonymous", "quantstack", "city96", "calcuis", "orabazes"}
)


def test_no_sd_cpp_asset_comes_from_a_community_repack():
    from core.inference.diffusion_families import _FAMILIES

    offenders = []
    for fam in _FAMILIES:
        specs = list(fam.sd_cpp_text_encoders)
        if fam.sd_cpp_vae:
            specs.append((*fam.sd_cpp_vae, "vae"))
        specs.extend(_FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS)
        for repo, _f, _k in specs:
            if repo.split("/", 1)[0].lower() in _REPACKER_ORGS:
                offenders.append((fam.name, repo))
    assert offenders == []


def test_the_moved_sd_cpp_assets_keep_their_upstream_relative_paths():
    want = {
        "flux.1": [
            ("unsloth/flux-text-encoders", "clip_l.safetensors"),
            ("unsloth/flux-text-encoders", "t5xxl_fp16.safetensors"),
        ],
        "flux.2-klein": [
            ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors"),
        ],
        "flux.2-dev": [
            (
                "unsloth/FLUX.2-dev-ComfyUI",
                "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
            ),
        ],
        "z-image": [
            ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors"),
        ],
    }
    for name, encoders in want.items():
        fam = detect_family(name)
        assert [(r, f) for r, f, _k in fam.sd_cpp_text_encoders] == encoders, name
    assert detect_family("qwen-image").sd_cpp_vae == (
        "unsloth/Qwen-Image-ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
    )
    assert detect_family("z-image").sd_cpp_vae == (
        "unsloth/Z-Image-Turbo-ComfyUI",
        "split_files/vae/ae.safetensors",
    )
    # The FLUX.2 autoencoder is Apache-2.0 while the conditioner beside it is not, so it is mirrored alone.
    for name in ("flux.2-klein", "flux.2-dev"):
        assert detect_family(name).sd_cpp_vae == (
            "unsloth/FLUX.2-VAE",
            "split_files/vae/flux2-vae.safetensors",
        ), name




def test_map_guidance_flux_uses_distilled_guidance():
    cfg, g = _map_guidance(detect_family("flux.1"), 3.5)
    assert cfg is None and g == 3.5


def test_map_guidance_cfg_family_off_when_distilled():
    assert _map_guidance(detect_family("qwen-image"), 0.0) == (1.0, None)
    assert _map_guidance(detect_family("qwen-image"), 4.0) == (4.0, None)




def test_status_unloaded_reports_sd_cpp_engine():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    st = b.status()
    assert st["loaded"] is False and st["engine"] == "sd_cpp"


def test_status_loaded_shape():
    b = _loaded_backend()
    st = b.status()
    assert st["loaded"] is True
    assert st["engine"] == "sd_cpp"
    assert st["family"] == "z-image"
    assert st["device"] == "cpu"
    assert st["gguf_variant"] is None
    for k in ("transformer_quant", "attention_backend", "transformer_cache", "text_encoder_quant"):
        assert st[k] is None




def test_generate_returns_images_and_seed():
    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 8, seed = 123, batch_size = 2)
    assert out["seed"] == 123
    assert out["repo_id"] == "unsloth/Z-Image-Turbo-GGUF"
    assert len(out["images"]) == 2
    assert all(isinstance(im, Image.Image) for im in out["images"])
    assert len(eng.calls) == 2
    seeds = [params.seed for _, params, _, _ in eng.calls]
    assert seeds == [123, 124]
    assert out["seeds"] == [123, 124]


def test_generate_qwen_passes_sampling_args():
    eng = _FakeEngine()
    b = _loaded_backend(fam_name = "qwen-image", engine = eng)
    b.generate(prompt = "x", steps = 20, guidance = 4.0, seed = 1)
    _, params, _, kw = eng.calls[0]
    assert params.sampling_method == "euler"
    assert "--flow-shift" in (kw.get("extra_args") or [])


def test_generate_refuses_a_snapshot_naming_another_model():
    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    st = b.status()
    loaded = bk.load_identity(st["repo_id"], st["base_repo"], st["family"])
    with pytest.raises(bk.DiffusionModelReplacedError) as replaced:
        stale = bk.load_identity("other/model", st["base_repo"], st["family"])
        b.generate(prompt = "stale", expected_load = stale)
    assert replaced.value.expected.repo_id == "other/model"
    assert replaced.value.actual == loaded
    assert eng.calls == []
    assert b.generate(prompt = "x", steps = 4, expected_load = loaded)
    assert b.generate(prompt = "x", steps = 4)


def test_generate_raises_when_not_loaded():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    with pytest.raises(RuntimeError, match = "No diffusion model is loaded"):
        b.generate(prompt = "x")


def test_generate_passes_vae_format_for_flux2():
    eng = _FakeEngine()
    b = _loaded_backend(fam_name = "flux.2-klein", engine = eng)
    b.generate(prompt = "x", steps = 4, seed = 1)
    _, _, _, kw = eng.calls[0]
    assert kw.get("extra_args") == ["--vae-format", "flux2"]


def test_oneshot_generate_refuses_a_binary_swapped_for_another_accelerator(monkeypatch):
    # The one-shot path re-resolves sd-cli per image, and existence is not identity: an install
    # landing mid-batch may target a DIFFERENT accelerator than the device and offload flags were
    # chosen for, spending unaccounted VRAM or running on CPU while the arbiter claims the GPU.
    import dataclasses

    b = _loaded_backend()
    b._state = dataclasses.replace(b._state, sd_accelerator = "cuda")
    monkeypatch.setattr(bk, "_installed_accelerator_of", lambda _binary: "cpu")
    with pytest.raises(RuntimeError, match = "different accelerator"):
        b.generate(prompt = "x", steps = 4, seed = 1)


def test_oneshot_generate_accepts_a_binary_for_the_same_accelerator(monkeypatch):
    import dataclasses

    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    b._state = dataclasses.replace(b._state, sd_accelerator = "cuda")
    monkeypatch.setattr(bk, "_installed_accelerator_of", lambda _binary: "cuda")
    b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(eng.calls) == 1


def test_generate_cancellation_raises_cancelled_not_failure():
    eng = _FakeEngine(cancel_on_call = True)
    b = _loaded_backend(engine = eng)
    with pytest.raises(RuntimeError, match = "cancelled"):
        b.generate(prompt = "x", steps = 8, seed = 5)


def test_cancel_generate_stops_a_running_native_run():
    # The cancel arrives from ANOTHER thread mid-run: the engine polls, kills the sd-cli tree, reports it.
    started = threading.Event()
    b = None

    class _BlockingEngine(_FakeEngine):
        def generate(
            self,
            files,
            params,
            *,
            output_path,
            cancel_event = None,
            **kw,
        ):
            started.set()
            assert cancel_event is not None and cancel_event.wait(5)
            raise SdCppCancelled("cancelled")

    b = _loaded_backend(engine = _BlockingEngine())
    assert b.cancel_generate() is False

    outcome: dict = {}

    def _run():
        try:
            b.generate(prompt = "x", steps = 8, seed = 5)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target = _run, daemon = True)
    worker.start()
    assert started.wait(5)

    assert b.cancel_generate() is True
    worker.join(10)
    assert isinstance(outcome["error"], RuntimeError)
    assert b.cancel_generate() is False


def test_cancel_generate_is_a_no_op_when_idle():
    assert SdCppDiffusionBackend(engine = _FakeEngine()).cancel_generate() is False


def test_generate_progress_tracks_parsed_steps():
    b = _loaded_backend()
    b._gen = bk._SdGen(total_steps = 8)
    b._on_log("  sampling 4/8 done")
    p = b.generate_progress()
    assert p["active"] is True and p["step"] == 4 and p["total_steps"] == 8
    b._on_log("loaded 1/3 tensors")
    assert b.generate_progress()["step"] == 4


def test_generate_publishes_progress_before_lora_resolution(monkeypatch):
    from core.inference import diffusion_lora

    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    monkeypatch.setattr(diffusion_lora, "supports_lora", lambda **_k: True)

    seen: dict = {}

    def _resolve(
        active,
        *,
        family = None,
        hf_token = None,
        cancel_event = None,
    ):
        seen["progress"] = b.generate_progress()
        return []

    monkeypatch.setattr(diffusion_lora, "resolve_specs", _resolve)

    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 8, loras = [("some/lora", 1.0)])
    assert out["images"]
    assert seen["progress"]["active"] is True
    assert seen["progress"]["total_steps"] == 8




def test_begin_load_rejects_unsupported_family(monkeypatch):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(bk, "family_sd_cpp_supported", lambda fam: False)
    with pytest.raises(ValueError, match = "no native sd.cpp asset mapping"):
        b.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z.gguf")


def test_begin_load_requires_gguf_filename():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "gguf_filename is required"):
        b.begin_load("unsloth/Z-Image-Turbo-GGUF")


def test_begin_load_resolves_family_from_filename_only(monkeypatch):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(b, "_run_load", lambda **kwargs: None)
    b.begin_load("/models/gguf-store", gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf")
    assert b._loading is not None and b._loading.repo_id == "/models/gguf-store"


def test_each_load_owns_its_cancel_event(monkeypatch):
    # unload() cancels the running asset pull through the event that worker holds: a clear() of one
    # shared event would un-cancel it, while a fresh Event per load leaves the loser cancelled.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    started = threading.Event()
    seen: list[threading.Event] = []

    def _capture(**kwargs):
        seen.append(kwargs["_cancel_event"])
        started.set()

    monkeypatch.setattr(b, "_run_load", _capture)

    b.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z.gguf")
    assert started.wait(5)
    first = seen[0]
    b.unload()
    assert first.is_set()

    started.clear()
    b.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z.gguf")
    assert started.wait(5)
    second = seen[1]
    assert second is not first, "each load needs its own event, not a clear() of the shared one"
    assert not second.is_set()
    assert first.is_set(), "the superseded worker's event must stay set"
    with pytest.raises(SdCppCancelled):
        b._fetch_assets(
            [("unsloth/Z-Image-Turbo-GGUF", "z.gguf", "diffusion_model")],
            None,
            cancel_event = first,
        )


def test_ensure_binary_returns_found(monkeypatch):
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: "/usr/bin/sd-cli")
    assert ensure_sd_cpp_binary() == "/usr/bin/sd-cli"


def test_ensure_binary_install_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: None)
    assert ensure_sd_cpp_binary(allow_install = False) is None


_PRE_H3_HELP = (
    "stable-diffusion.cpp version unknown, commit unknown\n"
    "  -M, --mode                    run mode, one of [img_gen, vid_gen, upscale, convert]\n"
    "  --audio-vae <string>          path to standalone LTX audio vae model\n"
)
_H3_HELP = _PRE_H3_HELP + (
    "  --ref-video                   MiniMax-H3 Ref2VA reference video frame directory at 24 fps\n"
)


def test_h3_binary_gate_replaces_a_stale_managed_install(monkeypatch, tmp_path):
    # Only runnability was probed, so an older managed sd-cli reported ready and failed on first generation.
    stale = tmp_path / "stale" / "sd-cli"
    fresh = tmp_path / "fresh" / "sd-cli"
    for p in (stale, fresh):
        p.parent.mkdir()
        p.write_text("binary")
    found = [str(stale), str(fresh)]

    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: found.pop(0))
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: True)
    monkeypatch.setattr(
        bk,
        "_sd_cpp_probe_output",
        lambda binary, *_args: _H3_HELP if binary == str(fresh) else _PRE_H3_HELP,
    )

    assert bk.ensure_h3_sd_cpp_binary() == str(fresh)
    assert not stale.exists()


def test_h3_binary_gate_defers_while_a_generation_holds_the_tree(monkeypatch, tmp_path):
    # Dropping the stale copy WRITES to the managed tree, so it needs an install's admission; Windows refuses.
    stale = tmp_path / "stale" / "sd-cli"
    stale.parent.mkdir()
    stale.write_text("binary")
    ensures: list[bool] = []

    monkeypatch.setattr(
        bk,
        "ensure_sd_cpp_binary",
        lambda **kwargs: (ensures.append(kwargs.get("allow_install", True)), str(stale))[1],
    )
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: True)
    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_args: _PRE_H3_HELP)
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)

    with bk._tree_reader(str(stale)):
        assert bk.ensure_h3_sd_cpp_binary() is None
    assert stale.exists()
    assert ensures == [True]


def test_h3_binary_gate_refuses_but_keeps_a_user_supplied_build(monkeypatch, tmp_path):
    # The user's own build is not ours to delete (install() then refuses it), so the load names the binary.
    own = tmp_path / "sd-cli"
    own.write_text("binary")
    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: str(own))
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: False)
    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_args: _PRE_H3_HELP)

    with pytest.raises(RuntimeError, match = "does not advertise MiniMax-H3") as excinfo:
        bk.ensure_h3_sd_cpp_binary()
    assert own.exists()
    assert "remove" not in str(excinfo.value)
    assert str(own) in str(excinfo.value)


def test_h3_binary_gate_offers_to_clear_an_unmarked_install_directory(monkeypatch, tmp_path):
    # The one case where clearing the path IS the fix: the user's own build at the installer's path, so MOVE it.
    root = tmp_path / "studio" / "stable-diffusion.cpp"
    own = root / "sd-cli"
    root.mkdir(parents = True)
    own.write_text("binary")
    monkeypatch.setattr(bk, "managed_install_root", lambda: root)
    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: str(own))
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: False)
    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_args: _PRE_H3_HELP)

    with pytest.raises(RuntimeError, match = "does not advertise MiniMax-H3") as excinfo:
        bk.ensure_h3_sd_cpp_binary()
    assert f"move {root} aside" in str(excinfo.value)
    assert "remove" not in str(excinfo.value)
    assert own.exists()


def test_h3_binary_gate_never_offers_to_delete_the_in_tree_developer_build(monkeypatch, tmp_path):
    root = tmp_path / "repo" / "stable-diffusion.cpp"
    own = root / "build" / "bin" / "sd-cli"
    own.parent.mkdir(parents = True)
    own.write_text("binary")
    monkeypatch.setattr(bk, "in_tree_install_root", lambda: root, raising = False)
    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: str(own))
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: False)
    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_args: _PRE_H3_HELP)

    with pytest.raises(RuntimeError, match = "does not advertise MiniMax-H3") as excinfo:
        bk.ensure_h3_sd_cpp_binary()
    assert "remove" not in str(excinfo.value)
    assert own.exists()


def test_h3_binary_gate_logs_the_real_fault_for_a_managed_non_sd_cpp_binary(
    monkeypatch, tmp_path, capsys
):
    # Calling a not-sd.cpp binary "does not advertise MiniMax-H3" is #8507's wrong diagnosis, in the log.
    own = tmp_path / "sd-cli"
    own.write_text("binary")
    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: str(own))
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: True)
    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_args: "sd 1.0.0\nFind & replace CLI\n")

    assert bk.ensure_h3_sd_cpp_binary(allow_install = False) is None
    logged = capsys.readouterr().out
    assert "is not stable-diffusion.cpp" in logged
    assert "MiniMax-H3" not in logged
    assert own.exists()


def test_h3_binary_gate_names_a_binary_that_is_not_sd_cpp_at_all(monkeypatch, tmp_path):
    # #8507: SD_CLI_PATH pointed at Debian's `sd` find-and-replace tool and the gate called it an
    # sd.cpp build predating MiniMax-H3, sending the user after a newer build of something they
    # never installed. Identity and capability are separate answers.
    own = tmp_path / "sd"
    own.write_text("binary")
    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: str(own))
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: False)
    monkeypatch.setattr(
        bk,
        "_sd_cpp_probe_output",
        lambda *_args: "sd 1.0.0\nFind & replace CLI\n\nUSAGE:\n    sd <find> <replace-with>\n",
    )

    with pytest.raises(RuntimeError, match = "is not stable-diffusion.cpp"):
        bk.ensure_h3_sd_cpp_binary()
    assert own.exists()


def test_h3_binary_gate_requires_identity_not_just_the_h3_marker(monkeypatch, tmp_path):
    own = tmp_path / "reference-video-cli"
    own.write_text("binary")
    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: str(own))
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: False)
    monkeypatch.setattr(
        bk,
        "_sd_cpp_probe_output",
        lambda *_args: "reference-video-cli 2.1\n  --ref-video PATH   reference clip\n",
    )

    with pytest.raises(RuntimeError, match = "is not stable-diffusion.cpp"):
        bk.ensure_h3_sd_cpp_binary()


def test_h3_binary_gate_probes_help_once_for_both_questions(monkeypatch, tmp_path):
    own = tmp_path / "sd-cli"
    own.write_text("binary")
    calls: list[tuple] = []

    def _probe(binary, *args):
        calls.append((binary, args))
        return "sd 1.0.0\nFind & replace CLI\n"

    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: str(own))
    monkeypatch.setattr(bk, "is_managed_binary", lambda _b: False)
    monkeypatch.setattr(bk, "_sd_cpp_probe_output", _probe)

    with pytest.raises(RuntimeError, match = "is not stable-diffusion.cpp"):
        bk.ensure_h3_sd_cpp_binary()
    assert [args for _b, args in calls] == [("--help",)]


def test_h3_binary_gate_keeps_a_binary_it_cannot_probe(monkeypatch):
    monkeypatch.setattr(bk, "ensure_sd_cpp_binary", lambda **_kwargs: "/usr/bin/sd-cli")
    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_args: None)
    assert bk.ensure_h3_sd_cpp_binary() == "/usr/bin/sd-cli"


def test_lists_accelerator_device_reads_the_ggml_device_list(monkeypatch):
    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_a: "CPU\tIntel(R) Xeon(R) Platinum\n")
    assert bk.sd_cpp_lists_accelerator_device("/existing/sd-cli") is False

    monkeypatch.setattr(
        bk,
        "_sd_cpp_probe_output",
        lambda *_a: "CUDA0\tNVIDIA GeForce RTX 4090\nCPU\tIntel(R) Xeon(R) Platinum\n",
    )
    assert bk.sd_cpp_lists_accelerator_device("/existing/sd-cli") is True

    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_a: None)
    assert bk.sd_cpp_lists_accelerator_device("/existing/sd-cli") is True
    assert bk.sd_cpp_lists_accelerator_device(None) is False


def test_supports_graph_cut_needs_both_flags_and_fails_closed(monkeypatch):
    monkeypatch.setattr(
        bk,
        "_sd_cpp_probe_output",
        lambda *_a: "  --max-vram         budget\n  --stream-layers    residency\n",
    )
    assert bk.sd_cpp_supports_graph_cut("/existing/sd-cli") is True

    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_a: "  --stream-layers    residency\n")
    assert bk.sd_cpp_supports_graph_cut("/existing/sd-cli") is False

    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_a: _PRE_H3_HELP)
    assert bk.sd_cpp_supports_graph_cut("/existing/sd-cli") is False

    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_a: None)
    assert bk.sd_cpp_supports_graph_cut("/existing/sd-cli") is False
    assert bk.sd_cpp_supports_graph_cut(None) is False


def test_device_name_for_ordinal_reads_the_ggml_device_list(monkeypatch):
    monkeypatch.setattr(
        bk,
        "_sd_cpp_probe_output",
        lambda *_a: "CUDA0\tRTX 4070 Ti\nCUDA1\tRTX 5060 Ti\nCPU\tAMD Ryzen 9\n",
    )
    assert bk.sd_cpp_device_name_for_ordinal("/existing/sd-cli", 1) == "CUDA1"
    assert bk.sd_cpp_device_name_for_ordinal("/existing/sd-cli", 0) == "CUDA0"
    assert bk.sd_cpp_device_name_for_ordinal("/existing/sd-cli", 3) is None
    assert bk.sd_cpp_device_name_for_ordinal("/existing/sd-cli", None) is None
    assert bk.sd_cpp_device_name_for_ordinal(None, 1) is None

    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_a: "ROCm1\tRadeon RX 7900\n")
    assert bk.sd_cpp_device_name_for_ordinal("/existing/sd-cli", 1) == "ROCm1"

    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_a: "Vulkan1\tRTX 5060 Ti\n")
    assert bk.sd_cpp_device_name_for_ordinal("/existing/sd-cli", 1) is None

    monkeypatch.setattr(bk, "_sd_cpp_probe_output", lambda *_a: None)
    assert bk.sd_cpp_device_name_for_ordinal("/existing/sd-cli", 1) is None


def test_offload_device_pin_is_probed_against_the_binary_it_is_given(monkeypatch):
    seen: list = []

    def _probe(binary, *_args):
        seen.append(binary)
        return "CUDA0\tA\nCUDA1\tB\n" if binary == "/new/sd-cli" else "CPU\tRyzen\n"

    monkeypatch.setattr(bk, "_sd_cpp_probe_output", _probe)
    base = ["--offload-to-cpu"]
    assert bk._offload_with_device_pin_impl(base, "/old/sd-cli", 1) == base
    assert bk._offload_with_device_pin_impl(base, "/new/sd-cli", 1) == [
        "--offload-to-cpu",
        "--backend",
        "diffusion=CUDA1,te=CUDA1,vae=CUDA1",
    ]
    assert seen == ["/old/sd-cli", "/new/sd-cli"]
    seen.clear()
    assert bk._offload_with_device_pin_impl(base, "/new/sd-cli", None) == base
    assert seen == []


def test_unload_clears_state_and_signals_cancel():
    cancel = threading.Event()
    b = _loaded_backend()
    b._active_generate_cancel = cancel
    st = b.unload()
    assert st["loaded"] is False
    assert cancel.is_set()
    assert b._cancel_event.is_set()


def test_status_reports_offload_when_flags_active():
    b = _loaded_backend()
    assert b.status()["offload_policy"] == "none" and b.status()["cpu_offload"] is False
    s = b._state
    b._state = bk._SdState(
        repo_id = s.repo_id,
        base_repo = s.base_repo,
        family = s.family,
        device = "cuda",
        files = s.files,
        offload_flags = ("--vae-on-cpu", "--clip-on-cpu"),
    )
    st = b.status()
    assert st["cpu_offload"] is True and st["offload_policy"] == "active"


def test_run_load_cancels_and_waits_for_inflight_generation(monkeypatch):
    # A generation started during the asset download runs the OLD model, so cancel and wait before committing.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family("z-image")
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )

    b._load_token = 5
    cancel = threading.Event()
    b._active_generate_cancel = cancel

    committed = threading.Event()

    def _load():
        b._run_load(
            repo_id = "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z.gguf",
            base = fam.base_repo,
            fam = fam,
            hf_token = None,
            _load_token = 5,
        )
        committed.set()

    b._generate_lock.acquire()
    try:
        threading.Thread(target = _load, daemon = True).start()
        assert not committed.wait(0.5)
        assert b._state is None
        assert cancel.is_set()
    finally:
        b._generate_lock.release()
    assert committed.wait(5)
    assert b._state is not None and b._state.repo_id == "unsloth/Z-Image-Turbo-GGUF"




def test_resolve_backend_prefers_server(monkeypatch):
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    mode, binary, engine = b._resolve_backend()
    assert mode == "server" and binary == "/x/sd-server" and engine is None


def test_resolve_backend_injected_engine_forces_oneshot():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    mode, binary, engine = b._resolve_backend()
    assert mode == "oneshot" and binary is None and engine is not None


def test_resolve_backend_falls_back_to_oneshot_without_server(monkeypatch):
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: None)
    monkeypatch.setattr(bk, "_install_allowed", lambda: False)
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: "/usr/bin/sd-cli")
    mode, binary, engine = b._resolve_backend()
    assert mode == "oneshot" and engine is not None


def test_resolve_backend_cached_fallback_engine_does_not_pin_oneshot(monkeypatch):
    b = SdCppDiffusionBackend()
    b._engine = _FakeEngine()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    mode, binary, engine = b._resolve_backend()
    assert mode == "server" and binary == "/x/sd-server" and engine is None


def _run_server_load(
    monkeypatch,
    b,
    servers,
    fam_name = "z-image",
    device = "cpu",
    gguf_filename = "z.gguf",
):
    fam = detect_family(fam_name)
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)

    def _factory(binary):
        s = _FakeServer(binary)
        servers.append(s)
        return s

    monkeypatch.setattr(bk, "SdCppServer", _factory)
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = device)
    )
    b._load_token = 1
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = gguf_filename,
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
    )


def test_server_load_spawns_once_and_status_reports_mode(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    assert len(servers) == 1
    assert servers[0].started is not None
    assert b._state is not None and b._state.mode == "server" and b._state.server is servers[0]
    assert b.status()["native_mode"] == "server"


def test_server_status_reports_selected_gguf_quant(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, gguf_filename = "z-image-turbo-Q8_0.gguf")
    assert b.status()["gguf_variant"] == "Q8_0"


def test_server_generate_uses_one_request_for_whole_batch(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 8, seed = 7, batch_size = 3)
    assert len(out["images"]) == 3
    assert all(isinstance(im, Image.Image) for im in out["images"])
    assert len(servers[0].payloads) == 1
    assert servers[0].payloads[0]["batch_count"] == 3
    assert out["seed"] == 7 and out["seeds"] == [7, 8, 9]
    assert b._gen is None


_GGML_ABORT = (
    "sd-server connection lost during img_gen poll (process exited, code -6)\n"
    "Last output:\n"
    "[ERROR] ggml_extend.hpp:70   - ggml_metal_op_encode_impl: error: unsupported op 'MUL_MAT'\n"
    "1   sd-server   0x00000001044f8df4 ggml_abort + 156\n"
    "10  sd-server   0x00000001043f6ce8 StableDiffusionGGML::sample"
)


def test_server_generation_restarts_on_the_cpu_backend_after_a_ggml_abort(monkeypatch):
    # ggml calls GGML_ABORT on an unimplemented op with no CPU fallback, so the load restarts on --backend cpu.
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "mps")
    servers[0].img_gen_error = RuntimeError(_GGML_ABORT)

    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 4, seed = 3)

    assert len(out["images"]) == 1
    assert len(servers) == 2 and servers[0].stopped is True
    assert servers[1].started["extra_args"] == ["--backend", "cpu"]
    assert servers[1].started["files"] is servers[0].started["files"]
    assert servers[1].started["native_speed"] == servers[0].started["native_speed"]
    assert b._state is not None and b._state.server is servers[1]


def test_cpu_backend_restart_happens_once_per_load(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "mps")
    servers[0].img_gen_error = RuntimeError(_GGML_ABORT)
    b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(servers) == 2

    servers[1].img_gen_error = RuntimeError(_GGML_ABORT)
    with pytest.raises(RuntimeError, match = "unsupported op"):
        b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(servers) == 2


def test_server_death_without_the_abort_signature_is_not_retried(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "mps")
    servers[0].img_gen_error = RuntimeError(
        "sd-server connection lost during img_gen poll (process exited, code -9)"
    )
    with pytest.raises(RuntimeError, match = "code -9"):
        b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(servers) == 1


def test_cpu_device_does_not_restart_on_an_abort(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "cpu")
    servers[0].img_gen_error = RuntimeError(_GGML_ABORT)
    with pytest.raises(RuntimeError, match = "unsupported op"):
        b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(servers) == 1


def test_server_generate_splits_batches_above_server_limit(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    out = b.generate(prompt = "x", width = 64, height = 64, steps = 4, seed = 100, batch_size = 10)
    assert len(out["images"]) == 10
    counts = [p["batch_count"] for p in servers[0].payloads]
    assert counts == [bk._MAX_SERVER_BATCH, 10 - bk._MAX_SERVER_BATCH]
    # Chunks share ONE request deadline: the split is only a per-job image cap, not a longer window.
    assert servers[0].timeouts[0] <= bk.NATIVE_GENERATION_TIMEOUT_S
    assert servers[0].timeouts[1] <= servers[0].timeouts[0]
    assert servers[0].timeouts[-1] > 1800.0
    assert out["seeds"] == list(range(100, 110))
    assert servers[0].payloads[1]["seed"] == 108


def test_server_generate_masks_large_seed(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    out = b.generate(prompt = "x", width = 64, height = 64, steps = 4, seed = 2**64 - 1, batch_size = 1)
    assert servers[0].payloads[0]["seed"] <= (1 << 63) - 1
    assert all(s <= (1 << 63) - 1 for s in out["seeds"])


def test_status_clears_when_server_died(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    assert b.status()["loaded"] is True
    servers[0].alive = False
    st = b.status()
    assert st["loaded"] is False
    assert b._state is None


def test_server_generate_progress_from_stdout(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)

    seen = {}

    class _WatchServer(_FakeServer):
        def img_gen(
            self,
            payload,
            *,
            on_step = None,
            cancel_event = None,
            total_timeout = None,
        ):
            on_step("  4/8")
            seen["mid"] = b.generate_progress()
            return super().img_gen(
                payload, on_step = on_step, cancel_event = cancel_event, total_timeout = total_timeout
            )

    b._state = bk._SdState(
        repo_id = b._state.repo_id,
        base_repo = b._state.base_repo,
        family = b._state.family,
        device = b._state.device,
        files = b._state.files,
        vae_format = b._state.vae_format,
        sampling_method = b._state.sampling_method,
        flow_shift = b._state.flow_shift,
        server = _WatchServer("/x/sd-server"),
        mode = "server",
    )
    b.generate(prompt = "x", steps = 8, seed = 1)
    assert seen["mid"]["step"] == 4 and seen["mid"]["total_steps"] == 8


def test_server_unload_stops_server(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    st = b.unload()
    assert st["loaded"] is False
    assert servers[0].stopped is True
    assert b._state is None


def test_server_reload_stops_old_server_before_new(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    b._load_token = 2
    fam = detect_family("z-image")
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 2,
    )
    assert len(servers) == 2
    assert servers[0].stopped is True
    assert b._state.server is servers[1] and servers[1].stopped is False


def test_a_cancel_during_server_revalidation_stops_before_the_process_spawns(monkeypatch):
    # _server_binary_runnable can sit for 20s and the publish happens after it returns, so an unload
    # in that window finds no _pending_server to stop and the load spawns anyway. Rechecking under
    # the publishing lock closes the gap.
    b = SdCppDiffusionBackend()
    cancel = threading.Event()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")

    def _revalidate_then_cancel(*_a, **_k):
        cancel.set()
        return True

    monkeypatch.setattr(bk, "_server_binary_runnable", _revalidate_then_cancel)

    started: list[str] = []

    class _RecordingServer:
        def __init__(self, binary):
            self.stopped = False

        def start(self, *a, **k):
            started.append("start")

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(bk, "SdCppServer", _RecordingServer)
    fake = _FakeEngine()
    monkeypatch.setattr(b, "_resolve_engine", lambda: fake)
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )
    fam = detect_family("z-image")
    b._load_token = 1
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
        _cancel_event = cancel,
    )
    assert started == [], "a cancelled load must not spawn the server process"
    assert b._pending_server is None
    assert bk._tree_in_use(b) is False


def test_server_start_failure_falls_back_to_oneshot(monkeypatch):
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)

    class _BadServer:
        def __init__(self, binary):
            self.stopped = False

        def start(self, *a, **k):
            raise RuntimeError("sd-server broken")

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(bk, "SdCppServer", _BadServer)
    fake = _FakeEngine()
    monkeypatch.setattr(b, "_resolve_engine", lambda: fake)
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )
    fam = detect_family("z-image")
    b._load_token = 1
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
    )
    assert b._state is not None and b._state.mode == "oneshot" and b._state.server is None
    # _pending_server means a native process runs from the managed tree, so a stale one pins the wrong build.
    assert b._pending_server is None
    assert bk._tree_in_use(b) is False
    out = b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(out["images"]) == 1 and len(fake.calls) == 1


def test_server_start_failure_keeps_the_engine_the_fallback_resolved(monkeypatch):
    # The fallback resolved an sd-cli then threw it away, so the accelerator was recorded off None.
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)

    class _BadServer:
        def __init__(self, binary):
            pass

        def start(self, *a, **k):
            raise RuntimeError("sd-server broken")

        def stop(self):
            pass

    monkeypatch.setattr(bk, "SdCppServer", _BadServer)
    fake = _FakeEngine()
    fake.binary = "/x/sd-cli"
    monkeypatch.setattr(b, "_resolve_engine", lambda: fake)
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )
    monkeypatch.setattr(
        bk, "_installed_accelerator_of", lambda binary: "cuda" if binary == "/x/sd-cli" else None
    )
    fam = detect_family("z-image")
    b._load_token = 1
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
    )
    assert b._state is not None and b._state.mode == "oneshot"
    assert b._state.sd_accelerator == "cuda"
    out = b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(out["images"]) == 1 and len(fake.calls) == 1


def test_server_unusable_after_the_download_keeps_the_engine_the_fallback_resolved(monkeypatch):
    # The other server-to-one-shot fallback: the pin was taken while the mode was still "server",
    # i.e. off an engine of None, then compared against the sd-cli the fallback resolved, so a clean
    # drop to one-shot is refused as a swapped binary.
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    runnable = [True]

    def _runnable(*_a, **_k):
        answer = runnable[0]
        runnable[0] = False
        return answer

    monkeypatch.setattr(bk, "_server_binary_runnable", _runnable)
    monkeypatch.setattr(bk, "ensure_sd_server_binary", lambda **_k: "/x/sd-server")
    fake = _FakeEngine()
    fake.binary = "/x/sd-cli"
    monkeypatch.setattr(b, "_resolve_engine", lambda: fake)
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )
    monkeypatch.setattr(bk, "_installed_accelerator_of", lambda binary: "cuda" if binary else None)
    fam = detect_family("z-image")
    b._load_token = 1
    b._loading = bk._SdLoading(repo_id = "unsloth/Z-Image-Turbo-GGUF", base_repo = fam.base_repo)
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
    )
    assert b._state is not None, f"the fallback was refused: {b.load_progress().get('error')}"
    assert b._state.mode == "oneshot" and b._state.server is None
    assert b._state.sd_accelerator == "cuda"
    out = b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(out["images"]) == 1 and len(fake.calls) == 1


def test_a_oneshot_load_refuses_a_cli_swapped_during_the_asset_download(monkeypatch):
    # The one-shot accelerator was sampled AFTER the multi-minute asset download, so an install
    # landing in that window was recorded as this load's own answer and the per-image check agreed
    # with it forever. Pinned where the engine is vetted instead.
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: None)
    monkeypatch.setattr(bk, "_install_allowed", lambda: False)
    fake = _FakeEngine()
    fake.binary = "/x/sd-cli"
    monkeypatch.setattr(b, "_resolve_engine", lambda: fake)
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )
    installed = ["cuda"]
    monkeypatch.setattr(
        bk, "_installed_accelerator_of", lambda binary: installed[0] if binary else None
    )

    def _fetch(*a, **k):
        installed[0] = "cpu"
        return {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"}

    monkeypatch.setattr(b, "_fetch_assets", _fetch)
    fam = detect_family("z-image")
    b._load_token = 1
    b._loading = bk._SdLoading(repo_id = "unsloth/Z-Image-Turbo-GGUF", base_repo = fam.base_repo)
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
    )

    assert b._state is None
    assert "different accelerator" in (b.load_progress()["error"] or "")


def test_run_load_redacts_paths_in_progress_error(monkeypatch):
    from utils import native_path_leases as npl

    secret_root = "/managed/native/root"
    npl._remember_native_path_for_redaction(secret_root, "model dir")
    try:
        b = SdCppDiffusionBackend(engine = _FakeEngine())
        fam = detect_family("z-image")
        monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
        monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)

        def _boom(*a, **k):
            raise RuntimeError(f"failed to read {secret_root}/z.gguf")

        monkeypatch.setattr(b, "_fetch_assets", _boom)

        b._load_token = 1
        b._loading = bk._SdLoading(repo_id = "unsloth/Z-Image-Turbo-GGUF", base_repo = fam.base_repo)
        b._run_load(
            repo_id = "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z.gguf",
            base = fam.base_repo,
            fam = fam,
            hf_token = None,
            _load_token = 1,
        )
        err = b.load_progress()["error"]
        assert err and secret_root not in err and "<native_path>" in err
    finally:
        with npl._REDACTION_LOCK:
            if secret_root in npl._NATIVE_PATH_REDACTIONS:
                npl._NATIVE_PATH_REDACTIONS.remove(secret_root)




def _fake_materialize(resolved, dest):
    """Stand-in for diffusion_lora.materialize_native_dir: write a stub file per adapter
    into ``dest`` and return the resolved list pointing at the written paths (mirroring the
    real helper's contract without touching the Hub / real weights)."""
    from pathlib import Path as _P

    from core.inference import diffusion_lora as dl

    dest.mkdir(parents = True, exist_ok = True)
    out = []
    for r in resolved:
        p = _P(dest) / f"{r.alias}.safetensors"
        p.write_bytes(b"stub")
        out.append(dl.ResolvedLora(r.id, r.alias, str(p), r.fmt, r.weight))
    return out


def _patch_lora(
    monkeypatch,
    resolved,
    supported = True,
):
    from core.inference import diffusion_lora as dl

    monkeypatch.setattr(dl, "supports_lora", lambda **k: supported)
    monkeypatch.setattr(dl, "resolve_specs", lambda specs, **k: list(resolved))
    monkeypatch.setattr(dl, "materialize_native_dir", _fake_materialize)


def test_generate_oneshot_applies_loras_via_prompt_tags(monkeypatch):
    from core.inference import diffusion_lora as dl

    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    _patch_lora(
        monkeypatch, [dl.ResolvedLora("id1", "myalias", "/x/a.safetensors", "safetensors", 0.8)]
    )
    b.generate(prompt = "a fox", steps = 4, seed = 1, loras = [("id1", 0.8)])
    _, params, _, _ = eng.calls[0]
    assert params.lora_dir is not None and params.lora_apply_mode == "auto"
    assert "<lora:myalias:0.8>" in params.prompt


def test_generate_server_stages_loras_and_sends_structured_field(monkeypatch, tmp_path):
    from pathlib import Path as _P

    from core.inference import diffusion_lora as dl

    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    servers[0].lora_dir = str(tmp_path)
    _patch_lora(
        monkeypatch, [dl.ResolvedLora("id1", "myalias", "/x/a.safetensors", "safetensors", 0.7)]
    )
    b.generate(prompt = "x", steps = 4, seed = 1, batch_size = 1, loras = [("id1", 0.7)])
    payload = servers[0].payloads[0]
    assert "lora" in payload and len(payload["lora"]) == 1
    assert payload["lora"][0]["multiplier"] == 0.7
    assert payload["lora"][0]["path"].endswith("myalias.safetensors")
    assert "<lora:" not in payload["prompt"]
    assert not list(_P(tmp_path).glob("gen_*"))


def test_generate_rejects_loras_on_unsupported_family(monkeypatch):
    b = _loaded_backend(engine = _FakeEngine())
    _patch_lora(monkeypatch, [], supported = False)
    with pytest.raises(ValueError, match = "LoRA is not supported"):
        b.generate(prompt = "x", steps = 4, seed = 1, loras = [("id1", 1.0)])


def test_generate_zero_weight_loras_are_noop(monkeypatch):
    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    _patch_lora(monkeypatch, [], supported = False)
    b.generate(prompt = "x", steps = 4, seed = 1, loras = [("id1", 0.0)])
    _, params, _, _ = eng.calls[0]
    assert params.lora_dir is None


def test_generate_rejects_controlnet_on_native_engine():
    b = _loaded_backend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "ControlNet is not yet supported on the native"):
        b.generate(prompt = "x", steps = 4, seed = 1, controlnet = ("id", "img", "canny", 1.0, 0.0, 1.0))


@pytest.mark.parametrize("cn_strength", [0, 0.0, None])
def test_generate_treats_zero_strength_controlnet_as_disabled(cn_strength):
    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    out = b.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        controlnet = ("id", "img", "canny", cn_strength, 0.0, 1.0),
    )
    assert len(out["images"]) == 1


def test_generate_rejects_image_conditioned_on_native_engine():
    b = _loaded_backend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "not yet supported on the native"):
        b.generate(prompt = "x", steps = 4, seed = 1, init_image = "data:image/png;base64,AAAA")


def test_status_native_reports_supports_controlnet_false():
    b = _loaded_backend()
    assert b.status()["supports_controlnet"] is False


def test_a_cached_community_repack_is_reused_instead_of_re_downloading_the_mirror(
    monkeypatch, tmp_path
):
    """Repointing the tables at unsloth mirrors would re-pull tens of GB on upgrade.

    The HF cache is keyed by repo id, so an install that already holds the byte-identical repack
    has it filed under the OLD id: the mirror's namespace is empty, the fetch re-downloads, and an
    offline load fails outright over bytes already on disk."""
    from core.inference.diffusion_families import prefer_cached_legacy_source
    from core.inference.sd_cpp_backend import _fetch_repo_map

    repack = "Comfy-Org/z_image_turbo"
    root = tmp_path / f"models--{repack.replace('/', '--')}"
    snapshot = root / "snapshots" / ("d" * 40)
    (snapshot / "split_files" / "vae").mkdir(parents = True)
    (snapshot / "split_files" / "vae" / "ae.safetensors").write_bytes(b"x" * 64)
    (root / "refs").mkdir(parents = True)
    (root / "refs" / "main").write_text("d" * 40, encoding = "utf-8")
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache", lambda: str(tmp_path), raising = False
    )

    assets = [("unsloth/Z-Image-Turbo-ComfyUI", "split_files/vae/ae.safetensors", "vae")]
    assert _fetch_repo_map(assets, None)["unsloth/Z-Image-Turbo-ComfyUI"] == repack

    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache",
        lambda: str(tmp_path / "empty"),
        raising = False,
    )
    assert _fetch_repo_map(assets, None)["unsloth/Z-Image-Turbo-ComfyUI"] == (
        "unsloth/Z-Image-Turbo-ComfyUI"
    )
    assert prefer_cached_legacy_source("unsloth/Qwen-Image", ["x"]) == "unsloth/Qwen-Image"


def test_a_repack_left_in_the_pre_change_cache_root_still_wins_over_the_mirror(
    monkeypatch, tmp_path
):
    """Changing Unsloth's cache folder must not cost the user the repack they already hold.

    The fetch passes reuse_other_cache_root, so a file cached only under huggingface_hub's
    import-time root resolves through that root -- but only under the repo id it was filed as.
    Picking the mirror because the LIVE root looks empty makes those bytes unreachable: several GB
    re-download online, and offline the load fails."""
    from core.inference.sd_cpp_backend import _fetch_repo_map

    repack = "Comfy-Org/z_image_turbo"
    other = tmp_path / "before_the_move"
    root = other / f"models--{repack.replace('/', '--')}"
    snapshot = root / "snapshots" / ("d" * 40)
    (snapshot / "split_files" / "vae").mkdir(parents = True)
    (snapshot / "split_files" / "vae" / "ae.safetensors").write_bytes(b"x" * 64)
    (root / "refs").mkdir(parents = True)
    (root / "refs" / "main").write_text("d" * 40, encoding = "utf-8")
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache",
        lambda: str(tmp_path / "after_the_move"),
        raising = False,
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(other))

    assets = [("unsloth/Z-Image-Turbo-ComfyUI", "split_files/vae/ae.safetensors", "vae")]
    assert _fetch_repo_map(assets, None)["unsloth/Z-Image-Turbo-ComfyUI"] == repack


def test_the_delete_guard_names_the_repack_as_well_as_the_mirror(monkeypatch):
    """The bytes land in whichever of the three the load chose, and that is re-decided per load, so
    a guard naming only one of them can delete a repo the next load needs."""
    from core.inference.sd_cpp_backend import _with_mirrors

    protected = _with_mirrors(["unsloth/Z-Image-Turbo-ComfyUI"])
    assert "unsloth/Z-Image-Turbo-ComfyUI" in protected
    assert "Comfy-Org/z_image_turbo" in protected


def test_begin_load_answers_without_waiting_on_the_header_probe(monkeypatch):
    """begin_load returns at once by contract: the route thread hands the UI a status and the
    multi-gigabyte pull happens on the worker. The FLUX.2 encoder pick needs the checkpoint's
    inner_dim, which for an uncached pick means a range request over the wire -- bounded, but
    bounded in SECONDS, and the route would wear every one of them on a load button press.

    So the pre-lock probe is offline-only. The guard it feeds is a hint at that moment; the worker
    re-probes with the network and publishes the real repos before it fetches a byte, which is the
    second half of this test."""
    import time as _time

    from core.inference import diffusion_compat

    diffusion_compat._reset_inner_dim_cache()
    probed: list[str] = []

    class _Slow:
        def get(
            self,
            url,
            headers = None,
            timeout = None,
            stream = False,
        ):
            probed.append(url)
            _time.sleep(30)
            raise AssertionError("unreachable")

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Slow())
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(b, "_run_load", lambda **kwargs: None)

    started = _time.monotonic()
    b.begin_load("unsloth/FLUX.2-klein-4B-GGUF", gguf_filename = "flux-2-klein-4b-Q4_K_M.gguf")
    elapsed = _time.monotonic() - started

    assert probed == [], "begin_load must not range-read the header on the route thread"
    assert elapsed < 5, f"begin_load blocked for {elapsed:.1f}s"


def test_the_worker_publishes_the_real_encoder_repos_before_fetching(monkeypatch):
    # The delete-cached guard reads _loading.asset_repos, which begin_load only guesses: 9B encoders stay deletable.
    from core.inference import diffusion_compat

    diffusion_compat._reset_inner_dim_cache()
    nine_b = {repo for repo, _f, _k in _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS}
    repo, filename = "unsloth/FLUX.2-klein-GGUF", "flux-2-klein-Q4_K_M.gguf"
    monkeypatch.setattr(
        bk.SdCppDiffusionBackend,
        "_flux2_inner_dim",
        staticmethod(
            lambda repo_id, fn, fam, tok, allow_network = True: 4096 if allow_network else None
        ),
    )

    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(b, "_run_load", lambda **kwargs: None)
    b.begin_load(repo, gguf_filename = filename)
    token = b._load_token
    assert not nine_b & set(
        b.loading_repo_ids()
    ), "fixture is not discriminating: begin_load already guessed the 9B encoders"

    monkeypatch.setattr(b, "_resolve_backend", lambda: ("oneshot", None, _FakeEngine()))
    monkeypatch.setattr(b, "_preflight_companion_repos", lambda *a, **k: None)
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    seen_at_fetch: list[tuple[str, ...]] = []

    def _fetch(*_a, **_k):
        seen_at_fetch.append(b.loading_repo_ids())
        raise SdCppCancelled("stop here; the publish already had to happen")

    monkeypatch.setattr(b, "_fetch_assets", _fetch)
    bk.SdCppDiffusionBackend._run_load(
        b,
        repo_id = repo,
        gguf_filename = filename,
        base = "black-forest-labs/FLUX.2-klein-9B",
        fam = detect_family(repo),
        hf_token = None,
        _load_token = token,
    )

    assert seen_at_fetch, "the fetch was never reached"
    assert nine_b <= set(
        seen_at_fetch[0]
    ), "the delete guard did not name the encoders this load is about to write"


def test_generate_reports_the_build_the_recipe_persists():
    # The route writes the recipe off these keys, so an omitted key persists as null and read "unknown".
    b = _loaded_backend()
    s = b._state
    b._state = bk._SdState(
        repo_id = s.repo_id,
        base_repo = s.base_repo,
        family = s.family,
        device = "cuda",
        files = s.files,
        vae_format = s.vae_format,
        sampling_method = s.sampling_method,
        flow_shift = s.flow_shift,
        mode = s.mode,
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        offload_flags = ("--vae-on-cpu", "--clip-on-cpu"),
    )
    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 4, seed = 1)
    assert out["model_kind"] == "gguf"
    assert out["gguf_filename"] == "z-image-turbo-Q4_K_M.gguf"
    assert out["offload_policy"] == "active"
    assert out["offload_policy"] == b.status()["offload_policy"]
    assert out["transformer_quant"] is None and out["text_encoder_quant"] is None
    assert out["memory_mode"] is None


def test_a_completed_native_generation_stops_advertising_itself_as_cancellable(monkeypatch):
    b = _loaded_backend()
    seen: list[bool] = []
    real_oneshot = b._generate_oneshot

    def _oneshot(*args, **kwargs):
        out = real_oneshot(*args, **kwargs)
        assert b.cancel_generate() is True
        return out

    monkeypatch.setattr(b, "_generate_oneshot", _oneshot)
    with pytest.raises(RuntimeError, match = "cancelled"):
        b.generate(prompt = "a fox", width = 64, height = 64, steps = 4, seed = 1)

    b2 = _loaded_backend()
    out = b2.generate(prompt = "a fox", width = 64, height = 64, steps = 4, seed = 1)
    assert out["images"]
    seen.append(b2.cancel_generate())
    assert seen == [False]


def _pinned_state(
    b,
    *,
    policy_flags = (),
    device = "cuda",
):
    """A resident native load whose argv carries the --backend device pin on top of `policy_flags`,
    exactly as _run_load builds it once a card has been selected."""
    from core.inference.sd_cpp_args import device_backend_flags

    s = b._state
    return bk._SdState(
        repo_id = s.repo_id,
        base_repo = s.base_repo,
        family = s.family,
        device = device,
        files = s.files,
        vae_format = s.vae_format,
        sampling_method = s.sampling_method,
        flow_shift = s.flow_shift,
        mode = s.mode,
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        offload_flags = (
            *policy_flags,
            *device_backend_flags("CUDA1", list(policy_flags)),
        ),
    )


def test_a_card_pick_is_not_reported_as_an_offload():
    # `fast` asks for no offload, so status() and the saved recipe read "nothing was offloaded" off
    # an empty flag list; the pin lands in the same tuple, so picking a GPU looked like CPU offload.
    b = _loaded_backend()
    b._state = _pinned_state(b, policy_flags = ())
    assert b._state.offload_flags, "the fixture must actually carry the pin"
    status = b.status()
    assert status["cpu_offload"] is False
    assert status["offload_policy"] == "none"
    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 4, seed = 1)
    assert out["offload_policy"] == "none"


def test_a_real_offload_still_reports_itself_when_a_card_is_pinned():
    b = _loaded_backend()
    b._state = _pinned_state(b, policy_flags = ("--offload-to-cpu", "--diffusion-fa"))
    assert b.status()["cpu_offload"] is True
    assert b.status()["offload_policy"] == "active"


def test_the_cpu_backend_restart_drops_the_device_pin(monkeypatch):
    # sd.cpp CONCATENATES --backend values and a per-module entry outranks `cpu`, so the pin must leave argv.
    started: dict = {}

    class _FakeServer:
        def __init__(self, binary):
            self.binary = binary

        def start(self, files, **kwargs):
            started.update(kwargs)

        def stop(self):
            pass

    b = _loaded_backend()
    b._state = _pinned_state(b, policy_flags = ("--offload-to-cpu", "--clip-on-cpu"))
    object.__setattr__(b._state, "server", _FakeServer("/bin/sd-server"))
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/bin/sd-server")
    monkeypatch.setattr(bk, "SdCppServer", _FakeServer)

    server = b._restart_server_on_cpu_backend(
        b._state,
        "ggml_metal_op_encode_impl: unsupported op 'MUL_MAT' -> ggml_abort",
        threading.Event(),
    )
    assert server is not None
    argv = [*started["offload"], *started["extra_args"]]
    backends = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--backend"]
    assert backends == ["cpu"]
    assert "--offload-to-cpu" in argv and "--clip-on-cpu" in argv


def test_the_native_engine_resolves_a_bare_gpu_selection_itself(monkeypatch):
    import core.inference.sd_cpp_backend as backend_module

    seen: dict = {}
    monkeypatch.setattr(
        backend_module,
        "resolve_diffusion_device_target",
        lambda **kw: types.SimpleNamespace(device = "cuda"),
    )
    monkeypatch.setattr(
        backend_module,
        "resolve_selected_cuda_ordinal",
        lambda ids: (seen.update(ids = list(ids)), 1)[1],
    )
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(b, "_start_load_thread", lambda *a, **k: None, raising = False)
    captured: dict = {}

    def _fake_thread(
        target = None,
        kwargs = None,
        **_,
    ):
        captured.update(kwargs or {})
        return types.SimpleNamespace(start = lambda: None, join = lambda *a, **k: None)

    monkeypatch.setattr(backend_module.threading, "Thread", _fake_thread)
    b.begin_load(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        gpu_ids = [3],
    )
    assert seen["ids"] == [3]
    assert captured["gpu_ordinal"] == 1


def test_an_unresolvable_device_pin_says_so_and_still_loads(tmp_path, capsys):
    binary = tmp_path / "sd-cli-old"
    binary.write_text("#!/usr/bin/env bash\nexit 1\n")
    binary.chmod(0o755)
    assert bk.sd_cpp_device_name_for_ordinal(str(binary), 1) is None
    assert "device_pin_unresolved" in capsys.readouterr().out
    assert bk.sd_cpp_device_name_for_ordinal(str(binary), None) is None
    assert "device_pin_unresolved" not in capsys.readouterr().out


def test_generation_in_flight_tracks_a_generation(monkeypatch):
    from core.inference import diffusion_lora

    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    monkeypatch.setattr(bk, "_sd_cpp_backend", b)
    monkeypatch.setattr(diffusion_lora, "supports_lora", lambda **_k: True)

    seen: dict = {}

    def _resolve(
        active,
        *,
        family = None,
        hf_token = None,
        cancel_event = None,
    ):
        seen["in_flight"] = bk.generation_in_flight()
        return []

    monkeypatch.setattr(diffusion_lora, "resolve_specs", _resolve)

    assert bk.generation_in_flight() is False
    b.generate(prompt = "a fox", width = 64, height = 64, steps = 8, loras = [("some/lora", 1.0)])
    assert (
        seen["in_flight"] is True
    ), "liveness cannot tell this backend from a dead one while the native engine renders"
    assert bk.generation_in_flight() is False


def test_generation_in_flight_never_builds_a_backend(monkeypatch):
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)
    monkeypatch.setattr(
        bk,
        "SdCppDiffusionBackend",
        lambda *a, **k: pytest.fail("liveness constructed a native diffusion backend"),
    )
    assert bk.generation_in_flight() is False

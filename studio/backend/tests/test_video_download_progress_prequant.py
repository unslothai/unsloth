# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video download progress counts hosted pre-cast DiT / text-encoder fetches made before the base pull."""

from __future__ import annotations

import inspect
import types

import pytest

from core.inference import video as video_mod
from core.inference.diffusion_prequant import PrequantSource
from core.inference.diffusion_te_prequant import TePrequantSource

BASE = "someone/fake-video-base"
PREQUANT = "someone/fake-video-prequant"
TE_KEY = "a1" * 32
DIT_KEY = "b2" * 32


def _sib(
    name,
    size,
    sha = None,
):
    return types.SimpleNamespace(
        rfilename = name,
        size = size,
        blob_id = "git-" + name,
        lfs = types.SimpleNamespace(sha256 = sha) if sha else None,
    )


class _FakeApi:
    def __init__(self, token = None):
        pass

    def model_info(
        self,
        repo,
        files_metadata = True,
        **_kw,
    ):
        if repo == PREQUANT:
            siblings = [
                _sib("text_encoder_fp8.safetensors", 400, TE_KEY),
                _sib("transformer_fp8.safetensors", 900, DIT_KEY),
            ]
        else:
            siblings = [
                _sib("model_index.json", 2),
                _sib("vae/config.json", 1),
                _sib("vae/diffusion_pytorch_model.safetensors", 30, "c3" * 32),
            ]
        return types.SimpleNamespace(sha = "0" * 40, siblings = siblings)


@pytest.fixture
def fake_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(tmp_path))
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(tmp_path))
    import huggingface_hub
    import huggingface_hub.constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "other"))
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)
    return tmp_path


def _partial(
    root,
    repo,
    key,
    nbytes,
    nonce = "1a2b3c4d",
):
    blobs = root / ("models--" + repo.replace("/", "--")) / "blobs"
    blobs.mkdir(parents = True, exist_ok = True)
    (blobs / f"{key}.{nonce}.incomplete").write_bytes(b"x" * nbytes)


def _start_load(backend, plan, expected):
    backend._loading = video_mod._VideoLoadingState(
        repo_id = BASE, base_repo = BASE, expected_bytes = expected
    )
    backend._loading.incomplete_baseline = video_mod.VideoBackend._plan_partials(plan)
    backend._loading.fetch_plan = plan


def _estimate(backend, **kwargs):
    planned: dict = {}
    expected = backend._estimate_download_bytes(
        BASE, None, BASE, None, "pipeline", planned_files = planned, **kwargs
    )
    return expected, planned


def test_te_prequant_fetch_is_counted_while_its_partial_grows(fake_cache):
    backend = video_mod.VideoBackend()
    te = TePrequantSource(kind = "repo", location = PREQUANT, filename = "text_encoder_fp8.safetensors")
    expected, plan = _estimate(backend, te_sources = {"text_encoder": te})
    assert expected == 2 + 1 + 30 + 400
    _start_load(backend, plan, expected)
    assert backend.load_progress()["downloaded_bytes"] == 0
    _partial(fake_cache, PREQUANT, TE_KEY, 150)
    assert backend.load_progress()["downloaded_bytes"] == 150


def test_denoiser_prequant_fetch_is_counted_while_its_partial_grows(fake_cache):
    backend = video_mod.VideoBackend()
    dit = PrequantSource(kind = "repo", location = PREQUANT, filename = "transformer_fp8.safetensors")
    expected, plan = _estimate(backend, denoiser_sources = [(dit, "fp8")])
    assert expected == 2 + 1 + 30 + 900
    _start_load(backend, plan, expected)
    _partial(fake_cache, PREQUANT, DIT_KEY, 600)
    assert backend.load_progress()["downloaded_bytes"] == 600


def test_a_mirrored_te_prequant_read_in_place_is_not_planned(fake_cache, monkeypatch):
    """A pre-cast encoder the loader reads in place from a mirror is never fetched, so it is not in the total."""
    monkeypatch.setattr(
        "core.inference.diffusion_te_prequant.te_prequant_mirror_path",
        lambda repo, name: str(fake_cache / name),
    )
    (fake_cache / "text_encoder_fp8.safetensors").write_bytes(b"m" * 400)
    backend = video_mod.VideoBackend()
    te = TePrequantSource(kind = "repo", location = PREQUANT, filename = "text_encoder_fp8.safetensors")
    expected, plan = _estimate(backend, te_sources = {"text_encoder": te})
    assert expected == 2 + 1 + 30
    assert all(entry.repo != PREQUANT for entry in plan.values())


def test_a_partial_present_at_plan_time_counts_its_growth(fake_cache):
    """A still-writing partial already on disk when the plan is made is counted from its baseline on."""
    backend = video_mod.VideoBackend()
    dit = PrequantSource(kind = "repo", location = PREQUANT, filename = "transformer_fp8.safetensors")
    expected, plan = _estimate(backend, denoiser_sources = [(dit, "fp8")])
    _partial(fake_cache, PREQUANT, DIT_KEY, 100, nonce = "0badf00d")
    _start_load(backend, plan, expected)
    assert backend.load_progress()["downloaded_bytes"] == 0
    _partial(fake_cache, PREQUANT, DIT_KEY, 250, nonce = "0badf00d")
    assert backend.load_progress()["downloaded_bytes"] == 150


def test_run_load_plans_the_same_prequant_sources_it_fetches():
    src = inspect.getsource(video_mod.VideoBackend._run_load)
    calls = src.split("self._estimate_download_bytes(")[1:]
    assert calls and all(
        "denoiser_sources = " in call.split("\n            )")[0] for call in calls
    )
    # The fetch reads the very list the plan was made from, not a second resolution.
    assert "self._fetch_denoiser_prequant(\n                    denoiser_sources," in src


FALLBACK_KEY = "c4" * 32


def test_a_fallback_denoiser_candidate_is_counted_while_it_downloads(fake_cache, monkeypatch):
    """The first listed checkpoint fails mid-transfer; progress follows the fallback file the loop then pulls."""
    from utils import hf_xet_fallback

    primary, fallback = "transformer_fp8.safetensors", "transformer_fp8.pt"
    siblings = [_sib(primary, 900, DIT_KEY), _sib(fallback, 300, FALLBACK_KEY)]
    listings = types.SimpleNamespace(info = lambda repo: types.SimpleNamespace(siblings = siblings))
    dit = PrequantSource(
        kind = "repo", location = PREQUANT, filename = primary, fallback_filenames = (fallback,)
    )
    backend = video_mod.VideoBackend()
    plan = backend._plan_prequant_fetches(listings, denoiser_sources = [(dit, None)])
    assert [e.files for e in plan.values()] == [((primary, DIT_KEY, 900),)]
    backend._load_token = token = 7
    _start_load(backend, plan, video_mod._plan_total(plan))
    seen = []

    def download(repo, name, *_a, **_kw):
        if name == primary:
            raise ConnectionError("transfer reset")
        _partial(fake_cache, PREQUANT, FALLBACK_KEY, 150)
        seen.append(backend.load_progress())

    monkeypatch.setattr(hf_xet_fallback, "hf_hub_download_with_xet_fallback", download)
    followed = {}

    def on_fallback(source, name):
        followed["plan"] = backend._publish_followed_candidate(token, plan, source, name)

    backend._fetch_denoiser_prequant([dit], None, on_fallback = on_fallback)
    assert seen and seen[0]["downloaded_bytes"] == 150
    assert seen[0]["expected_bytes"] == 300
    # A re-plan made from the followed source names the same file.
    moved = video_mod._source_starting_at(dit, fallback)
    replanned = backend._plan_prequant_fetches(listings, denoiser_sources = [(moved, None)])
    assert [e.files for e in replanned.values()] == [((fallback, FALLBACK_KEY, 300),)]
    assert list(replanned) == list(followed["plan"])


def test_run_load_follows_a_denoiser_fallback():
    src = inspect.getsource(video_mod.VideoBackend._run_load)
    assert src.count("on_fallback = _follow_denoiser_fallback") == 2


def test_a_completed_hosted_file_in_a_symlinkless_cache_stays_counted(fake_cache):
    """Windows without symlinks moves the finished blob into snapshots/<commit>/: still the full size."""
    backend = video_mod.VideoBackend()
    te = TePrequantSource(kind = "repo", location = PREQUANT, filename = "text_encoder_fp8.safetensors")
    expected, plan = _estimate(backend, te_sources = {"text_encoder": te})
    _start_load(backend, plan, expected)
    folder = fake_cache / ("models--" + PREQUANT.replace("/", "--"))
    (folder / "blobs").mkdir(parents = True)
    stable = folder / "blobs" / f"{TE_KEY}.incomplete"
    stable.write_bytes(b"x" * 150)
    assert backend.load_progress()["downloaded_bytes"] == 150
    commit = "9" * 40
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text(commit)
    snap = folder / "snapshots" / commit / "text_encoder_fp8.safetensors"
    snap.parent.mkdir(parents = True)
    stable.unlink()
    snap.write_bytes(b"x" * 400)
    assert backend.load_progress()["downloaded_bytes"] == 400


def test_a_snapshot_symlink_to_another_blob_is_not_counted(fake_cache):
    """With symlinks, the snapshot entry points at a blob read by key; an older revision's blob must not count."""
    backend = video_mod.VideoBackend()
    te = TePrequantSource(kind = "repo", location = PREQUANT, filename = "text_encoder_fp8.safetensors")
    expected, plan = _estimate(backend, te_sources = {"text_encoder": te})
    _start_load(backend, plan, expected)
    folder = fake_cache / ("models--" + PREQUANT.replace("/", "--"))
    (folder / "blobs").mkdir(parents = True)
    old = folder / "blobs" / ("d5" * 32)
    old.write_bytes(b"o" * 400)
    commit = "8" * 40
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text(commit)
    snap = folder / "snapshots" / commit / "text_encoder_fp8.safetensors"
    snap.parent.mkdir(parents = True)
    snap.symlink_to(old)
    assert backend.load_progress()["downloaded_bytes"] == 0


def test_the_encoder_pick_and_the_plan_share_one_listing(fake_cache, monkeypatch):
    """The hosted-encoder choice and the plan that sizes it list the pre-cast repo once, so they read one commit."""
    import huggingface_hub

    calls = []

    class _CountingApi(_FakeApi):
        def model_info(
            self,
            repo,
            files_metadata = True,
            **kw,
        ):
            calls.append(repo)
            return super().model_info(repo, files_metadata = files_metadata, **kw)

    monkeypatch.setattr(huggingface_hub, "HfApi", _CountingApi)
    backend = video_mod.VideoBackend()
    te = TePrequantSource(kind = "repo", location = PREQUANT, filename = "text_encoder_fp8.safetensors")
    expected, plan = _estimate(backend, te_sources = {"text_encoder": te})
    assert expected == 2 + 1 + 30 + 400
    assert calls.count(PREQUANT) == 1
    assert calls.count(BASE) == 1

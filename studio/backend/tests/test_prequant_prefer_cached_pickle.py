# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A user who already holds a hosted ``.pt`` checkpoint must not download its ``.safetensors`` twin.

The candidate chain prefers safetensors, so once a repo publishes the twin every path that walks
the chain in order (the transformer resolver, the text-encoder resolver, the download planners and
the video prefetch) would fetch it again next to the multi-GB pickle already in the cache. These
tests drive a fake Hub cache through ``try_to_load_from_cache`` / ``hf_hub_download``."""

from __future__ import annotations

import types

import huggingface_hub
import pytest
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

import core.inference.diffusion_prequant as pq
import core.inference.diffusion_te_prequant as tpq

REPO = "unsloth/Model-FP8"
KILL_SWITCH = "UNSLOTH_PREQUANT_PREFER_SAFETENSORS"
NO_EXIST = object()  # stands in for huggingface_hub's recorded-404 sentinel


class FakeHub:
    """A Hub cache keyed by (repo, name) plus a Hub listing; records every download."""

    def __init__(
        self,
        tmp_path,
        cached = (),
        hosted = None,
    ):
        self.tmp = tmp_path
        self.cached: dict = {}
        self.hosted = set(hosted) if hosted is not None else None
        # names a 404 was recorded for, as huggingface_hub's .no_exist markers
        self.absent: set = set()
        self.downloads: list = []
        for name in cached:
            self.cache(name)

    def cache(
        self,
        name,
        repo = REPO,
    ):
        path = self.tmp / "cache" / repo.replace("/", "--") / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
        self.cached[(repo, name)] = str(path)

    def try_to_load_from_cache(
        self,
        repo_id,
        filename,
        cache_dir = None,
        **_,
    ):
        if (repo_id, filename) in self.absent:
            return NO_EXIST
        return self.cached.get((repo_id, filename))

    def hf_hub_download(
        self,
        repo_id,
        filename,
        token = None,
        cache_dir = None,
        local_files_only = False,
        **_,
    ):
        self.downloads.append((filename, bool(local_files_only)))
        hit = self.cached.get((repo_id, filename))
        if local_files_only:
            if hit is None:
                raise LocalEntryNotFoundError(f"{filename} not cached")
            return hit
        if self.hosted is not None and filename not in self.hosted:
            self.absent.add((repo_id, filename))
            raise EntryNotFoundError(f"404 {filename}")
        if hit is None:
            self.cache(filename, repo_id)
        return self.cached[(repo_id, filename)]

    @property
    def fetched(self):
        return [n for n, _ in self.downloads]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake.try_to_load_from_cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    monkeypatch.setattr(
        pq, "restricted_prequant_load_supported", lambda scheme = None, filename = None: True
    )
    monkeypatch.setattr(tpq, "te_candidate_is_readable", lambda name: bool(name))
    monkeypatch.delenv(pq.PREQUANT_MIRROR_ENV, raising = False)
    monkeypatch.delenv(KILL_SWITCH, raising = False)
    getattr(pq, "_logged_twin_choices", set()).clear()
    return fake


def _dit(*names):
    names = names or ("Model-INT8.safetensors", "Model-INT8.pt", "transformer_int8.pt")
    return pq.PrequantSource(
        kind = "repo", location = REPO, filename = names[0], fallback_filenames = tuple(names[1:])
    )


def _te(*names):
    names = names or ("Model-text_encoder-FP8.safetensors", "Model-text_encoder-FP8.pt")
    return tpq.TePrequantSource(
        kind = "repo", location = REPO, filename = names[0], fallback_filenames = tuple(names[1:])
    )


def _resolve(source, **kw):
    return pq._resolve_checkpoint_path(source, None, None, scheme = "int8", **kw)


# ---- transformer resolver ----


def test_cached_pickle_is_used_and_revalidated_online(hub, caplog):
    hub.cache("Model-INT8.pt")
    caplog.set_level("INFO")
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.pt")]
    # Revalidated through hf_hub_download for the PICKLE name; the safetensors twin is never asked.
    assert hub.downloads == [("Model-INT8.pt", False)]
    assert "Model-INT8.safetensors" not in hub.fetched
    assert "using the cached Model-INT8.pt" in caplog.text


def test_cached_pickle_offline_uses_cache_without_a_network_call(hub):
    hub.cache("Model-INT8.pt")
    assert _resolve(_dit(), local_files_only = True) == hub.cached[(REPO, "Model-INT8.pt")]
    assert hub.downloads == [("Model-INT8.pt", True)]


def test_new_user_downloads_safetensors(hub):
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == ["Model-INT8.safetensors"]


def test_both_cached_safetensors_wins(hub):
    hub.cache("Model-INT8.pt")
    hub.cache("Model-INT8.safetensors")
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == ["Model-INT8.safetensors"]


def test_kill_switch_forces_safetensors(hub, monkeypatch):
    hub.cache("Model-INT8.pt")
    monkeypatch.setenv(KILL_SWITCH, "1")
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == ["Model-INT8.safetensors"]


def test_twin_removed_from_hub_falls_back_to_the_chain(hub):
    hub.cache("Model-INT8.pt")
    hub.hosted = {"Model-INT8.safetensors"}
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == ["Model-INT8.pt", "Model-INT8.safetensors"]


def test_a_cached_different_artifact_does_not_preempt_a_better_one(hub):
    """INT8-ConvRot is a different artifact than INT8: a cached INT8 pickle must not win."""
    hub.cache("Model-INT8.pt")
    src = _dit(
        "Model-INT8-ConvRot.safetensors",
        "Model-INT8-ConvRot.pt",
        "Model-INT8.safetensors",
        "Model-INT8.pt",
    )
    assert _resolve(src) == hub.cached[(REPO, "Model-INT8-ConvRot.safetensors")]
    assert hub.fetched == ["Model-INT8-ConvRot.safetensors"]


def test_unreadable_twin_is_not_preferred(hub, monkeypatch):
    hub.cache("Model-INT8.pt")
    monkeypatch.setattr(
        pq,
        "restricted_prequant_load_supported",
        lambda scheme = None, filename = None: not str(filename).endswith(".pt"),
    )
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == ["Model-INT8.safetensors"]


def test_mirror_still_answers_first(hub, tmp_path, monkeypatch):
    hub.cache("Model-INT8.pt")
    mirrored = tmp_path / "mirror" / "unsloth" / "Model-FP8" / "Model-INT8.safetensors"
    mirrored.parent.mkdir(parents = True)
    mirrored.write_bytes(b"x")
    monkeypatch.setenv(pq.PREQUANT_MIRROR_ENV, str(tmp_path / "mirror"))
    assert _resolve(_dit()) == str(mirrored.resolve())
    assert hub.downloads == []


def test_cache_probe_agrees_with_the_resolver(hub):
    hub.cache("Model-INT8.pt")
    src = _dit()
    assert pq.cached_checkpoint_path(src) == _resolve(src)


# ---- text encoder resolver ----


def test_te_cached_pickle_is_used(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    got = tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live")
    assert got == hub.cached[(REPO, "Model-text_encoder-FP8.pt")]
    assert hub.downloads == [("Model-text_encoder-FP8.pt", False)]


def test_te_offline_cached_pickle(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    got = tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live", local_files_only = True)
    assert got == hub.cached[(REPO, "Model-text_encoder-FP8.pt")]
    assert hub.downloads == [("Model-text_encoder-FP8.pt", True)]


def test_te_new_user_downloads_safetensors(hub):
    tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live")
    assert hub.fetched == ["Model-text_encoder-FP8.safetensors"]


def test_te_cached_fp8_does_not_preempt_int8_convrot(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    src = _te(
        "Model-text_encoder-INT8-ConvRot.safetensors",
        "Model-text_encoder-INT8-ConvRot.pt",
        "Model-text_encoder-FP8.safetensors",
        "Model-text_encoder-FP8.pt",
    )
    tpq._resolve_checkpoint_path(src, None, cache_dir = "/live")
    assert hub.fetched == ["Model-text_encoder-INT8-ConvRot.safetensors"]


def test_te_twin_removed_from_hub_falls_back(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    hub.hosted = {"Model-text_encoder-FP8.safetensors"}
    got = tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live")
    assert got == hub.cached[(REPO, "Model-text_encoder-FP8.safetensors")]
    assert hub.fetched == ["Model-text_encoder-FP8.pt", "Model-text_encoder-FP8.safetensors"]


# ---- planners: price / stage the same name the resolver uses ----


class _Api:
    def __init__(self, names):
        self.names = names

    def model_info(
        self,
        repo_id,
        files_metadata = False,
    ):
        return types.SimpleNamespace(
            siblings = [
                types.SimpleNamespace(rfilename = n, size = 100 + i) for i, n in enumerate(self.names)
            ]
        )


def test_te_hub_files_planner_prices_the_cached_pickle(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    api = _Api(["Model-text_encoder-FP8.safetensors", "Model-text_encoder-FP8.pt"])
    files = tpq.te_prequant_hub_files({"text_encoder": _te()}, api, None)
    assert files == {"text_encoder": [("Model-text_encoder-FP8.pt", 101)]}
    hub.cached.clear()
    files = tpq.te_prequant_hub_files({"text_encoder": _te()}, api, None)
    assert files == {"text_encoder": [("Model-text_encoder-FP8.safetensors", 100)]}


def test_dit_planner_stages_the_cached_pickle(hub, monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    api = _Api(["Model-INT8.safetensors", "Model-INT8.pt"])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token = None: api)
    hub.cache("Model-INT8.pt")
    entry = DiffusionBackend._prequant_source_hub_entry(_dit(), None, scheme = "int8")
    assert entry == (REPO, "Model-INT8.pt", 101)
    assert entry[1] == _resolve(_dit()).rsplit("/", 1)[-1]
    hub.cached.clear()
    entry = DiffusionBackend._prequant_source_hub_entry(_dit(), None, scheme = "int8")
    assert entry == (REPO, "Model-INT8.safetensors", 100)


def test_video_denoiser_prefetch_fetches_the_cached_pickle(hub, monkeypatch):
    import utils.hf_xet_fallback as xet
    from core.inference.video import VideoBackend

    hub.cache("Model-INT8.pt")
    fetched = []

    def _dl(repo, name, token, **kw):
        fetched.append(name)
        return hub.cached.get((repo, name)) or "/new/" + name

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _dl)
    backend = VideoBackend.__new__(VideoBackend)
    import threading

    VideoBackend._fetch_denoiser_prequant(backend, [_dit()], None, cancel_event = threading.Event())
    assert fetched == ["Model-INT8.pt"]


def test_video_denoiser_prefetch_skips_an_unreadable_cached_pickle(hub, monkeypatch):
    import threading

    import utils.hf_xet_fallback as xet
    from core.inference.video import VideoBackend

    hub.cache("Model-INT8.pt")
    monkeypatch.setattr(
        pq,
        "restricted_prequant_load_supported",
        lambda scheme = None, filename = None: not str(filename).endswith(".pt"),
    )
    fetched = []

    def _dl(repo, name, token, **kw):
        fetched.append(name)
        return hub.cached.get((repo, name)) or "/new/" + name

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _dl)
    backend = VideoBackend.__new__(VideoBackend)
    VideoBackend._fetch_denoiser_prequant(
        backend, [_dit()], None, cancel_event = threading.Event(), scheme = "int8"
    )
    # the loader cannot open the cached .pt, so the prefetch stages the safetensors it will open
    assert fetched == ["Model-INT8.safetensors"]


def test_video_te_prefetch_fetches_the_cached_pickle(hub, monkeypatch):
    import threading

    import utils.hf_xet_fallback as xet
    from core.inference.video import VideoBackend

    hub.cache("Model-text_encoder-FP8.pt")
    fetched = []

    def _dl(repo, name, token, **kw):
        fetched.append(name)
        return hub.cached.get((repo, name)) or "/new/" + name

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _dl)
    backend = VideoBackend.__new__(VideoBackend)
    got = VideoBackend._fetch_te_prequant(
        backend, {"text_encoder": _te()}, None, cancel_event = threading.Event()
    )
    assert got == ("text_encoder",) and fetched == ["Model-text_encoder-FP8.pt"]


# ---- policy: new users only ever get the safetensors container ----


def test_new_user_never_requests_the_pickle_when_the_twin_is_hosted(hub, monkeypatch):
    """Both containers on the Hub, nothing cached: only the .safetensors is fetched, on every path."""
    import threading

    import utils.hf_xet_fallback as xet
    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import VideoBackend

    hub.hosted = {
        "Model-INT8.safetensors",
        "Model-INT8.pt",
        "Model-text_encoder-FP8.safetensors",
        "Model-text_encoder-FP8.pt",
    }
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live")
    assert hub.fetched == ["Model-INT8.safetensors", "Model-text_encoder-FP8.safetensors"]
    hub.cached.clear()

    api = _Api(sorted(hub.hosted))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token = None: api)
    assert DiffusionBackend._prequant_source_hub_entry(_dit(), None, scheme = "int8")[1] == (
        "Model-INT8.safetensors"
    )
    assert tpq.te_prequant_hub_files({"text_encoder": _te()}, api, None)["text_encoder"][0][0] == (
        "Model-text_encoder-FP8.safetensors"
    )
    fetched = []
    monkeypatch.setattr(
        xet, "hf_hub_download_with_xet_fallback", lambda repo, name, tok, **kw: fetched.append(name)
    )
    backend = VideoBackend.__new__(VideoBackend)
    VideoBackend._fetch_denoiser_prequant(backend, [_dit()], None, cancel_event = threading.Event())
    VideoBackend._fetch_te_prequant(
        backend, {"text_encoder": _te()}, None, cancel_event = threading.Event()
    )
    assert fetched == ["Model-INT8.safetensors", "Model-text_encoder-FP8.safetensors"]
    assert not any(n.endswith(".pt") for n in hub.fetched + fetched)


def test_pickle_only_repo_downloads_the_pickle_and_says_why(hub, caplog):
    hub.hosted = {"Model-INT8.pt"}
    caplog.set_level("INFO")
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.pt")]
    assert hub.fetched == ["Model-INT8.safetensors", "Model-INT8.pt"]
    assert "does not host the .safetensors twin yet" in caplog.text


def test_install_without_safetensors_support_downloads_the_pickle_and_says_why(
    hub, monkeypatch, caplog
):
    monkeypatch.setattr(
        pq,
        "restricted_prequant_load_supported",
        lambda scheme = None, filename = None: not str(filename).endswith(".safetensors"),
    )
    caplog.set_level("INFO")
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.pt")]
    assert hub.fetched == ["Model-INT8.pt"]
    assert "cannot read the .safetensors container" in caplog.text


# ---- cache probes: "nothing to download" must mean the file the load will open ----

CONVROT = ("Model-INT8-ConvRot.safetensors",)
PLAIN = ("Model-INT8.safetensors", "Model-INT8.pt", "transformer_int8.pt")


def _convrot_dit(declared = True):
    names = CONVROT + PLAIN
    return pq.PrequantSource(
        kind = "repo",
        location = REPO,
        filename = names[0],
        fallback_filenames = names[1:],
        declared_filenames = CONVROT if declared else (),
    )


def test_cached_int8_is_not_free_when_a_declared_convrot_is_ahead(hub):
    """An existing user's INT8 .pt is cached and the family now declares an INT8-ConvRot build.
    Online the load downloads the ConvRot file first, so the planner and the disk gate must not be
    told this costs nothing."""
    hub.cache("Model-INT8.pt")
    src = _convrot_dit()
    assert pq.cached_checkpoint_path(src, online = True) is None
    assert pq.prequant_checkpoint_cached(src, online = True) is False
    # and that is what the load really does
    _resolve(src)
    assert hub.fetched == ["Model-INT8-ConvRot.safetensors"]


def test_offline_the_cached_int8_is_what_loads(hub):
    hub.cache("Model-INT8.pt")
    src = _convrot_dit()
    assert pq.cached_checkpoint_path(src, online = False) == hub.cached[(REPO, "Model-INT8.pt")]
    assert _resolve(src, local_files_only = True) == hub.cached[(REPO, "Model-INT8.pt")]


def test_an_unpublished_convrot_stops_blocking_once_the_resolver_saw_its_404(hub):
    """The code names a ConvRot build the repo does not host yet (merged before the upload). The
    first load asks, gets a 404, and opens the cached INT8 .pt; from then on the probe agrees with
    the resolver instead of planning a download that never happens on every load."""
    hub.cache("Model-INT8.pt")
    src = _convrot_dit()
    # never asked: online, it may be hosted, so a download is planned
    assert pq.cached_checkpoint_path(src, online = True) is None
    hub.hosted = set(PLAIN)
    got = _resolve(src)
    assert got == hub.cached[(REPO, "Model-INT8.pt")]
    assert hub.fetched == ["Model-INT8-ConvRot.safetensors", "Model-INT8.pt"]
    assert pq.cached_checkpoint_path(src, online = True) == got
    assert pq.prequant_checkpoint_cached(src, online = True) is True


def test_a_derived_name_is_cleared_by_the_resolvers_own_404(hub):
    """Same rule for a name derived from the repo id rather than declared by the family."""
    hub.cache("Model-INT8.pt")
    src = _convrot_dit(declared = False)
    assert pq.cached_checkpoint_path(src, online = True) is None
    hub.hosted = {"Model-INT8.pt"}
    got = _resolve(src)
    assert got == hub.cached[(REPO, "Model-INT8.pt")]
    assert pq.cached_checkpoint_path(src, online = True) == got


def test_the_cached_pickle_twin_still_counts_online(hub):
    """The safetensors twin of a cached .pt never blocks it: the resolver opens the .pt."""
    hub.cache("Model-INT8.pt")
    src = _dit()
    assert pq.cached_checkpoint_path(src, online = True) == hub.cached[(REPO, "Model-INT8.pt")]
    assert pq.cached_checkpoint_path(src, online = True) == _resolve(src)
    assert hub.fetched == ["Model-INT8.pt"]


def test_a_cached_convrot_is_free(hub):
    hub.cache("Model-INT8.pt")
    hub.cache("Model-INT8-ConvRot.safetensors")
    got = pq.cached_checkpoint_path(_convrot_dit(), online = True)
    assert got == hub.cached[(REPO, "Model-INT8-ConvRot.safetensors")]


def test_video_cached_repo_probe_follows_the_same_rule(hub, monkeypatch):
    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import VideoBackend

    hub.cache("Model-INT8.pt")
    monkeypatch.setattr(
        VideoBackend,
        "_denoiser_prequant_source_list",
        staticmethod(lambda *a, **k: [_convrot_dit()]),
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo, name, *a, **k: (repo, name) in hub.cached),
    )
    probe = VideoBackend._denoiser_prequant_cached_repo
    assert probe(None, "int8", "base/Model", online = True) is None
    assert probe(None, "int8", "base/Model", online = False) == REPO


def test_video_cached_repo_probe_ignores_a_cached_unreadable_file(hub, monkeypatch):
    """Offline, a cached safetensors this install cannot read is not what the loader opens."""
    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import VideoBackend

    hub.cache("Model-INT8.safetensors")
    monkeypatch.setattr(
        pq,
        "restricted_prequant_load_supported",
        lambda scheme = None, filename = None: not str(filename).endswith(".safetensors"),
    )
    monkeypatch.setattr(
        VideoBackend, "_denoiser_prequant_source_list", staticmethod(lambda *a, **k: [_dit()])
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo, name, *a, **k: (repo, name) in hub.cached),
    )
    assert VideoBackend._denoiser_prequant_cached_repo(None, "int8", "b", online = False) is None
    hub.cache("Model-INT8.pt")
    assert VideoBackend._denoiser_prequant_cached_repo(None, "int8", "b", online = False) == REPO


def test_te_pricing_does_not_call_a_cached_fp8_encoder_free(hub, monkeypatch, tmp_path):
    """The text-encoder size estimate: a cached FP8 encoder is not what loads when an uncached
    INT8-ConvRot encoder is ahead of it and no 404 for it is recorded, so the size is not exact."""
    import core.inference.diffusion as diffusion
    from core.inference.diffusion import DiffusionBackend

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "Model-text_encoder-FP8.pt").write_bytes(b"x" * 4096)
    hub.cache("Model-text_encoder-FP8.pt")
    convrot = ("Model-text_encoder-INT8-ConvRot.safetensors",)

    def _sources(*a, **k):
        names = convrot + ("Model-text_encoder-FP8.safetensors", "Model-text_encoder-FP8.pt")
        return {
            "text_encoder": tpq.TePrequantSource(
                kind = "repo",
                location = REPO,
                filename = names[0],
                fallback_filenames = names[1:],
            )
        }

    monkeypatch.setattr(tpq, "te_prequant_sources_for_base", _sources)
    monkeypatch.setattr(
        DiffusionBackend,
        "_union_over_cached_revs",
        staticmethod(lambda base, fn, staged_dir = None: sum(fn(snap).values())),
    )
    monkeypatch.setattr(diffusion, "family_bf16_components_gb", lambda fam, base: (10.0, 16.0))
    monkeypatch.setattr(pq, "hub_offline", lambda: False)
    fam = types.SimpleNamespace(name = "probe")
    mib, components, exact = DiffusionBackend._precast_text_encoder_mib(
        fam, "base/Model", None, "int8"
    )
    assert components == ("text_encoder",)
    assert exact is False
    # Not hosted yet: once the resolver's 404 for it is recorded, the cached fp8 file is what loads.
    hub.absent.add((REPO, convrot[0]))
    mib, components, exact = DiffusionBackend._precast_text_encoder_mib(
        fam, "base/Model", None, "int8"
    )
    assert exact is True and mib == 1
    hub.absent.clear()
    # Once the ConvRot encoder is cached it is what loads, and the size is the file's own.
    (snap / "Model-text_encoder-INT8-ConvRot.safetensors").write_bytes(b"x" * (3 << 20))
    mib, components, exact = DiffusionBackend._precast_text_encoder_mib(
        fam, "base/Model", None, "int8"
    )
    assert exact is True and mib == 3


def test_kill_switch_probe_plans_the_safetensors_download(hub, monkeypatch):
    """With the kill switch the resolver fetches the uncached safetensors, so the probe must not call it free."""
    hub.cache("Model-INT8.pt")
    monkeypatch.setenv(KILL_SWITCH, "1")
    assert pq.cached_checkpoint_path(_dit(), online = True) is None
    assert pq.cached_checkpoint_path(_dit(), online = False) == hub.cached[(REPO, "Model-INT8.pt")]


def test_te_pickle_in_the_default_root_is_reused_through_that_root(hub, monkeypatch):
    """A pickle cached only under huggingface_hub's default root is what the planner and the video
    prefetch count, so the TE resolver reuses it through that root instead of fetching the twin."""
    hub.cache("Model-text_encoder-FP8.pt")
    real = hub.try_to_load_from_cache
    real_dl = hub.hf_hub_download
    roots = []

    def _default_root_only(
        repo_id,
        filename,
        cache_dir = None,
        **kw,
    ):
        return real(repo_id, filename) if cache_dir is None else None

    def _dl(
        repo_id,
        filename,
        cache_dir = None,
        **kw,
    ):
        roots.append(cache_dir)
        return real_dl(repo_id, filename, cache_dir = cache_dir, **kw)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _default_root_only)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _dl)
    got = tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live")
    assert got == hub.cached[(REPO, "Model-text_encoder-FP8.pt")]
    assert hub.fetched == ["Model-text_encoder-FP8.pt"] and roots == [None]


def test_local_files_only_reachability_uses_the_cached_fallback(hub, monkeypatch):
    """local_files_only skips the uncached ConvRot and opens the cached INT8, so the prequant is reachable."""
    import core.inference.diffusion as dmod

    hub.cache("Model-INT8.pt")
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda *a, **k: _convrot_dit())
    backend = dmod.DiffusionBackend.__new__(dmod.DiffusionBackend)
    assert backend._hosted_prequant_reachable(None, "int8", {"local_files_only": True}) is True
    assert _resolve(_convrot_dit(), local_files_only = True) == hub.cached[(REPO, "Model-INT8.pt")]


def test_a_local_files_only_load_probes_like_an_offline_one(hub, monkeypatch):
    """Inside a local_files_only load every default probe (auto policy, retry rung, plan source) opens
    the cached fallback, as the resolver will; outside it the uncached ConvRot still plans a download."""
    monkeypatch.setattr(pq, "hub_offline", lambda: False)
    hub.cache("Model-INT8.pt")
    seen = {}

    @pq.scoped_local_files_only
    def _load(*, local_files_only = False):
        seen["inner"] = pq.prequant_checkpoint_cached(_convrot_dit())
        return pq.prequant_checkpoint_cached(_convrot_dit())

    assert _load(local_files_only = True) is True and seen["inner"] is True
    assert _load(local_files_only = False) is False
    assert pq.prequant_checkpoint_cached(_convrot_dit()) is False

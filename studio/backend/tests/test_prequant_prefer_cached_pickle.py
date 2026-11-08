# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-quant checkpoints load from .safetensors only: a cached .pt is never promoted, priced, prefetched or opened,
whatever the chain a caller hands in still names; a repo or install that cannot serve the .safetensors falls back to
runtime quantisation of the dense weights, never to the pickle. ``UNSLOTH_PREQUANT_PREFER_SAFETENSORS`` is inert."""

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


def test_a_cached_pickle_is_ignored_and_the_twin_fetched_online(hub, caplog):
    hub.cache("Model-INT8.pt")
    caplog.set_level("INFO")
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.downloads == [("Model-INT8.safetensors", False)]
    assert "Model-INT8.pt" not in hub.fetched
    assert "using the cached Model-INT8.pt" not in caplog.text


def test_a_cached_pickle_is_never_opened_offline(hub):
    hub.cache("Model-INT8.pt")
    with pytest.raises(LocalEntryNotFoundError):
        _resolve(_dit(), local_files_only = True)
    assert hub.downloads == [("Model-INT8.safetensors", True)]


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


def test_a_cached_pickle_never_shadows_the_hosted_twin(hub):
    hub.cache("Model-INT8.pt")
    hub.hosted = {"Model-INT8.safetensors"}
    assert _resolve(_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == ["Model-INT8.safetensors"]


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
    # the cached pickle is not what loads, so it is not "free"
    assert pq.cached_checkpoint_path(src) is None
    got = _resolve(src)
    assert got == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert pq.cached_checkpoint_path(src) == got


def test_te_cached_pickle_is_ignored(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    got = tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live")
    assert got == hub.cached[(REPO, "Model-text_encoder-FP8.safetensors")]
    assert hub.downloads == [("Model-text_encoder-FP8.safetensors", False)]


def test_te_offline_cached_pickle_is_not_used(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    with pytest.raises(LocalEntryNotFoundError):
        tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live", local_files_only = True)
    assert "Model-text_encoder-FP8.pt" not in hub.fetched


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


def test_te_cached_pickle_never_shadows_the_hosted_twin(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    hub.hosted = {"Model-text_encoder-FP8.safetensors"}
    got = tpq._resolve_checkpoint_path(_te(), None, cache_dir = "/live")
    assert got == hub.cached[(REPO, "Model-text_encoder-FP8.safetensors")]
    assert hub.fetched == ["Model-text_encoder-FP8.safetensors"]


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


def test_te_hub_files_planner_never_prices_a_pickle(hub):
    hub.cache("Model-text_encoder-FP8.pt")
    api = _Api(["Model-text_encoder-FP8.safetensors", "Model-text_encoder-FP8.pt"])
    files = tpq.te_prequant_hub_files({"text_encoder": _te()}, api, None)
    assert files == {"text_encoder": [("Model-text_encoder-FP8.safetensors", 100)]}
    # a repo that hosts only the pickle keeps the dense encoder
    files = tpq.te_prequant_hub_files(
        {"text_encoder": _te()}, _Api(["Model-text_encoder-FP8.pt"]), None
    )
    assert files == {}


def test_dit_planner_stages_the_safetensors_even_with_a_cached_pickle(hub, monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    api = _Api(["Model-INT8.safetensors", "Model-INT8.pt"])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token = None: api)
    hub.cache("Model-INT8.pt")
    entry = DiffusionBackend._prequant_source_hub_entry(_dit(), None, scheme = "int8")
    assert entry == (REPO, "Model-INT8.safetensors", 100)
    assert entry[1] == _resolve(_dit()).rsplit("/", 1)[-1]
    # a listing with only the pickle stages nothing: the dense shards stay in the plan
    api.names = ["Model-INT8.pt"]
    failures = []
    assert (
        DiffusionBackend._prequant_source_hub_entry(_dit(), None, failures, scheme = "int8") is None
    )
    assert failures


def test_video_denoiser_prefetch_never_fetches_the_cached_pickle(hub, monkeypatch):
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
    assert fetched == ["Model-INT8.safetensors"]


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


def test_video_te_prefetch_never_fetches_the_cached_pickle(hub, monkeypatch):
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
    assert got == ("text_encoder",) and fetched == ["Model-text_encoder-FP8.safetensors"]


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


def test_pickle_only_repo_is_never_loaded_and_the_probe_refuses_it(hub, monkeypatch):
    import core.inference.prequant_header_probe as hp

    hub.hosted = {"Model-INT8.pt"}
    with pytest.raises(EntryNotFoundError):
        _resolve(_dit())
    assert hub.fetched == ["Model-INT8.safetensors"]
    # planning learns the same up front: every .safetensors name 404s, so the dense weights are staged
    monkeypatch.setattr(
        hp,
        "read_hub_checkpoint_header",
        lambda repo, name, token = None, **_: (None, True if name in hub.hosted else False),
    )
    monkeypatch.setattr(pq, "hub_offline", lambda: False)
    # the install's real rule: a pickle name is never readable, so the probe walks only the .safetensors names
    monkeypatch.setattr(
        pq,
        "restricted_prequant_load_supported",
        lambda scheme = None, filename = None: str(filename).endswith(".safetensors"),
    )
    pq._HEADER_PROBE_MEMO.clear()
    reason = pq.hosted_prequant_refusal(_dit(), "int8", hf_token = "tok-pickle-only")
    assert reason and "no .safetensors checkpoint is hosted" in reason


def test_install_without_safetensors_support_quantises_at_runtime(hub, monkeypatch):
    monkeypatch.setattr(
        pq,
        "restricted_prequant_load_supported",
        lambda scheme = None, filename = None: not str(filename).endswith(".safetensors"),
    )
    with pytest.raises(ValueError, match = "cannot be read here"):
        _resolve(_dit())
    assert hub.fetched == []


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
    _resolve(src)
    assert hub.fetched == ["Model-INT8-ConvRot.safetensors"]


def test_offline_a_cached_int8_pickle_is_not_what_loads(hub):
    hub.cache("Model-INT8.pt")
    src = _convrot_dit()
    assert pq.cached_checkpoint_path(src, online = False) is None
    with pytest.raises(LocalEntryNotFoundError):
        _resolve(src, local_files_only = True)
    assert "Model-INT8.pt" not in hub.fetched


def test_an_unpublished_convrot_stops_blocking_once_the_resolver_saw_its_404(hub):
    """The code names a ConvRot build the repo does not host yet. The first load asks, gets a 404, and opens the
    plain INT8 .safetensors (never the cached INT8 .pt); from then on the probe agrees with the resolver."""
    hub.cache("Model-INT8.pt")
    src = _convrot_dit()
    assert pq.cached_checkpoint_path(src, online = True) is None
    hub.hosted = set(PLAIN)
    got = _resolve(src)
    assert got == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == ["Model-INT8-ConvRot.safetensors", "Model-INT8.safetensors"]
    assert pq.cached_checkpoint_path(src, online = True) == got
    assert pq.prequant_checkpoint_cached(src, online = True) is True


def test_a_repo_hosting_only_the_pickle_leaves_nothing_cached_after_its_404s(hub):
    """Same rule for derived names: after the .safetensors 404s nothing loads, and the cached .pt is not credited."""
    hub.cache("Model-INT8.pt")
    src = _convrot_dit(declared = False)
    assert pq.cached_checkpoint_path(src, online = True) is None
    hub.hosted = {"Model-INT8.pt"}
    with pytest.raises(EntryNotFoundError):
        _resolve(src)
    assert "Model-INT8.pt" not in hub.fetched
    assert pq.cached_checkpoint_path(src, online = True) is None


def test_a_cached_pickle_twin_never_counts(hub):
    """The safetensors twin of a cached .pt is what loads, so the cached .pt makes nothing free."""
    hub.cache("Model-INT8.pt")
    src = _dit()
    assert pq.cached_checkpoint_path(src, online = True) is None
    assert pq.cached_checkpoint_path(src, online = False) is None
    assert _resolve(src) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == ["Model-INT8.safetensors"]


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
    # a cached pickle alone never makes the hosted denoiser "cached"
    assert probe(None, "int8", "base/Model", online = True) is None
    assert probe(None, "int8", "base/Model", online = False) is None
    hub.cache("Model-INT8.safetensors")
    assert probe(None, "int8", "base/Model", online = True) is None
    assert probe(None, "int8", "base/Model", online = False) == REPO


def test_video_cached_repo_probe_ignores_a_cached_unreadable_file(hub, monkeypatch):
    """Offline, a cached safetensors this install cannot read is not what the loader opens, nor is a cached .pt."""
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
    assert VideoBackend._denoiser_prequant_cached_repo(None, "int8", "b", online = False) is None


def test_te_pricing_does_not_call_a_cached_fp8_encoder_free(hub, monkeypatch, tmp_path):
    """The text-encoder size estimate: a cached FP8 encoder is not what loads when an uncached
    INT8-ConvRot encoder is ahead of it and no 404 for it is recorded, so the size is not exact.
    A cached FP8 .pt is never priced at all."""
    import core.inference.diffusion as diffusion
    from core.inference.diffusion import DiffusionBackend

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "Model-text_encoder-FP8.pt").write_bytes(b"x" * (2 << 20))
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
    hub.absent.add((REPO, convrot[0]))
    mib, components, exact = DiffusionBackend._precast_text_encoder_mib(
        fam, "base/Model", None, "int8"
    )
    # only the pickle is cached: nothing loads from the cache, so nothing is priced from it
    assert components == ("text_encoder",)
    assert exact is False
    hub.absent.clear()
    (snap / "Model-text_encoder-FP8.safetensors").write_bytes(b"x" * 4096)
    hub.cache("Model-text_encoder-FP8.safetensors")
    mib, components, exact = DiffusionBackend._precast_text_encoder_mib(
        fam, "base/Model", None, "int8"
    )
    assert components == ("text_encoder",)
    assert exact is False
    hub.absent.add((REPO, convrot[0]))
    mib, components, exact = DiffusionBackend._precast_text_encoder_mib(
        fam, "base/Model", None, "int8"
    )
    assert exact is True and mib == 1
    hub.absent.clear()
    (snap / "Model-text_encoder-INT8-ConvRot.safetensors").write_bytes(b"x" * (3 << 20))
    mib, components, exact = DiffusionBackend._precast_text_encoder_mib(
        fam, "base/Model", None, "int8"
    )
    assert exact is True and mib == 3


@pytest.mark.parametrize("value", [None, "1", "0"])
def test_the_prefer_safetensors_switch_is_inert_for_the_cache_probe(hub, monkeypatch, value):
    """Set or not, a cached pickle never makes the probe call the load free."""
    hub.cache("Model-INT8.pt")
    if value is not None:
        monkeypatch.setenv(KILL_SWITCH, value)
    assert pq.cached_checkpoint_path(_dit(), online = True) is None
    assert pq.cached_checkpoint_path(_dit(), online = False) is None


def test_te_pickle_in_the_default_root_is_never_reused(hub, monkeypatch):
    """A pickle cached only under huggingface_hub's default root is not what loads: the TE resolver fetches the
    .safetensors into the live root instead."""
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
    assert got == hub.cached[(REPO, "Model-text_encoder-FP8.safetensors")]
    assert hub.fetched == ["Model-text_encoder-FP8.safetensors"] and roots == ["/live"]


def test_local_files_only_with_only_a_cached_pickle_is_unreachable(hub, monkeypatch):
    """local_files_only with only the INT8 .pt cached: nothing loadable, so the hosted prequant is not reachable;
    a cached INT8 .safetensors fallback is."""
    import core.inference.diffusion as dmod

    hub.cache("Model-INT8.pt")
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda *a, **k: _convrot_dit())
    backend = dmod.DiffusionBackend.__new__(dmod.DiffusionBackend)
    assert backend._hosted_prequant_reachable(None, "int8", {"local_files_only": True}) is False
    with pytest.raises(LocalEntryNotFoundError):
        _resolve(_convrot_dit(), local_files_only = True)
    hub.cache("Model-INT8.safetensors")
    assert backend._hosted_prequant_reachable(None, "int8", {"local_files_only": True}) is True
    assert (
        _resolve(_convrot_dit(), local_files_only = True)
        == hub.cached[(REPO, "Model-INT8.safetensors")]
    )


def test_a_local_files_only_load_probes_like_an_offline_one(hub, monkeypatch):
    """Inside a local_files_only load every default probe opens the cached .safetensors fallback, as the resolver
    will; outside it the uncached ConvRot still plans a download. A cached .pt alone counts for neither."""
    monkeypatch.setattr(pq, "hub_offline", lambda: False)
    hub.cache("Model-INT8.pt")
    seen = {}

    @pq.scoped_local_files_only
    def _load(*, local_files_only = False):
        seen["inner"] = pq.prequant_checkpoint_cached(_convrot_dit())
        return pq.prequant_checkpoint_cached(_convrot_dit())

    assert _load(local_files_only = True) is False and seen["inner"] is False
    hub.cache("Model-INT8.safetensors")
    assert _load(local_files_only = True) is True and seen["inner"] is True
    assert _load(local_files_only = False) is False
    assert pq.prequant_checkpoint_cached(_convrot_dit()) is False


# ---- ComfyUI-format twin: the default download, never a re-download of a cached artifact ----

COMFY = "Model-INT8-ComfyUI.safetensors"


def _comfy_dit():
    return _dit(COMFY, "Model-INT8.safetensors", "Model-INT8.pt", "transformer_int8.pt")


def test_new_user_downloads_the_comfy_twin(hub):
    assert _resolve(_comfy_dit()) == hub.cached[(REPO, COMFY)]
    assert hub.fetched == [COMFY]


def test_cached_own_safetensors_is_kept_over_the_comfy_twin(hub, caplog):
    hub.cache("Model-INT8.safetensors")
    caplog.set_level("INFO")
    assert _resolve(_comfy_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert COMFY not in hub.fetched
    assert "its ComfyUI-format twin" in caplog.text


def test_a_cached_own_pickle_is_dropped_and_the_twin_leads(hub):
    hub.cache("Model-INT8.pt")
    assert pq.prefer_cached_pickle_twins(REPO, _comfy_dit().candidate_filenames) == [
        COMFY,
        "Model-INT8.safetensors",
    ]
    assert _resolve(_comfy_dit()) == hub.cached[(REPO, COMFY)]
    assert hub.fetched == [COMFY]


def test_a_cached_own_pickle_offline_is_not_opened(hub):
    hub.cache("Model-INT8.pt")
    with pytest.raises(LocalEntryNotFoundError):
        _resolve(_comfy_dit(), local_files_only = True)
    assert "Model-INT8.pt" not in hub.fetched
    assert all(offline for _n, offline in hub.downloads)


def test_cached_comfy_twin_stays_first(hub):
    hub.cache(COMFY)
    hub.cache("Model-INT8.pt")
    assert _resolve(_comfy_dit()) == hub.cached[(REPO, COMFY)]
    assert hub.fetched == [COMFY]


def test_repo_without_the_twin_falls_back_to_the_old_names(hub):
    """A repo that does not host the twin yet (or an older publish): one 404, then today's artifact."""
    hub.hosted = {"Model-INT8.safetensors", "Model-INT8.pt"}
    assert _resolve(_comfy_dit()) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert hub.fetched == [COMFY, "Model-INT8.safetensors"]


def test_a_cached_other_artifact_does_not_demote_the_twin(hub):
    """INT8 cached does not stand in for the INT8-ConvRot twin (different weights); only its own twin moves."""
    hub.cache("Model-INT8.safetensors")
    names = (
        "Model-INT8-ConvRot-ComfyUI.safetensors",
        "Model-INT8-ConvRot.safetensors",
        COMFY,
        "Model-INT8.safetensors",
    )
    assert pq.prefer_cached_pickle_twins(REPO, names) == [
        "Model-INT8-ConvRot-ComfyUI.safetensors",
        "Model-INT8-ConvRot.safetensors",
        "Model-INT8.safetensors",
        COMFY,
    ]


def test_cache_probe_reads_a_cached_own_file_as_free_online(hub):
    """The planner's no-download answer: the cached own file, not 'must fetch the twin'."""
    hub.cache("Model-INT8.safetensors")
    src = _comfy_dit()
    assert pq.cached_checkpoint_path(src) == hub.cached[(REPO, "Model-INT8.safetensors")]
    assert pq.cached_checkpoint_path(src) == _resolve(src)


def test_dit_planner_stages_the_twin_only_when_nothing_is_cached(hub, monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    api = _Api([COMFY, "Model-INT8.safetensors", "Model-INT8.pt"])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token = None: api)
    assert DiffusionBackend._prequant_source_hub_entry(_comfy_dit(), None, scheme = "int8") == (
        REPO,
        COMFY,
        100,
    )
    hub.cache("Model-INT8.safetensors")
    assert DiffusionBackend._prequant_source_hub_entry(_comfy_dit(), None, scheme = "int8") == (
        REPO,
        "Model-INT8.safetensors",
        101,
    )
    # a repo listing without the twin stages today's artifact
    hub.cached.clear()
    api.names = ["Model-INT8.safetensors", "Model-INT8.pt"]
    assert DiffusionBackend._prequant_source_hub_entry(_comfy_dit(), None, scheme = "int8") == (
        REPO,
        "Model-INT8.safetensors",
        100,
    )


def test_a_cached_trailing_fp8_twin_is_used_before_an_uncached_own_file(hub):
    """fp8 twins trail Studio's own artifact; one that is already cached still wins over downloading the other."""
    fp8 = _dit(
        "Model-FP8.safetensors",
        "Model-FP8.pt",
        "Model-FP8-ComfyUI.safetensors",
        "transformer_fp8.pt",
    )
    hub.cache("Model-FP8-ComfyUI.safetensors")
    assert _resolve(fp8) == hub.cached[(REPO, "Model-FP8-ComfyUI.safetensors")]
    assert hub.fetched == ["Model-FP8-ComfyUI.safetensors"]
    # nothing cached: Studio's own per-row fp8 is the download
    hub.cached.clear()
    hub.downloads.clear()
    assert _resolve(fp8) == hub.cached[(REPO, "Model-FP8.safetensors")]
    assert hub.fetched == ["Model-FP8.safetensors"]


# nvfp4 is the exception: its hosted repos hold only the .safetensors, and a .pt is never resolved, cached or opened.
NV_REPO = "unsloth/Z-Image-Turbo-NVFP4"
NV_ST, NV_PT = "Z-Image-Turbo-NVFP4.safetensors", "Z-Image-Turbo-NVFP4.pt"
# Captured before any fixture stubs it: the nvfp4 cases ask the install's real readability question.
_REAL_LOAD_SUPPORTED = pq.restricted_prequant_load_supported


@pytest.fixture
def nv_hub(hub, monkeypatch):
    from core.inference import diffusion_nvfp4_flag

    monkeypatch.setenv(diffusion_nvfp4_flag.NVFP4_DIFFUSION_ENV, "1")
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", _REAL_LOAD_SUPPORTED)
    hub.hosted = {NV_ST, NV_PT}
    return hub


def _nv_source():
    """The chain Studio resolves for Z-Image's hosted NVFP4 repo, not a hand-written one."""
    from core.inference.diffusion_families import detect_family

    fam = detect_family("Tongyi-MAI/Z-Image-Turbo")
    source = pq.resolve_prequant_source(fam, "nvfp4")
    assert source.location == NV_REPO
    chain = pq.candidate_filenames_of(source)
    # the hosted safetensors leads; no .pt / .pth name anywhere (a ComfyUI-format twin may trail)
    assert chain[0] == NV_ST and all(n.endswith(".safetensors") for n in chain), chain
    assert source.declared_filenames == (NV_ST,)
    return source


def _nv_resolve(source, **kw):
    return pq._resolve_checkpoint_path(source, None, None, scheme = "nvfp4", **kw)


def _needs_nvfp4_safetensors():
    if not _REAL_LOAD_SUPPORTED("nvfp4", NV_ST):
        pytest.skip("this install cannot read nvfp4 safetensors checkpoints")


def test_nvfp4_pickle_is_never_readable():
    assert _REAL_LOAD_SUPPORTED("nvfp4", NV_PT) is False
    assert _REAL_LOAD_SUPPORTED("nvfp4", "Model-NVFP4.pth") is False
    assert _REAL_LOAD_SUPPORTED("nvfp4", "transformer_nvfp4.pt") is False


def test_nvfp4_cached_pickle_is_ignored(nv_hub):
    _needs_nvfp4_safetensors()
    nv_hub.cache(NV_PT, NV_REPO)
    assert _nv_resolve(_nv_source()) == nv_hub.cached[(NV_REPO, NV_ST)]
    assert nv_hub.fetched == [NV_ST]


def test_nvfp4_cached_pickle_offline_is_not_used(nv_hub):
    _needs_nvfp4_safetensors()
    nv_hub.cache(NV_PT, NV_REPO)
    with pytest.raises(LocalEntryNotFoundError):
        _nv_resolve(_nv_source(), local_files_only = True)
    assert NV_PT not in nv_hub.fetched


def test_nvfp4_cached_pickle_is_not_preferred_even_when_offered(nv_hub):
    """A stale chain naming the .pt (an older caller) never moves the cached pickle ahead; it is dropped."""
    nv_hub.cache(NV_PT, NV_REPO)
    names = [NV_ST, NV_PT]
    assert pq.prefer_cached_pickle_twins(NV_REPO, names, readable = None) == [NV_ST]
    out = pq.prefer_cached_pickle_twins(
        NV_REPO, names, readable = lambda n: _REAL_LOAD_SUPPORTED("nvfp4", n)
    )
    assert out == [NV_ST]
    assert pq.without_nvfp4_pickles("nvfp4", out) == [NV_ST]


def test_nvfp4_stale_pickle_chain_is_filtered_at_download(nv_hub):
    """Even a source that still lists the .pt (built by older code) never downloads it."""
    _needs_nvfp4_safetensors()
    nv_hub.cache(NV_PT, NV_REPO)
    stale = pq.PrequantSource(
        kind = "repo", location = NV_REPO, filename = NV_ST, fallback_filenames = (NV_PT,)
    )
    assert _nv_resolve(stale) == nv_hub.cached[(NV_REPO, NV_ST)]
    nv_hub.hosted = {NV_PT}
    nv_hub.cached.clear()
    nv_hub.downloads.clear()
    with pytest.raises(EntryNotFoundError):
        _nv_resolve(stale)
    assert nv_hub.fetched == [NV_ST]


def test_nvfp4_new_user_downloads_only_the_safetensors(nv_hub):
    _needs_nvfp4_safetensors()
    assert _nv_resolve(_nv_source()) == nv_hub.cached[(NV_REPO, NV_ST)]
    assert nv_hub.fetched == [NV_ST]


def test_nvfp4_pickle_only_repo_is_not_loaded(nv_hub):
    _needs_nvfp4_safetensors()
    nv_hub.hosted = {NV_PT}
    with pytest.raises(EntryNotFoundError):
        _nv_resolve(_nv_source())
    assert nv_hub.fetched[0] == NV_ST and NV_PT not in nv_hub.fetched


def test_nvfp4_torchao_without_nvfp4_unflatten_is_refused(nv_hub, monkeypatch):
    """torchao 0.16 cannot rebuild the NVFP4 safetensors: a clear refusal, never the hosted .pt."""
    utils = pytest.importorskip("torchao.prototype.safetensors.safetensors_utils")
    monkeypatch.setattr(
        utils,
        "ALLOWED_TENSORS_SUBCLASSES",
        [n for n in utils.ALLOWED_TENSORS_SUBCLASSES if n not in ("NVFP4Tensor", "MXTensor")],
    )
    with pytest.raises(ValueError, match = "torchao >= 0.17"):
        _nv_resolve(_nv_source())
    assert nv_hub.fetched == []

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A video family's declared ``<stem>.pt`` is preceded by its ``<stem>.safetensors`` twin, with the image rules:
a new user fetches only the twin, a cached .pt is kept, an unreadable twin is skipped, offline is unchanged."""

from __future__ import annotations

import threading
import types

import huggingface_hub
import pytest
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

import core.inference.diffusion_prequant as pq
from core.inference import video_families as vf

TWINS_ENV = "UNSLOTH_PREQUANT_SAFETENSORS_TWINS"
NO_EXIST = object()


def _fam(name):
    return next(f for f in vf._FAMILIES if f.name == name)


# (family, scheme, base, task) -> (chain before, chain now). Before = origin/main, read with the real tables.
CHAINS = {
    ("minimax-h3", "fp8", None, "ref2va"): (
        ["MiniMax-H3-Ref2VA-FP8.pt"],
        ["MiniMax-H3-Ref2VA-FP8.safetensors", "MiniMax-H3-Ref2VA-FP8.pt"],
    ),
    ("minimax-h3", "int8", None, None): (
        [
            "MiniMax-H3-INT8-ConvRot.pt",
            "MiniMax-H3-INT8.safetensors",
            "MiniMax-H3-INT8.pt",
            "transformer_int8.pt",
        ],
        [
            "MiniMax-H3-INT8-ConvRot.safetensors",
            "MiniMax-H3-INT8-ConvRot.pt",
            "MiniMax-H3-INT8.safetensors",
            "MiniMax-H3-INT8.pt",
            "transformer_int8.pt",
        ],
    ),
    ("minimax-h3", "int8", None, "ref2va"): (
        ["MiniMax-H3-Ref2VA-INT8-ConvRot.pt"],
        ["MiniMax-H3-Ref2VA-INT8-ConvRot.safetensors", "MiniMax-H3-Ref2VA-INT8-ConvRot.pt"],
    ),
    ("wan2.2-t2v-a14b", "fp8", None, "transformer_2"): (
        ["Wan2.2-T2V-A14B-FP8-2.pt"],
        ["Wan2.2-T2V-A14B-FP8-2.safetensors", "Wan2.2-T2V-A14B-FP8-2.pt"],
    ),
    ("wan2.2-t2v-a14b", "int8", None, "transformer_2"): (
        ["Wan2.2-T2V-A14B-INT8-2.pt"],
        ["Wan2.2-T2V-A14B-INT8-2.safetensors", "Wan2.2-T2V-A14B-INT8-2.pt"],
    ),
    ("wan2.2-ti2v-5b", "nvfp4", None, None): (
        ["Wan2.2-TI2V-5B-NVFP4.pt", "Wan2.2-TI2V-5B-NVFP4.safetensors", "transformer_nvfp4.pt"],
        ["Wan2.2-TI2V-5B-NVFP4.safetensors", "Wan2.2-TI2V-5B-NVFP4.pt", "transformer_nvfp4.pt"],
    ),
    ("wan2.2-t2v-a14b", "nvfp4", None, None): (
        ["Wan2.2-T2V-A14B-NVFP4.pt", "Wan2.2-T2V-A14B-NVFP4.safetensors", "transformer_nvfp4.pt"],
        ["Wan2.2-T2V-A14B-NVFP4.safetensors", "Wan2.2-T2V-A14B-NVFP4.pt", "transformer_nvfp4.pt"],
    ),
    ("wan2.2-t2v-a14b", "nvfp4", None, "transformer_2"): (
        ["Wan2.2-T2V-A14B-transformer_2-NVFP4.pt"],
        [
            "Wan2.2-T2V-A14B-transformer_2-NVFP4.safetensors",
            "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt",
        ],
    ),
    ("hunyuanvideo-1.5", "nvfp4", None, None): (
        [
            "HunyuanVideo-1.5-Diffusers-480p_t2v-NVFP4.pt",
            "HunyuanVideo-1.5-NVFP4.safetensors",
            "HunyuanVideo-1.5-NVFP4.pt",
            "transformer_nvfp4.pt",
        ],
        [
            "HunyuanVideo-1.5-Diffusers-480p_t2v-NVFP4.safetensors",
            "HunyuanVideo-1.5-Diffusers-480p_t2v-NVFP4.pt",
            "HunyuanVideo-1.5-NVFP4.safetensors",
            "HunyuanVideo-1.5-NVFP4.pt",
            "transformer_nvfp4.pt",
        ],
    ),
    ("hunyuanvideo-1.5-720p", "nvfp4", None, None): (
        [
            "HunyuanVideo-1.5-Diffusers-720p_t2v-NVFP4.pt",
            "HunyuanVideo-1.5-NVFP4.safetensors",
            "HunyuanVideo-1.5-NVFP4.pt",
            "transformer_nvfp4.pt",
        ],
        [
            "HunyuanVideo-1.5-Diffusers-720p_t2v-NVFP4.safetensors",
            "HunyuanVideo-1.5-Diffusers-720p_t2v-NVFP4.pt",
            "HunyuanVideo-1.5-NVFP4.safetensors",
            "HunyuanVideo-1.5-NVFP4.pt",
            "transformer_nvfp4.pt",
        ],
    ),
    # Already twin-first (derived chain or declared safetensors): unchanged.
    ("minimax-h3", "fp8", None, None): (
        ["MiniMax-H3-FP8.safetensors", "MiniMax-H3-FP8.pt", "transformer_fp8.pt"],
        ["MiniMax-H3-FP8.safetensors", "MiniMax-H3-FP8.pt", "transformer_fp8.pt"],
    ),
    ("wan2.2-ti2v-5b", "fp8", None, None): (
        ["Wan2.2-TI2V-5B-FP8.safetensors", "Wan2.2-TI2V-5B-FP8.pt", "transformer_fp8.pt"],
        ["Wan2.2-TI2V-5B-FP8.safetensors", "Wan2.2-TI2V-5B-FP8.pt", "transformer_fp8.pt"],
    ),
    ("wan2.2-t2v-a14b", "int8", None, None): (
        ["Wan2.2-T2V-A14B-INT8.safetensors", "Wan2.2-T2V-A14B-INT8.pt", "transformer_int8.pt"],
        ["Wan2.2-T2V-A14B-INT8.safetensors", "Wan2.2-T2V-A14B-INT8.pt", "transformer_int8.pt"],
    ),
    ("ltx-2.5", "int8", "lightricks/ltx-2.5-diffusers", None): (
        [
            "LTX-2.5-INT8-ConvRot.safetensors",
            "LTX-2.5-INT8.safetensors",
            "LTX-2.5-INT8.pt",
            "transformer_int8.pt",
        ],
        [
            "LTX-2.5-INT8-ConvRot.safetensors",
            "LTX-2.5-INT8.safetensors",
            "LTX-2.5-INT8.pt",
            "transformer_int8.pt",
        ],
    ),
    ("ltx-2", "fp8", "lightricks/ltx-2.3", None): (
        ["LTX-2.3-FP8.safetensors", "LTX-2.3-FP8.pt", "transformer_fp8.pt"],
        ["LTX-2.3-FP8.safetensors", "LTX-2.3-FP8.pt", "transformer_fp8.pt"],
    ),
}


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    for name in (TWINS_ENV, pq.PREFER_SAFETENSORS_ENV, pq.PREQUANT_MIRROR_ENV):
        monkeypatch.delenv(name, raising = False)
    pq._logged_twin_choices.clear()


def _chain(key):
    fam, scheme, base, task = key
    src = pq.resolve_prequant_source(_fam(fam), scheme, base_repo = base, task = task)
    assert src is not None and src.kind == "repo"
    return src, list(pq.candidate_filenames_of(src))


@pytest.mark.parametrize("key", sorted(CHAINS, key = str), ids = lambda k: "-".join(map(str, k)))
def test_chain_requests_the_twin_right_before_each_declared_pickle(key):
    src, chain = _chain(key)
    assert chain == CHAINS[key][1]
    # The twin is a guess, never a declared name: planners still require proof it exists.
    added = set(chain) - set(CHAINS[key][0])
    assert not added.intersection(src.declared_filenames)


@pytest.mark.parametrize("key", sorted(CHAINS, key = str), ids = lambda k: "-".join(map(str, k)))
def test_kill_switch_restores_the_old_chain(key, monkeypatch):
    monkeypatch.setenv(TWINS_ENV, "0")
    assert _chain(key)[1] == CHAINS[key][0]


def test_every_declared_video_pickle_gets_its_twin_first():
    """Over the whole table, so a new family row cannot skip the rule."""
    seen = 0
    for fam in vf._FAMILIES:
        for entry in getattr(fam, "prequant_filenames", ()) or ():
            scheme, name = entry[0], entry[-1]
            if not name.endswith(".pt"):
                continue
            task = entry[1] if len(entry) == 3 else None
            src = pq.resolve_prequant_source(fam, scheme, task = task)
            if src is None or src.kind != "repo":
                continue
            chain = list(pq.candidate_filenames_of(src))
            twin = name[: -len(".pt")] + ".safetensors"
            assert chain.index(twin) == chain.index(name) - 1, (fam.name, chain)
            seen += 1
    assert seen >= 8


class FakeHub:
    def __init__(self, tmp_path, repo, hosted):
        self.tmp, self.repo, self.hosted = tmp_path, repo, set(hosted)
        self.cached: dict = {}
        self.absent: set = set()
        self.downloads: list = []

    def cache(self, name):
        path = self.tmp / "cache" / self.repo.replace("/", "--") / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
        self.cached[(self.repo, name)] = str(path)

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
        if filename not in self.hosted:
            self.absent.add((repo_id, filename))
            raise EntryNotFoundError(f"404 {filename}")
        if hit is None:
            self.cache(filename)
        return self.cached[(repo_id, filename)]

    @property
    def fetched(self):
        return [n for n, _ in self.downloads]


# One cell per shape: agnostic declared .pt, task partition (H3 ref2va), secondary expert (transformer_2).
CELLS = [
    ("minimax-h3", "int8", None, None, "MiniMax-H3-INT8-ConvRot"),
    ("minimax-h3", "fp8", None, "ref2va", "MiniMax-H3-Ref2VA-FP8"),
    ("wan2.2-t2v-a14b", "fp8", None, "transformer_2", "Wan2.2-T2V-A14B-FP8-2"),
    ("wan2.2-t2v-a14b", "nvfp4", None, "transformer_2", "Wan2.2-T2V-A14B-transformer_2-NVFP4"),
    ("hunyuanvideo-1.5", "nvfp4", None, None, "HunyuanVideo-1.5-Diffusers-480p_t2v-NVFP4"),
]
CELL_IDS = [f"{c[0]}-{c[1]}-{c[3]}" for c in CELLS]


def _setup(
    cell,
    tmp_path,
    monkeypatch,
    *,
    hosted_twin = True,
    readable = None,
):
    fam, scheme, base, task, stem = cell
    src = pq.resolve_prequant_source(_fam(fam), scheme, base_repo = base, task = task)
    hosted = {stem + ".pt"} | ({stem + ".safetensors"} if hosted_twin else set())
    hub = FakeHub(tmp_path, src.location, hosted)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", hub.try_to_load_from_cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(
        pq,
        "restricted_prequant_load_supported",
        readable or (lambda scheme = None, filename = None: True),
    )
    return src, scheme, stem, hub


def _resolve(src, scheme, **kw):
    return pq._resolve_checkpoint_path(src, None, None, scheme = scheme, **kw)


@pytest.mark.parametrize("cell", CELLS, ids = CELL_IDS)
def test_new_user_fetches_only_the_twin(cell, tmp_path, monkeypatch):
    src, scheme, stem, hub = _setup(cell, tmp_path, monkeypatch)
    assert _resolve(src, scheme) == hub.cached[(src.location, stem + ".safetensors")]
    assert hub.fetched == [stem + ".safetensors"]


@pytest.mark.parametrize("cell", CELLS, ids = CELL_IDS)
@pytest.mark.parametrize("offline", [False, True], ids = ["online", "offline"])
def test_cached_pickle_is_kept_and_the_twin_never_downloaded(cell, offline, tmp_path, monkeypatch):
    src, scheme, stem, hub = _setup(cell, tmp_path, monkeypatch)
    hub.cache(stem + ".pt")
    got = _resolve(src, scheme, local_files_only = offline)
    assert got == hub.cached[(src.location, stem + ".pt")]
    assert hub.downloads == [(stem + ".pt", offline)]
    # the cache probe agrees with what the resolver opened
    assert pq.cached_checkpoint_path(src, online = not offline) == got


@pytest.mark.parametrize("cell", CELLS, ids = CELL_IDS)
def test_prefer_safetensors_still_fetches_the_twin_over_a_cached_pickle(
    cell, tmp_path, monkeypatch
):
    src, scheme, stem, hub = _setup(cell, tmp_path, monkeypatch)
    hub.cache(stem + ".pt")
    monkeypatch.setenv(pq.PREFER_SAFETENSORS_ENV, "1")
    assert _resolve(src, scheme) == hub.cached[(src.location, stem + ".safetensors")]
    assert hub.fetched == [stem + ".safetensors"]


@pytest.mark.parametrize("cell", CELLS, ids = CELL_IDS)
def test_unreadable_twin_is_skipped_for_the_pickle(cell, tmp_path, monkeypatch):
    src, scheme, stem, hub = _setup(
        cell,
        tmp_path,
        monkeypatch,
        readable = lambda scheme = None, filename = None: not str(filename).endswith(".safetensors"),
    )
    assert _resolve(src, scheme) == hub.cached[(src.location, stem + ".pt")]
    assert hub.fetched == [stem + ".pt"]


@pytest.mark.parametrize("cell", CELLS, ids = CELL_IDS)
def test_twin_not_hosted_falls_back_to_the_pickle(cell, tmp_path, monkeypatch):
    src, scheme, stem, hub = _setup(cell, tmp_path, monkeypatch, hosted_twin = False)
    assert _resolve(src, scheme) == hub.cached[(src.location, stem + ".pt")]
    assert hub.fetched == [stem + ".safetensors", stem + ".pt"]
    # the recorded 404 lets a later offline plan count the cached .pt
    assert pq.cached_checkpoint_path(src, online = True) == hub.cached[(src.location, stem + ".pt")]


@pytest.mark.parametrize("cell", CELLS, ids = CELL_IDS)
def test_offline_with_nothing_cached_still_fails_without_a_download(cell, tmp_path, monkeypatch):
    src, scheme, stem, hub = _setup(cell, tmp_path, monkeypatch)
    with pytest.raises(LocalEntryNotFoundError):
        _resolve(src, scheme, local_files_only = True)
    assert all(offline for _n, offline in hub.downloads)


@pytest.mark.parametrize("cell", CELLS, ids = CELL_IDS)
def test_offline_cached_twin_is_used(cell, tmp_path, monkeypatch):
    src, scheme, stem, hub = _setup(cell, tmp_path, monkeypatch)
    hub.cache(stem + ".safetensors")
    assert (
        _resolve(src, scheme, local_files_only = True)
        == hub.cached[(src.location, stem + ".safetensors")]
    )
    assert hub.downloads == [(stem + ".safetensors", True)]


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
                types.SimpleNamespace(rfilename = n, size = 1000 + i) for i, n in enumerate(self.names)
            ]
        )


def _video_sources(monkeypatch, fam_name, scheme):
    import core.inference.video_denoiser_prequant as vdp
    monkeypatch.setattr(vdp, "hosted_denoiser_refused", lambda *a, **k: False)
    return vdp.denoiser_prequant_sources(_fam(fam_name), scheme, None)


@pytest.mark.parametrize("cached_pt", [False, True], ids = ["new_user", "cached_pt"])
def test_planner_prices_the_file_each_expert_will_open(cached_pt, tmp_path, monkeypatch):
    """Wan A14B: both experts. Disk / memory planning counts the twin for a new user, the .pt for a cached one."""
    from core.inference.video import VideoBackend

    stems = ["Wan2.2-T2V-A14B-NVFP4", "Wan2.2-T2V-A14B-transformer_2-NVFP4"]
    sources = _video_sources(monkeypatch, "wan2.2-t2v-a14b", "nvfp4")
    assert sources is not None and set(sources) == {"transformer", "transformer_2"}
    repo = sources["transformer"].location
    hub = FakeHub(tmp_path, repo, {s + suf for s in stems for suf in (".pt", ".safetensors")})
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", hub.try_to_load_from_cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(
        pq, "restricted_prequant_load_supported", lambda scheme = None, filename = None: True
    )
    monkeypatch.setattr(
        VideoBackend,
        "_denoiser_prequant_source_list",
        staticmethod(lambda *a, **k: list(sources.values())),
    )
    import core.inference.diffusion_nvfp4_flag as flag

    monkeypatch.setattr(flag, "nvfp4_repo_blocked", lambda repo: False)
    if cached_pt:
        for s in stems:
            hub.cache(s + ".pt")
    suffix = ".pt" if cached_pt else ".safetensors"
    api = _Api(sorted(hub.hosted))
    location, files = VideoBackend._denoiser_prequant_hub_files(None, "nvfp4", None, api)
    assert location == repo
    assert [n for n, _ in files] == [s + suffix for s in stems]

    fetched = []
    import utils.hf_xet_fallback as xet

    monkeypatch.setattr(
        xet, "hf_hub_download_with_xet_fallback", lambda repo, name, tok, **kw: fetched.append(name)
    )
    backend = VideoBackend.__new__(VideoBackend)
    VideoBackend._fetch_denoiser_prequant(
        backend, list(sources.values()), None, cancel_event = threading.Event(), scheme = "nvfp4"
    )
    assert fetched == [s + suffix for s in stems]
    for src, stem in zip(sources.values(), stems):
        assert _resolve(src, "nvfp4") == hub.cached[(repo, stem + suffix)]

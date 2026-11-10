# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Safetensors first; a .pt / .bin already on disk is reused only when its sha256 matches the Hub's LFS record.

The Hub boundary is faked at huggingface_hub itself (``HfApi.get_paths_info``, ``try_to_load_from_cache``,
``hf_hub_download``) and the local files are really hashed, so these run unchanged against any tree."""

from __future__ import annotations

import hashlib
import json
import os
import types

import huggingface_hub
import pytest
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

import core.inference.diffusion_prequant as pq

REPO = "unsloth/Model-INT8"
PT_BYTES = b"pickle checkpoint bytes"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Hub:
    """Cache + listing + LFS records for one repo; records downloads and metadata calls."""

    def __init__(self, root):
        self.root = root
        self.cached: dict = {}
        self.lfs: dict = {}
        self.hosted: set = set()
        self.downloads: list = []
        self.info_calls: list = []
        self.offline = False

    def put(
        self,
        name,
        data,
        *,
        cache = True,
        record = True,
    ):
        self.hosted.add(name)
        if record:
            self.lfs[name] = _sha(data)
        if cache:
            path = self.root / "cache" / name
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_bytes(data)
            self.cached[name] = str(path)

    def try_to_load_from_cache(
        self,
        repo_id,
        filename,
        cache_dir = None,
        **_,
    ):
        return self.cached.get(filename) if repo_id == REPO else None

    def hf_hub_download(
        self,
        repo_id,
        filename,
        local_files_only = False,
        **_,
    ):
        hit = self.cached.get(filename)
        if local_files_only or self.offline:
            self.downloads.append((filename, "local"))
            if hit is None:
                raise LocalEntryNotFoundError(f"{filename} not cached")
            return hit
        if filename not in self.hosted:
            raise EntryNotFoundError(f"404 {filename}")
        if hit is None:
            self.downloads.append((filename, "fetched"))
            path = self.root / "cache" / filename
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_bytes(b"fresh " + filename.encode())
            self.cached[filename] = str(path)
            return str(path)
        self.downloads.append((filename, "revalidated"))
        return hit

    def get_paths_info(self, repo_id, paths, **_):
        self.info_calls.append(tuple(paths))
        if self.offline:
            raise OSError("no network")
        out = []
        for p in paths:
            if p in self.hosted:
                sha = self.lfs.get(p)
                lfs = {"size": 1, "sha256": sha, "pointer_size": 1} if sha else None
                out.append(types.SimpleNamespace(path = p, lfs = lfs))
        return out

    @property
    def fetched(self):
        return [n for n, how in self.downloads if how == "fetched"]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = Hub(tmp_path)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake.try_to_load_from_cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    monkeypatch.setattr(
        huggingface_hub.HfApi, "get_paths_info", lambda self, *a, **k: fake.get_paths_info(*a, **k)
    )
    monkeypatch.setattr(
        pq, "restricted_prequant_load_supported", lambda scheme = None, filename = None: True
    )
    monkeypatch.delenv(pq.PREQUANT_MIRROR_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_PREQUANT_PREFER_SAFETENSORS", raising = False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    getattr(pq, "_logged_twin_choices", set()).clear()
    try:
        import core.inference.cached_weight_sha as cws

        cws._hub.clear()
        cws._local.clear()
        monkeypatch.setattr(cws, "_persisted", None)
    except ImportError:
        pass
    return fake


def _src():
    return pq.PrequantSource(
        kind = "repo",
        location = REPO,
        filename = "Model-INT8.safetensors",
        fallback_filenames = ("Model-INT8.pt",),
    )


def _resolve(**kw):
    return pq._resolve_checkpoint_path(_src(), None, None, scheme = "int8", **kw)


def test_safetensors_is_downloaded_when_nothing_is_cached(hub):
    hub.put("Model-INT8.safetensors", b"st", cache = False)
    hub.put("Model-INT8.pt", PT_BYTES, cache = False)
    assert _resolve().endswith("Model-INT8.safetensors")
    assert hub.fetched == ["Model-INT8.safetensors"]


def test_a_cached_pickle_matching_the_hub_sha_is_reused_without_a_download(hub):
    hub.put("Model-INT8.safetensors", b"st", cache = False)
    hub.put("Model-INT8.pt", PT_BYTES)
    assert _resolve() == hub.cached["Model-INT8.pt"]
    assert hub.fetched == []


def test_a_cached_pickle_with_a_different_sha_is_not_reused(hub):
    hub.put("Model-INT8.safetensors", b"st", cache = False)
    hub.put("Model-INT8.pt", PT_BYTES)
    hub.lfs["Model-INT8.pt"] = _sha(b"the Hub has a newer upload")
    assert _resolve().endswith("Model-INT8.safetensors")
    assert hub.fetched == ["Model-INT8.safetensors"]


def test_a_cached_pickle_without_an_lfs_record_is_not_reused(hub):
    hub.put("Model-INT8.safetensors", b"st", cache = False)
    hub.put("Model-INT8.pt", PT_BYTES, record = False)
    assert _resolve().endswith("Model-INT8.safetensors")
    assert hub.fetched == ["Model-INT8.safetensors"]


def test_offline_the_hash_is_unanswerable_so_safetensors_is_tried_first(hub, monkeypatch):
    hub.put("Model-INT8.safetensors", b"st", cache = False)
    hub.put("Model-INT8.pt", PT_BYTES)
    hub.offline = True
    # Fail open: the safetensors name is asked for first; it cannot be fetched offline, so the cached pickle loads.
    assert _resolve() == hub.cached["Model-INT8.pt"]
    assert [n for n, _ in hub.downloads][0] == "Model-INT8.safetensors"
    assert hub.fetched == []


def test_a_local_files_only_load_never_asks_the_hub_for_the_hash(hub):
    hub.put("Model-INT8.safetensors", b"st", cache = False)
    hub.put("Model-INT8.pt", PT_BYTES)
    token = pq._LOCAL_FILES_ONLY.set(True)
    try:
        assert _resolve(local_files_only = True) == hub.cached["Model-INT8.pt"]
    finally:
        pq._LOCAL_FILES_ONLY.reset(token)
    assert hub.info_calls == []
    assert hub.fetched == []


def test_a_pickle_only_repo_still_loads_its_pickle(hub):
    hub.put("Model-INT8.pt", PT_BYTES, cache = False)
    assert _resolve().endswith("Model-INT8.pt")
    assert hub.fetched == ["Model-INT8.pt"]


def test_both_cached_safetensors_wins_and_nothing_is_hashed(hub):
    hub.put("Model-INT8.safetensors", b"st")
    hub.put("Model-INT8.pt", PT_BYTES)
    assert _resolve() == hub.cached["Model-INT8.safetensors"]
    assert hub.info_calls == []


# --- the hash helper ---------------------------------------------------------------------------------------------


def test_the_local_hash_is_computed_once_and_remembered(hub, tmp_path, monkeypatch):
    import core.inference.cached_weight_sha as cws

    path = tmp_path / "w.bin"
    path.write_bytes(PT_BYTES)
    opened = []
    real_open = open

    def counting_open(p, *a, **k):
        if str(p) == os.path.realpath(path):
            opened.append(p)
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", counting_open)
    assert cws.file_sha256(str(path)) == _sha(PT_BYTES)
    assert cws.file_sha256(str(path)) == _sha(PT_BYTES)
    assert len(opened) == 1
    # A fresh process reads the persisted record instead of rehashing.
    cws._local.clear()
    monkeypatch.setattr(cws, "_persisted", None)
    assert cws.file_sha256(str(path)) == _sha(PT_BYTES)
    assert len(opened) == 1
    record = json.loads(cws._store_path().read_text())
    assert record[os.path.realpath(path)][2] == _sha(PT_BYTES)
    # A changed file is rehashed.
    path.write_bytes(b"changed!" * 10)
    assert cws.file_sha256(str(path)) == _sha(b"changed!" * 10)


def test_a_blob_is_judged_by_its_bytes_not_its_name(hub, tmp_path):
    import core.inference.cached_weight_sha as cws

    # huggingface_hub 1.x names cache blobs by a Xet hash, not the sha256.
    blob = tmp_path / "blobs" / "c5" / ("c" * 64)
    blob.parent.mkdir(parents = True)
    blob.write_bytes(PT_BYTES)
    assert cws.cached_copy_matches_hub(REPO, "x.pt", str(blob), expected = _sha(PT_BYTES))
    assert not cws.cached_copy_matches_hub(REPO, "x.pt", str(blob), expected = _sha(b"other"))


def test_the_reuse_switch_turns_verification_off(hub, monkeypatch):
    import core.inference.cached_weight_sha as cws

    hub.put("Model-INT8.safetensors", b"st", cache = False)
    hub.put("Model-INT8.pt", PT_BYTES)
    monkeypatch.setenv(cws.ENV_DISABLE, "0")
    assert _resolve().endswith("Model-INT8.safetensors")
    assert hub.fetched == ["Model-INT8.safetensors"]


def test_hub_lookups_are_cached_per_process(hub):
    import core.inference.cached_weight_sha as cws

    hub.put("Model-INT8.pt", PT_BYTES)
    assert cws.hub_sha256s(REPO, ["Model-INT8.pt"]) == {"Model-INT8.pt": _sha(PT_BYTES)}
    assert cws.hub_sha256s(REPO, ["Model-INT8.pt"]) == {"Model-INT8.pt": _sha(PT_BYTES)}
    assert len(hub.info_calls) == 1


# --- diffusers pipelines: a cached, verified .bin is loaded instead of downloading safetensors --------------------


def _snapshot(tmp_path, components):
    root = tmp_path / "hub" / "models--o--pipe" / "snapshots" / ("c" * 40)
    root.mkdir(parents = True)
    index = {
        "_class_name": "FluxPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    for name, files in components.items():
        index[name] = ["diffusers", "Y"]
        (root / name).mkdir()
        (root / name / "config.json").write_text("{}")
        for fname, data in files.items():
            (root / name / fname).write_bytes(data)
    (root / "scheduler").mkdir()
    (root / "model_index.json").write_text(json.dumps(index))
    return root


def _hub_records(
    monkeypatch,
    records,
    calls = None,
):
    def info(self, repo_id, paths, **kw):
        if calls is not None:
            calls.append((repo_id, kw.get("revision"), tuple(paths)))
        return [
            types.SimpleNamespace(path = p, lfs = {"sha256": records[p]} if records.get(p) else None)
            for p in paths
        ]

    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", info)


def test_a_snapshot_of_verified_bins_loads_them_instead_of_downloading_safetensors(
    hub, tmp_path, monkeypatch
):
    from core.inference import diffusers_safetensors as ds

    root = _snapshot(
        tmp_path,
        {
            "transformer": {"diffusion_pytorch_model.bin": b"t"},
            "text_encoder": {"pytorch_model.bin": b"e"},
        },
    )
    calls = []
    _hub_records(
        monkeypatch,
        {
            "transformer/diffusion_pytorch_model.bin": _sha(b"t"),
            "text_encoder/pytorch_model.bin": _sha(b"e"),
        },
        calls,
    )
    assert ds.pipeline_safetensors_kwargs(str(root)) == {"use_safetensors": False}
    # A snapshot folder is read as is: checked against its own commit.
    assert calls and calls[0][0] == "o/pipe" and calls[0][1] == "c" * 40
    assert ds.component_safetensors_kwargs(str(root), "transformer") == {"use_safetensors": False}


def test_one_mismatched_bin_keeps_the_library_default(hub, tmp_path, monkeypatch):
    from core.inference import diffusers_safetensors as ds

    root = _snapshot(
        tmp_path,
        {
            "transformer": {"diffusion_pytorch_model.bin": b"t"},
            "text_encoder": {"pytorch_model.bin": b"e"},
        },
    )
    _hub_records(
        monkeypatch,
        {
            "transformer/diffusion_pytorch_model.bin": _sha(b"t"),
            "text_encoder/pytorch_model.bin": _sha(b"other"),
        },
    )
    assert ds.pipeline_safetensors_kwargs(str(root)) == {}
    assert ds.component_safetensors_kwargs(str(root), "text_encoder") == {}


def test_a_partly_safetensors_snapshot_never_forces_bins(hub, tmp_path, monkeypatch):
    from core.inference import diffusers_safetensors as ds

    root = _snapshot(
        tmp_path,
        {
            "transformer": {"diffusion_pytorch_model.bin": b"t"},
            "vae": {"diffusion_pytorch_model.safetensors": b"v"},
        },
    )
    _hub_records(monkeypatch, {"transformer/diffusion_pytorch_model.bin": _sha(b"t")})
    # The vae has no .bin to reuse, so use_safetensors=False would fetch one: keep the default.
    assert ds.pipeline_safetensors_kwargs(str(root)) == {}


def test_offline_a_snapshot_cached_only_as_bins_loads_them(hub, tmp_path, monkeypatch):
    from core.inference import diffusers_safetensors as ds
    import huggingface_hub.constants as constants

    root = _snapshot(tmp_path, {"transformer": {"diffusion_pytorch_model.bin": b"t"}})
    calls = []
    _hub_records(monkeypatch, {}, calls)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    # Nothing can be fetched or hashed: the cached .bin is the only way the load succeeds.
    assert ds.pipeline_safetensors_kwargs(str(root)) == {"use_safetensors": False}
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    assert ds.pipeline_safetensors_kwargs(str(root), passed = {"local_files_only": True}) == {
        "use_safetensors": False
    }
    assert calls == []
    # A variant load (fp16 twins) is never pointed at the default-name .bin files.
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    assert ds.pipeline_safetensors_kwargs(str(root), passed = {"variant": "fp16"}) == {}
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    # Online with no matching record the library default stays.
    assert ds.pipeline_safetensors_kwargs(str(root)) == {}


def test_a_plain_local_folder_of_bins_is_never_hub_verified(hub, tmp_path, monkeypatch):
    from core.inference import diffusers_safetensors as ds

    root = tmp_path / "local_pipe"
    (root / "transformer").mkdir(parents = True)
    (root / "transformer" / "diffusion_pytorch_model.bin").write_bytes(b"t")
    (root / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "Y"]}))
    _hub_records(monkeypatch, {})
    assert ds.pipeline_safetensors_kwargs(str(root)) == {}


def test_the_dense_encoder_fetch_skips_safetensors_for_a_verified_cached_bin(
    hub, tmp_path, monkeypatch
):
    from core.inference import diffusion_te_prequant as tp

    root = _snapshot(tmp_path, {"text_encoder": {"pytorch_model.bin": b"e"}})
    listing = [
        "model_index.json",
        "text_encoder/model.safetensors",
        "text_encoder/pytorch_model.bin",
    ]
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda self, *a, **k: listing)
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, filename, **k: str(root / filename)
        if filename == "model_index.json"
        else None,
    )
    _hub_records(monkeypatch, {"text_encoder/pytorch_model.bin": _sha(b"e")})
    patterns = tp._dense_component_pickles("o/pipe", ["text_encoder"], None, None)
    assert "text_encoder/model.safetensors" in patterns
    assert "text_encoder/pytorch_model.bin" not in patterns
    _hub_records(monkeypatch, {"text_encoder/pytorch_model.bin": _sha(b"different")})
    import core.inference.cached_weight_sha as cws

    cws._hub.clear()
    patterns = tp._dense_component_pickles("o/pipe", ["text_encoder"], None, None)
    assert "text_encoder/pytorch_model.bin" in patterns
    assert "text_encoder/model.safetensors" not in patterns


def test_an_unpinned_hub_id_is_checked_against_the_hubs_current_main(hub, tmp_path, monkeypatch):
    from core.inference import diffusers_safetensors as ds

    root = _snapshot(tmp_path, {"transformer": {"diffusion_pytorch_model.bin": b"t"}})
    (root.parents[1] / "refs").mkdir()
    (root.parents[1] / "refs" / "main").write_text("c" * 40)
    import huggingface_hub.file_download as fd
    import core.inference.cached_weight_sha as cws

    cache = str(tmp_path / "hub")
    calls = []
    # The Hub's current main no longer lists the .bin (it moved to safetensors only): no reuse.
    _hub_records(monkeypatch, {}, calls)
    real = fd.try_to_load_from_cache
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", real)
    assert ds.pipeline_safetensors_kwargs("o/pipe", cache_dir = cache) == {}
    assert calls and calls[-1][1] is None  # current main, not the cached commit
    cws._hub.clear()  # the unanswered lookup above is otherwise remembered for a minute
    _hub_records(monkeypatch, {"transformer/diffusion_pytorch_model.bin": _sha(b"t")}, calls)
    assert ds.pipeline_safetensors_kwargs("o/pipe", cache_dir = cache) == {"use_safetensors": False}
    assert calls[-1][1] is None


def test_a_main_answer_is_refreshed_but_a_commit_answer_is_kept(hub, monkeypatch):
    import core.inference.cached_weight_sha as cws

    hub.put("Model-INT8.pt", PT_BYTES)
    clock = [1000.0]
    monkeypatch.setattr(cws.time, "monotonic", lambda: clock[0])
    cws.hub_sha256s(REPO, ["Model-INT8.pt"])
    cws.hub_sha256s(REPO, ["Model-INT8.pt"], revision = "c" * 40)
    assert len(hub.info_calls) == 2
    clock[0] += cws._MUTABLE_TTL_S + 1
    cws.hub_sha256s(REPO, ["Model-INT8.pt"], revision = "c" * 40)
    assert len(hub.info_calls) == 2
    cws.hub_sha256s(REPO, ["Model-INT8.pt"])
    assert len(hub.info_calls) == 3

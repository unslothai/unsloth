# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Restyle setup from the UI: engine install, weight fetch, shard merge, VAE rename, refusals."""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

from core.inference import video_restyle as vr
from core.inference import video_restyle_setup as rs

_GIB = 1024**3


def _write_st(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, blob, offset = {"__metadata__": {"format": "pt"}}, b"", 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blob += data
        offset += len(data)
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)


def _read_st(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    header, start = rs.read_safetensors_header(path)
    data = path.read_bytes()
    return {
        k: (
            v["dtype"],
            v["shape"],
            data[start + v["data_offsets"][0] : start + v["data_offsets"][1]],
        )
        for k, v in header.items()
    }


def _bf16(values) -> bytes:
    return (np.asarray(values, dtype = np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.delenv(vr.MODEL_ENV, raising = False)
    monkeypatch.delenv(vr.VAE_ENV, raising = False)
    monkeypatch.setenv(rs.DIR_ENV, str(tmp_path / "restyle"))
    # _FakeHub serves its files from tmp_path/hub, standing in for Studio's Hub cache entry of the repo.
    monkeypatch.setattr(rs, "_hub_repo_dir", lambda repo = rs.MODEL_ID: tmp_path / "hub")
    # The fake VAE has 2 tensors, not the release's 196.
    monkeypatch.setattr(rs, "VAE_TENSORS", 2)
    return tmp_path / "restyle"


# ---- merge and rename -----------------------------------------------------------------------------------------


def test_merge_matches_the_fork_converter(tmp_path):
    a = tmp_path / "s-00001.safetensors"
    b = tmp_path / "s-00002.safetensors"
    _write_st(
        a,
        {
            "z.weight": ("BF16", [2], b"\x01\x02\x03\x04"),
            "lm_head.weight": ("BF16", [1], b"\xff\xff"),
        },
    )
    _write_st(
        b,
        {"a.weight": ("F32", [1], b"\x00\x00\x80\x3f"), "norm.weight": ("BF16", [1], b"\xee\xee")},
    )
    out = tmp_path / "merged.safetensors"
    seen = []
    payload = rs.merge_shards([a, b], out, on_bytes = seen.append)
    got = _read_st(out)
    # Sorted, the two unused tensors dropped, bytes unchanged.
    assert list(got) == ["a.weight", "z.weight"]
    assert got["z.weight"] == ("BF16", [2], b"\x01\x02\x03\x04")
    assert got["a.weight"] == ("F32", [1], b"\x00\x00\x80\x3f")
    assert payload == 8 and seen[-1] == 8
    assert not out.with_name(out.name + ".partial").exists()


def test_merge_refuses_a_truncated_shard_and_leaves_nothing(tmp_path):
    a = tmp_path / "s.safetensors"
    _write_st(a, {"w": ("BF16", [4], b"\x00" * 8)})
    a.write_bytes(a.read_bytes()[:-4])
    out = tmp_path / "merged.safetensors"
    with pytest.raises(rs.SetupError, match = "truncated"):
        rs.merge_shards([a], out)
    assert not out.exists() and not out.with_name(out.name + ".partial").exists()


def test_merge_is_cancellable(tmp_path):
    a = tmp_path / "s.safetensors"
    _write_st(a, {"w": ("BF16", [4], b"\x00" * 8)})
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(rs.SetupCancelled):
        rs.merge_shards([a], tmp_path / "m.safetensors", cancel = cancel)
    assert not (tmp_path / "m.safetensors").exists()


def _write_vae_pth(path: Path, tensors: dict) -> None:
    import torch
    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save({k: torch.tensor(v, dtype = torch.float32) for k, v in tensors.items()}, path)


def test_convert_vae_keeps_the_original_names_and_stores_fp16(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "VAE_TENSORS", 2)
    src = tmp_path / "Wan2.2_VAE.pth"
    values = [0.07421875, -0.0168457, 1.5, 0.0]
    _write_vae_pth(src, {"conv1.weight": values, "decoder.head.2.bias": [-3.0]})
    out = tmp_path / "wan.safetensors"
    rs.convert_vae(src, out)
    got = _read_st(out)
    assert set(got) == {"conv1.weight", "decoder.head.2.bias"}
    dtype, shape, raw = got["conv1.weight"]
    assert dtype == "F16" and shape == [4]
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype = np.float16), np.array(values, dtype = np.float32).astype(np.float16)
    )
    assert not out.with_name(out.name + ".partial").exists()


def test_convert_vae_refuses_an_out_of_range_weight_or_another_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "VAE_TENSORS", 2)
    src = tmp_path / "Wan2.2_VAE.pth"
    _write_vae_pth(src, {"conv1.weight": [1e6], "x": [0.0]})
    with pytest.raises(rs.SetupError, match = "fp16"):
        rs.convert_vae(src, tmp_path / "out.safetensors")
    _write_vae_pth(src, {"conv1.weight": [1.0]})
    with pytest.raises(rs.SetupError, match = "not the Wan2.2 VAE"):
        rs.convert_vae(src, tmp_path / "out.safetensors")
    assert not (tmp_path / "out.safetensors").exists()


# ---- hardware refusals ----------------------------------------------------------------------------------------


def _card(gib: float):
    return ("0000:01:00.0", False, int(gib * _GIB), "card")


@pytest.mark.parametrize(
    "platform, devices, weights, refused",
    [
        ("linux", [], None, "NVIDIA GPU"),
        ("linux", [_card(22.0)], None, "at least 31 GB"),
        ("linux", [_card(39.4)], "q8_0", None),
        ("linux", [_card(44.5)], "bf16", None),
        ("win32", [_card(95.0)], "bf16", None),
        ("linux", None, "bf16", None),  # unknown: never refused
        ("linux", [None], "bf16", None),
    ],
)
def test_hardware_plan(platform, devices, weights, refused):
    plan = rs.hardware_plan(platform = platform, devices = devices)
    assert plan["weights"] == weights
    if refused is None:
        assert plan["refusal"] is None
    else:
        assert refused in plan["refusal"]


@pytest.mark.parametrize(
    "working_set_gib, weights, refused",
    [
        (None, None, "Apple Silicon"),
        (27.0, None, "this Mac's GPU can use 27.0 GB"),
        (36.0, "q8_0", None),
        (96.0, "bf16", None),
    ],
)
def test_hardware_plan_on_a_mac_sizes_from_the_metal_working_set(working_set_gib, weights, refused):
    working_set = None if working_set_gib is None else int(working_set_gib * _GIB)
    # A CUDA answer is ignored on a Mac.
    plan = rs.hardware_plan(platform = "darwin", devices = [], working_set = working_set)
    assert plan["weights"] == weights
    if refused is None:
        assert plan["refusal"] is None
    else:
        assert refused in plan["refusal"]


# ---- availability ---------------------------------------------------------------------------------------------


def _cosmos3_server(tmp_path) -> str:
    binary = tmp_path / "bin" / "sd-server"
    binary.parent.mkdir(parents = True, exist_ok = True)
    binary.write_bytes(b"\x7fELF" + vr._COSMOS3_MARKER)
    return str(binary)


def test_fresh_install_offers_setup(clean_env, monkeypatch):
    monkeypatch.setattr("core.inference.sd_cpp_engine.find_sd_server_binary", lambda: None)
    monkeypatch.setattr(
        rs, "hardware_plan", lambda: {"weights": "bf16", "vram_gib": 95.0, "refusal": None}
    )
    out = vr.availability()
    assert out["available"] is False
    assert out["missing"] == ["weights", "engine"]
    assert out["can_setup"] is True and out["refusal"] is None


def test_small_card_is_refused_up_front(clean_env, monkeypatch):
    monkeypatch.setattr("core.inference.sd_cpp_engine.find_sd_server_binary", lambda: None)
    monkeypatch.setattr(
        rs,
        "hardware_plan",
        lambda real = rs.hardware_plan: real(platform = "linux", devices = [_card(22.0)]),
    )
    out = vr.availability()
    assert out["can_setup"] is False and "31 GB" in out["refusal"]


def test_missing_override_weights_are_not_set_up(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv(vr.MODEL_ENV, str(tmp_path / "nope.safetensors"))
    monkeypatch.setattr(
        "core.inference.sd_cpp_engine.find_sd_server_binary", lambda: _cosmos3_server(tmp_path)
    )
    monkeypatch.setattr(
        rs, "hardware_plan", lambda: {"weights": "bf16", "vram_gib": 95.0, "refusal": None}
    )
    out = vr.availability()
    assert out["missing"] == ["weights"] and out["can_setup"] is False
    assert vr.VAE_ENV in out["reason"]


def test_model_files_prefers_env_then_the_setup_record(tmp_path, clean_env, monkeypatch):
    clean_env.mkdir(parents = True)
    (clean_env / rs.MODEL_BF16).write_bytes(b"m" * 3)
    (clean_env / rs.VAE_OUT).write_bytes(b"v" * 2)
    record = {
        "version": rs.RECORD_VERSION,
        "model": {"file": rs.MODEL_BF16, "bytes": 3},
        "vae": {"file": rs.VAE_OUT, "bytes": 2},
    }
    (clean_env / rs.RECORD_NAME).write_text(json.dumps(record))
    files = vr.model_files()
    assert files.diffusion_model == str(clean_env / rs.MODEL_BF16)
    assert files.vae == str(clean_env / rs.VAE_OUT)
    # A file that changed size since setup is not trusted.
    (clean_env / rs.VAE_OUT).write_bytes(b"v")
    with pytest.raises(vr.RestyleError, match = "Set up Restyle"):
        vr.model_files()
    own = tmp_path / "own.gguf"
    own.write_bytes(b"g")
    monkeypatch.setenv(vr.MODEL_ENV, str(own))
    monkeypatch.setenv(vr.VAE_ENV, str(own))
    assert vr.model_files().diffusion_model == str(own)


# ---- the setup job --------------------------------------------------------------------------------------------


class _FakeHub:
    """A repo with two transformer shards, an index and a VAE, served from tmp files."""

    def __init__(self, root: Path) -> None:
        snap = root / "hub" / "snapshots" / "rev"
        self.paths = {
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": snap
            / "transformer"
            / "s1.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": snap
            / "transformer"
            / "s2.safetensors",
            rs.VAE_FILE: root / "hub" / "snapshots" / "vae-rev" / rs.VAE_FILE,
        }
        _write_st(
            self.paths["transformer/diffusion_pytorch_model-00001-of-00002.safetensors"],
            {
                "blocks.1.w": ("BF16", [2], b"\x01\x00\x02\x00"),
                "lm_head.weight": ("BF16", [1], b"\x09\x09"),
            },
        )
        _write_st(
            self.paths["transformer/diffusion_pytorch_model-00002-of-00002.safetensors"],
            {"blocks.0.w": ("BF16", [1], b"\x03\x00")},
        )
        _write_vae_pth(
            self.paths[rs.VAE_FILE], {"conv1.weight": [0.5], "decoder.head.2.bias": [0.25]}
        )
        index = snap / rs.TRANSFORMER_INDEX
        index.write_text(
            json.dumps(
                {
                    "weight_map": {
                        "blocks.1.w": "diffusion_pytorch_model-00001-of-00002.safetensors",
                        "lm_head.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
                        "blocks.0.w": "diffusion_pytorch_model-00002-of-00002.safetensors",
                    }
                }
            )
        )
        self.paths[rs.TRANSFORMER_INDEX] = index
        self.calls: list[tuple[str, object, object]] = []

    def files(self, repo, token):
        self.calls.append(("info", token, None))
        names = (
            [rs.VAE_FILE]
            if repo == rs.VAE_REPO
            else [n for n in self.paths if n not in (rs.TRANSFORMER_INDEX, rs.VAE_FILE)]
        )
        rev = "vae-rev" if repo == rs.VAE_REPO else "rev"
        return rev, [
            {"name": n, "size": self.paths[n].stat().st_size, "sha256": None} for n in names
        ]

    def download(self, repo, filename, token, *, revision, cancel_event):
        assert (repo == rs.VAE_REPO) == (filename == rs.VAE_FILE) and repo in (
            rs.MODEL_ID,
            rs.VAE_REPO,
        )
        self.calls.append((filename, token, revision))
        return str(self.paths[filename])


def _runner(
    hub,
    engine_calls = None,
    **kw,
):
    return rs.RestyleSetup(
        download = hub.download,
        repo_files = hub.files,
        ensure_engine = lambda cancel: (engine_calls.append(1) if engine_calls is not None else None)
        or "/x/sd-server",
        plan = kw.pop("plan", lambda: {"weights": "bf16", "vram_gib": 95.0, "refusal": None}),
        **kw,
    )


def test_setup_installs_fetches_merges_and_records(tmp_path, clean_env, monkeypatch):
    hub = _FakeHub(tmp_path)
    engine = []
    runner = _runner(hub, engine)
    state = runner.start(hf_token = "hf_x", background = False)
    assert state["phase"] == "done" and state["active"] is False, state
    assert engine == [1]
    assert all(token == "hf_x" for _, token, _ in hub.calls)
    assert {(name == rs.VAE_FILE, rev) for name, _, rev in hub.calls if name != "info"} == {
        (False, "rev"),
        (True, "vae-rev"),
    }
    model, vae = rs.installed_files()
    merged = _read_st(Path(model))
    assert list(merged) == ["blocks.0.w", "blocks.1.w"]
    assert sorted(_read_st(Path(vae))) == ["conv1.weight", "decoder.head.2.bias"]
    assert vr.model_files().diffusion_model == model
    # The verified outputs replaced what setup downloaded (shards and the fp32 VAE); the index stays.
    assert not hub.paths["transformer/diffusion_pytorch_model-00001-of-00002.safetensors"].exists()
    assert not hub.paths[rs.VAE_FILE].exists()
    assert hub.paths[rs.TRANSFORMER_INDEX].exists()
    # A second run with the record in place downloads nothing.
    hub.calls.clear()
    assert runner.start(need_engine = False, background = False)["phase"] == "done"
    assert hub.calls == []


def test_setup_quantizes_for_a_40gb_card(tmp_path, clean_env):
    hub = _FakeHub(tmp_path)
    made = []

    def quantize(src, out, cancel):
        made.append(src.name)
        out.write_bytes(b"GGUF")

    runner = _runner(
        hub, plan = lambda: {"weights": "q8_0", "vram_gib": 39.4, "refusal": None}, quantize = quantize
    )
    assert runner.start(background = False)["phase"] == "done"
    model, _ = rs.installed_files()
    assert model.endswith(rs.MODEL_Q8) and made == [rs.MODEL_BF16]
    assert not (clean_env / rs.MODEL_BF16).exists()


def test_setup_refuses_when_the_card_cannot_run_it(tmp_path, clean_env):
    runner = _runner(
        _FakeHub(tmp_path), plan = lambda: {"weights": None, "vram_gib": 22.0, "refusal": "too small"}
    )
    with pytest.raises(rs.SetupError, match = "too small"):
        runner.start(background = False)
    assert runner.progress()["active"] is False


def test_setup_refuses_without_disk_space(tmp_path, clean_env, monkeypatch):
    monkeypatch.setattr(rs, "_free_bytes", lambda path: (1 * _GIB, 1))
    runner = _runner(_FakeHub(tmp_path))
    state = runner.start(background = False)
    assert state["phase"] == "error" and "GB free" in state["error"]
    assert rs.installed_files() is None


def test_setup_engine_failure_is_reported(tmp_path, clean_env):
    def broken(cancel):
        raise rs.SetupError("no server")

    runner = rs.RestyleSetup(
        download = _FakeHub(tmp_path).download,
        repo_files = _FakeHub(tmp_path).files,
        ensure_engine = broken,
        plan = lambda: {"weights": "bf16", "vram_gib": 95.0, "refusal": None},
    )
    state = runner.start(background = False)
    assert state["phase"] == "error" and state["error"] == "no server"


def test_setup_cancel_stops_before_the_merge(tmp_path, clean_env):
    hub = _FakeHub(tmp_path)
    runner = _runner(hub)
    real = hub.download

    def download(*a, **k):
        runner.cancel()
        return real(*a, **k)

    runner._download = download
    state = runner.start(background = False)
    assert state["phase"] == "cancelled"
    assert rs.installed_files() is None
    assert hub.paths["transformer/diffusion_pytorch_model-00001-of-00002.safetensors"].exists()


def test_ensure_engine_uses_studios_installer(monkeypatch, tmp_path):
    calls = []
    binary = _cosmos3_server(tmp_path)

    def ensure(*, allow_install, accelerator):
        calls.append((allow_install, accelerator))
        return binary

    monkeypatch.setattr("core.inference.sd_cpp_backend.ensure_sd_server_binary", ensure)
    monkeypatch.setattr(rs, "install_accelerator", lambda: "cuda")
    assert rs._ensure_engine(threading.Event()) == binary
    assert calls == [(True, "cuda")]
    plain = tmp_path / "plain" / "sd-server"
    plain.parent.mkdir()
    plain.write_bytes(b"\x7fELF")
    monkeypatch.setattr(
        "core.inference.sd_cpp_backend.ensure_sd_server_binary", lambda **k: str(plain)
    )
    with pytest.raises(rs.SetupError, match = "no Cosmos3 support"):
        rs._ensure_engine(threading.Event())
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    with pytest.raises(rs.SetupError, match = "turned off"):
        rs._ensure_engine(threading.Event())


def test_setup_route_starts_and_status_reports_progress(tmp_path, clean_env, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import video_restyle as route

    hub = _FakeHub(tmp_path)
    runner = _runner(hub)
    monkeypatch.setattr(rs, "setup_runner", runner)
    monkeypatch.setattr(
        rs, "hardware_plan", lambda: {"weights": "bf16", "vram_gib": 95.0, "refusal": None}
    )
    found = {"binary": None}
    monkeypatch.setattr(
        "core.inference.sd_cpp_engine.find_sd_server_binary", lambda: found["binary"]
    )

    real_start = runner.start

    def start(**kw):
        found["binary"] = _cosmos3_server(tmp_path)
        return real_start(background = False, **kw)

    monkeypatch.setattr(runner, "start", start)
    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    client = TestClient(app)
    status = client.get("/api/inference/video/restyle/status").json()
    assert status["can_setup"] is True and status["missing"] == ["weights", "engine"]
    res = client.post("/api/inference/video/restyle/setup", json = {"hf_token": "hf_y"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["available"] is True and body["setup"]["phase"] == "done"
    assert ("info", "hf_y", None) in hub.calls
    assert client.post("/api/inference/video/restyle/setup/cancel").json() == {"cancelled": False}


def test_setup_route_refuses_a_small_card(clean_env, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import video_restyle as route

    monkeypatch.setattr("core.inference.sd_cpp_engine.find_sd_server_binary", lambda: None)
    monkeypatch.setattr(
        rs,
        "hardware_plan",
        lambda real = rs.hardware_plan: real(platform = "linux", devices = [_card(22.0)]),
    )
    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    res = TestClient(app).post("/api/inference/video/restyle/setup", json = {})
    assert res.status_code == 400 and "31 GB" in res.json()["detail"]


def test_setup_keeps_shards_that_were_cached_before_it_ran(tmp_path, clean_env, monkeypatch):
    hub = _FakeHub(tmp_path)
    first = "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
    second = "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
    # The Hub's record of an earlier download of `first`, in the cache holding the fake repo dir.
    earlier = tmp_path / ("models--" + rs.MODEL_ID.replace("/", "--")) / "snapshots" / "rev" / first
    earlier.parent.mkdir(parents = True)
    earlier.write_bytes(b"x")
    assert _runner(hub).start(background = False)["phase"] == "done"
    # The user's own earlier download stays; only what setup fetched is dropped.
    assert hub.paths[first].exists() and not hub.paths[second].exists()


def test_was_cached_looks_in_the_given_cache(tmp_path):
    # Not the Hub's import-time cache, which Studio stops using after a cache-home change.
    shard = tmp_path / "hub" / "models--o--r" / "snapshots" / "abc" / "s.safetensors"
    shard.parent.mkdir(parents = True)
    shard.write_bytes(b"x")
    repo = tmp_path / "hub" / "models--o--r"
    assert rs._was_cached("o/r", "s.safetensors", "abc", repo)
    assert not rs._was_cached("o/r", "other.safetensors", "abc", repo)


def test_setup_cleans_only_the_cache_it_checked(tmp_path, clean_env, monkeypatch):
    hub = _FakeHub(tmp_path)
    active = [tmp_path / "before"]
    monkeypatch.setattr(rs, "_hub_repo_dir", lambda repo = rs.MODEL_ID: active[0])
    fetch = hub.download

    def download(repo, name, *args, **kwargs):
        # The cache home changes mid-setup to where the files already were.
        if name != rs.TRANSFORMER_INDEX:
            active[0] = tmp_path / "hub"
        return fetch(repo, name, *args, **kwargs)

    hub.download = download
    assert _runner(hub).start(background = False)["phase"] == "done"
    assert all(p.exists() for p in hub.paths.values())


def test_remove_cached_drops_only_this_repos_unshared_blobs(tmp_path):
    repo = tmp_path / "models--nvidia--Cosmos3-Nano-Sim-Transfer"
    blob = repo / "blobs" / "abc"
    blob.parent.mkdir(parents = True)
    blob.write_bytes(b"x")
    links = []
    for rev in ("r1", "r2"):
        link = repo / "snapshots" / rev / "transformer" / "s.safetensors"
        link.parent.mkdir(parents = True)
        link.symlink_to(blob)
        links.append(link)
    rs._remove_cached(str(links[0]), repo)
    assert (
        not links[0].exists() and not links[0].is_symlink() and blob.exists()
    )  # r2 still links it
    rs._remove_cached(str(links[1]), repo)
    assert not blob.exists()
    # A file served from another cache root (the user's own) is never touched.
    elsewhere = tmp_path / "user_cache" / "s.safetensors"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"y")
    rs._remove_cached(str(elsewhere), repo)
    assert elsewhere.exists()


def test_remove_cached_keeps_a_blob_another_revision_links_under_another_name(tmp_path):
    repo = tmp_path / "models--nvidia--Cosmos3-Nano-Sim-Transfer"
    blob = repo / "blobs" / "abc"
    blob.parent.mkdir(parents = True)
    blob.write_bytes(b"x")
    # The same content-addressed blob, renamed between revisions.
    old = repo / "snapshots" / "r1" / "transformer" / "old_name.safetensors"
    new = repo / "snapshots" / "r2" / "transformer" / "s.safetensors"
    for link in (old, new):
        link.parent.mkdir(parents = True)
        link.symlink_to(blob)
    rs._remove_cached(str(new), repo)
    assert not new.is_symlink() and blob.exists() and old.read_bytes() == b"x"
    rs._remove_cached(str(old), repo)
    assert not blob.exists()


def test_a_failure_after_the_merge_leaves_no_orphaned_outputs(tmp_path, clean_env):
    hub = _FakeHub(tmp_path)

    def quantize(src, out, cancel):
        out.with_name(out.name + ".partial").write_bytes(b"GG")
        raise rs.SetupError("convert failed")

    runner = _runner(
        hub, plan = lambda: {"weights": "q8_0", "vram_gib": 39.4, "refusal": None}, quantize = quantize
    )
    state = runner.start(background = False)
    assert state["phase"] == "error" and "convert failed" in state["error"]
    assert rs.installed_files() is None
    assert [p for p in rs._outputs(clean_env) if p.exists()] == []
    # Nothing verified, so the shards stay cached for a retry.
    assert hub.paths["transformer/diffusion_pytorch_model-00001-of-00002.safetensors"].exists()


def test_leftover_outputs_are_removed_before_the_disk_check(tmp_path, clean_env, monkeypatch):
    """A failed run's outputs would sit beside their .partial replacements, so they are removed, not counted free."""
    clean_env.mkdir(parents = True)
    leftovers = [clean_env / rs.MODEL_BF16, clean_env / (rs.VAE_OUT + ".partial")]
    for path in leftovers:
        path.write_bytes(b"0" * 4096)
    seen = []

    def check(self, blobs, left, root, out):
        seen.append((out, [p.exists() for p in leftovers]))

    monkeypatch.setattr(rs.RestyleSetup, "_check_disk", check)
    assert _runner(_FakeHub(tmp_path)).start(background = False)["phase"] == "done"
    out, present = seen[0]
    assert present == [False, False]
    assert out > 0


def test_setup_routes_are_owner_only(tmp_path, clean_env, monkeypatch):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    from auth import policy
    from auth.authentication import get_current_subject
    from routes import video_restyle as route

    async def not_owner():
        raise HTTPException(status_code = 403, detail = "Only the installation owner can do this")

    monkeypatch.setattr(policy, "require_owner", not_owner)
    monkeypatch.setattr(
        rs, "hardware_plan", lambda: {"weights": "bf16", "vram_gib": 95.0, "refusal": None}
    )
    monkeypatch.setattr("core.inference.sd_cpp_engine.find_sd_server_binary", lambda: None)
    started = []
    monkeypatch.setattr(rs.setup_runner, "start", lambda **kw: started.append(kw))
    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "member"
    client = TestClient(app)
    assert client.post("/api/inference/video/restyle/setup", json = {}).status_code == 403
    assert client.post("/api/inference/video/restyle/setup/cancel").status_code == 403
    assert started == []
    status = client.get("/api/inference/video/restyle/status").json()
    assert status["can_setup"] is False and "owner" in status["refusal"]


def test_a_setup_error_shown_to_every_account_carries_no_credentials(tmp_path, clean_env):
    hub = _FakeHub(tmp_path)

    def download(*a, **k):
        raise OSError(
            "GET https://bob:s3cretpw@mirror.example/x failed for hf_abcdefghijklmnopqrstuvwxyz123456"
        )

    runner = _runner(hub)
    runner._download = download
    state = runner.start(background = False)
    assert state["phase"] == "error"
    assert "s3cretpw" not in state["error"] and "abcdefghijklmnopqrstuvwxyz" not in state["error"]

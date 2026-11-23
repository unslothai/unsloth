# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inkling 2 DSpark drafters in both llama.cpp layouts: what each file is, what each llama-server loads, which
drafter Studio picks, and the depth it launches at."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import llama_cpp as lc
from core.inference.llama_cpp import LlamaCppBackend
from utils.models import model_config as mc
from utils.models.drafters import dspark_schema as ds

_FIXTURES = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "dspark_drafter_headers.json").read_text()
)
PADDED, UNPADDED = 201024, 200058


def test_this_tree_is_the_one_under_test():
    assert Path(ds.__file__).resolve().is_relative_to(Path(_BACKEND_DIR).resolve())


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value) -> bytes:
    if isinstance(value, bool):
        return _s(key) + struct.pack("<IB", 7, int(value))
    if isinstance(value, int):
        return _s(key) + struct.pack("<IQ", 10, value)
    if isinstance(value, float):
        return _s(key) + struct.pack("<If", 6, value)
    if isinstance(value, str):
        return _s(key) + struct.pack("<I", 8) + _s(value)
    raise TypeError(value)


def _write_gguf(path: Path, kvs: dict, tensors: dict) -> Path:
    body = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs))
    body += b"".join(_kv(k, v) for k, v in kvs.items())
    for name, shape in tensors.items():
        body += _s(name) + struct.pack("<I", len(shape))
        body += b"".join(struct.pack("<Q", d) for d in shape) + struct.pack("<IQ", 0, 0)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(body + b"\0" * 64)
    return path


def _drafter(path: Path, fixture: str) -> Path:
    """A header-only copy of a published drafter (scalar KVs, every tensor name and shape)."""
    facts = _FIXTURES[fixture]
    return _write_gguf(path, facts["kvs"], {k: tuple(v) for k, v in facts["tensors"].items()})


def _target(path: Path) -> Path:
    return _write_gguf(
        path,
        {
            "general.architecture": "inkling2",
            "general.name": "Inkling-2-Mini",
            "inkling2.embedding_length": 2304,
            "inkling2.vocab_size_unpadded": UNPADDED,
        },
        {"token_embd.weight": (2304, PADDED)},
    )


@pytest.fixture(autouse = True)
def _hermetic(monkeypatch):
    monkeypatch.delenv(ds.DSPARK_SCHEMAS_ENV, raising = False)
    monkeypatch.setattr(ds, "_default_llama_server_binary", lambda: None)
    ds._scan_cache.clear()
    ds._remote_cache.clear()


# ---- what a drafter file is ---------------------------------------------------


@pytest.mark.parametrize(
    "fixture, schema",
    [
        ("inkling2_fused_q8_0", "fused"),
        ("inkling2_markov2_q8_0", "markov2"),
        ("inkling2_markov2_bf16", "markov2"),
        # Both carry markov_w2, like every generic DSpark drafter; neither may ever be filtered.
        ("generic_qwen3_4b_bf16", "generic"),
        ("generic_minimax_m31_flash_q8_0", "generic"),
    ],
)
def test_published_drafters_classify(tmp_path, fixture, schema):
    path = _drafter(tmp_path / f"{fixture}.gguf", fixture)
    assert ds.dspark_drafter_schema(str(path)) == schema
    for loads in (frozenset({"fused"}), frozenset({"markov2"}), frozenset()):
        reason = ds.dspark_drafter_incompatibility(str(path), supported = loads)
        assert (reason is None) == (schema == "generic" or schema in loads)


def test_vocab_rows_checked_against_the_target_per_layout(tmp_path):
    target = str(_target(tmp_path / "Inkling-2-Mini-UD-Q4_K_XL.gguf"))
    both = frozenset({"fused", "markov2"})
    fused = _drafter(tmp_path / "f.gguf", "inkling2_fused_q8_0")
    markov2 = _drafter(tmp_path / "m.gguf", "inkling2_markov2_q8_0")
    assert ds.dspark_drafter_incompatibility(str(fused), target, supported = both) is None
    assert ds.dspark_drafter_incompatibility(str(markov2), target, supported = both) is None


# ---- what a llama-server loads -----------------------------------------------------


def _server(
    tmp_path: Path,
    lib: bytes | None,
    *,
    static: bytes = b"\x7fELF",
) -> str:
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_bytes(static)
    if lib is not None:
        real = bin_dir / "libllama.so.1.0.0"
        real.write_bytes(b"\0" * 1000 + lib + b"\0" * 1000)
        os.symlink(real.name, bin_dir / "libllama.so.1")
    return str(binary)


@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"markov_w1\0dspark_hidden_out\0", frozenset({"fused"})),
        (b"markov_w1\0markov_ctx_mix\0", frozenset({"markov2"})),
        (b"markov_w1\0dspark_hidden_out\0markov_ctx_mix\0", frozenset({"fused", "markov2"})),
        # Neither Inkling 2 head named anywhere: inconclusive (nothing filtered), never "loads none".
        (b"markov_w1\0markov_w2\0", None),
        (b"llama\0", None),
    ],
)
def test_library_scan(tmp_path, payload, expected):
    assert ds.binary_dspark_schemas(_server(tmp_path, payload)) == expected


def test_a_split_library_build_is_read_as_a_whole(tmp_path):
    """The tensor-name table may sit in libllama-common or the server impl library, not libllama."""
    binary = _server(tmp_path, b"markov_w1\0")
    bin_dir = Path(binary).parent
    (bin_dir / "libllama-common.so.1.0.0").write_bytes(b"\0" * 64 + b"markov_ctx_mix\0")
    (bin_dir / "libllama-server-impl.so").write_bytes(b"\0" * 64 + b"dspark_hidden_out\0")
    # Tool impl libraries are not scanned.
    (bin_dir / "libllama-bench-impl.so").write_bytes(b"\0" * 64)
    assert ds.binary_dspark_schemas(binary) == frozenset({"fused", "markov2"})
    names = [p.name for p in ds._library_candidates(binary)]
    assert "libllama-bench-impl.so" not in names and names[-1] == "llama-server"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("libllama.so.1.0.0", True),
        ("libllama-common.so.1", True),
        ("libllama.0.dylib", True),
        ("libllama-common.dylib", True),
        ("llama.dll", True),
        ("llama-common.dll", True),
        ("libllama-server-impl.so", True),
        ("libllama-cli-impl.so", False),
        ("libggml-base.so.0", False),
        ("llama-server", False),
    ],
)
def test_llama_library_names(name, expected):
    assert ds._is_llama_library(name) is expected


def test_library_scan_finds_a_marker_split_across_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "_SCAN_CHUNK", 1003)
    binary = _server(tmp_path, b"markov_w1\0markov_ctx_mix\0")
    assert ds.binary_dspark_schemas(binary) == frozenset({"markov2"})


def test_static_binary_is_scanned_when_no_library_sits_beside_it(tmp_path):
    binary = _server(tmp_path, None, static = b"\x7fELF markov_w1 dspark_hidden_out")
    assert ds.binary_dspark_schemas(binary) == frozenset({"fused"})


def test_rescan_after_the_library_is_replaced(tmp_path):
    binary = _server(tmp_path, b"markov_w1\0dspark_hidden_out\0")
    assert ds.binary_dspark_schemas(binary) == frozenset({"fused"})
    lib = Path(binary).parent / "libllama.so.1.0.0"
    lib.write_bytes(b"\0" * 1500 + b"markov_w1\0dspark_hidden_out\0markov_ctx_mix\0")
    assert ds.binary_dspark_schemas(binary) == frozenset({"fused", "markov2"})


def test_published_release_tag_wins_over_the_scan(tmp_path):
    binary = _server(tmp_path, b"llama\0")
    marker = Path(binary).parent.parent.parent / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(
        json.dumps({"release_tag": "b12631-mix-acc0843", "published_repo": "unslothai/llama.cpp"})
    )
    assert ds.binary_dspark_schemas(binary) == frozenset({"fused"})
    # A tag the table does not know falls through to the scan.
    marker.write_text(json.dumps({"release_tag": "b12700-mix-0000000"}))
    from utils import llama_cpp_freshness

    getattr(llama_cpp_freshness, "_marker_cache", {}).clear()
    assert ds.binary_dspark_schemas(binary) is None


def test_env_override_is_only_an_escape_hatch(tmp_path, monkeypatch, caplog):
    """The env setting names schemas for a build neither the release table nor the scan can read; it never overrides
    what the launch binary itself shows (that would hand llama-server a drafter it refuses)."""
    readable = _server(tmp_path / "a", b"markov_w1\0dspark_hidden_out\0")
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "markov2")
    with caplog.at_level("WARNING"):
        assert ds.binary_dspark_schemas(readable) == frozenset({"fused"})
    assert any("ignored" in r.getMessage() for r in caplog.records)
    unreadable = _server(tmp_path / "b", b"markov_w1\0")
    assert ds.binary_dspark_schemas(unreadable) == frozenset({"markov2"})
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "none")
    assert ds.binary_dspark_schemas(unreadable) == frozenset()
    assert ds.binary_dspark_schemas(None) == frozenset()


def test_a_known_release_beats_the_env_setting(tmp_path, monkeypatch):
    binary = _server(tmp_path, b"markov_w1\0")
    marker = Path(binary).parent.parent.parent / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(json.dumps({"release_tag": "b12631-mix-acc0843"}))
    from utils import llama_cpp_freshness

    getattr(llama_cpp_freshness, "_marker_cache", {}).clear()
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "markov2")
    assert ds.binary_dspark_schemas(binary) == frozenset({"fused"})


def test_scan_never_raises(tmp_path):
    assert ds.binary_dspark_schemas(str(tmp_path / "missing" / "llama-server")) is None
    assert ds.binary_dspark_schemas(None) is None


# ---- which drafter Studio picks --------------------------------------------------


def _local_layout(tmp_path: Path, *drafters: tuple[str, str]) -> str:
    weight = _target(tmp_path / "Inkling-2-Mini-UD-Q4_K_XL.gguf")
    for fixture, name in drafters:
        _drafter(tmp_path / name, fixture)
    return str(weight)


@pytest.mark.parametrize(
    "loads, expected",
    [
        ("fused", "dspark-Inkling-2-Mini-BF16.gguf"),
        ("markov2", "dspark-Inkling-2-Mini-Q8_0.gguf"),
        # A build that loads both takes the preferred layout over the better-ranked precision.
        ("fused,markov2", "dspark-Inkling-2-Mini-Q8_0.gguf"),
    ],
)
def test_detect_picks_a_loadable_drafter(tmp_path, monkeypatch, loads, expected):
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, loads)
    monkeypatch.setattr(ds, "PREFERRED_INKLING_DSPARK_SCHEMA", "markov2")
    weight = _local_layout(
        tmp_path,
        ("inkling2_markov2_q8_0", "dspark-Inkling-2-Mini-Q8_0.gguf"),
        ("inkling2_fused_q8_0", "dspark-Inkling-2-Mini-BF16.gguf"),
    )
    assert Path(mc.detect_dspark_file(weight)).name == expected


def test_detect_prefers_the_constant_layout_when_both_load(tmp_path, monkeypatch):
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused,markov2")
    monkeypatch.setattr(ds, "PREFERRED_INKLING_DSPARK_SCHEMA", "fused")
    weight = _local_layout(
        tmp_path,
        ("inkling2_markov2_q8_0", "dspark-Inkling-2-Mini-Q8_0.gguf"),
        ("inkling2_fused_q8_0", "dspark-Inkling-2-Mini-BF16.gguf"),
    )
    assert Path(mc.detect_dspark_file(weight)).name == "dspark-Inkling-2-Mini-BF16.gguf"


@pytest.mark.parametrize("loads", ["fused", "markov2", "none"])
def test_generic_drafters_are_never_filtered(tmp_path, monkeypatch, loads):
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, loads)
    weight = _write_gguf(
        tmp_path / "Qwen3-4B-Q4_K_M.gguf",
        {"general.architecture": "qwen3"},
        {"token_embd.weight": (2560, 151936)},
    )
    _drafter(tmp_path / "dspark-Qwen3-4B-BF16.gguf", "generic_qwen3_4b_bf16")
    assert Path(mc.detect_dspark_file(str(weight))).name == "dspark-Qwen3-4B-BF16.gguf"


def _snapshot(tmp_path: Path, sha: str = "a" * 40) -> Path:
    snap = tmp_path / "hub" / "models--unsloth--Inkling-2-Mini-GGUF" / "snapshots" / sha
    snap.mkdir(parents = True)
    return snap


def test_a_cached_drafter_wins_over_any_download(tmp_path, monkeypatch):
    """The old user's fused drafter beside the old target is reused, never re-fetched, on a fused or dual build."""
    snap = _snapshot(tmp_path)
    weight = str(_target(snap / "Inkling-2-Mini-UD-Q4_K_XL.gguf"))
    _drafter(snap / "dspark-Inkling-2-Mini-Q8_0.gguf", "inkling2_fused_q8_0")
    backend = LlamaCppBackend()

    def _no_download(**_kwargs):
        raise AssertionError("a cached loadable drafter must not be downloaded again")

    monkeypatch.setattr(backend, "_download_companion_gguf", _no_download)
    for loads in ("fused", "fused,markov2"):
        monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, loads)
        got = backend._download_dspark(
            hf_repo = "unsloth/Inkling-2-Mini-GGUF",
            near_path = weight,
            binary = "/fake/llama-server",
            caps_probe = lambda _b: {"supports_dspark": True},
        )
        assert Path(got).name == "dspark-Inkling-2-Mini-Q8_0.gguf"


def test_hub_candidates_are_checked_before_download(tmp_path, monkeypatch):
    """A fused-only build never downloads the markov2 drafter: the next loadable Hub copy is fetched instead."""
    snap = _snapshot(tmp_path)
    weight = str(_target(snap / "Inkling-2-Mini-UD-Q4_K_XL.gguf"))
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused")
    remote = {
        "dspark-Inkling-2-Mini-Q8_0.gguf": "markov2",
        "Unsloth_Layout/dspark-Inkling-2-Mini-Q8_0.gguf": "fused",
    }
    asked: list[str] = []

    def _remote(
        repo,
        name,
        token = None,
        **_kw,
    ):
        asked.append(name)
        return remote.get(name)

    monkeypatch.setattr(ds, "remote_dspark_drafter_schema", _remote)
    import utils.models.drafters as drafters

    monkeypatch.setattr(drafters, "remote_dspark_drafter_schema", _remote)
    monkeypatch.setattr(lc, "_is_dspark_drafter_path", lambda n: Path(n).name.startswith("dspark-"))
    backend = LlamaCppBackend()
    picked: list[str] = []

    def _fake_companion(
        *,
        pick,
        outcome = None,
        **_kw,
    ):
        name = pick(list(remote))
        picked.append(name)
        if outcome is not None:
            outcome["listed"] = name is not None
        if name is None:
            return None
        path = snap / name
        _drafter(
            path, "inkling2_fused_q8_0" if remote[name] == "fused" else "inkling2_markov2_q8_0"
        )
        return str(path)

    monkeypatch.setattr(backend, "_download_companion_gguf", _fake_companion)
    got = backend._download_dspark(
        hf_repo = "unsloth/Inkling-2-Mini-GGUF",
        near_path = weight,
        binary = "/fake/llama-server",
        caps_probe = lambda _b: {"supports_dspark": True},
    )
    assert picked == ["Unsloth_Layout/dspark-Inkling-2-Mini-Q8_0.gguf"]
    assert got.endswith("Unsloth_Layout/dspark-Inkling-2-Mini-Q8_0.gguf")
    # The current layout is asked first and skipped before any download; the previous-layout copy is then fetched.
    assert asked == [
        "dspark-Inkling-2-Mini-Q8_0.gguf",
        "Unsloth_Layout/dspark-Inkling-2-Mini-Q8_0.gguf",
    ]
    assert backend._dspark_sidecar_absent is False

    # A dual build fetches the preferred (fused) layout when the repo publishes one, wherever it sits.
    asked.clear()
    picked.clear()
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused,markov2")
    for f in snap.glob("**/dspark-*.gguf"):
        f.unlink()
    got = backend._download_dspark(
        hf_repo = "unsloth/Inkling-2-Mini-GGUF",
        near_path = weight,
        binary = "/fake/llama-server",
        caps_probe = lambda _b: {"supports_dspark": True},
    )
    assert picked == ["Unsloth_Layout/dspark-Inkling-2-Mini-Q8_0.gguf"]
    assert got.endswith("Unsloth_Layout/dspark-Inkling-2-Mini-Q8_0.gguf")

    # ... and the markov2 drafter only when no fused one is published.
    del remote["Unsloth_Layout/dspark-Inkling-2-Mini-Q8_0.gguf"]
    asked.clear()
    picked.clear()
    for f in snap.glob("**/dspark-*.gguf"):
        f.unlink()
    got = backend._download_dspark(
        hf_repo = "unsloth/Inkling-2-Mini-GGUF",
        near_path = weight,
        binary = "/fake/llama-server",
        caps_probe = lambda _b: {"supports_dspark": True},
    )
    assert picked == ["dspark-Inkling-2-Mini-Q8_0.gguf"]
    # A single candidate needs no header read.
    assert asked == []


def test_only_unloadable_hub_drafters_count_as_absent(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path)
    weight = str(_target(snap / "Inkling-2-Mini-UD-Q4_K_XL.gguf"))
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused")
    import utils.models.drafters as drafters

    monkeypatch.setattr(drafters, "remote_dspark_drafter_schema", lambda *a, **k: "markov2")
    backend = LlamaCppBackend()

    def _fake_companion(
        *,
        pick,
        outcome = None,
        **_kw,
    ):
        name = pick(["dspark-Inkling-2-Mini-Q8_0.gguf"])
        if outcome is not None:
            outcome["listed"] = name is not None
        return None

    monkeypatch.setattr(backend, "_download_companion_gguf", _fake_companion)
    assert (
        backend._download_dspark(
            hf_repo = "unsloth/Inkling-2-Mini-GGUF",
            near_path = weight,
            binary = "/fake/llama-server",
            caps_probe = lambda _b: {"supports_dspark": True},
        )
        is None
    )
    assert backend._dspark_sidecar_absent is True


# ---- the depth it launches at ---------------------------------------------------


@pytest.mark.parametrize(
    "architecture, fixture, expected",
    [
        ("inkling2", "inkling2_fused_q8_0", 2),
        ("inkling2_loop", "inkling2_fused_q8_0", 2),
        ("inkling2", "inkling2_markov2_q8_0", 3),
        ("inkling2_loop", "inkling2_markov2_bf16", 3),
        ("inkling2", None, 2),
        ("deepseek4", "generic_minimax_m31_flash_q8_0", 3),
        (None, "inkling2_markov2_q8_0", 3),
    ],
)
def test_default_depth_per_layout(tmp_path, architecture, fixture, expected):
    path = str(_drafter(tmp_path / "d.gguf", fixture)) if fixture else None
    assert lc._dspark_default_draft_n_max(architecture, path) == expected


@pytest.mark.parametrize(
    "fixture, requested, expected",
    [
        ("inkling2_fused_q8_0", None, "2"),
        ("inkling2_markov2_q8_0", None, "3"),
        ("inkling2_markov2_q8_0", 5, "5"),
    ],
)
def test_launch_depth_follows_the_drafter(monkeypatch, tmp_path, fixture, requested, expected):
    fake = {
        "found": True,
        "mtp_token": "draft-mtp",
        "supports_mtp": True,
        "supports_dspark": True,
        "mtp_probe_inconclusive": False,
        "ngram_mod_flavor": "new",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, binary = None: fake)
    )
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    backend._architecture = "inkling2"
    sidecar = _drafter(tmp_path / "dspark-Inkling-2-Mini-Q8_0.gguf", fixture)
    flags = backend._build_speculative_flags(
        speculative_type = "dspark",
        spec_draft_n_max = requested,
        extra_args = None,
        model_identifier = "unsloth/Inkling-2-Mini-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = str(sidecar),
    )
    i = flags.index("--spec-draft-n-max")
    assert flags[i + 1] == expected


@pytest.mark.parametrize(
    "studio, extras, expected",
    [
        ("inkling2_fused_q8_0", "inkling2_markov2_q8_0", "3"),
        ("inkling2_markov2_q8_0", "inkling2_fused_q8_0", "2"),
    ],
)
def test_launch_depth_follows_an_extras_drafter(monkeypatch, tmp_path, studio, extras, expected):
    """An extras --model-draft replaces Studio's sidecar on the child (last wins): the default depth is the one for
    the file the child opens."""
    fake = {
        "found": True,
        "mtp_token": "draft-mtp",
        "supports_mtp": True,
        "supports_dspark": True,
        "mtp_probe_inconclusive": False,
        "ngram_mod_flavor": "new",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, binary = None: fake)
    )
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    backend._architecture = "inkling2"
    sidecar = _drafter(tmp_path / "dspark-Inkling-2-Mini-Q8_0.gguf", studio)
    mine = _drafter(tmp_path / "mine" / "dspark-mine.gguf", extras)
    flags = backend._build_speculative_flags(
        speculative_type = "dspark",
        spec_draft_n_max = None,
        extra_args = ["--model-draft", str(mine)],
        model_identifier = "unsloth/Inkling-2-Mini-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = str(sidecar),
    )
    i = flags.index("--spec-draft-n-max")
    assert flags[i + 1] == expected


# ---- a drafter cached at another revision, after the repo moved -----------------------


def _cache_blob(repo_dir: Path, snap_sha: str, name: str, fixture: str) -> tuple[Path, str]:
    """A Hub-cache entry: snapshots/<rev>/<name> -> ../../blobs/<sha256>, as huggingface_hub lays it out."""
    import hashlib

    blobs = repo_dir / "blobs"
    blobs.mkdir(parents = True, exist_ok = True)
    tmp = blobs / "tmp.gguf"
    _drafter(tmp, fixture)
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.rename(blobs / sha)
    link = repo_dir / "snapshots" / snap_sha / name
    link.parent.mkdir(parents = True, exist_ok = True)
    link.symlink_to(os.path.relpath(blobs / sha, link.parent))
    return link, sha


def _moved_repo(tmp_path: Path, drafter_fixture: str):
    """Target cached at c0, the drafter at c1 under the root name; the live repo moved since."""
    repo_dir = tmp_path / "hub" / "models--unsloth--Inkling-2-Mini-GGUF"
    weight = _target(repo_dir / "snapshots" / ("0" * 40) / "Inkling-2-Mini-UD-Q4_K_XL.gguf")
    link, sha = _cache_blob(repo_dir, "1" * 40, "dspark-Inkling-2-Mini-Q8_0.gguf", drafter_fixture)
    return str(weight), link, sha


def _forbid_download(monkeypatch, backend):
    def _no_download(**_kw):
        raise AssertionError("a loadable drafter cached at another revision must not be downloaded")

    monkeypatch.setattr(backend, "_download_companion_gguf", _no_download)


@pytest.mark.parametrize("loads", ["fused", "fused,markov2"])
def test_online_reuses_a_loadable_drafter_cached_at_another_revision(tmp_path, monkeypatch, loads):
    """The old user's fused drafter (fetched at c1) is still published under Unsloth_Layout/ after the move: reused
    with no download, judged on the cached file even though the live root name now holds the markov2 drafter."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, loads)
    weight, link, sha = _moved_repo(tmp_path, "inkling2_fused_q8_0")
    monkeypatch.setattr(lc, "_published_lfs_hashes", lambda repo, token, **_kw: frozenset({sha}))
    import utils.models.drafters as drafters

    monkeypatch.setattr(drafters, "remote_dspark_drafter_schema", lambda *a, **k: "markov2")
    backend = LlamaCppBackend()
    _forbid_download(monkeypatch, backend)
    got = backend._download_dspark(
        hf_repo = "unsloth/Inkling-2-Mini-GGUF",
        near_path = weight,
        binary = "/fake/llama-server",
        caps_probe = lambda _b: {"supports_dspark": True},
    )
    assert got == str(link)


def test_online_cached_drafter_the_build_cannot_load_is_not_reused(tmp_path, monkeypatch):
    """A markov2 drafter cached at another revision is never handed to a fused-only build; the Hub is asked instead."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused")
    weight, link, sha = _moved_repo(tmp_path, "inkling2_markov2_q8_0")
    monkeypatch.setattr(lc, "_published_lfs_hashes", lambda repo, token, **_kw: frozenset({sha}))
    backend = LlamaCppBackend()
    calls: list[dict] = []

    def _fake_companion(**kw):
        calls.append(kw)
        return None

    monkeypatch.setattr(backend, "_download_companion_gguf", _fake_companion)
    backend._download_dspark(
        hf_repo = "unsloth/Inkling-2-Mini-GGUF",
        near_path = weight,
        binary = "/fake/llama-server",
        caps_probe = lambda _b: {"supports_dspark": True},
    )
    assert len(calls) == 1
    # The generic reuse steps judge a cached file by the live name; DSpark already ran its own.
    assert calls[0]["reuse_snapshot_sibling"] is False


def test_online_cached_drafter_no_longer_published_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused")
    weight, _link, _sha = _moved_repo(tmp_path, "inkling2_fused_q8_0")
    monkeypatch.setattr(
        lc, "_published_lfs_hashes", lambda repo, token, **_kw: frozenset({"f" * 64})
    )
    backend = LlamaCppBackend()
    calls: list[dict] = []
    monkeypatch.setattr(backend, "_download_companion_gguf", lambda **kw: calls.append(kw))
    backend._download_dspark(
        hf_repo = "unsloth/Inkling-2-Mini-GGUF",
        near_path = weight,
        binary = "/fake/llama-server",
        caps_probe = lambda _b: {"supports_dspark": True},
    )
    assert len(calls) == 1


def test_remote_schema_answers_expire(monkeypatch):
    """A header read pinned to a commit is kept for a while, not for the process."""
    calls = []

    def _layout(fetch, total, cap, first):
        calls.append(1)

        class L:
            tensors = [(b"markov_ctx_norm.weight",)]

        return L(), b""

    import hub.utils.gguf_header_delta as ghd

    monkeypatch.setattr(ghd, "_layout_of_remote", _layout)
    monkeypatch.setattr(ghd, "hub_range_fetcher", lambda *a, **k: None)
    clock = [1000.0]
    monkeypatch.setattr(ds.time, "monotonic", lambda: clock[0])
    assert ds.remote_dspark_drafter_schema("o/r", "d.gguf", revision = "c1") == "markov2"
    assert ds.remote_dspark_drafter_schema("o/r", "d.gguf", revision = "c1") == "markov2"
    assert len(calls) == 1
    clock[0] += ds._REMOTE_TTL_S + 1
    assert ds.remote_dspark_drafter_schema("o/r", "d.gguf", revision = "c1") == "markov2"
    assert len(calls) == 2


def test_remote_schema_on_a_moving_ref_is_never_reused(monkeypatch):
    """A name on main can be replaced by the other layout (a repo move): without a pinned commit every probe reads
    the header again, and a different commit never sees another commit's answer."""
    answers = [
        [(b"dspark_hidden_out.weight",)],
        [(b"markov_ctx_norm.weight",)],
        [(b"dspark_fuse.weight",)],
    ]
    calls = []

    def _layout(fetch, total, cap, first):
        tensors = answers[len(calls)]
        calls.append(1)

        class L:
            pass

        L.tensors = tensors
        return L(), b""

    import hub.utils.gguf_header_delta as ghd

    monkeypatch.setattr(ghd, "_layout_of_remote", _layout)
    monkeypatch.setattr(ghd, "hub_range_fetcher", lambda *a, **k: None)
    assert ds.remote_dspark_drafter_schema("o/r", "d.gguf") == "fused"
    assert ds.remote_dspark_drafter_schema("o/r", "d.gguf") == "markov2"
    assert ds.remote_dspark_drafter_schema("o/r", "d.gguf", revision = "c2") == "fused"
    assert len(calls) == 3


def test_remote_schema_reads_the_markov_context_flag(monkeypatch):
    """A drafter marked markov2 only by dflash.has_markov_context (no markov_ctx_* tensor names) is classified the same
    from a ranged header read as from the local header."""
    import struct as _st

    key = b"dflash.has_markov_context"
    header = b"GGUF" + b"\x00" * 20 + _st.pack("<Q", len(key)) + key + _st.pack("<I", 7) + b"\x01"

    def _layout(fetch, total, cap, first):
        class L:
            tensors = [(b"markov_w1.weight",), (b"markov_w2.weight",)]

        return L(), header

    import hub.utils.gguf_header_delta as ghd

    monkeypatch.setattr(ghd, "_layout_of_remote", _layout)
    monkeypatch.setattr(ghd, "hub_range_fetcher", lambda *a, **k: None)
    assert ds.remote_dspark_drafter_schema("o/r", "m.gguf") == "markov2"
    off = header[:-1] + b"\x00"
    monkeypatch.setattr(ghd, "_layout_of_remote", lambda *a: (_layout(*a)[0], off))
    assert ds.remote_dspark_drafter_schema("o/r", "m.gguf") == "generic"


def test_a_kept_listing_from_before_the_move_is_refreshed(tmp_path, monkeypatch):
    """The published-hash listing is kept for a while; one taken before the drafter existed must not send the cached
    drafter back to the Hub: a miss lists again before giving up."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused")
    weight, link, sha = _moved_repo(tmp_path, "inkling2_fused_q8_0")
    asked: list = []

    def _hashes(
        repo,
        token,
        max_age_s = None,
    ):
        asked.append(max_age_s)
        return frozenset({"0" * 64}) if max_age_s is None else frozenset({sha})

    monkeypatch.setattr(lc, "_published_lfs_hashes", _hashes)
    backend = LlamaCppBackend()
    _forbid_download(monkeypatch, backend)
    got = backend._download_dspark(
        hf_repo = "unsloth/Inkling-2-Mini-GGUF",
        near_path = weight,
        binary = "/fake/llama-server",
        caps_probe = lambda _b: {"supports_dspark": True},
    )
    assert got == str(link)
    assert asked == [None, 0]


def test_measured_defaults():
    """B200 timing of a build that loads both: fused at depth 2 beats markov2 at its best depth (3) at both Studio
    sampling defaults, and markov2 at 5 trails no drafter at temperature 1.0."""
    assert ds.PREFERRED_INKLING_DSPARK_SCHEMA == ds.DSPARK_SCHEMA_FUSED
    assert lc._INKLING2_DSPARK_DEPTH == {"fused": 2, "markov2": 3}


@pytest.mark.parametrize(
    "fixture, depth", [("inkling2_fused_q8_0", 2), ("inkling2_markov2_q8_0", 3)]
)
def test_the_public_memory_estimate_prices_the_launch_depth(tmp_path, fixture, depth):
    """The /memory estimate charges the depth the launch emits for an Inkling 2 sidecar (fused 2, markov2 3), not a
    flat 3; a non-Inkling DSpark target keeps 3."""
    from types import SimpleNamespace

    import routes.inference as ri

    target = _write_gguf(
        tmp_path / "Inkling-2-Mini-UD-Q4_K_XL.gguf",
        {"general.architecture": "inkling2", "inkling2.embedding_length": 2304},
        {"token_embd.weight": (2304, PADDED)},
    )
    drafter = str(_drafter(tmp_path / "dspark-Inkling-2-Mini-Q8_0.gguf", fixture))
    config = SimpleNamespace(gguf_file = str(target), gguf_dspark_file = drafter)
    assert ri._estimate_draft_n_max(config, drafter, requested = None, extras = []) == depth
    other = _write_gguf(tmp_path / "qwen.gguf", {"general.architecture": "qwen3"}, {})
    config = SimpleNamespace(gguf_file = str(other), gguf_dspark_file = drafter)
    assert ri._estimate_draft_n_max(config, drafter, requested = None, extras = []) == 3

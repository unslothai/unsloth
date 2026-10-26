# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lookup-only tables (per-layer embeddings / Engram) priced apart in the listing and the panel."""

from __future__ import annotations

import asyncio
import io
import os
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import test_kv_cache_estimation  # noqa: E402,F401
import test_llama_cpp_placement  # noqa: E402,F401 -- stubs jwt for routes.inference

import routes.inference as ri  # noqa: E402
import utils.hardware as hw  # noqa: E402
from core.inference import llama_cpp as lc  # noqa: E402
from hub.schemas.inventory import GgufVariantDetail  # noqa: E402
from hub.services.models import gguf_variants as gv  # noqa: E402
from hub.utils import gguf_header_delta  # noqa: E402
from hub.utils import gguf_lookup_tables as glt  # noqa: E402

GIB = 1024**3
IQ4_NL, Q5_1, F16 = 20, 7, 1
PLE = "per_layer_token_embd.weight"

DSV41_TABLE = 256 * 768022850 // 32 * 18
FLASH_NEXT_TABLE = 160 * 320001536 // 32 * 18
GEMMA_TABLE = 8960 * 262144 // 32 * 24

_ATTN = {
    "block_count": 4,
    "context_length": 4096,
    "embedding_length": 256,
    "attention.head_count": 4,
    "attention.head_count_kv": 2,
    "attention.key_length": 64,
    "attention.value_length": 64,
}


def _header(
    arch: str,
    tensors: list[tuple[str, list[int], int]],
    ints: dict | None = None,
    offsets: list[int] | None = None,
):
    """A GGUF v3 header: the architecture, uint32 metadata, and tensor infos with no data."""
    kv: list[tuple[str, object]] = [("general.architecture", arch)]
    kv += [(f"{arch}.{k}", int(v)) for k, v in (ints or {}).items()]
    buf = io.BytesIO()
    buf.write(b"GGUF")
    buf.write(struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for key, value in kv:
        raw = key.encode()
        buf.write(struct.pack("<Q", len(raw)) + raw)
        if isinstance(value, str):
            data = value.encode()
            buf.write(struct.pack("<IQ", 8, len(data)) + data)
        else:
            buf.write(struct.pack("<II", 4, value))
    offset = 0
    for i, (name, dims, ggml_type) in enumerate(tensors):
        raw = name.encode()
        buf.write(struct.pack("<Q", len(raw)) + raw)
        buf.write(struct.pack("<I", len(dims)))
        buf.write(struct.pack(f"<{len(dims)}Q", *dims))
        buf.write(struct.pack("<IQ", ggml_type, offsets[i] if offsets else offset))
        offset += 1024
    return buf.getvalue()


_LAZY_BLOB = _header("qwen4exp", [(PLE, [160, 320001536], IQ4_NL)], _ATTN)
_DSV41_EMBD = ("token_embd.weight", [4096, 129280], IQ4_NL)
_DSV41_PLE = (PLE, [256, 768022850], IQ4_NL)


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(data)
    return str(path)


def _dsv41(tmp_path: Path) -> str:
    return _write(
        tmp_path / "DeepSeek-V4.1-Flash-UD-IQ1_S.gguf",
        _header("deepseek41", [_DSV41_EMBD, _DSV41_PLE]),
    )


def _flash_next(tmp_path: Path) -> str:
    tensors = [
        ("token_embd.weight", [2048, 151936], IQ4_NL),
        (PLE, [160, 320001536], IQ4_NL),
        ("blk.0.attn_q.weight", [256, 256], F16),
    ]
    return _write(
        tmp_path / "Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf", _header("qwen4exp", tensors, _ATTN)
    )


def _gemma(tmp_path: Path, arch = "gemma3n") -> str:
    tensors = [(PLE, [8960, 262144], Q5_1)]
    if arch == "gemma3n":
        tensors.insert(0, ("token_embd.weight", [2048, 262144], IQ4_NL))
    return _write(tmp_path / f"{arch}.gguf", _header(arch, tensors))


def _blob_fetcher(
    blobs: dict,
    calls: list,
    delay: float = 0.0,
):
    """A Range fetcher over in-memory files that reports each file's size like Content-Range."""

    def fetcher(
        repo_id,
        name,
        token,
        revision = None,
    ):
        blob, size = blobs[name]

        def fetch(start, end):
            calls.append((name, start, end))
            if delay:
                time.sleep(delay)
            fetch.file_size = size
            return blob[start : end + 1]

        return fetch

    return fetcher


def _answer(variants, sources = None):
    from hub.schemas.inventory import GgufVariantsResponse
    from hub.services.models.gguf_variants import VariantsAnswer
    return VariantsAnswer(
        GgufVariantsResponse(repo_id = "unsloth/X-GGUF", variants = variants),
        None,
        variant_context_sources = sources or {},
    )


def _row(
    filename = "a.gguf",
    quant = "Q4",
    size = 40 * GIB,
    **kw,
):
    return GgufVariantDetail(filename = filename, quant = quant, size_bytes = size, **kw)


def _listing(
    rows,
    repo,
    *,
    token = None,
    offline = False,
):
    """The coroutine; ``rows`` may be a ready answer."""
    answer = rows if hasattr(rows, "response") else _answer(rows)
    return gv._with_lookup_tables(
        answer, repo, hf_token = token, offline = offline, prefer_local_cache = False
    )


def _list(rows, repo, **kw):
    return asyncio.run(_listing(rows, repo, **kw))


def _tables(out):
    return [v.lookup_table_bytes for v in out.response.variants]


def _arch(
    monkeypatch,
    known,
    arch = "qwen4exp",
):
    monkeypatch.setattr(gv, "_cached_hub_repo_architecture", lambda *a, **k: (known, arch))


def _remote(monkeypatch, fn = None):
    monkeypatch.setattr(
        glt, "remote_lookup_info", fn or (lambda *a, **k: ("qwen4exp", FLASH_NEXT_TABLE))
    )


def _overlap(
    first,
    second,
    wait = None,
):
    """Run ``second`` once ``first`` is underway (after 0.1 s, or once ``wait`` is set)."""

    async def both():
        a = asyncio.ensure_future(first)
        if wait is None:
            await asyncio.sleep(0.1)
        while wait is not None and not wait.is_set():
            await asyncio.sleep(0.01)
        return await asyncio.gather(a, second)

    return asyncio.run(both())


def _stream_backend(monkeypatch, *, vulkan, gpus):
    backend = lc.LlamaCppBackend.__new__(lc.LlamaCppBackend)
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda: "/fake/llama-server")
    monkeypatch.setattr(
        backend,
        "probe_server_capabilities",
        lambda binary = None: {"supports_lazy_mode": True, "lazy_mode_flag": "--lazy-mode"},
    )
    monkeypatch.setattr(backend, "_is_vulkan_backend", lambda binary = None: vulkan)
    monkeypatch.setattr(backend, "_get_gpu_memory", lambda *a, **k: list(gpus))
    return backend


def _no_gpu_backend(monkeypatch, cpu_build):
    monkeypatch.setattr(
        lc.LlamaCppBackend, "_binary_ships_no_gpu_backend", staticmethod(lambda *a, **k: cpu_build)
    )


@pytest.fixture(autouse = True)
def _streams_here(monkeypatch):
    monkeypatch.setattr(gv, "_host_streams_lookup_tables", lambda: True)


def test_split_names_keep_the_extension_as_written():
    assert glt.shard_names("d/M-00001-of-00002.GGUF") == [
        "d/M-00001-of-00002.GGUF",
        "d/M-00002-of-00002.GGUF",
    ]


def test_a_posix_class_pattern_is_not_proof_the_table_stays_on_the_host():
    """std::regex reads [[:alpha:]] as a class."""
    on_device = lc._override_places_tensor_on_device
    assert on_device(PLE, ["-ot", "per_[[:alpha:]_]+=CUDA0"], {})
    assert on_device(PLE, ["-ot", "per_[[:alpha:]_]+=CPU,per_layer=CUDA0"], {})
    assert not on_device(PLE, ["-ot", "per_[[:alpha:]_]+=CPU"], {})


class TestHeaderScan:
    def test_real_table_sizes(self, tmp_path):
        assert glt.local_lookup_info(_dsv41(tmp_path)) == ("deepseek41", DSV41_TABLE)
        assert DSV41_TABLE / GIB == pytest.approx(103.00, abs = 0.01)
        assert glt.local_lookup_info(_flash_next(tmp_path)) == ("qwen4exp", FLASH_NEXT_TABLE)
        assert FLASH_NEXT_TABLE / GIB == pytest.approx(26.82, abs = 0.01)

    def test_gemma3n_is_not_lazily_read(self, tmp_path):
        assert glt.local_lookup_info(_gemma(tmp_path)) == ("gemma3n", 0)

    def test_a_small_table_stays_resident_under_auto(self, tmp_path):
        assert glt.local_lookup_table_bytes(_gemma(tmp_path, "gemma4")) == 0

    def test_an_explicit_lazy_mode_on_reads_a_small_table_lazily(self, tmp_path):
        small = _gemma(tmp_path, "gemma4")
        assert glt.local_lookup_info(small) == ("gemma4", 0)
        assert glt.local_lookup_info(small, "on") == ("gemma4", GEMMA_TABLE)

    @pytest.mark.parametrize(
        "second, want",
        [(True, DSV41_TABLE), (False, 0)],
        ids = ["split_file_finds_the_table_in_a_later_shard", "missing_shard"],
    )
    def test_split_file_finds_the_table_in_a_later_shard(self, tmp_path, second, want):
        first = _write(tmp_path / "m-00001-of-00002.gguf", _header("deepseek41", [_DSV41_EMBD]))
        if second:
            _write(tmp_path / "m-00002-of-00002.gguf", _header("", [_DSV41_PLE]))
        assert glt.local_lookup_table_bytes(first) == want

    def test_a_missing_shard_or_garbage_is_zero(self, tmp_path):
        assert glt.local_lookup_table_bytes(_write(tmp_path / "junk.gguf", b"not a gguf")) == 0
        assert glt.local_lookup_table_bytes(None) == 0

    def test_remote_reads_only_the_header_range(self, tmp_path):
        calls = []
        fetcher = _blob_fetcher({"Q2/flash-Q2.gguf": (_LAZY_BLOB, 40 * GIB)}, calls)

        def read():
            return glt.remote_lookup_info(
                "unsloth/Flash-Next-GGUF", "Q2/flash-Q2.gguf", identity = 1, fetcher_fn = fetcher
            )

        assert read() == ("qwen4exp", FLASH_NEXT_TABLE)
        assert calls and all(start == 0 for _name, start, _end in calls)
        calls.clear()
        assert read() == ("qwen4exp", FLASH_NEXT_TABLE)
        assert not calls
        glt._REMOTE_CACHE.clear()
        glt._REMOTE_TTL_S, ttl = -1.0, glt._REMOTE_TTL_S
        try:
            read()
            assert glt.remote_cached("unsloth/Flash-Next-GGUF", "Q2/flash-Q2.gguf", None, 1) is None
        finally:
            glt._REMOTE_TTL_S = ttl
        assert calls

    def test_remote_failure_is_zero_and_not_cached(self):
        def broken(*a, **k):
            def fetch(start, end):
                raise OSError("network down")

            return fetch

        assert glt.remote_lookup_info("o/r", "f.gguf", identity = 7, fetcher_fn = broken) == (None, 0)
        assert glt.remote_cached("o/r", "f.gguf", None, 7) is None

    def test_a_remote_answer_is_cached_per_lazy_mode(self):
        fetcher = _blob_fetcher({"f.gguf": (b"", 0)}, [])
        glt.remote_lookup_info("o/m", "f.gguf", identity = 11, fetcher_fn = fetcher)
        assert glt.remote_cached("o/m", "f.gguf", None, 11, mode = "on") is None


class TestListing:
    @pytest.mark.parametrize(
        "streams, want",
        [(False, None), (True, DSV41_TABLE)],
        ids = [
            "a_host_that_reads_the_tables_resident_lists_no_discount",
            "a_downloaded_variant_is_read_from_its_own_header",
        ],
    )
    def test_a_downloaded_variant_is_read_from_its_own_header(
        self, tmp_path, monkeypatch, streams, want
    ):
        """An iGPU / APU resolves lazy auto to off, so the listing prices the whole file."""
        monkeypatch.setattr(gv, "_host_streams_lookup_tables", lambda: streams)
        monkeypatch.setattr(
            gv, "_hub_repo_architecture", lambda *a, **k: pytest.fail("no Hub call")
        )
        path = _dsv41(tmp_path)
        row = _row(os.path.basename(path), "UD-IQ1_S", 216 * GIB, downloaded = True)
        answer = _answer([row], {"ud-iq1_s": path})
        out = _list(answer, "unsloth/X-GGUF", offline = True)
        if want is None:
            assert out is answer
        else:
            assert _tables(out) == [want]
            assert _tables(answer) == [0]

    def test_both_variant_models_carry_the_field(self):
        from models.models import GgufVariantDetail as RouteDetail
        for model in (GgufVariantDetail, RouteDetail):
            assert model.model_fields["lookup_table_bytes"].default == 0

    def test_hub_only_quants_of_a_lazy_arch_are_range_read(self, monkeypatch):
        _arch(monkeypatch, True)
        _remote(monkeypatch)
        rows = [_row("Q2/a-00001-of-00003.gguf", "Q2"), _row("README.md", "x", 1)]
        assert _tables(_list(rows, "unsloth/Flash-Next-GGUF")) == [FLASH_NEXT_TABLE, 0]

    def test_a_small_repo_never_asks_the_hub_for_its_architecture(self, monkeypatch):
        """No row can hold a streamed table (each is under 4 GiB)."""
        _arch(monkeypatch, False, None)
        monkeypatch.setattr(
            gv, "_hub_repo_architecture", lambda *a, **k: pytest.fail("no Hub call")
        )
        answer = _answer([_row(size = GIB)])
        assert _list(answer, "o/small") is answer

    def test_a_cold_architecture_is_priced_on_the_first_listing(self, monkeypatch):
        fetched = []
        monkeypatch.setattr(
            gv, "_hub_repo_architecture", lambda repo, token: fetched.append(repo) or "qwen4exp"
        )
        _arch(monkeypatch, False, None)
        _remote(monkeypatch)
        assert _tables(_list([_row()], "o/cold")) == [FLASH_NEXT_TABLE]
        assert fetched == ["o/cold"]

    def test_a_slow_architecture_lookup_is_bounded_by_the_deadline(self, monkeypatch):
        """A late answer never holds the listing past its deadline, and is asked once."""
        release = threading.Event()
        fetched = []

        def slow_arch(repo, token):
            fetched.append(repo)
            release.wait(5)
            return "qwen4exp"

        monkeypatch.setattr(gv, "_LOOKUP_DEADLINE_S", 0.2)
        monkeypatch.setattr(gv, "_hub_repo_architecture", slow_arch)
        _arch(monkeypatch, False, None)
        _remote(monkeypatch, lambda *a, **k: pytest.fail("no header read yet"))
        answer = _answer([_row()])
        try:
            for _ in range(3):
                started = time.monotonic()
                assert _list(answer, "o/slow") is answer
                assert time.monotonic() - started < 2.0
        finally:
            release.set()
        for _ in range(50):
            if not gv._lookup_inflight:
                break
            time.sleep(0.05)
        assert fetched == ["o/slow"]

    def test_a_direct_file_source_finds_its_row_beside_it(self, tmp_path):
        model = tmp_path / "Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf"
        model.write_bytes(b"GGUF")
        row = SimpleNamespace(filename = model.name, quant = "UD-Q2_K_XL", downloaded = True)
        for answer, repo in (
            (SimpleNamespace(variant_context_sources = None, context_source = str(model)), "o/r"),
            (SimpleNamespace(variant_context_sources = None, context_source = None), str(model)),
        ):
            assert gv._local_variant_gguf(answer, repo, row) == str(model)

    def test_a_requested_cache_copy_is_found_where_the_listing_found_it(self, tmp_path):
        """local_path naming a second cache root, or a snapshot other than refs/main, is searched."""
        repo_dir = tmp_path / "second_cache" / "models--o--lookup-cache-test"
        main_snap, pinned_snap = repo_dir / "snapshots" / "aaaa", repo_dir / "snapshots" / "bbbb"
        for snap, name in ((main_snap, "m-Q4.gguf"), (pinned_snap, "p-Q8.gguf")):
            snap.mkdir(parents = True)
            (snap / name).write_bytes(b"GGUF")
        (repo_dir / "refs").mkdir()
        (repo_dir / "refs" / "main").write_text("aaaa")
        answer = SimpleNamespace(
            variant_context_sources = None, context_source = None, cache_authorized = True
        )
        for name, root, snap in (
            ("m-Q4.gguf", repo_dir, main_snap),
            ("p-Q8.gguf", pinned_snap, pinned_snap),
        ):
            row = SimpleNamespace(filename = name, quant = "Q", downloaded = True)
            got = gv._local_variant_gguf(answer, "o/lookup-cache-test", row, str(root))
            assert got == str(snap / name)

    def test_a_read_the_pool_refuses_after_the_deadline_releases_its_key(self, monkeypatch):
        """Late refusals (the pool shut down at exit) never reach the read's own finally."""

        async def refused(fn, /, *args):
            await asyncio.sleep(0.01)
            raise RuntimeError("cannot schedule new futures after shutdown")

        monkeypatch.setattr(gv, "_lookup_thread", refused)
        key = ("arch", "o/refused")

        async def go():
            loop = asyncio.get_running_loop()
            assert await gv._lookup_await_once(key, loop.time(), lambda: 1) is None
            assert key in gv._lookup_inflight
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert key not in gv._lookup_inflight

    def test_a_cached_non_lazy_variant_does_not_veto_the_rest_of_the_repo(
        self, tmp_path, monkeypatch
    ):
        local = _write(
            tmp_path / "llama.gguf",
            _header("llama", [("token_embd.weight", [256, 1000], F16)], _ATTN),
        )
        _arch(monkeypatch, True)
        _remote(monkeypatch)
        rows = [_row("llama.gguf", "Q8_0", 1, downloaded = True), _row("flash/f.gguf", "Q2")]
        out = _list(_answer(rows, {"q8_0": local}), "o/mixed")
        assert _tables(out) == [0, FLASH_NEXT_TABLE]

    def test_one_listing_queues_every_uncached_header(self, monkeypatch):
        calls = []
        lock = threading.Lock()

        def read(repo, name, *a, **k):
            with lock:
                calls.append(name)
            return ("qwen4exp", FLASH_NEXT_TABLE)

        _arch(monkeypatch, True)
        _remote(monkeypatch, read)
        rows = [_row(f"q{i}/m.gguf", f"Q{i}", 40 * GIB + i) for i in range(10)]
        out = _list(rows, "o/many")
        assert sorted(calls) == sorted(r.filename for r in rows)
        assert all(_tables(out))

    @pytest.mark.parametrize("offline, want", [(False, 10 * GIB), (True, 40 * GIB)])
    def test_an_update_is_priced_from_the_published_header(
        self, monkeypatch, tmp_path, offline, want
    ):
        """The cached file is the old revision: online, an update's table comes from the Hub's header."""
        on_disk = tmp_path / "old.gguf"
        on_disk.write_bytes(b"x")
        _arch(monkeypatch, True)
        monkeypatch.setattr(gv, "_local_variant_gguf", lambda *a: str(on_disk))
        monkeypatch.setattr(
            glt, "local_lookup_info", lambda path, mode = "auto": ("qwen4exp", 40 * GIB)
        )
        _remote(monkeypatch, lambda *a, **k: ("qwen4exp", 10 * GIB))
        row = _row("u.gguf", "U", 70 * GIB + offline, downloaded = True, update_available = True)
        out = _list([row], f"o/updated-{offline}", offline = offline)
        assert _tables(out) == [want]

    def test_a_forced_lazy_mode_sizes_the_tables_it_streams(self, monkeypatch, tmp_path):
        modes = []

        def local(path, mode = "auto"):
            modes.append(("local", mode))
            return ("qwen4exp", FLASH_NEXT_TABLE)

        def read(
            *a,
            mode = "auto",
            **k,
        ):
            modes.append(("remote", mode))
            return ("qwen4exp", FLASH_NEXT_TABLE)

        monkeypatch.setattr(gv, "_listing_lazy_mode", lambda: "on")
        _arch(monkeypatch, True)
        _remote(monkeypatch, read)
        monkeypatch.setattr(glt, "local_lookup_info", local)
        on_disk = tmp_path / "local.gguf"
        on_disk.write_bytes(b"x")
        monkeypatch.setattr(
            gv, "_local_variant_gguf", lambda a, r, v, *_: str(on_disk) if v.quant == "L" else None
        )
        _list([_row("l.gguf", "L", 1, downloaded = True), _row("r.gguf", "R", 2)], "o/forced")
        assert sorted(modes) == [("local", "on"), ("remote", "on")]

    def test_a_forced_lazy_mode_looks_up_a_cold_repo_of_small_files(self, monkeypatch):
        monkeypatch.setattr(gv, "_listing_lazy_mode", lambda: "on")
        _arch(monkeypatch, False, None)
        monkeypatch.setattr(gv, "_hub_repo_architecture", lambda *a, **k: "gemma4")
        _remote(monkeypatch, lambda *a, **k: ("gemma4", 2 * GIB))
        assert _tables(_list([_row("s.gguf", size = 3 * GIB)], "o/small")) == [2 * GIB]

    @pytest.mark.parametrize(
        "stage",
        ["arch", "table"],
        ids = [
            "an_architecture_lookup_in_flight_for_another_token_does_not_block",
            "a_table_scan_in_flight_for_another_token_does_not_skip_the_row",
        ],
    )
    def test_an_architecture_lookup_in_flight_for_another_token_does_not_block(
        self, monkeypatch, stage
    ):
        asked = []
        second = threading.Event()

        def gate(token, answer):
            asked.append(token)
            if token == "hf_first":
                second.wait(1.5)
            else:
                second.set()
            return answer

        if stage == "arch":
            _arch(monkeypatch, False, None)
            monkeypatch.setattr(
                gv, "_hub_repo_architecture", lambda repo, token: gate(token, "qwen4exp")
            )
            _remote(monkeypatch)
        else:
            _arch(monkeypatch, True)
            _remote(
                monkeypatch,
                lambda repo, name, token, *a, **k: gate(token, ("qwen4exp", FLASH_NEXT_TABLE)),
            )
        rows = [_row(size = 40 * GIB + 7)]
        repo = f"o/token-{stage}"
        _first, other = _overlap(
            _listing(rows, repo, token = "hf_first"), _listing(rows, repo, token = "hf_second")
        )
        assert sorted(asked) == ["hf_first", "hf_second"]
        assert _tables(other) == [FLASH_NEXT_TABLE]

    @pytest.mark.parametrize(
        "vulkan, node, streams",
        [(False, True, False), (False, False, True), (True, False, False), (True, True, False)],
    )
    def test_an_empty_inventory_beside_a_gpu_node_prices_the_table_resident(
        self, monkeypatch, vulkan, node, streams
    ):
        """Without the torch fallback a ROCm APU can read as no rows; an empty Vulkan inventory may hide an iGPU."""
        backend = _stream_backend(monkeypatch, vulkan = vulkan, gpus = [])
        monkeypatch.setattr(backend, "_lazy_auto_resolves_off", lambda **k: False)
        monkeypatch.setattr(lc, "_gpu_device_node_present", lambda: node)
        assert backend.lookup_tables_stream_by_default(None, {}) is streams

    def test_a_stalled_hub_leaves_no_queue_after_the_deadline(self, monkeypatch):
        release = threading.Event()
        started = []
        lock = threading.Lock()

        def stalled(repo, name, *a, **k):
            with lock:
                started.append(name)
            release.wait(10)
            return ("qwen4exp", FLASH_NEXT_TABLE)

        monkeypatch.setattr(gv, "_LOOKUP_DEADLINE_S", 0.3)
        _arch(monkeypatch, True)
        _remote(monkeypatch, stalled)
        rows = [_row(f"s{i}/m.gguf", f"S{i}", 40 * GIB + i) for i in range(12)]

        async def listing_then_release():
            await _listing(rows, "o/stalled")
            await asyncio.sleep(0.3)
            release.set()
            await asyncio.sleep(0.5)

        try:
            asyncio.run(listing_then_release())
        finally:
            release.set()
        assert len(started) <= gv._LOOKUP_CONCURRENCY, started
        with gv._lookup_arch_lock:
            assert not [k for k in gv._lookup_inflight if k[:2] == ("table", "o/stalled")]

    def test_an_expired_architecture_lookup_leaves_no_queue(self, monkeypatch):
        release = threading.Event()
        ran = []

        def slow(name):
            ran.append(name)
            release.wait(10)
            return name

        async def main():
            loop = asyncio.get_running_loop()
            busy = [
                asyncio.ensure_future(gv._lookup_thread(slow, f"busy{i}"))
                for i in range(gv._LOOKUP_CONCURRENCY)
            ]
            await asyncio.sleep(0.2)
            got = await gv._lookup_await_once(
                ("arch", "o/late", "fp"), loop.time() + 0.2, slow, "late"
            )
            release.set()
            await asyncio.gather(*busy)
            await asyncio.sleep(0.3)
            return got

        try:
            got = asyncio.run(main())
        finally:
            release.set()
        assert got is None
        assert "late" not in ran
        with gv._lookup_arch_lock:
            assert ("arch", "o/late", "fp") not in gv._lookup_inflight

    def test_work_the_deadline_drops_never_runs_if_a_worker_reaches_it_late(self, monkeypatch):
        """A worker can pick up queued work before the cancel reaches the executor."""
        captured = []
        reads = []

        async def queued(fn, *args):
            captured.append((fn, args))
            await asyncio.sleep(10)

        def read(repo, name, *a, **k):
            reads.append(name)
            return ("qwen4exp", FLASH_NEXT_TABLE)

        monkeypatch.setattr(gv, "_lookup_thread", queued)
        monkeypatch.setattr(gv, "_LOOKUP_DEADLINE_S", 0.1)
        _arch(monkeypatch, True)
        _remote(monkeypatch, read)
        rows = [_row(f"q{i}/m.gguf", f"Q{i}", 40 * GIB + i) for i in range(3)]

        async def main():
            await _listing(rows, "o/late")
            loop = asyncio.get_running_loop()
            return await gv._lookup_await_once(
                ("arch", "o/late", "fp"), loop.time() + 0.1, read, "o/late", "arch", None
            )

        assert asyncio.run(main()) is None
        assert len(captured) == 4
        for fn, args in captured:
            fn(*args)
        assert reads == []

    def test_a_capability_read_never_falls_back_to_a_torch_context(self, monkeypatch):
        from utils.hardware import gpu_query

        B = lc.LlamaCppBackend
        monkeypatch.setattr(B, "_is_vulkan_backend", staticmethod(lambda binary = None: False))
        monkeypatch.setattr(B, "_get_gpu_memory_nvml", staticmethod(lambda: []))
        monkeypatch.setattr(B, "_get_gpu_memory_amd_smi", staticmethod(lambda binary = None, **k: []))
        monkeypatch.setattr(
            B,
            "_get_gpu_memory_torch",
            staticmethod(lambda *a, **k: pytest.fail("torch probe in a capability read")),
        )

        def no_smi(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(gpu_query, "run_nvidia_smi", no_smi)
        with lc._without_torch_gpu_probe():
            assert B._get_gpu_memory("/fake/llama-server", for_llama_server = True) == []
        assert lc._GPU_MEMORY_SKIP_TORCH.get() is False

    def test_offline_or_a_non_lazy_arch_never_reads_the_hub(self, monkeypatch):
        _remote(monkeypatch, lambda *a, **k: pytest.fail("no header read"))
        answer = _answer([_row(quant = "Q4_K_M", size = 1)])
        assert _list(answer, "o/r", offline = True) is answer
        monkeypatch.setattr(gv, "_hub_repo_architecture", lambda *a, **k: "llama")
        assert _list(answer, "o/r") is answer

    def test_a_failure_returns_the_listing_unchanged(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(gv, "_local_variant_gguf", boom)
        answer = _answer([_row(quant = "Q", size = 1)])
        assert _list(answer, "o/r") is answer


def _static(monkeypatch, name, fn):
    monkeypatch.setattr(ri.LlamaCppBackend, name, staticmethod(fn))


def _ram(monkeypatch, gib):
    _static(monkeypatch, "_available_system_memory_mib", lambda: int(gib * 1024))


def _gpus(monkeypatch, *rows):
    _static(monkeypatch, "_get_gpu_memory", lambda binary = None, **k: list(rows))


def _caps(monkeypatch, **caps):
    _static(monkeypatch, "probe_server_capabilities", lambda binary = None: {"found": True, **caps})


def _apple(monkeypatch, budget_gib):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    _static(monkeypatch, "_apple_metal_memory_budget_bytes", lambda: int(budget_gib * GIB))


class TestEstimate:
    WEIGHTS = 73.45 * GIB

    @pytest.fixture(autouse = True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.delenv("LLAMA_ARG_OVERRIDE_TENSOR", raising = False)

    @pytest.fixture
    def breakdown(self, tmp_path, monkeypatch):
        path = _flash_next(tmp_path)
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: self.WEIGHTS / GIB)
        _static(monkeypatch, "_get_gguf_size_bytes", lambda p: int(self.WEIGHTS))
        _caps(monkeypatch, supports_lazy_mode = True)
        _static(monkeypatch, "_is_vulkan_backend", lambda b = None: False)
        _gpus(monkeypatch, (0, 22000, 24576))
        _static(monkeypatch, "_integrated_cuda_gpu_ids", lambda: set())
        monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
        _static(monkeypatch, "_apple_metal_memory_budget_bytes", lambda: 0)
        _ram(monkeypatch, 16)
        ri._estimate_files_cache.clear()
        ri._estimate_config_cache.clear()
        config = SimpleNamespace(
            identifier = "local/flash-next", gguf_file = path, is_gguf = True, gguf_mmproj_file = None
        )

        def run(gguf = None, **kw):
            cfg = config if gguf is None else SimpleNamespace(**{**vars(config), "gguf_file": gguf})
            return ri._gguf_memory_breakdown(
                cfg, gguf or path, **{"n_ctx": 4096, "gpu_memory_mode": "auto", **kw}
            )

        yield run
        ri._estimate_files_cache.clear()
        ri._estimate_config_cache.clear()

    def test_streamed_table_leaves_the_gpu_and_the_total(self, breakdown):
        b = breakdown()
        assert (b.lookup_table_bytes, b.lookup_table_on_disk) == (FLASH_NEXT_TABLE, True)
        assert b.weights_bytes == int(round(self.WEIGHTS)) - FLASH_NEXT_TABLE
        assert b.gpu_bytes == b.total_bytes == b.weights_bytes + b.kv_bytes + b.compute_bytes

    def test_lazy_mode_off_keeps_it_resident_but_still_off_the_gpu(self, breakdown):
        b = breakdown(llama_extra_args = ["-lzm", "off"])
        assert (b.lookup_table_bytes, b.lookup_table_on_disk) == (FLASH_NEXT_TABLE, False)
        assert b.weights_bytes == int(round(self.WEIGHTS))
        assert b.gpu_bytes == b.total_bytes - FLASH_NEXT_TABLE

    @pytest.mark.parametrize("extras", [[], ["-lzm", "auto"]], ids = ["managed", "user_auto"])
    def test_ram_that_holds_the_table_reads_it_resident_as_the_launch_does(
        self, breakdown, monkeypatch, extras
    ):
        """A user's own lazy mode wins over the managed choice, as at launch."""
        _ram(monkeypatch, 256)
        b = breakdown(llama_extra_args = extras)
        assert (b.lookup_table_bytes, b.lookup_table_on_disk) == (FLASH_NEXT_TABLE, bool(extras))

    @pytest.mark.parametrize(
        "extras, on_disk",
        [
            ([], False),
            (["--parallel", "1"], False),
            (["-c", "0"], True),
            (["--parallel", "8"], True),
        ],
        ids = ["none", "same_parallel", "ctx_zero", "larger_parallel"],
    )
    def test_a_pass_through_that_reprices_the_cache_keeps_the_table_streamed(
        self, breakdown, monkeypatch, extras, on_disk
    ):
        _ram(monkeypatch, 256)
        assert breakdown(llama_extra_args = extras).lookup_table_on_disk is on_disk

    @pytest.mark.parametrize("ram_gib", [66, 256])
    @pytest.mark.parametrize("cpu_build", [False, True])
    def test_a_cpu_child_keeps_the_table_streamed_as_the_launch_does(
        self, breakdown, monkeypatch, cpu_build, ram_gib
    ):
        _ram(monkeypatch, ram_gib)
        _static(monkeypatch, "_binary_ships_no_gpu_backend", lambda *a, **k: cpu_build)
        assert breakdown().lookup_table_on_disk is cpu_build
        assert breakdown(llama_extra_args = ["--device", "none"]).lookup_table_on_disk is True

    def test_a_mig_slice_reporting_no_total_keeps_its_vram_credit(self, breakdown, monkeypatch):
        """Total 0 means an iGPU only on Vulkan; a CUDA MIG or vGPU slice can report it with VRAM of its own."""
        _ram(monkeypatch, 66)
        _gpus(monkeypatch, (0, 22000, 0))
        assert breakdown().lookup_table_on_disk is False

    @pytest.mark.parametrize("pool", ["soc", "apu", "discrete"])
    def test_a_unified_pool_beside_a_card_is_priced_once_as_the_launch_does(
        self, breakdown, monkeypatch, pool
    ):
        """An integrated CUDA SoC or ROCm APU's free memory is the host RAM the proof already counts."""
        _ram(monkeypatch, 58)
        _gpus(monkeypatch, (0, 16000, 16384), (1, 50000, 65536))
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_INTEGRATED_CUDA_IDS",
            {ri.LlamaCppBackend._integrated_cuda_mask_key(): (1,) if pool == "soc" else ()},
        )
        monkeypatch.setattr(
            ri.get_llama_cpp_backend(),
            "_amd_apu_wants_unified_memory",
            lambda ids = None, *a, **k: (pool == "apu" and ids is not None and bool(set(ids) & {1})),
            raising = False,
        )
        assert breakdown().lookup_table_on_disk is (pool != "discrete")

    @pytest.mark.parametrize("tensor", [False, True])
    def test_an_inherited_split_mode_the_launch_drops_keeps_the_table_resident(
        self, breakdown, monkeypatch, tensor
    ):
        """Layer launches drop an inherited split mode none and tensor launches override it: neither child is on the CPU."""
        monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "none")
        monkeypatch.setenv("LLAMA_ARG_MAIN_GPU", "-1")
        _ram(monkeypatch, 256)
        if tensor:
            _gpus(monkeypatch, (0, 22000, 24576), (1, 22000, 24576))
        assert breakdown(tensor_parallel = tensor).lookup_table_on_disk is False

    @pytest.mark.parametrize(
        "ram_gib, kw, on_disk",
        [
            (66, {}, False),
            (66, {"gpu_memory_mode": "manual", "gpu_layers": -1}, True),
            (66, {"gpu_memory_mode": "manual", "gpu_layers": 20}, True),
            (256, {"gpu_memory_mode": "manual", "gpu_layers": -1}, False),
            (256, {"gpu_memory_mode": "manual", "gpu_layers": -1, "n_ctx": 0}, True),
        ],
        ids = [
            "auto_gpu_credit",
            "manual_ram_only",
            "manual_layers_ram_only",
            "manual_fits",
            "manual_no_ctx",
        ],
    )
    def test_manual_mode_proves_residency_as_the_launch_does(
        self, breakdown, monkeypatch, ram_gib, kw, on_disk
    ):
        """Manual launches plan no cards (gpus = []), and Auto layers with no context leave it to --fit."""
        _ram(monkeypatch, ram_gib)
        assert breakdown(**kw).lookup_table_on_disk is on_disk

    def test_a_paravirtual_metal_estimate_keeps_the_table_streamed(self, breakdown, monkeypatch):
        _ram(monkeypatch, 256)
        assert breakdown().lookup_table_on_disk is False
        monkeypatch.setattr(ri, "_gpu_placement_paravirtual", lambda: True)
        assert breakdown().lookup_table_on_disk is True

    @pytest.mark.parametrize(
        "wired_gib, on_disk", [(None, False), (0, True), (200, False), (60, True)]
    )
    def test_the_apple_estimate_obeys_the_wired_headroom(
        self, breakdown, monkeypatch, wired_gib, on_disk
    ):
        _apple(monkeypatch, 120)
        _ram(monkeypatch, 256)
        _static(
            monkeypatch,
            "_apple_metal_wired_headroom_bytes",
            lambda: None if wired_gib is None else wired_gib * GIB,
        )
        assert breakdown().lookup_table_on_disk is on_disk

    def test_a_cpu_pinned_projector_is_charged_to_ram_in_the_residency_proof(
        self, breakdown, monkeypatch
    ):
        """A 96 GB card holds everything but the table, so 34 GiB of RAM keeps the table resident."""
        MM = 6 * GIB
        monkeypatch.setattr(
            ri, "_gguf_resident_file_gb", lambda cfg, **kw: (self.WEIGHTS + MM) / GIB
        )
        monkeypatch.setattr(ri, "_charged_projector_bytes", lambda *a, **k: MM)
        _gpus(monkeypatch, (0, 96000, 98304))
        _ram(monkeypatch, 34)
        assert breakdown().lookup_table_on_disk is False
        assert breakdown(llama_extra_args = ["--no-mmproj-offload"]).lookup_table_on_disk is True

    def test_ram_the_spilled_weights_crowd_out_keeps_the_table_streamed(
        self, breakdown, monkeypatch
    ):
        monkeypatch.delenv("LLAMA_ARG_CACHE_RAM", raising = False)
        _ram(monkeypatch, 58)
        _caps(monkeypatch, supports_lazy_mode = True, supports_cache_ram = True)
        assert breakdown().lookup_table_on_disk is True
        _gpus(monkeypatch, (0, 96_000, 98_304))
        assert breakdown().lookup_table_on_disk is False

    @pytest.mark.parametrize("budget_gib, ram_gib, resident", [(48, 64, False), (180, 256, True)])
    def test_apple_prices_the_whole_load_against_unified_memory(
        self, breakdown, monkeypatch, budget_gib, ram_gib, resident
    ):
        _apple(monkeypatch, budget_gib)
        _ram(monkeypatch, ram_gib)
        assert breakdown().lookup_table_on_disk is (not resident)

    @pytest.mark.parametrize(
        "indexed",
        [True, False],
        ids = [
            "an_indexed_table_name_keeps_its_override",
            "a_device_override_keeps_the_table_streamed_as_the_launch_does",
        ],
    )
    def test_an_indexed_table_name_keeps_its_override(self, breakdown, monkeypatch, indexed):
        _ram(monkeypatch, 256)
        name = "per_layer_token_embd_0" if indexed else "per_layer_token_embd"
        if indexed:
            real = ri.LlamaCppBackend._gguf_tensor_scan

            def scan(self, path):
                arch, named, experts = real(self, path)
                return (
                    arch,
                    {k.replace(PLE, f"{name}.weight"): v for k, v in named.items()},
                    experts,
                )

            monkeypatch.setattr(ri.LlamaCppBackend, "_gguf_tensor_scan", scan)
        assert breakdown(llama_extra_args = ["-ot", f"{name}=CUDA0"]).lookup_table_on_disk is True

    def test_a_device_override_charges_a_resident_table_wholly_to_the_gpu(
        self, breakdown, tmp_path
    ):
        """A layer fraction does not apply to a tensor an -ot pins."""
        b = breakdown(
            gguf = _gemma(tmp_path),
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            llama_extra_args = ["-ot", "per_layer_token_embd=CUDA0"],
        )
        assert b.lookup_table_bytes == 0
        assert b.gpu_bytes >= GEMMA_TABLE

    @pytest.mark.parametrize(
        "igpu",
        [False, True],
        ids = [
            "a_build_without_lazy_mode_keeps_the_table_resident",
            "an_integrated_gpu_reads_the_table_resident",
        ],
    )
    def test_a_build_without_lazy_mode_keeps_the_table_resident(self, breakdown, monkeypatch, igpu):
        """A Vulkan iGPU has no mmap support, so llama.cpp resolves auto to off."""
        if igpu:
            _static(monkeypatch, "_is_vulkan_backend", lambda b = None: True)
            _gpus(monkeypatch, (0, 30000, 0))
        else:
            _caps(monkeypatch)
        b = breakdown()
        assert b.lookup_table_on_disk is False
        assert b.weights_bytes == int(round(self.WEIGHTS))

    def test_a_pin_on_the_integrated_gpu_reads_the_table_resident(self, breakdown, monkeypatch):
        """A mixed host streams by default (llama.cpp drops the iGPU), not when pinned to it."""
        _static(monkeypatch, "_is_vulkan_backend", lambda b = None: True)
        _gpus(monkeypatch, (0, 30000, 0), (1, 22000, 24576))
        assert breakdown().lookup_table_on_disk is True
        assert breakdown(selected_gpu_ids = [0]).lookup_table_on_disk is False
        assert breakdown(selected_gpu_ids = [1]).lookup_table_on_disk is True
        stale = breakdown(selected_gpu_ids = [0], llama_extra_args = ["--device", "Vulkan1"])
        assert stale.lookup_table_on_disk is False

    def test_an_explicit_lazy_mode_on_prices_every_flagged_table(self, breakdown, monkeypatch):
        modes = []
        real = glt.local_lookup_info
        monkeypatch.setattr(
            glt,
            "local_lookup_info",
            lambda path, mode = "auto": modes.append(mode) or real(path, mode),
        )
        breakdown(llama_extra_args = ["-lzm", "on"])
        breakdown()
        assert modes == ["on", "auto"]

    def test_a_resident_gemma3n_table_leaves_the_gpu_share_not_the_total(self, breakdown, tmp_path):
        """Not lazily read, but still a host-pinned input like the launcher prices it."""
        b = breakdown(gguf = _gemma(tmp_path))
        assert (b.lookup_table_bytes, b.lookup_table_on_disk) == (GEMMA_TABLE, False)
        assert b.weights_bytes == int(round(self.WEIGHTS))
        assert b.gpu_bytes == b.total_bytes - b.lookup_table_bytes

    def test_a_resident_table_moved_to_the_gpu_stays_in_the_gpu_share(self, breakdown):
        b = breakdown(llama_extra_args = ["-lzm", "off", "-ot", "per_layer_token_embd=CUDA0"])
        assert (b.lookup_table_bytes, b.lookup_table_on_disk) == (0, False)
        assert b.gpu_bytes == b.total_bytes

    def test_the_load_panel_response_carries_the_table(self, breakdown):
        from core.inference.memory_contract import (
            build_memory_estimate,
            project_estimate_memory_response,
        )
        from models.inference import EstimateMemoryResponse

        est = build_memory_estimate(breakdown(), quant_file_bytes = int(self.WEIGHTS))
        out = project_estimate_memory_response(est)
        assert (out["lookup_table_bytes"], out["lookup_table_on_disk"]) == (FLASH_NEXT_TABLE, True)
        assert EstimateMemoryResponse(**out).lookup_table_bytes == FLASH_NEXT_TABLE

    def test_the_memory_bar_response_carries_the_table(self, breakdown):
        """Its weights_bytes is the whole file, so the bar needs the table to take it off the VRAM segment."""
        from core.inference.memory_contract import build_memory_estimate, project_kv_cache_estimate

        est = build_memory_estimate(breakdown(), quant_file_bytes = int(self.WEIGHTS))
        out = project_kv_cache_estimate(est)
        assert (out["weights_bytes"], out["lookup_table_bytes"]) == (
            int(self.WEIGHTS),
            FLASH_NEXT_TABLE,
        )


def _split_blobs(first_arch = "qwen4exp", first_embd = ("token_embd.weight", [2048, 151936], IQ4_NL)):
    names = [f"Q4/m-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    headers = [
        _header(first_arch, [first_embd], _ATTN),
        _header("", [(PLE, [160, 320001536], IQ4_NL)]),
        _header("", [("blk.3.attn_q.weight", [256, 256], F16)]),
    ]
    return names, {n: (h, 40 * GIB) for n, h in zip(names, headers)}


class TestHubDiscovery:
    # Engram stored in a quant type gguf predates (ggml type 43); the next tensor bounds it.
    UNKNOWN_TYPE = 43

    def test_an_unknown_quant_type_is_sized_from_the_next_offset(self, tmp_path):
        path = _write(
            tmp_path / "dsv41-unknown.gguf",
            _header(
                "deepseek41",
                [
                    (PLE, [256, 768022850], self.UNKNOWN_TYPE),
                    ("blk.0.attn_q.weight", [256, 256], F16),
                ],
                offsets = [0, DSV41_TABLE],
            ),
        )
        assert glt.local_lookup_info(path) == ("deepseek41", DSV41_TABLE)

    def test_an_unknown_quant_type_last_in_the_file_is_sized_from_the_file(self, tmp_path):
        header = _header(
            "deepseek41",
            [_DSV41_EMBD, (PLE, [256, 768022850], self.UNKNOWN_TYPE)],
            offsets = [0, 4096],
        )
        size = -(-len(header) // 32) * 32 + 4096 + DSV41_TABLE
        path = tmp_path / "dsv41-last.gguf"
        with open(path, "wb") as f:
            f.write(header)
            f.truncate(size)
        assert glt.local_lookup_info(str(path)) == ("deepseek41", DSV41_TABLE)
        got = glt.remote_lookup_info(
            "o/unknown-type",
            "m.gguf",
            identity = size,
            fetcher_fn = _blob_fetcher({"m.gguf": (header, size)}, []),
        )
        assert got == ("deepseek41", DSV41_TABLE)

    def test_local_discovery_reads_every_shard(self, tmp_path):
        first = _write(tmp_path / "big-00001-of-00010.gguf", _header("deepseek41", [_DSV41_EMBD]))
        for i in range(2, 10):
            _write(
                tmp_path / f"big-{i:05d}-of-00010.gguf",
                _header("", [(f"blk.{i}.attn_q.weight", [256, 256], F16)]),
            )
        _write(tmp_path / "big-00010-of-00010.gguf", _header("", [_DSV41_PLE]))
        assert glt.local_lookup_table_bytes(first) == DSV41_TABLE

    def test_a_later_split_shard_starts_with_a_small_range(self):
        names, blobs = _split_blobs()
        calls = []
        got = glt.remote_lookup_info(
            "o/small-range", names[0], identity = 1, fetcher_fn = _blob_fetcher(blobs, calls)
        )
        assert got == ("qwen4exp", FLASH_NEXT_TABLE)
        later = [end + 1 for name, _start, end in calls if name != names[0]]
        assert later and max(later) <= 64 << 10, calls

    def test_a_known_architecture_reads_the_later_shards_before_the_tokenizer(self):
        names, blobs = _split_blobs()
        calls = []
        got = glt.remote_lookup_info(
            "o/known-arch",
            names[0],
            identity = 1,
            fetcher_fn = _blob_fetcher(blobs, calls),
            known_arch = "qwen4exp",
        )
        assert got == ("qwen4exp", FLASH_NEXT_TABLE)
        # The table's shard, then only the first shard's metadata for its architecture.
        assert [name for name, _s, _e in calls] == [names[1], names[0]]
        assert calls[1][2] + 1 <= 64 << 10, calls

    def test_the_listing_forwards_the_known_architecture(self, monkeypatch):
        seen = []
        _arch(monkeypatch, True)
        _remote(monkeypatch, lambda *a, **k: seen.append(k.get("known_arch")) or ("qwen4exp", 1))
        _list([_row("Q4/m.gguf")], "o/forward")
        assert seen == ["qwen4exp"]

    def test_auto_mode_skips_files_too_small_to_hold_a_streamed_table(self, monkeypatch):
        read_names = []

        def read(repo, name, *a, **k):
            read_names.append(name)
            return ("gemma4", 5 * GIB if name == "BF16.gguf" else 0)

        _arch(monkeypatch, True, "gemma4")
        _remote(monkeypatch, read)
        rows = [_row("Q4_K_M.gguf", "Q4_K_M", 3 * GIB), _row("BF16.gguf", "BF16", 15 * GIB)]
        out = _list(rows, "o/small-files")
        assert read_names == ["BF16.gguf"]
        assert _tables(out) == [0, 5 * GIB]

    @pytest.mark.parametrize(
        "repos",
        [("o/shared-read", "o/shared-read"), ("O/Cased-GGUF", "o/cased-gguf")],
        ids = [
            "a_listing_shares_a_table_read_already_in_flight",
            "differently_cased_listings_share_one_header_read",
        ],
    )
    def test_a_listing_shares_a_table_read_already_in_flight(self, monkeypatch, repos):
        calls = []
        monkeypatch.setattr(
            gguf_header_delta,
            "hub_range_fetcher",
            _blob_fetcher({"Q4/shared.gguf": (_LAZY_BLOB, 40 * GIB)}, calls, delay = 0.3),
        )
        _arch(monkeypatch, True)
        glt._REMOTE_CACHE.clear()
        # Room for a cold gguf import under a loaded runner; the overlap is what is tested.
        monkeypatch.setattr(gv, "_LOOKUP_DEADLINE_S", 10.0)
        rows = [_row("Q4/shared.gguf")]
        outs = _overlap(_listing(rows, repos[0]), _listing(rows, repos[1]))
        assert [_tables(o) for o in outs] == [[FLASH_NEXT_TABLE]] * 2
        assert len(calls) == 1

    def test_a_listing_shares_an_architecture_lookup_already_in_flight(self, monkeypatch):
        cache: dict = {}
        lock = threading.Lock()
        asked = []

        def arch(repo, token):
            with lock:
                if repo in cache:
                    return cache[repo]
            asked.append(repo)
            time.sleep(0.3)
            with lock:
                cache[repo] = "qwen4exp"
            return "qwen4exp"

        _arch(monkeypatch, False, None)
        monkeypatch.setattr(gv, "_hub_repo_architecture", arch)
        _remote(monkeypatch)
        monkeypatch.setattr(gv, "_LOOKUP_DEADLINE_S", 10.0)
        rows = [_row()]
        outs = _overlap(
            _listing(rows, "o/shared-arch", token = "hf_same"),
            _listing(rows, "o/shared-arch", token = "hf_same"),
        )
        assert [_tables(o) for o in outs] == [[FLASH_NEXT_TABLE]] * 2
        assert asked == ["o/shared-arch"]


class TestListingFollowups:
    def test_a_repo_hint_never_names_another_familys_split_variant(self):
        names, blobs = _split_blobs("gemma3n", ("token_embd.weight", [2048, 262144], IQ4_NL))
        got = glt.remote_lookup_info(
            "o/mixed-families",
            names[0],
            identity = 1,
            fetcher_fn = _blob_fetcher(blobs, []),
            known_arch = "qwen4exp",
        )
        assert got == ("gemma3n", 0)

    def test_the_remote_cache_ignores_repo_id_case(self):
        fetcher = _blob_fetcher({"Q4/case.gguf": (_LAZY_BLOB, 40 * GIB)}, [])
        glt.remote_lookup_info("Unsloth/Case-GGUF", "Q4/case.gguf", identity = 3, fetcher_fn = fetcher)
        assert glt.remote_cached("unsloth/case-gguf", "Q4/case.gguf", None, 3) == (
            "qwen4exp",
            FLASH_NEXT_TABLE,
        )

    def test_nested_metadata_arrays_are_skipped(self):
        # Splice one array-of-uint32-arrays KV in after the header's counts.
        n_tensors, n_kv = struct.unpack_from("<QQ", _LAZY_BLOB, 8)
        key = b"x.nested"
        inner = struct.pack("<IQ", 4, 3) + struct.pack("<3I", 1, 2, 3)
        kv = struct.pack("<Q", len(key)) + key + struct.pack("<IIQ", 9, 9, 2) + inner * 2
        spliced = _LAZY_BLOB[:8] + struct.pack("<QQ", n_tensors, n_kv + 1) + kv + _LAZY_BLOB[24:]
        assert glt.parse_header(spliced) == ("qwen4exp", {PLE: FLASH_NEXT_TABLE})

    def test_a_late_release_keeps_a_newer_claim(self):
        key = ("table", "o/owner")
        _held, first = gv._claim_lookup(key)
        gv._release_lookup(key, first)
        _held, second = gv._claim_lookup(key)
        gv._release_lookup(key, first)  # the cancelled worker, running late
        assert gv._lookup_inflight.get(key) is second
        gv._release_lookup(key, second)
        assert key not in gv._lookup_inflight

    def test_a_listing_retries_a_read_that_failed_while_it_waited(self, monkeypatch):
        attempts = []
        lock = threading.Lock()
        reading = threading.Event()

        def read(repo, name, *a, **k):
            with lock:
                attempts.append(name)
                first = len(attempts) == 1
            reading.set()
            time.sleep(0.3)
            if first:
                raise OSError("transient")
            return ("qwen4exp", FLASH_NEXT_TABLE)

        _arch(monkeypatch, True)
        _remote(monkeypatch, read)
        monkeypatch.setattr(gv, "_LOOKUP_DEADLINE_S", 10.0)
        rows = [_row("Q4/retry.gguf")]
        outs = _overlap(_listing(rows, "o/retry"), _listing(rows, "o/retry"), wait = reading)
        assert [_tables(o) for o in outs] == [[0], [FLASH_NEXT_TABLE]]
        assert len(attempts) == 2

    def test_an_unreadable_local_copy_falls_back_to_the_hub_header(self, tmp_path, monkeypatch):
        """The first shard is on disk, the second still downloading."""
        first = _write(
            tmp_path / "m-00001-of-00002.gguf",
            _header("qwen4exp", [("token_embd.weight", [2048, 151936], IQ4_NL)], _ATTN),
        )
        monkeypatch.setattr(gv, "_local_variant_gguf", lambda *a, **k: first)
        _arch(monkeypatch, True)
        _remote(monkeypatch)
        out = _list([_row("m-00001-of-00002.gguf")], "o/partial")
        assert _tables(out) == [FLASH_NEXT_TABLE]

    def test_a_host_that_reads_tables_resident_reads_no_header(self, monkeypatch):
        looked = []
        monkeypatch.setattr(gv, "_host_streams_lookup_tables", lambda: False)
        monkeypatch.setattr(gv, "_local_variant_gguf", lambda *a, **k: looked.append(a) or None)
        answer = _answer([_row("Q4/m.gguf")])
        assert _list(answer, "o/resident") is answer and looked == []

    def test_a_cpu_build_streams_the_tables_beside_an_empty_gpu_inventory(self, monkeypatch):
        """No model device on a CPU-only build, so lazy auto stays on whatever the GPU probe says."""
        backend = _stream_backend(monkeypatch, vulkan = False, gpus = [])
        monkeypatch.setattr(lc, "_gpu_device_node_present", lambda: True)
        _no_gpu_backend(monkeypatch, True)
        assert backend.lookup_tables_stream_by_default(None, {}) is True
        _no_gpu_backend(monkeypatch, False)
        assert backend.lookup_tables_stream_by_default(["--device", "none"], {}) is True
        assert backend.lookup_tables_stream_by_default(None, {}) is False
        cpu_split = ["--split-mode", "none", "--main-gpu", "-1"]
        assert backend.lookup_tables_stream_by_default(cpu_split, {}) is True
        # Not under tensor parallel: Studio's --split-mode tensor beats the env.
        assert (
            backend.lookup_tables_stream_by_default(
                ["--main-gpu", "-1"], {"LLAMA_ARG_SPLIT_MODE": "none"}, tensor_parallel = True
            )
            is False
        )

    @pytest.mark.parametrize(
        "env",
        [
            {},
            {"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_MAIN_GPU": "-1"},
            {"LLAMA_ARG_SPLIT_MODE": "row", "LLAMA_ARG_MAIN_GPU": "-1"},
        ],
        ids = ["clean", "inherited_none", "inherited_row"],
    )
    def test_a_listing_drops_the_split_mode_the_launch_drops(self, monkeypatch, env):
        """A layer launch clears an inherited non-layer split mode, so the child runs on the shared Vulkan iGPU."""
        backend = _stream_backend(monkeypatch, vulkan = True, gpus = [(0, 8000, 0)])
        _no_gpu_backend(monkeypatch, False)
        assert backend.lookup_tables_stream_by_default(None, env) is False

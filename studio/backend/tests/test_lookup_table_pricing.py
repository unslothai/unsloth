# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lookup-only tables (per-layer embeddings / Engram) priced apart in the listing and the panel.

llama.cpp reads a TENSOR_READ_LAZY table over 4 GiB lazily under its default ``--lazy-mode
auto``: it stays mmap'd from disk and only the rows a token touches page in. The model listing
reports those bytes per variant (``lookup_table_bytes``) and the Load Model estimate keeps them
out of the GPU share, and out of the resident total while they stream.

The headers are hand-built with the real tensor shapes and quant types read off the published
GGUFs: DeepSeek-V4.1-Flash UD-IQ1_S (``per_layer_token_embd`` IQ4_NL [256, 768022850], 103.00
GiB), Qwen3.8-Flash-Next UD-Q2_K_XL (arch qwen4exp, IQ4_NL [160, 320001536], 26.82 GiB) and
gemma-3n-E4B Q4_K_M (Q5_1 [8960, 262144], 1.64 GiB, not lazily read upstream). Only the header
is written, so a 103 GiB table costs a few hundred bytes on disk.
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Installs the process-wide loggers/structlog/httpx stubs the route module needs.
import test_kv_cache_estimation  # noqa: E402,F401

GIB = 1024**3
IQ4_NL, Q5_1, F16 = 20, 7, 1

DSV41_TABLE = 256 * 768022850 // 32 * 18
FLASH_NEXT_TABLE = 160 * 320001536 // 32 * 18

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
    for name, dims, ggml_type in tensors:
        raw = name.encode()
        buf.write(struct.pack("<Q", len(raw)) + raw)
        buf.write(struct.pack("<I", len(dims)))
        buf.write(struct.pack(f"<{len(dims)}Q", *dims))
        buf.write(struct.pack("<IQ", ggml_type, offset))
        offset += 1024
    return buf.getvalue()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(data)
    return str(path)


def _dsv41(tmp_path: Path, name = "DeepSeek-V4.1-Flash-UD-IQ1_S.gguf") -> str:
    return _write(
        tmp_path / name,
        _header(
            "deepseek41",
            [
                ("token_embd.weight", [4096, 129280], IQ4_NL),
                ("per_layer_token_embd.weight", [256, 768022850], IQ4_NL),
            ],
        ),
    )


def _flash_next(tmp_path: Path, name = "Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf") -> str:
    return _write(
        tmp_path / name,
        _header(
            "qwen4exp",
            [
                ("token_embd.weight", [2048, 151936], IQ4_NL),
                ("per_layer_token_embd.weight", [160, 320001536], IQ4_NL),
                ("blk.0.attn_q.weight", [256, 256], F16),
            ],
            _ATTN,
        ),
    )


def _gemma3n(tmp_path: Path) -> str:
    return _write(
        tmp_path / "gemma-3n-E4B-it-Q4_K_M.gguf",
        _header(
            "gemma3n",
            [
                ("token_embd.weight", [2048, 262144], IQ4_NL),
                ("per_layer_token_embd.weight", [8960, 262144], Q5_1),
            ],
        ),
    )


# A. The header scan


def test_split_names_keep_the_extension_as_written():
    from hub.utils.gguf_lookup_tables import shard_names
    assert shard_names("d/M-00001-of-00002.GGUF") == [
        "d/M-00001-of-00002.GGUF",
        "d/M-00002-of-00002.GGUF",
    ]


class TestHeaderScan:
    def test_real_table_sizes(self, tmp_path):
        from hub.utils.gguf_lookup_tables import local_lookup_info

        assert local_lookup_info(_dsv41(tmp_path)) == ("deepseek41", DSV41_TABLE)
        assert DSV41_TABLE / GIB == pytest.approx(103.00, abs = 0.01)
        assert local_lookup_info(_flash_next(tmp_path)) == ("qwen4exp", FLASH_NEXT_TABLE)
        assert FLASH_NEXT_TABLE / GIB == pytest.approx(26.82, abs = 0.01)

    def test_gemma3n_is_not_lazily_read(self, tmp_path):
        # Same tensor name, but gemma3n does not flag it TENSOR_READ_LAZY (and it is under
        # 4 GiB anyway): it is priced as weights, as before.
        from hub.utils.gguf_lookup_tables import local_lookup_info
        assert local_lookup_info(_gemma3n(tmp_path)) == ("gemma3n", 0)

    def test_a_small_table_stays_resident_under_auto(self, tmp_path):
        from hub.utils.gguf_lookup_tables import local_lookup_table_bytes
        small = _write(
            tmp_path / "gemma4-E2B.gguf",
            _header("gemma4", [("per_layer_token_embd.weight", [8960, 262144], Q5_1)]),
        )
        assert local_lookup_table_bytes(small) == 0

    def test_an_explicit_lazy_mode_on_reads_a_small_table_lazily(self, tmp_path):
        from hub.utils.gguf_lookup_tables import local_lookup_info

        small = _write(
            tmp_path / "gemma4-E2B.gguf",
            _header("gemma4", [("per_layer_token_embd.weight", [8960, 262144], Q5_1)]),
        )
        assert local_lookup_info(small) == ("gemma4", 0)
        assert local_lookup_info(small, "on") == ("gemma4", 8960 * 262144 // 32 * 24)

    def test_split_file_finds_the_table_in_a_later_shard(self, tmp_path):
        from hub.utils.gguf_lookup_tables import local_lookup_table_bytes

        first = _write(
            tmp_path / "m-00001-of-00002.gguf",
            _header("deepseek41", [("token_embd.weight", [4096, 129280], IQ4_NL)]),
        )
        _write(
            tmp_path / "m-00002-of-00002.gguf",
            _header("", [("per_layer_token_embd.weight", [256, 768022850], IQ4_NL)]),
        )
        assert local_lookup_table_bytes(first) == DSV41_TABLE

    def test_a_missing_shard_or_garbage_is_zero(self, tmp_path):
        from hub.utils.gguf_lookup_tables import local_lookup_table_bytes

        first = _write(
            tmp_path / "x-00001-of-00002.gguf",
            _header("deepseek41", [("token_embd.weight", [4096, 129280], IQ4_NL)]),
        )
        assert local_lookup_table_bytes(first) == 0
        assert local_lookup_table_bytes(_write(tmp_path / "junk.gguf", b"not a gguf")) == 0
        assert local_lookup_table_bytes(None) == 0

    def test_remote_reads_only_the_header_range(self, tmp_path):
        from hub.utils import gguf_lookup_tables as glt

        blob = _header(
            "qwen4exp", [("per_layer_token_embd.weight", [160, 320001536], IQ4_NL)], _ATTN
        )
        calls = []

        def fetcher(
            repo_id,
            name,
            token,
            revision = None,
        ):
            def fetch(start, end):
                calls.append((name, start, end))
                return blob[start : end + 1]

            return fetch

        got = glt.remote_lookup_info(
            "unsloth/Flash-Next-GGUF", "Q2/flash-Q2.gguf", identity = 1, fetcher_fn = fetcher
        )
        assert got == ("qwen4exp", FLASH_NEXT_TABLE)
        assert calls and all(start == 0 for _name, start, _end in calls)
        # Cached: no second read for the same file identity.
        calls.clear()
        assert glt.remote_lookup_info(
            "unsloth/Flash-Next-GGUF", "Q2/flash-Q2.gguf", identity = 1, fetcher_fn = fetcher
        ) == ("qwen4exp", FLASH_NEXT_TABLE)
        assert not calls
        # The listed size is not content identity: an expired answer is read again.
        glt._REMOTE_CACHE.clear()
        glt._REMOTE_TTL_S, ttl = -1.0, glt._REMOTE_TTL_S
        try:
            glt.remote_lookup_info(
                "unsloth/Flash-Next-GGUF", "Q2/flash-Q2.gguf", identity = 1, fetcher_fn = fetcher
            )
            assert glt.remote_cached("unsloth/Flash-Next-GGUF", "Q2/flash-Q2.gguf", None, 1) is None
        finally:
            glt._REMOTE_TTL_S = ttl
        assert calls

    def test_remote_failure_is_zero_and_not_cached(self):
        from hub.utils import gguf_lookup_tables as glt

        def broken(
            repo_id,
            name,
            token,
            revision = None,
        ):
            def fetch(start, end):
                raise OSError("network down")

            return fetch

        assert glt.remote_lookup_info("o/r", "f.gguf", identity = 7, fetcher_fn = broken) == (
            None,
            0,
        )
        assert glt.remote_cached("o/r", "f.gguf", None, 7) is None


# B. The listing


def _answer(variants, sources = None):
    from hub.schemas.inventory import GgufVariantsResponse
    from hub.services.models.gguf_variants import VariantsAnswer
    return VariantsAnswer(
        GgufVariantsResponse(repo_id = "unsloth/X-GGUF", variants = variants),
        None,
        variant_context_sources = sources or {},
    )


class TestListing:
    @pytest.fixture(autouse = True)
    def _streams_here(self, monkeypatch):
        from hub.services.models import gguf_variants as gv
        monkeypatch.setattr(gv, "_host_streams_lookup_tables", lambda: True)

    def test_a_host_that_reads_the_tables_resident_lists_no_discount(self, tmp_path, monkeypatch):
        """An iGPU / APU resolves lazy auto to off, so the listing prices the whole file."""
        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv

        monkeypatch.setattr(gv, "_host_streams_lookup_tables", lambda: False)
        path = _dsv41(tmp_path)
        answer = _answer(
            [
                GgufVariantDetail(
                    filename = os.path.basename(path),
                    quant = "UD-IQ1_S",
                    size_bytes = 216 * GIB,
                    downloaded = True,
                )
            ],
            {"ud-iq1_s": path},
        )
        out = asyncio.run(
            gv._with_lookup_tables(
                answer, "unsloth/X-GGUF", hf_token = None, offline = True, prefer_local_cache = False
            )
        )
        assert out is answer

    def test_both_variant_models_carry_the_field(self):
        from hub.schemas.inventory import GgufVariantDetail as HubDetail
        from models.models import GgufVariantDetail as RouteDetail
        for model in (HubDetail, RouteDetail):
            assert model.model_fields["lookup_table_bytes"].default == 0

    def test_a_downloaded_variant_is_read_from_its_own_header(self, tmp_path, monkeypatch):
        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv

        path = _dsv41(tmp_path)
        # A local answer never asks the Hub.
        monkeypatch.setattr(
            gv, "_hub_repo_architecture", lambda *a, **k: pytest.fail("no Hub call")
        )
        answer = _answer(
            [
                GgufVariantDetail(
                    filename = os.path.basename(path),
                    quant = "UD-IQ1_S",
                    size_bytes = 216 * GIB,
                    downloaded = True,
                )
            ],
            {"ud-iq1_s": path},
        )
        out = asyncio.run(
            gv._with_lookup_tables(
                answer,
                "unsloth/X-GGUF",
                hf_token = None,
                offline = True,
                prefer_local_cache = False,
            )
        )
        assert out.response.variants[0].lookup_table_bytes == DSV41_TABLE
        # The shared answer is not mutated in place.
        assert answer.response.variants[0].lookup_table_bytes == 0

    def test_hub_only_quants_of_a_lazy_arch_are_range_read(self, monkeypatch):
        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv
        from hub.utils import gguf_lookup_tables as glt

        # The repo's architecture is already known (an earlier listing fetched it).
        monkeypatch.setattr(gv, "_cached_hub_repo_architecture", lambda *a, **k: (True, "qwen4exp"))
        monkeypatch.setattr(
            glt,
            "remote_lookup_info",
            lambda repo, name, token, identity = None, **k: ("qwen4exp", FLASH_NEXT_TABLE),
        )
        rows = [
            GgufVariantDetail(filename = "Q2/a-00001-of-00003.gguf", quant = "Q2", size_bytes = 1),
            GgufVariantDetail(filename = "README.md", quant = "x", size_bytes = 1),
        ]
        out = asyncio.run(
            gv._with_lookup_tables(
                _answer(rows),
                "unsloth/Flash-Next-GGUF",
                hf_token = None,
                offline = False,
                prefer_local_cache = False,
            )
        )
        assert [v.lookup_table_bytes for v in out.response.variants] == [FLASH_NEXT_TABLE, 0]

    def test_a_small_repo_never_asks_the_hub_for_its_architecture(self, monkeypatch):
        """No row can hold a streamed table (each is under 4 GiB): no Hub round trip."""
        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv

        monkeypatch.setattr(gv, "_cached_hub_repo_architecture", lambda *a, **k: (False, None))
        monkeypatch.setattr(
            gv, "_hub_repo_architecture", lambda *a, **k: pytest.fail("no Hub call")
        )
        answer = _answer([GgufVariantDetail(filename = "a.gguf", quant = "Q2", size_bytes = GIB)])
        out = asyncio.run(
            gv._with_lookup_tables(
                answer, "o/small", hf_token = None, offline = False, prefer_local_cache = False
            )
        )
        assert out is answer

    def test_a_cold_architecture_is_priced_on_the_first_listing(self, monkeypatch):
        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv
        from hub.utils import gguf_lookup_tables as glt

        fetched = []

        def arch(repo, token):
            fetched.append(repo)
            return "qwen4exp"

        monkeypatch.setattr(gv, "_hub_repo_architecture", arch)
        monkeypatch.setattr(gv, "_cached_hub_repo_architecture", lambda *a, **k: (False, None))
        monkeypatch.setattr(
            glt,
            "remote_lookup_info",
            lambda repo, name, token, identity = None, **k: ("qwen4exp", FLASH_NEXT_TABLE),
        )
        rows = [GgufVariantDetail(filename = "a.gguf", quant = "Q2", size_bytes = 40 * GIB)]
        out = asyncio.run(
            gv._with_lookup_tables(
                _answer(rows), "o/cold", hf_token = None, offline = False, prefer_local_cache = False
            )
        )
        assert [v.lookup_table_bytes for v in out.response.variants] == [FLASH_NEXT_TABLE]
        assert fetched == ["o/cold"]

    def test_a_slow_architecture_lookup_is_bounded_by_the_deadline(self, monkeypatch):
        """A late answer never holds the listing past its deadline, and is asked once."""
        import threading
        import time

        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv
        from hub.utils import gguf_lookup_tables as glt

        release = threading.Event()
        fetched = []

        def slow_arch(repo, token):
            fetched.append(repo)
            release.wait(5)
            return "qwen4exp"

        monkeypatch.setattr(gv, "_LOOKUP_DEADLINE_S", 0.2)
        monkeypatch.setattr(gv, "_hub_repo_architecture", slow_arch)
        monkeypatch.setattr(gv, "_cached_hub_repo_architecture", lambda *a, **k: (False, None))
        monkeypatch.setattr(
            glt, "remote_lookup_info", lambda *a, **k: pytest.fail("no header read yet")
        )
        rows = [GgufVariantDetail(filename = "a.gguf", quant = "Q2", size_bytes = 40 * GIB)]
        answer = _answer(rows)
        try:
            for _ in range(3):
                started = time.monotonic()
                out = asyncio.run(
                    gv._with_lookup_tables(
                        answer, "o/slow", hf_token = None, offline = False, prefer_local_cache = False
                    )
                )
                assert out is answer
                assert time.monotonic() - started < 2.0
        finally:
            release.set()
        for _ in range(50):
            if not gv._lookup_inflight:
                break
            time.sleep(0.05)
        assert fetched == ["o/slow"]

    def test_a_direct_file_source_finds_its_row_beside_it(self, tmp_path):
        from types import SimpleNamespace

        from hub.services.models import gguf_variants as gv

        model = tmp_path / "Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf"
        model.write_bytes(b"GGUF")
        row = SimpleNamespace(filename = model.name, quant = "UD-Q2_K_XL", downloaded = True)
        for answer, repo in (
            (SimpleNamespace(variant_context_sources = None, context_source = str(model)), "o/r"),
            (SimpleNamespace(variant_context_sources = None, context_source = None), str(model)),
        ):
            assert gv._local_variant_gguf(answer, repo, row) == str(model)

    def test_a_read_the_pool_refuses_after_the_deadline_releases_its_key(self, monkeypatch):
        """Late refusals (the pool shut down at exit) never reach the read's own finally."""
        from hub.services.models import gguf_variants as gv

        async def refused(fn, /, *args):
            await asyncio.sleep(0.01)
            raise RuntimeError("cannot schedule new futures after shutdown")

        monkeypatch.setattr(gv, "_lookup_thread", refused)
        key = ("arch", "o/refused")

        async def go():
            loop = asyncio.get_running_loop()
            assert await gv._lookup_await_once(key, loop.time(), lambda: 1) is None
            assert key in gv._lookup_inflight  # still queued at the deadline
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert key not in gv._lookup_inflight

    def test_a_cached_non_lazy_variant_does_not_veto_the_rest_of_the_repo(
        self, tmp_path, monkeypatch
    ):
        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv
        from hub.utils import gguf_lookup_tables as glt

        local = _write(
            tmp_path / "llama.gguf",
            _header("llama", [("token_embd.weight", [256, 1000], F16)], _ATTN),
        )
        monkeypatch.setattr(gv, "_cached_hub_repo_architecture", lambda *a, **k: (True, "qwen4exp"))
        monkeypatch.setattr(
            glt,
            "remote_lookup_info",
            lambda repo, name, token, identity = None, **k: ("qwen4exp", FLASH_NEXT_TABLE),
        )
        rows = [
            GgufVariantDetail(filename = "llama.gguf", quant = "Q8_0", size_bytes = 1, downloaded = True),
            GgufVariantDetail(filename = "flash/f.gguf", quant = "Q2", size_bytes = 40 * GIB),
        ]
        out = asyncio.run(
            gv._with_lookup_tables(
                _answer(rows, {"q8_0": local}),
                "o/mixed",
                hf_token = None,
                offline = False,
                prefer_local_cache = False,
            )
        )
        assert [v.lookup_table_bytes for v in out.response.variants] == [0, FLASH_NEXT_TABLE]

    def test_one_listing_reads_at_most_a_pool_of_headers(self, monkeypatch):
        import threading

        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv
        from hub.utils import gguf_lookup_tables as glt

        calls = []
        lock = threading.Lock()

        def read(
            repo,
            name,
            token,
            identity = None,
            **k,
        ):
            with lock:
                calls.append(name)
            return ("qwen4exp", FLASH_NEXT_TABLE)

        monkeypatch.setattr(gv, "_cached_hub_repo_architecture", lambda *a, **k: (True, "qwen4exp"))
        monkeypatch.setattr(glt, "remote_lookup_info", read)
        rows = [
            GgufVariantDetail(filename = f"q{i}/m.gguf", quant = f"Q{i}", size_bytes = 40 * GIB + i)
            for i in range(10)
        ]
        out = asyncio.run(
            gv._with_lookup_tables(
                _answer(rows), "o/many", hf_token = None, offline = False, prefer_local_cache = False
            )
        )
        assert len(calls) == gv._LOOKUP_CONCURRENCY
        assert sum(1 for v in out.response.variants if v.lookup_table_bytes) == len(calls)

    def test_offline_or_a_non_lazy_arch_never_reads_the_hub(self, monkeypatch):
        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv
        from hub.utils import gguf_lookup_tables as glt

        monkeypatch.setattr(
            glt, "remote_lookup_info", lambda *a, **k: pytest.fail("no header read")
        )
        rows = [GgufVariantDetail(filename = "a.gguf", quant = "Q4_K_M", size_bytes = 1)]
        answer = _answer(rows)
        kwargs = dict(hf_token = None, prefer_local_cache = False)
        assert asyncio.run(gv._with_lookup_tables(answer, "o/r", offline = True, **kwargs)) is answer
        monkeypatch.setattr(gv, "_hub_repo_architecture", lambda *a, **k: "llama")
        assert asyncio.run(gv._with_lookup_tables(answer, "o/r", offline = False, **kwargs)) is answer

    def test_a_failure_returns_the_listing_unchanged(self, monkeypatch):
        from hub.schemas.inventory import GgufVariantDetail
        from hub.services.models import gguf_variants as gv

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(gv, "_local_variant_gguf", boom)
        answer = _answer([GgufVariantDetail(filename = "a.gguf", quant = "Q", size_bytes = 1)])
        assert (
            asyncio.run(
                gv._with_lookup_tables(
                    answer, "o/r", hf_token = None, offline = False, prefer_local_cache = False
                )
            )
            is answer
        )


# C. The Load Model estimate


class TestEstimate:
    WEIGHTS = 73.45 * GIB

    @pytest.fixture
    def breakdown(self, tmp_path, monkeypatch):
        import routes.inference as ri

        path = _flash_next(tmp_path)
        monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: self.WEIGHTS / GIB)
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_get_gguf_size_bytes",
            staticmethod(lambda p: int(self.WEIGHTS)),
        )
        # A build with the lazy mode, as the Studio prebuilt is, on one discrete card.
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, binary = None: {"found": True, "supports_lazy_mode": True}),
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False)
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None, **k: [(0, 22000, 24576)]),
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_integrated_cuda_gpu_ids", staticmethod(lambda: set())
        )
        # A discrete Linux host on every runner: a macOS runner would take the Apple path.
        import utils.hardware as hw

        monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
        )
        # 16 GiB of free RAM cannot hold the 26.82 GiB table: it streams.
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 16 * 1024)
        )
        ri._estimate_files_cache.clear()
        ri._estimate_config_cache.clear()
        config = SimpleNamespace(
            identifier = "local/flash-next",
            gguf_file = path,
            is_gguf = True,
            gguf_mmproj_file = None,
        )

        def run(gguf = None, **kw):
            cfg = config if gguf is None else SimpleNamespace(**{**vars(config), "gguf_file": gguf})
            return ri._gguf_memory_breakdown(
                cfg, gguf or path, n_ctx = 4096, **{"gpu_memory_mode": "auto", **kw}
            )

        yield run
        ri._estimate_files_cache.clear()
        ri._estimate_config_cache.clear()

    def test_streamed_table_leaves_the_gpu_and_the_total(self, breakdown, monkeypatch):
        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        b = breakdown()
        assert b.lookup_table_bytes == FLASH_NEXT_TABLE
        assert b.lookup_table_on_disk is True
        assert b.weights_bytes == int(round(self.WEIGHTS)) - FLASH_NEXT_TABLE
        assert b.total_bytes == b.weights_bytes + b.kv_bytes + b.compute_bytes
        # Auto puts everything else on the GPU; the table is not part of it.
        assert b.gpu_bytes == b.total_bytes

    def test_lazy_mode_off_keeps_it_resident_but_still_off_the_gpu(self, breakdown, monkeypatch):
        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        b = breakdown(llama_extra_args = ["-lzm", "off"])
        assert b.lookup_table_bytes == FLASH_NEXT_TABLE
        assert b.lookup_table_on_disk is False
        assert b.weights_bytes == int(round(self.WEIGHTS))
        assert b.gpu_bytes == b.total_bytes - FLASH_NEXT_TABLE

    @pytest.mark.parametrize("extras", [[], ["-lzm", "auto"]], ids = ["managed", "user_auto"])
    def test_ram_that_holds_the_table_reads_it_resident_as_the_launch_does(
        self, breakdown, monkeypatch, extras
    ):
        import routes.inference as ri

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 256 * 1024)
        )
        b = breakdown(llama_extra_args = extras)
        # A user's own lazy mode wins over the managed choice, as at launch.
        assert b.lookup_table_on_disk is bool(extras)
        assert b.lookup_table_bytes == FLASH_NEXT_TABLE

    def test_ram_the_spilled_weights_crowd_out_keeps_the_table_streamed(
        self, breakdown, monkeypatch
    ):
        """24 GiB card + 58 GiB free RAM: Auto spills ~25 GiB of weights to RAM, and with the
        8 GiB prompt cache the 26.8 GiB table no longer fits beside them (the launch streams
        it). The request's every-layer offload alone would have read it resident."""
        import routes.inference as ri

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.delenv("LLAMA_ARG_CACHE_RAM", raising = False)
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 58 * 1024)
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(
                lambda cls, binary = None: {
                    "found": True,
                    "supports_lazy_mode": True,
                    "supports_cache_ram": True,
                }
            ),
        )
        assert breakdown().lookup_table_on_disk is True
        # A card that holds every weight leaves RAM to the table.
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None, **k: [(0, 96_000, 98_304)]),
        )
        assert breakdown().lookup_table_on_disk is False

    @pytest.mark.parametrize("budget_gib, ram_gib, resident", [(48, 64, False), (180, 256, True)])
    def test_apple_prices_the_whole_load_against_unified_memory(
        self, breakdown, monkeypatch, budget_gib, ram_gib, resident
    ):
        """A 64 GB Mac holds the 27 GiB table only beside nothing else; the launch's Apple
        discount checks the whole load, so the panel does too."""
        import routes.inference as ri
        import utils.hardware as hw

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_apple_metal_memory_budget_bytes",
            staticmethod(lambda: int(budget_gib * GIB)),
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_available_system_memory_mib",
            staticmethod(lambda: ram_gib * 1024),
        )
        assert breakdown().lookup_table_on_disk is (not resident)

    def test_an_indexed_table_name_keeps_its_override(self, breakdown, monkeypatch):
        import routes.inference as ri

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.delenv("LLAMA_ARG_OVERRIDE_TENSOR", raising = False)
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 256 * 1024)
        )
        real = ri.LlamaCppBackend._gguf_tensor_scan

        def indexed(self, path):
            arch, named, experts = real(self, path)
            return (
                arch,
                {
                    k.replace("per_layer_token_embd.", "per_layer_token_embd_0."): v
                    for k, v in named.items()
                },
                experts,
            )

        monkeypatch.setattr(ri.LlamaCppBackend, "_gguf_tensor_scan", indexed)
        b = breakdown(llama_extra_args = ["-ot", "per_layer_token_embd_0=CUDA0"])
        assert b.lookup_table_on_disk is True

    def test_a_device_override_charges_a_resident_table_wholly_to_the_gpu(
        self, breakdown, tmp_path, monkeypatch
    ):
        """A layer fraction does not apply to a tensor an -ot pins: --gpu-layers 0 with the
        table on CUDA0 still has the whole table on the card."""
        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.delenv("LLAMA_ARG_OVERRIDE_TENSOR", raising = False)
        table = 8960 * 262144 // 32 * 24
        b = breakdown(
            gguf = _gemma3n(tmp_path),
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            llama_extra_args = ["-ot", "per_layer_token_embd=CUDA0"],
        )
        assert b.lookup_table_bytes == 0
        assert b.gpu_bytes >= table

    def test_a_device_override_keeps_the_table_streamed_as_the_launch_does(
        self, breakdown, monkeypatch
    ):
        import routes.inference as ri

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.delenv("LLAMA_ARG_OVERRIDE_TENSOR", raising = False)
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 256 * 1024)
        )
        b = breakdown(llama_extra_args = ["-ot", "per_layer_token_embd=CUDA0"])
        assert b.lookup_table_on_disk is True

    def test_a_build_without_lazy_mode_keeps_the_table_resident(self, breakdown, monkeypatch):
        import routes.inference as ri

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, binary = None: {"found": True}),
        )
        b = breakdown()
        assert b.lookup_table_on_disk is False
        assert b.weights_bytes == int(round(self.WEIGHTS))

    def test_an_integrated_gpu_reads_the_table_resident(self, breakdown, monkeypatch):
        """A Vulkan iGPU has no mmap support, so llama.cpp resolves auto to off."""
        import routes.inference as ri

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None, **k: [(0, 30000, 0)]),
        )
        b = breakdown()
        assert b.lookup_table_on_disk is False
        assert b.weights_bytes == int(round(self.WEIGHTS))

    def test_a_pin_on_the_integrated_gpu_reads_the_table_resident(self, breakdown, monkeypatch):
        """A mixed host streams by default (llama.cpp drops the iGPU), not when pinned to it."""
        import routes.inference as ri

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.setattr(
            ri.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
        )
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None, **k: [(0, 30000, 0), (1, 22000, 24576)]),
        )
        assert breakdown().lookup_table_on_disk is True
        assert breakdown(selected_gpu_ids = [0]).lookup_table_on_disk is False
        assert breakdown(selected_gpu_ids = [1]).lookup_table_on_disk is True
        # The pin owns placement: a stale --device naming the discrete card is stripped.
        stale = breakdown(selected_gpu_ids = [0], llama_extra_args = ["--device", "Vulkan1"])
        assert stale.lookup_table_on_disk is False

    def test_an_explicit_lazy_mode_on_prices_every_flagged_table(self, breakdown, monkeypatch):
        from hub.utils import gguf_lookup_tables as glt

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
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

    def test_a_resident_gemma3n_table_leaves_the_gpu_share_not_the_total(
        self, breakdown, tmp_path, monkeypatch
    ):
        """Not lazily read, but still a host-pinned input like the launcher prices it."""
        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.delenv("LLAMA_ARG_OVERRIDE_TENSOR", raising = False)
        b = breakdown(gguf = _gemma3n(tmp_path))
        assert b.lookup_table_on_disk is False
        assert b.lookup_table_bytes == 8960 * 262144 // 32 * 24
        assert b.weights_bytes == int(round(self.WEIGHTS))
        assert b.gpu_bytes == b.total_bytes - b.lookup_table_bytes

    def test_a_resident_table_moved_to_the_gpu_stays_in_the_gpu_share(self, breakdown, monkeypatch):
        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        monkeypatch.delenv("LLAMA_ARG_OVERRIDE_TENSOR", raising = False)
        b = breakdown(llama_extra_args = ["-lzm", "off", "-ot", "per_layer_token_embd=CUDA0"])
        assert b.lookup_table_on_disk is False
        assert b.lookup_table_bytes == 0
        assert b.gpu_bytes == b.total_bytes

    def test_the_load_panel_response_carries_the_table(self, breakdown, monkeypatch):
        from core.inference.memory_contract import (
            build_memory_estimate,
            project_estimate_memory_response,
        )
        from models.inference import EstimateMemoryResponse

        monkeypatch.delenv("LLAMA_ARG_LAZY_MODE", raising = False)
        out = project_estimate_memory_response(
            build_memory_estimate(breakdown(), quant_file_bytes = int(self.WEIGHTS))
        )
        assert out["lookup_table_bytes"] == FLASH_NEXT_TABLE
        assert out["lookup_table_on_disk"] is True
        assert EstimateMemoryResponse(**out).lookup_table_bytes == FLASH_NEXT_TABLE

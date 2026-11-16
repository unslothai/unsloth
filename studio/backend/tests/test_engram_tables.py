# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Standalone Engram tables swapped in for a GGUF quant's own (llama-server --ple)."""

from __future__ import annotations

import io
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

import test_llama_cpp_placement  # noqa: E402,F401 -- stubs jwt for routes.inference

from core.inference import llama_cpp as lc  # noqa: E402
from core.inference.engram_tables import (  # noqa: E402
    bundled_table,
    engram_table_precision,
    engram_tables,
    files_llama_reads,
    find_engram_table,
)
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402
from hub.utils.gguf_lookup_tables import ple_only_local  # noqa: E402
from hub.utils.gguf_plan import build_gguf_variant_plans  # noqa: E402
from models.inference import LoadRequest  # noqa: E402
from utils.openai_auto_switch_settings import (  # noqa: E402
    model_override_load_kwargs,
    normalize_model_override,
)

PLE = "per_layer_token_embd.weight"


@pytest.fixture(autouse = True)
def real_repo_snapshots(monkeypatch):
    """Table lookups see only what a test puts in reach, never this machine's Hub cache."""
    from core.inference import engram_tables as et

    real = et.repo_snapshots
    monkeypatch.setattr(et, "repo_snapshots", lambda repo_id: [])
    return real


def _tables_in(monkeypatch, *snapshots):
    from core.inference import engram_tables as et
    monkeypatch.setattr(et, "repo_snapshots", lambda repo_id: list(snapshots))


IQ4_NL, Q8_0, F16 = 20, 8, 1

# Sizes as published by unsloth/Qwen3.8-Flash-Next-GGUF.
QWEN_TABLES = {
    "BF16": 102400491712,
    "Q8_0": 54400261312,
    "Q5_1": 38400184512,
    "IQ4_NL": 28800138432,
}
REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"


def _table_name(precision: str) -> str:
    return f"engram/Qwen3.8-Flash-Next-engram-{precision}.gguf_file"


def _qwen_listing() -> list[tuple[str, int]]:
    files = [(_table_name(p), size) for p, size in QWEN_TABLES.items()]
    files += [
        ("UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf", 9_000_000),
        ("UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf", 37_570_000_000),
        ("UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf", QWEN_TABLES["IQ4_NL"]),
        ("UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf", 9_000_000),
        ("UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf", 680_000_000),
        ("UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf", QWEN_TABLES["Q8_0"]),
        ("UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf", 49_810_000_000),
        ("UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf", 49_420_000_000),
        ("UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf", 14_840_000_000),
    ]
    return files


def _gguf(
    tensors: list[tuple[str, list[int], int]],
    arch: str | None = None,
    split_tensors: int | None = None,
) -> bytes:
    kv: list[tuple[str, object]] = [("general.architecture", arch)] if arch else []
    kv += [("split.no", 0)]
    if split_tensors is not None:
        kv += [("split.tensors.count", (5, "<i", split_tensors))]
    buf = io.BytesIO()
    buf.write(b"GGUF")
    buf.write(struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for key, value in kv:
        raw = key.encode()
        buf.write(struct.pack("<Q", len(raw)) + raw)
        if isinstance(value, str):
            data = value.encode()
            buf.write(struct.pack("<IQ", 8, len(data)) + data)
        elif isinstance(value, tuple):
            buf.write(struct.pack("<I", value[0]) + struct.pack(value[1], value[2]))
        else:
            buf.write(struct.pack("<IH", 2, value))
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


class TestDiscovery:
    def test_precision_comes_from_the_table_file_name_only(self):
        assert engram_table_precision(_table_name("Q8_0")) == "Q8_0"
        assert (
            engram_table_precision("engram/DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file")
            == "MXFP8"
        )
        assert engram_table_precision("engram/README.md") is None
        assert engram_table_precision("engram/set_split.py") is None
        assert engram_table_precision("UD-Q8_0/model-Q8_0-00001-of-00002.gguf") is None
        assert engram_table_precision("engram\\M-engram-q8_0.gguf") == "q8_0"

    def test_tables_are_listed_most_precise_first_and_found_case_insensitively(self):
        tables = engram_tables(_qwen_listing())
        assert [t.precision for t in tables] == ["BF16", "Q8_0", "Q5_1", "IQ4_NL"]
        assert find_engram_table(tables, "q8_0").filename == _table_name("Q8_0")
        assert find_engram_table(tables, "FP4") is None
        assert find_engram_table(tables, None) is None

    def test_a_repeated_or_empty_table_is_listed_once_or_not_at_all(self):
        tables = engram_tables(
            [
                (_table_name("Q8_0"), 3),
                ("engram/Other-engram-q8_0.gguf", 2),
                (_table_name("F16"), 0),
            ]
        )
        assert [(t.precision, t.size_bytes) for t in tables] == [("Q8_0", 3)]

    @pytest.mark.parametrize(
        "quant, shard, precision",
        [
            ("UD-IQ1_S", "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf", "IQ4_NL"),
            # Not the last shard: the guide's 3 of 6 slot.
            ("UD-Q6_K_XL", "UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf", "Q8_0"),
        ],
    )
    def test_the_bundled_table_shard_is_the_one_sized_like_a_table(self, quant, shard, precision):
        tables = engram_tables(_qwen_listing())
        shards = [(n, s) for n, s in _qwen_listing() if n.startswith(f"{quant}/")]
        found = bundled_table(shards, tables)
        assert found is not None
        assert found[0] == shard and found[1].precision == precision

    def test_an_ambiguous_or_single_file_quant_has_no_bundled_table(self):
        tables = engram_tables(_qwen_listing())
        size = QWEN_TABLES["Q8_0"]
        assert (
            bundled_table(
                [("a-00001-of-00002.gguf", size), ("a-00002-of-00002.gguf", size)], tables
            )
            is None
        )
        assert bundled_table([("a.gguf", size)], tables) is None


class TestFilesLlamaReads:
    def test_the_missing_or_table_only_shard_gives_way_to_the_external_table(self, tmp_path):
        first = _write(
            tmp_path / "m-00001-of-00003.gguf",
            _gguf([("token_embd.weight", [8, 8], F16)], "qwen4exp"),
        )
        second = _write(
            tmp_path / "m-00002-of-00003.gguf", _gguf([("blk.0.attn_q.weight", [8, 8], F16)])
        )
        third = str(tmp_path / "m-00003-of-00003.gguf")
        ple = str(tmp_path / "engram-Q8_0.gguf_file")
        shards = [first, second, third]

        assert files_llama_reads(shards, None) == shards
        # Absent: skipped, as llama.cpp does.
        assert files_llama_reads(shards, ple) == [first, second, ple]
        # Present but holding only the table: skipped too.
        _write(Path(third), _gguf([(PLE, [160, 1024], IQ4_NL)]))
        assert ple_only_local(third) and not ple_only_local(second)
        assert files_llama_reads(shards, ple) == [first, second, ple]


class TestPlan:
    def _siblings(self):
        return [
            SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": f"sha-{i}"})
            for i, (name, size) in enumerate(_qwen_listing())
        ]

    def test_with_engram_swaps_the_quant_table_shard_for_the_standalone_file(self):
        plan = build_gguf_variant_plans(self._siblings())["ud-iq1_s"]
        shard = "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf"
        assert plan.engram_precision == "IQ4_NL"
        assert plan.engram_shard.path == shard

        swapped = plan.with_engram("q8_0")
        assert shard not in swapped.target_filenames
        assert _table_name("Q8_0") in swapped.target_filenames
        assert swapped.download_size_bytes == (
            plan.download_size_bytes - QWEN_TABLES["IQ4_NL"] + QWEN_TABLES["Q8_0"]
        )
        shard_sha = next(f.sha256 for f in plan.expected_files if f.path == shard)
        assert shard_sha not in swapped.required_hashes

    def test_with_engram_keeps_the_plan_for_its_own_or_an_unknown_precision(self):
        plan = build_gguf_variant_plans(self._siblings())["ud-iq1_s"]
        assert plan.with_engram("IQ4_NL") is plan
        assert plan.with_engram("FP4") is plan
        assert plan.with_engram(None) is plan

    def test_swapping_a_swapped_plan_keeps_exactly_one_table(self):
        plan = build_gguf_variant_plans(self._siblings())["ud-iq1_s"]
        swapped = plan.with_engram("Q8_0")
        assert swapped.with_engram("Q8_0").expected_files == swapped.expected_files
        again = [f.path for f in swapped.with_engram("BF16").expected_files]
        assert [p for p in again if engram_table_precision(p)] == [_table_name("BF16")]


class TestResolveSwap:
    def _hub(
        self,
        monkeypatch,
        files,
        *,
        fail = None,
    ):
        names = [n for n, _ in files if n.startswith("UD-IQ1_S/")]
        monkeypatch.setattr(lc, "_resolve_variant_gguf_files", lambda *a: (names[0], names[1:]))

        class _Api:
            def __init__(self, token = None):
                pass

            def model_info(
                self,
                repo,
                files_metadata = False,
            ):
                if fail is not None:
                    raise fail
                return SimpleNamespace(
                    siblings = [SimpleNamespace(rfilename = n, size = s) for n, s in files]
                )

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

    def test_another_precision_swaps_and_names_the_shard_not_to_fetch(self, monkeypatch):
        self._hub(monkeypatch, _qwen_listing())
        swap = lc._resolve_engram_swap(REPO, "UD-IQ1_S", None, "q5_1")
        assert swap.table.precision == "Q5_1"
        assert swap.bundled_shard == "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf"

    def test_the_quant_own_precision_swaps_nothing(self, monkeypatch):
        self._hub(monkeypatch, _qwen_listing())
        assert lc._resolve_engram_swap(REPO, "UD-IQ1_S", None, "IQ4_NL") is None

    def test_an_unpublished_precision_is_an_error_naming_the_others(self, monkeypatch):
        self._hub(monkeypatch, _qwen_listing())
        with pytest.raises(
            ValueError,
            match = r"publishes no FP4 Engram table \(available: BF16, Q8_0, Q5_1, IQ4_NL\)",
        ):
            lc._resolve_engram_swap(REPO, "UD-IQ1_S", None, "FP4")

    def test_offline_reads_the_cache_and_never_claims_the_repo_lacks_a_table(
        self, monkeypatch, tmp_path
    ):
        self._hub(monkeypatch, _qwen_listing(), fail = OSError("no route to host"))
        snap = tmp_path / "snapshots" / "abc"
        for name in (
            "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
            "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf",
            _table_name("Q5_1"),
        ):
            _write(snap / name, b"x")
        _tables_in(monkeypatch, snap)
        swap = lc._resolve_engram_swap(REPO, "UD-IQ1_S", None, "Q5_1")
        # The one absent shard is the one llama.cpp tolerates missing.
        assert swap.bundled_shard == "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf"
        with pytest.raises(ValueError, match = "has no cached Q8_0 Engram table"):
            lc._resolve_engram_swap(REPO, "UD-IQ1_S", None, "Q8_0")


class TestRequestAndOverrides:
    def test_a_blank_precision_is_the_quant_own_table(self):
        assert LoadRequest(model_path = REPO, engram_precision = "  ").engram_precision is None
        assert LoadRequest(model_path = REPO, engram_precision = " Q8_0 ").engram_precision == "Q8_0"

    def test_a_malformed_precision_is_rejected(self):
        with pytest.raises(ValueError):
            LoadRequest(model_path = REPO, engram_precision = "../x")
        with pytest.raises(ValueError):
            LoadRequest(model_path = REPO, engram_precision = 0)

    def test_the_override_store_keeps_it_for_gguf_loads_only(self):
        entry = normalize_model_override({"engram_precision": "Q8_0"})
        assert entry["engram_precision"] == "Q8_0"
        assert model_override_load_kwargs(entry, is_gguf = True)["engram_precision"] == "Q8_0"
        assert "engram_precision" not in model_override_load_kwargs(entry, is_gguf = False)
        assert "engram_precision" not in normalize_model_override({"engram_precision": "Q8 0"})
        assert "engram_precision" not in normalize_model_override({"engram_precision": 8})


class TestResidentIdentity:
    def _resident(self, precision):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = SimpleNamespace(poll = lambda: None)
        backend._retired_process = None
        backend._model_identifier = REPO
        backend._gguf_path = None
        backend._hf_variant = "UD-IQ1_S"
        backend._engram_precision = precision
        return backend

    @pytest.mark.parametrize(
        "precision, asked, matches",
        [
            (None, None, True),
            (None, "Q8_0", False),
            ("Q8_0", "q8_0", True),
            ("Q8_0", None, False),
            ("Q8_0", "Q5_1", False),
        ],
    )
    def test_the_table_is_part_of_the_resident_weights(
        self, monkeypatch, precision, asked, matches
    ):
        monkeypatch.setattr(LlamaCppBackend, "is_loaded", property(lambda self: True))
        intent = GgufLoadIntent(
            model_identifier = REPO, hf_repo = REPO, hf_variant = "UD-IQ1_S", engram_precision = asked
        )
        assert self._resident(precision).matches_load_source(intent) is matches


@pytest.mark.skipif(sys.platform == "win32", reason = "shell script stands in for llama-server")
@pytest.mark.parametrize("has_ple", [True, False])
def test_the_probe_reads_the_ple_flag(tmp_path, monkeypatch, has_ple):
    monkeypatch.setattr(LlamaCppBackend, "_capability_cache", {})
    help_text = "-m,   --model FNAME                    model path\n"
    if has_ple:
        help_text += "--ple, --shard FNAME,...               extra GGUF shards for the model\n"
    script = tmp_path / f"llama-server-{has_ple}"
    script.write_text("#!/bin/sh\ncat <<'HELP'\n" + help_text + "\nHELP\n")
    script.chmod(0o755)
    assert LlamaCppBackend.probe_server_capabilities(str(script))["supports_ple"] is has_ple


class TestHubDownload:
    def _siblings(self):
        return [
            SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": f"{i:064x}"})
            for i, (name, size) in enumerate(_qwen_listing())
        ]

    def test_the_worker_fetches_the_table_in_place_of_the_quant_shard(self, monkeypatch):
        from hub.workers import hf_download as w

        siblings = self._siblings()
        monkeypatch.setattr(
            w, "_model_info_with_retry", lambda *a, **k: SimpleNamespace(siblings = siblings)
        )
        seen = {}

        class _Stop(Exception):
            pass

        def manifest(repo_type, repo_id, variant, expected, *a, **k):
            seen["files"] = [f.path for f in expected]
            raise _Stop()

        monkeypatch.setattr("hub.utils.download_manifest.write_manifest", manifest)
        with pytest.raises(_Stop):
            w._download_gguf_variant(REPO, "UD-IQ1_S", None, "http", engram_precision = "Q8_0")
        assert _table_name("Q8_0") in seen["files"]
        assert "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf" not in seen["files"]

    def test_the_worker_refuses_an_unpublished_table(self, monkeypatch):
        from hub.workers import hf_download as w

        siblings = self._siblings()
        monkeypatch.setattr(
            w, "_model_info_with_retry", lambda *a, **k: SimpleNamespace(siblings = siblings)
        )
        with pytest.raises(SystemExit):
            w._download_gguf_variant(REPO, "UD-IQ1_S", None, "http", engram_precision = "FP4")

    def test_the_request_needs_a_quant(self):
        import asyncio

        from fastapi import HTTPException

        from hub.schemas.downloads import DownloadModelRequest
        from hub.services.models import downloads

        body = DownloadModelRequest(repo_id = REPO, engram_precision = "Q8_0")
        with pytest.raises(HTTPException) as raised:
            asyncio.run(downloads.download_model_response(body))
        assert raised.value.status_code == 400
        assert "engram_precision needs a gguf_variant" in str(raised.value.detail)


class TestDownloadJobCarriesTheTable:
    """The table is part of a download job's plan, so every path that rebuilds the job fetches it."""

    KEY = "unsloth/qwen3.8-flash-next-gguf::ud-iq1_s"

    def _claim(
        self,
        registry,
        precision,
        transport = None,
    ):
        from hub.utils import download_registry
        return registry.claim(
            self.KEY,
            transport or download_registry.TRANSPORT_XET,
            repo_type = "model",
            repo_id = REPO,
            variant = "UD-IQ1_S",
            blob_hashes = frozenset({"a"}),
            engram_precision = precision,
        )

    def test_a_live_job_is_joined_only_by_the_same_table(self):
        from hub.utils import download_registry

        registry = download_registry.DownloadRegistry()
        assert self._claim(registry, "Q8_0")[0]
        assert registry.get_job_metadata(self.KEY).engram_precision == "Q8_0"
        assert self._claim(registry, "q8_0") == (False, "running")
        assert self._claim(registry, None) == (False, "engram_mismatch")
        assert self._claim(registry, "Q5_1") == (False, "engram_mismatch")
        # A claim that does not choose a table (a load, a delete) is not a mismatch.
        assert registry.claim(self.KEY, download_registry.TRANSPORT_XET) == (False, "running")

    def test_the_transport_fallback_worker_fetches_the_same_table(self, monkeypatch, tmp_path):
        from hub.services import download_lifecycle as dl
        from hub.utils import download_registry, state_dir

        monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
        registry = download_registry.DownloadRegistry()
        assert self._claim(registry, "Q8_0")[0]
        registry.set_job(self.KEY, "error", "xet failed")
        spawned = []

        def spawn(args, *a, **k):
            spawned.append(list(args))
            raise OSError("stop here")

        monkeypatch.setattr(dl, "spawn_worker", spawn)
        dl._try_transport_retry(
            registry,
            self.KEY,
            hf_token = None,
            label = "x",
            log_prefix = "t",
            logger = SimpleNamespace(
                **{n: (lambda *a, **k: None) for n in ("debug", "info", "warning", "error")}
            ),
            repo_type = "model",
            repo_id = REPO,
            watch_name = "x",
        )
        assert spawned and spawned[0][spawned[0].index("--engram") + 1] == "Q8_0"
        assert registry.get_job_metadata(self.KEY).engram_precision == "Q8_0"

    def test_the_transport_is_chosen_by_the_largest_file_this_job_fetches(self, monkeypatch):
        from hub.services import download_lifecycle as dl
        from hub.utils import download_registry

        siblings = [
            SimpleNamespace(rfilename = n, size = s, lfs = {"sha256": f"{i:064x}"})
            for i, (n, s) in enumerate(_qwen_listing())
        ]
        monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(siblings))
        monkeypatch.setattr(download_registry, "finalized_blob_hashes", lambda *a, **k: frozenset())
        own = dl.largest_download_file_bytes("model", REPO, variant = "UD-IQ1_S")
        swapped = dl.largest_download_file_bytes(
            "model", REPO, variant = "UD-IQ1_S", engram_precision = "Q8_0"
        )
        assert own == 37_570_000_000
        assert swapped == QWEN_TABLES["Q8_0"]
        job = SimpleNamespace(
            variant = "UD-IQ1_S", scoped_files = (), hub_cache = None, engram_precision = "Q8_0"
        )
        assert dl._largest_file_bytes_for_job("model", REPO, job) == QWEN_TABLES["Q8_0"]

    def test_a_partial_of_the_table_is_purged_like_the_quants_own(self):
        siblings = [
            SimpleNamespace(rfilename = n, size = s, lfs = {"sha256": f"{i:064x}"})
            for i, (n, s) in enumerate(_qwen_listing())
        ]
        plan = build_gguf_variant_plans(siblings)["ud-iq1_s"]
        swapped = plan.with_engram("Q8_0")
        table_hash = next(f.sha256 for f in swapped.expected_files if f.path == _table_name("Q8_0"))
        assert table_hash in swapped.main_hashes
        # The built-in shard keeps its protection too.
        assert plan.engram_shard.sha256 in swapped.main_hashes

    def test_the_active_download_report_names_the_table(self, monkeypatch):
        from hub.schemas.downloads import ActiveDownload
        assert ActiveDownload(state = "running", engram_precision = "Q8_0").engram_precision == "Q8_0"


class TestListing:
    """The variants listing reports each quant's own table and what loads with nothing to fetch."""

    SIZES = {"Q8_0": 500, "Q5_1": 400, "IQ4_NL": 300}
    SHARDS = {
        "UD-IQ1_S/M-UD-IQ1_S-00001-of-00003.gguf": 64,
        "UD-IQ1_S/M-UD-IQ1_S-00002-of-00003.gguf": 900,
        "UD-IQ1_S/M-UD-IQ1_S-00003-of-00003.gguf": 300,
    }

    @pytest.fixture
    def repo(self, monkeypatch, tmp_path):
        from hub.services.models import gguf_variants as gv
        from hub.utils import inventory_scan
        from hub.utils.gguf import GgufVariantInfo
        from utils import hf_cache_settings

        store = {}
        monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(
            "storage.studio_db.get_app_setting",
            lambda key, fallback = None: store.get(key, fallback),
        )
        monkeypatch.setattr("storage.studio_db.upsert_app_settings", store.update)
        monkeypatch.setattr("hub.utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
        monkeypatch.setattr("hub.utils.paths.hf_default_cache_dir", lambda: tmp_path / "unused")
        root = hf_cache_settings.get_hf_cache_paths().hub_cache
        cache = root / "models--Org--M-GGUF"
        snapshot = cache / "snapshots" / ("e" * 40)
        snapshot.mkdir(parents = True)
        (cache / "refs").mkdir()
        (cache / "refs" / "main").write_text("e" * 40)

        files = dict(self.SHARDS)
        files.update({f"engram/M-engram-{p}.gguf_file": s for p, s in self.SIZES.items()})
        siblings = [
            SimpleNamespace(rfilename = n, size = s, lfs = {"sha256": f"{i:064x}"})
            for i, (n, s) in enumerate(files.items())
        ]
        variant = GgufVariantInfo(
            filename = "UD-IQ1_S/M-UD-IQ1_S-00001-of-00003.gguf",
            quant = "UD-IQ1_S",
            size_bytes = sum(self.SHARDS.values()),
        )
        monkeypatch.setattr(gv, "list_gguf_variants", lambda *a, **k: ([variant], False, siblings))
        monkeypatch.setattr(gv, "_llama_loads_engram_tables", lambda: True)

        def cache_files(*names):
            for name in names:
                path = snapshot / name
                path.parent.mkdir(parents = True, exist_ok = True)
                path.write_bytes(b"\0" * files[name])
            inventory_scan.invalidate_hf_cache_scans()

        return cache_files

    def _row(self):
        import asyncio

        from hub.services.models import gguf_variants as gv

        response = asyncio.run(gv.get_gguf_variants_response("Org/M-GGUF"))
        return response, response.variants[0]

    def test_a_quant_missing_only_its_table_loads_with_a_cached_standalone_one(self, repo):
        repo(
            "UD-IQ1_S/M-UD-IQ1_S-00001-of-00003.gguf",
            "UD-IQ1_S/M-UD-IQ1_S-00002-of-00003.gguf",
            "engram/M-engram-Q5_1.gguf_file",
        )
        response, row = self._row()
        assert row.engram_precision == "IQ4_NL"
        # On disk: it loads with the cached table, though its own table shard is absent.
        assert row.downloaded is True
        assert row.engram_ready == ["Q5_1"]
        assert [(t.precision, t.downloaded) for t in response.engram_tables] == [
            ("Q8_0", False),
            ("Q5_1", True),
            ("IQ4_NL", False),
        ]
        assert response.engram_supported is True

    def test_a_complete_quant_is_ready_with_its_own_table_too(self, repo):
        repo(*self.SHARDS, "engram/M-engram-Q8_0.gguf_file")
        _, row = self._row()
        assert row.downloaded is True
        assert row.engram_ready == ["IQ4_NL", "Q8_0"]

    def test_nothing_is_ready_without_the_rest_of_the_quant(self, repo):
        repo("UD-IQ1_S/M-UD-IQ1_S-00001-of-00003.gguf", "engram/M-engram-Q5_1.gguf_file")
        _, row = self._row()
        assert row.engram_ready == []
        assert row.downloaded is False


class TestManualQaFindings:
    def test_the_model_size_counts_the_table_llama_reads_not_the_quants_own(self, tmp_path):
        first = _write(
            tmp_path / "m-00001-of-00003.gguf",
            _gguf([("token_embd.weight", [8, 8], F16)], "qwen4exp"),
        )
        second = _write(
            tmp_path / "m-00002-of-00003.gguf", _gguf([("blk.0.attn_q.weight", [8, 8], F16)])
        )
        own = _write(tmp_path / "m-00003-of-00003.gguf", _gguf([(PLE, [160, 1024], IQ4_NL)]))
        with open(own, "ab") as f:
            f.truncate(3_000_000)
        table = tmp_path / "engram-Q5_1.gguf_file"
        table.write_bytes(_gguf([(PLE, [160, 1024], IQ4_NL)]))
        with open(table, "ab") as f:
            f.truncate(4_000_000)
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._launch_ple = None
        size = lambda p: Path(p).stat().st_size  # noqa: E731
        assert backend._gguf_read_bytes(first) == size(first) + size(second) + size(own)
        backend._launch_ple = (first, str(table))
        assert backend._gguf_read_bytes(first) == size(first) + size(second) + size(table)

    @pytest.mark.parametrize(
        "xet_off, cached, size, refused",
        [
            (True, False, 54_400_261_312, True),
            (False, False, 54_400_261_312, False),
            (True, True, 54_400_261_312, False),
            (True, False, 38_400_184_512, False),
        ],
    )
    def test_a_table_https_cannot_fetch_is_refused_up_front_when_xet_is_off(
        self, monkeypatch, xet_off, cached, size, refused
    ):
        from core.inference.engram_tables import EngramTable
        from hub.utils import download_registry
        import huggingface_hub.constants as hf_constants

        monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_XET", xet_off, raising = False)
        monkeypatch.setattr(
            download_registry, "download_transport_unavailable_reason", lambda t: None
        )
        monkeypatch.setattr(download_registry, "_modelscope_is_source", lambda: False)
        monkeypatch.setattr(lc, "cached_table_path", lambda *a: "/x" if cached else None)
        reason = lc._engram_table_unfetchable_reason(
            REPO, EngramTable("Q8_0", _table_name("Q8_0"), size)
        )
        assert (reason is not None) is refused
        if refused:
            assert "Xet is turned off" in reason and "54.4GB" in reason


def _swapped_snapshot(
    tmp_path: Path,
    *,
    absent: int,
    table: bool = True,
):
    """UD-IQ1_S as a --ple download leaves it: shard ``absent`` missing (3 holds the table)."""
    snap = tmp_path / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / "abc"
    names = [n for n, _ in _qwen_listing() if n.startswith("UD-IQ1_S/")]
    shards = [
        _gguf([("token_embd.weight", [8, 8], F16)], "qwen4exp", split_tensors = 4),
        _gguf([("blk.0.attn_q.weight", [8, 8], F16), ("blk.0.attn_k.weight", [8, 8], F16)]),
        _gguf([(PLE, [160, 1024], IQ4_NL)]),
    ]
    for index, (name, data) in enumerate(zip(names, shards), start = 1):
        if index != absent:
            _write(snap / name, data)
    if table:
        _write(snap / _table_name("Q5_1"), _gguf([(PLE, [160, 1024], Q8_0)]))
    return snap, names


def _no_download(**kwargs):
    raise AssertionError("a path load downloaded an Engram table")


class TestCachedQuantLoads:
    def test_offline_a_quant_missing_its_table_shard_resolves_to_its_split_names(
        self, monkeypatch, tmp_path
    ):
        import huggingface_hub

        snap, names = _swapped_snapshot(tmp_path, absent = 3)
        import utils.models.model_config as mc

        monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo, *a, **k: [snap])
        monkeypatch.setattr(lc, "_cached_gguf_snapshots", lambda repo: [snap])
        _tables_in(monkeypatch, snap)

        def offline(*a, **k):
            raise OSError("no route to host")

        monkeypatch.setattr(huggingface_hub, "list_repo_files", offline)
        monkeypatch.setattr(huggingface_hub.HfApi, "model_info", offline)
        assert lc._resolve_variant_gguf_files(REPO, "UD-IQ1_S") == (names[0], names[1:])
        swap = lc._resolve_engram_swap(REPO, "UD-IQ1_S", None, "Q5_1")
        assert swap.bundled_shard == names[2]
        # Without a cached standalone table, a missing shard is an incomplete download.
        (snap / _table_name("Q5_1")).unlink()
        assert lc._resolve_variant_gguf_files(REPO, "UD-IQ1_S")[1] == []

    def test_a_hub_quant_loaded_by_its_cache_path_takes_the_saved_table(
        self, monkeypatch, tmp_path
    ):
        from core.inference.engram_tables import EngramTable
        from test_llama_cpp_placement import _backend, _launch

        backend, _ = _backend(tmp_path, vulkan = False, memory = [(0, 24_000, 24_000)])
        snap = tmp_path / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / "abc"
        gguf = _write(snap / "M-UD-IQ1_S.gguf", (tmp_path / "model.gguf").read_bytes())
        table = _write(snap / _table_name("Q5_1"), b"x")
        seen = {}

        def resolve(repo, variant, token, precision):
            seen["repo"] = repo
            return lc.EngramSwap(EngramTable(precision, _table_name(precision), 1), None)

        monkeypatch.setattr(lc, "_resolve_engram_swap", resolve)
        monkeypatch.setattr(lc, "_engram_table_unfetchable_reason", lambda *a: None)
        backend._download_engram_table = _no_download
        probe = backend.probe_server_capabilities
        backend.probe_server_capabilities = lambda b: {**probe(b), "supports_ple": True}
        cmd = _launch(backend, gguf, hf_variant = "UD-IQ1_S", engram_precision = "Q5_1")["cmd"]
        assert seen["repo"] == REPO
        assert cmd[cmd.index("--ple") + 1] == table
        assert backend._engram_precision == "Q5_1"

        # A path load never downloads: an uncached table is refused before the resident model goes.
        Path(table).unlink()
        with pytest.raises(ValueError, match = "Q5_1 Engram table .* is not downloaded"):
            backend.load_model(
                GgufLoadIntent(
                    gguf_path = gguf,
                    model_identifier = "other",
                    hf_variant = "UD-IQ1_S",
                    engram_precision = "Q5_1",
                )
            )

    def test_a_path_load_of_a_table_skipped_quant_needs_its_table(self, monkeypatch, tmp_path):
        from test_llama_cpp_placement import _backend

        backend, _ = _backend(tmp_path, vulkan = False, memory = [(0, 24_000, 24_000)])
        snap, names = _swapped_snapshot(tmp_path, absent = 3)
        backend._download_engram_table = _no_download
        with pytest.raises(ValueError, match = "Choose that table"):
            backend.load_model(
                GgufLoadIntent(
                    gguf_path = str(snap / names[0]),
                    model_identifier = "m",
                    hf_variant = "UD-IQ1_S",
                )
            )

    @pytest.mark.parametrize(
        "absent, table, complete",
        [(3, True, True), (2, True, False), (3, False, False), (0, False, True)],
    )
    def test_the_local_index_serves_a_quant_missing_only_its_table_shard(
        self, tmp_path, absent, table, complete
    ):
        from hub.utils.inventory_scan import _completed_gguf_variants
        snap, _ = _swapped_snapshot(tmp_path, absent = absent, table = table)
        assert ("UD-IQ1_S" in _completed_gguf_variants(snap)) is complete

    def test_tables_are_looked_for_in_every_known_cache_root(
        self, monkeypatch, tmp_path, real_repo_snapshots
    ):
        import hub.utils.hf_cache_state as hcs
        import utils.hf_cache_settings as hcset

        repo_dir = "models--" + REPO.replace("/", "--")
        snaps = [tmp_path / root / repo_dir / "snapshots" / "r" for root in ("a", "b")]
        for snap in snaps:
            snap.mkdir(parents = True)
        monkeypatch.setattr(hcs, "hf_cache_roots", lambda *a, **k: [tmp_path / "b"])
        monkeypatch.setattr(
            hcset, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path / "a")
        )
        assert real_repo_snapshots(REPO.lower()) == snaps

    @pytest.mark.parametrize("requested, resumes", [("Q5_1", True), ("Q8_0", False), (None, False)])
    def test_without_metadata_the_worker_resumes_only_the_same_table(
        self, monkeypatch, requested, resumes
    ):
        from hub.utils import download_manifest as dm
        from hub.utils.download_manifest import ExpectedFile
        from hub.workers import hf_download as w

        def unavailable(*a, **k):
            raise RuntimeError("Hub unreachable")

        recorded = [n for n, _ in _qwen_listing() if n.startswith("UD-IQ1_S/")][:2]
        manifest = SimpleNamespace(
            expected_files = [ExpectedFile(n, 1) for n in (*recorded, _table_name("Q5_1"))],
            metadata_derived = False,
            commit_hash = None,
        )
        monkeypatch.setattr(w, "_gguf_variant_target_plan", unavailable)
        monkeypatch.setattr(dm, "read_manifest", lambda *a, **k: manifest)

        class _Resumed(Exception):
            pass

        def resumed(*a, **k):
            raise _Resumed()

        monkeypatch.setattr(dm, "write_manifest", resumed)
        with pytest.raises(_Resumed if resumes else SystemExit):
            w._download_gguf_variant(REPO, "UD-IQ1_S", None, "http", engram_precision = requested)

    def test_a_large_table_cached_in_a_remembered_root_is_not_refused(self, monkeypatch, tmp_path):
        from core.inference.engram_tables import EngramTable
        from hub.utils import download_registry
        import huggingface_hub.constants as hf_constants

        snap, _ = _swapped_snapshot(tmp_path, absent = 3)
        # A small file over a lowered ceiling: truncate() allocates on NTFS, so 54 GB filled Windows CI.
        size = 4096
        _write(snap / _table_name("Q8_0"), b"\0" * size)
        monkeypatch.setattr(download_registry, "http_max_file_bytes", lambda: size - 1)
        monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_XET", True, raising = False)
        monkeypatch.setattr(download_registry, "_modelscope_is_source", lambda: False)
        _tables_in(monkeypatch, snap)
        table = EngramTable("Q8_0", _table_name("Q8_0"), size)
        assert lc._engram_table_unfetchable_reason(REPO, table) is None
        assert lc._engram_table_unfetchable_reason(REPO, table.__class__("Q8_0", "x", size))

    def test_a_resumed_plan_owns_its_standalone_table_like_a_shard(self):
        from hub.utils.download_manifest import ExpectedFile
        from hub.utils.gguf_plan import plan_from_expected_files

        shard = "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
        plan = plan_from_expected_files(
            "UD-IQ1_S",
            [ExpectedFile(shard, 1, "a" * 64), ExpectedFile(_table_name("Q5_1"), 1, "b" * 64)],
        )
        assert plan.main_hashes == {"a" * 64, "b" * 64}

    @pytest.mark.parametrize("last, reclaimed", [(True, True), (False, False)])
    def test_deleting_the_last_quant_reclaims_its_standalone_table(self, tmp_path, last, reclaimed):
        from hub.services.models.deletion import _delete_gguf_variant_from_repos
        from test_mtp_drafter_companion import _cache_repo

        names = ["M-UD-IQ1_S.gguf", "engram/M-engram-Q5_1.gguf_file"]
        repo, snap = _cache_repo(tmp_path, REPO, names + ([] if last else ["M-Q8_0.gguf"]))
        _delete_gguf_variant_from_repos(REPO, "UD-IQ1_S", [repo], None, root = tmp_path)
        assert not (snap / names[0]).is_symlink()
        assert (snap / names[1]).is_symlink() is not reclaimed

    @pytest.mark.parametrize("sibling_active", [False, True])
    def test_deleting_a_quant_drops_the_partial_table_its_job_left(
        self, monkeypatch, tmp_path, sibling_active
    ):
        from hub.services.models import deletion, gguf_variants
        from hub.utils import download_manifest as dm
        from hub.utils.download_manifest import ExpectedFile
        from test_mtp_drafter_companion import _cache_repo

        table_hash = "c" * 64
        repo, _ = _cache_repo(tmp_path, REPO, ["M-UD-IQ1_S.gguf"])
        partial = repo.repo_path / "blobs" / f"{table_hash}.incomplete"
        partial.write_bytes(b"x")
        manifest = SimpleNamespace(
            expected_files = [ExpectedFile(_table_name("Q8_0"), 1, table_hash)], commit_hash = None
        )
        monkeypatch.setattr(dm, "read_manifest", lambda *a, **k: manifest)
        monkeypatch.setattr(gguf_variants, "gguf_variant_blob_hashes", lambda *a, **k: frozenset())
        deletion._delete_gguf_variant_from_repos(
            REPO, "UD-IQ1_S", [repo], None, root = tmp_path, sibling_active = sibling_active
        )
        assert partial.exists() is sibling_active

    def test_switching_the_table_invalidates_saved_kv(self, tmp_path):
        backend = LlamaCppBackend()
        first = _write(tmp_path / "m-00001-of-00002.gguf", b"x")
        tables = [_write(tmp_path / f"t{i}.gguf_file", b"x" * (i + 1)) for i in range(2)]
        backend._gguf_path, backend._extra_args = first, []
        backend._launch_ple = None
        own = backend._slot_launch_fingerprint()
        fingerprints = []
        for table in tables:
            backend._launch_ple = (first, table)
            fingerprints.append(backend._slot_launch_fingerprint())
        assert len({own, *fingerprints}) == 3

    def test_a_table_cached_in_another_root_is_reused_not_downloaded(self, monkeypatch, tmp_path):
        from core.inference.engram_tables import EngramTable

        other = _write(tmp_path / "b" / "snapshots" / "r" / _table_name("Q5_1"), b"x" * 7)
        monkeypatch.setattr(lc, "_cached_hf_snapshot_file", lambda *a, **k: None)
        _tables_in(monkeypatch, Path(other).parents[1])

        def download(*a, **k):
            raise AssertionError("downloaded a table that is cached")

        monkeypatch.setattr(lc, "hf_hub_download_with_xet_fallback", download)
        backend = LlamaCppBackend()
        near = _write(tmp_path / "a" / "snapshots" / "r" / "m.gguf", b"x")
        path = backend._download_engram_table(
            hf_repo = REPO, table = EngramTable("Q5_1", _table_name("Q5_1"), 7), near_path = near
        )
        assert path == other


class TestOwnTableAndOtherCacheRoots:
    @pytest.mark.parametrize("requested, recorded", [("iq4_nl", None), ("Q8_0", "Q8_0")])
    def test_a_download_naming_the_quant_own_table_is_a_plain_download(
        self, monkeypatch, requested, recorded
    ):
        import asyncio

        from hub.schemas.downloads import DownloadModelRequest
        from hub.services import download_lifecycle
        from hub.services.models import downloads, gguf_variants

        siblings = [
            SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": f"{i:064x}"})
            for i, (name, size) in enumerate(_qwen_listing())
        ]
        plan = build_gguf_variant_plans(siblings)["ud-iq1_s"]
        monkeypatch.setattr(download_lifecycle, "largest_download_file_bytes", lambda *a, **k: None)
        monkeypatch.setattr(
            download_lifecycle, "resolve_requested_use_xet", lambda *a, **k: (True, "test")
        )
        monkeypatch.setattr(gguf_variants, "gguf_variant_requirements", lambda *a, **k: plan)
        monkeypatch.setattr(gguf_variants, "gguf_variant_blob_hashes", lambda *a, **k: frozenset())
        seen = {}

        class _Claimed(Exception):
            pass

        def claim(*a, **k):
            seen.update(k)
            raise _Claimed()

        monkeypatch.setattr(downloads._registry, "claim", claim)
        body = DownloadModelRequest(
            repo_id = REPO, gguf_variant = "UD-IQ1_S", engram_precision = requested
        )
        with pytest.raises(_Claimed):
            asyncio.run(downloads.download_model_response(body))
        assert seen["engram_precision"] == recorded

    def test_a_table_cached_in_another_root_completes_the_quant(
        self, monkeypatch, tmp_path, real_repo_snapshots
    ):
        from core.inference import engram_tables as et

        import hub.utils.hf_cache_state as hcs
        import utils.hf_cache_settings as hcset
        from hub.utils.inventory_scan import _completed_gguf_variants

        snap, _ = _swapped_snapshot(tmp_path / "a", absent = 3, table = False)
        other = tmp_path / "b"
        _write(
            other / snap.parent.parent.name / "snapshots" / "def" / _table_name("Q5_1"),
            _gguf([(PLE, [160, 1024], Q8_0)]),
        )
        monkeypatch.setattr(et, "repo_snapshots", real_repo_snapshots)
        monkeypatch.setattr(hcs, "hf_cache_roots", lambda *a, **k: [other])
        monkeypatch.setattr(
            hcset, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path / "a")
        )
        assert "UD-IQ1_S" in _completed_gguf_variants(snap)


class TestSnapshotPathsAndRoots:
    def test_a_symlinked_quant_missing_its_table_shard_loads_from_its_snapshot(
        self, monkeypatch, tmp_path
    ):
        import os

        from utils.models.model_config import ModelConfig

        snap, names = _swapped_snapshot(tmp_path, absent = 3)
        blobs = snap.parent.parent / "blobs"
        blobs.mkdir()
        for index, name in enumerate(names[:2]):
            os.replace(snap / name, blobs / f"h{index}")
            (snap / name).symlink_to(blobs / f"h{index}")
        config = ModelConfig.from_identifier(model_id = str(snap), gguf_variant = "UD-IQ1_S")
        assert config.gguf_file == str(snap / names[0])

    def test_an_unreadable_root_does_not_end_the_search(
        self, monkeypatch, tmp_path, real_repo_snapshots
    ):
        import hub.utils.hf_cache_state as hcs
        import utils.hf_cache_settings as hcset

        snap = tmp_path / "b" / ("models--" + REPO.replace("/", "--")) / "snapshots" / "r"
        snap.mkdir(parents = True)
        monkeypatch.setattr(hcs, "hf_cache_roots", lambda *a, **k: [tmp_path / "b"])
        monkeypatch.setattr(
            hcset, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path / "gone")
        )
        assert real_repo_snapshots(REPO) == [snap]

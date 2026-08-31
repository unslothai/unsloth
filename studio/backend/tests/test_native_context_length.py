# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the native_context_length feature (PR #4746).

Verifies the `native_context_length` property on LlamaCppBackend and the
matching Pydantic fields. The raw GGUF `_context_length` must never be
overwritten by VRAM-capping logic.

Needs no GPU, network, or libraries beyond pytest and pydantic.
"""

import io
import json
import re
import struct
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable deps before importing the module under test.
# Same pattern as test_kv_cache_estimation.py.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **kw: __import__("logging").getLogger(
    a[0] if a else __name__
)
sys.modules.setdefault("structlog", _structlog_stub)

# httpx -- stub only names referenced at import / class-definition time
_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
# Only when the real library is absent. sys.modules holds what has been IMPORTED, not
# what is installed, so setdefault does not defer to a real httpx that nothing in this
# process has touched yet: the stub wins and shadows it for the whole session. This stub
# has no Response, and starlette.testclient reads httpx.Response at import, so every
# module collected afterwards that reaches fastapi.testclient or routes.inference dies.
try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend
from models.inference import LoadResponse, InferenceStatusResponse


# ── Helpers ──────────────────────────────────────────────────────────


def _write_kv(buf: io.BytesIO, key: str, value, vtype: int) -> None:
    """Append a single GGUF KV pair to *buf*."""
    key_bytes = key.encode("utf-8")
    buf.write(struct.pack("<Q", len(key_bytes)))
    buf.write(key_bytes)
    buf.write(struct.pack("<I", vtype))
    if vtype == 4:  # UINT32
        buf.write(struct.pack("<I", value))
    elif vtype == 10:  # UINT64
        buf.write(struct.pack("<Q", value))
    elif vtype == 8:  # STRING
        val_bytes = value.encode("utf-8")
        buf.write(struct.pack("<Q", len(val_bytes)))
        buf.write(val_bytes)
    else:
        raise ValueError(f"Unsupported vtype in test helper: {vtype}")


def make_gguf(
    tmp_path: Path,
    arch: str,
    kvs: list,
    *,
    arch_first: bool = True,
    filename: str = "test.gguf",
) -> str:
    """Create a minimal valid GGUF v3 binary in *tmp_path*."""
    buf = io.BytesIO()
    buf.write(struct.pack("<I", 0x46554747))  # GGUF magic
    buf.write(struct.pack("<I", 3))  # version 3
    buf.write(struct.pack("<Q", 0))  # tensor count = 0

    ordered = []
    arch_entry = ("general.architecture", arch, 8)

    if arch_first:
        ordered.append(arch_entry)
    for suffix, val, vt in kvs:
        ordered.append((f"{arch}.{suffix}", val, vt))
    if not arch_first:
        ordered.append(arch_entry)

    buf.write(struct.pack("<Q", len(ordered)))
    for key, val, vt in ordered:
        _write_kv(buf, key, val, vt)

    path = tmp_path / filename
    path.write_bytes(buf.getvalue())
    return str(path)


@pytest.fixture
def backend():
    """Create a fresh LlamaCppBackend with side effects disabled."""
    with patch.object(LlamaCppBackend, "_kill_orphaned_servers"):
        with patch("atexit.register"):
            return LlamaCppBackend()


# =====================================================================
# A. TestNativeContextLengthProperty -- the new property
# =====================================================================


class TestNativeContextLengthProperty:
    """Tests the new `native_context_length` property on LlamaCppBackend."""

    def test_none_on_fresh_backend(self, backend):
        """Returns None when no model loaded."""
        assert backend.native_context_length is None

    def test_returns_raw_gguf_value(self, backend):
        """Directly returns _context_length when set."""
        backend._context_length = 131072
        assert backend.native_context_length == 131072

    def test_not_capped_by_effective(self, backend):
        """native_context_length ignores _effective_context_length."""
        backend._context_length = 131072
        backend._effective_context_length = 32768
        assert backend.native_context_length == 131072

    def test_not_capped_by_max(self, backend):
        """native_context_length ignores _max_context_length."""
        backend._context_length = 131072
        backend._max_context_length = 65536
        assert backend.native_context_length == 131072

    def test_none_after_unload(self, backend):
        """After unload_model(), returns None."""
        backend._context_length = 131072
        assert backend.native_context_length == 131072
        backend.unload_model()
        assert backend.native_context_length is None

    def test_after_gguf_parse(self, tmp_path, backend):
        """Synthetic GGUF with context_length=16384 populates the property."""
        path = make_gguf(
            tmp_path,
            "llama",
            [("context_length", 16384, 4)],
        )
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 16384

    def test_resets_between_parses(self, tmp_path, backend):
        """Second GGUF without context_length resets native to None."""
        path_a = make_gguf(
            tmp_path,
            "llama",
            [("context_length", 16384, 4)],
            filename = "a.gguf",
        )
        backend._read_gguf_metadata(path_a)
        assert backend.native_context_length == 16384

        path_b = make_gguf(
            tmp_path,
            "gpt2",
            [("block_count", 12, 4)],
            filename = "b.gguf",
        )
        backend._read_gguf_metadata(path_b)
        assert backend.native_context_length is None


# =====================================================================
# B. TestContextValueSeparation -- core invariant
# =====================================================================


class TestContextValueSeparation:
    """_context_length is never overwritten by VRAM logic."""

    def test_preserved_after_effective_set(self, backend):
        """Setting _effective_context_length does not change _context_length."""
        backend._context_length = 131072
        backend._effective_context_length = 32768
        assert backend._context_length == 131072
        assert backend.native_context_length == 131072

    def test_ordering_when_capped(self, backend):
        """native >= max >= effective holds when VRAM-capped."""
        backend._context_length = 131072
        backend._max_context_length = 65536
        backend._effective_context_length = 32768
        assert backend.native_context_length >= backend.max_context_length
        assert backend.max_context_length >= backend.context_length

    def test_all_equal_when_uncapped(self, backend):
        """All three equal when no VRAM constraint."""
        backend._context_length = 8192
        # No effective/max set -- properties fall back to _context_length.
        assert backend.native_context_length == 8192
        assert backend.max_context_length == 8192
        assert backend.context_length == 8192

    def test_fit_context_does_not_modify(self, backend):
        """_fit_context_to_vram() does not touch _context_length."""
        backend._context_length = 131072
        backend._n_layers = 32
        backend._n_kv_heads = 8
        backend._n_heads = 32
        backend._embedding_length = 4096
        original = backend._context_length

        # Tiny VRAM budget forces capping.
        result = backend._fit_context_to_vram(
            requested_ctx = 131072,
            available_mib = 512,  # very small
            model_size_bytes = 0,
        )
        # Returns the capped value without modifying _context_length.
        assert backend._context_length == original
        assert backend.native_context_length == original
        # Capped value must be <= requested.
        assert result <= 131072

    def test_native_gt_context_when_capped(self, backend):
        """native_context_length > context_length after VRAM capping."""
        backend._context_length = 131072
        backend._effective_context_length = 16384
        assert backend.native_context_length > backend.context_length


# =====================================================================
# C. TestPydanticModels -- LoadResponse & InferenceStatusResponse
# =====================================================================


class TestPydanticModels:
    """Tests native_context_length field on Pydantic models."""

    def test_load_response_has_field(self):
        """Field exists in LoadResponse.model_fields."""
        assert "native_context_length" in LoadResponse.model_fields
        assert "context_length" in LoadResponse.model_fields

    def test_load_response_defaults_none(self):
        """Omitting native_context_length defaults to None."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
        )
        assert resp.native_context_length is None

    def test_load_response_accepts_int(self):
        """native_context_length=131072 stores correctly."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            native_context_length = 131072,
        )
        assert resp.native_context_length == 131072

    def test_load_response_json_null(self):
        """None serializes to JSON null."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
        )
        data = json.loads(resp.model_dump_json())
        assert data["native_context_length"] is None

    def test_load_response_json_int(self):
        """131072 serializes to JSON number."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            native_context_length = 131072,
        )
        data = json.loads(resp.model_dump_json())
        assert data["native_context_length"] == 131072

    def test_status_response_has_field(self):
        """Field exists in InferenceStatusResponse.model_fields."""
        assert "native_context_length" in InferenceStatusResponse.model_fields
        assert "context_length" in InferenceStatusResponse.model_fields

    def test_status_response_has_chat_template_field(self):
        """Status includes chat_template so the UI can rehydrate after refresh."""
        assert "chat_template" in InferenceStatusResponse.model_fields

    def test_status_response_defaults_none(self):
        """Omitting native_context_length defaults to None."""
        resp = InferenceStatusResponse()
        assert resp.native_context_length is None

    def test_status_response_chat_template_roundtrip(self):
        """chat_template serializes and validates as part of status."""
        resp = InferenceStatusResponse(chat_template = "{{ messages }}")
        roundtripped = InferenceStatusResponse.model_validate_json(resp.model_dump_json())
        assert roundtripped.chat_template == "{{ messages }}"

    def test_roundtrip_preserves_value(self):
        """model_validate_json(model_dump_json()) round-trips."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            native_context_length = 131072,
        )
        roundtripped = LoadResponse.model_validate_json(resp.model_dump_json())
        assert roundtripped.native_context_length == 131072

    def test_context_length_roundtrip(self):
        """Runtime context_length serializes for non-GGUF/hub models."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            context_length = 8192,
        )
        roundtripped = LoadResponse.model_validate_json(resp.model_dump_json())
        assert roundtripped.context_length == 8192


# =====================================================================
# D. TestRouteCompleteness -- source-level verification
# =====================================================================


class TestRouteCompleteness:
    """All response construction sites in routes/inference.py include native_context_length."""

    @pytest.fixture(autouse = True)
    def _load_source(self):
        """Read routes/inference.py source once."""
        routes_path = Path(__file__).resolve().parent.parent / "routes" / "inference.py"
        self._source = routes_path.read_text(encoding = "utf-8")

    def _find_construction_blocks(self, class_name: str) -> list[str]:
        """Extract all code blocks that construct a given response class."""
        blocks = []
        idx = 0
        while True:
            start = self._source.find(f"{class_name}(", idx)
            if start == -1:
                break
            # Find the matching closing paren via a depth counter.
            depth = 0
            end = start
            for i, ch in enumerate(self._source[start:], start):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            blocks.append(self._source[start:end])
            idx = end
        return blocks

    def test_gguf_load_responses_have_field(self):
        """Every GGUF LoadResponse (is_gguf = True) includes native_context_length."""
        blocks = self._find_construction_blocks("LoadResponse")
        gguf_blocks = [b for b in blocks if "is_gguf = True" in b or "is_gguf=True" in b]
        assert (
            len(gguf_blocks) == 1
        ), f"Expected one shared GGUF LoadResponse block, found {len(gguf_blocks)}"
        for i, block in enumerate(gguf_blocks):
            assert (
                "_llama_runtime_fields(llama_backend)" in block
            ), f"GGUF LoadResponse block #{i} missing runtime fields:\n{block[:200]}"
        assert "for name in _InferenceRuntimeFields.model_fields" in self._source

    def test_non_gguf_load_responses_report_the_native_window(self):
        """Non-GGUF LoadResponse blocks carry native_context_length and max_context_length.

        These once had to be absent, back when only GGUF knew its own window; a non-GGUF
        block that leaves them off now defaults them to None and blanks the control.
        """
        blocks = self._find_construction_blocks("LoadResponse")
        non_gguf = [b for b in blocks if "is_gguf = True" not in b and "is_gguf=True" not in b]
        assert non_gguf, "Expected at least one non-GGUF LoadResponse block"
        for block in non_gguf:
            for field in ("native_context_length", "max_context_length"):
                assert re.search(
                    rf"{field} = _positive_int_or_none\(\s*_model_info\.get\(", block
                ), f"Non-GGUF LoadResponse should read {field} from _model_info:\n{block[:200]}"

    def test_non_gguf_load_responses_set_runtime_context_length(self):
        """Non-GGUF LoadResponse blocks report runtime context_length."""
        blocks = self._find_construction_blocks("LoadResponse")
        non_gguf = [b for b in blocks if "is_gguf = True" not in b and "is_gguf=True" not in b]
        assert non_gguf, "Expected at least one non-GGUF LoadResponse block"
        for block in non_gguf:
            assert (
                "context_length" in block
            ), f"Non-GGUF LoadResponse should set context_length:\n{block[:200]}"

    def test_status_path(self):
        """InferenceStatusResponse construction with llama_backend has the field.

        The route may splat the helper's result straight in, or bind it first
        and adjust a field before passing it on. Both carry the runtime fields.
        """
        blocks = self._find_construction_blocks("InferenceStatusResponse")
        found = False
        for block in blocks:
            if "llama_backend" not in block:
                continue
            if "_llama_runtime_fields(llama_backend)" in block:
                found = True
                break
            if "**_runtime_fields" in block:
                # Only counts if that dict is the helper's, not any local name.
                assert (
                    "_runtime_fields = _llama_runtime_fields(llama_backend)" in self._source
                ), "**_runtime_fields is not built from _llama_runtime_fields(llama_backend)"
                found = True
                break
        assert found, "No InferenceStatusResponse block with llama_backend has runtime fields"
        assert "for name in _InferenceRuntimeFields.model_fields" in self._source

    def test_non_gguf_status_path_reports_runtime_context_length(self):
        """Non-GGUF InferenceStatusResponse reports context_length from model_info."""
        blocks = self._find_construction_blocks("InferenceStatusResponse")
        found = False
        for block in blocks:
            if "is_gguf = False" in block and "context_length" in block:
                found = True
                break
        assert found, "No non-GGUF InferenceStatusResponse block with context_length"

    def test_openai_models_listing_reports_context_length(self):
        """/v1/models includes context_length when the backend knows it."""
        assert 'entry["context_length"]' in self._source
        assert 'model_info.get("context_length")' in self._source


# =====================================================================
# E. TestEdgeCases
# =====================================================================


class TestNativeContextEdgeCases:
    """Edge cases for native_context_length."""

    def test_context_length_zero(self, tmp_path, backend):
        """GGUF context_length=0 returns 0, not None."""
        path = make_gguf(tmp_path, "llama", [("context_length", 0, 4)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 0

    def test_context_length_uint32_max(self, tmp_path, backend):
        """2^32 - 1 survives without truncation."""
        val = 2**32 - 1
        path = make_gguf(tmp_path, "llama", [("context_length", val, 4)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == val

    def test_context_length_uint64(self, tmp_path, backend):
        """UINT64 type context_length parsed correctly."""
        val = 2**33  # exceeds UINT32 range
        path = make_gguf(tmp_path, "llama", [("context_length", val, 10)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == val

    def test_no_context_length_in_gguf(self, tmp_path, backend):
        """GGUF without context_length key yields None."""
        path = make_gguf(tmp_path, "llama", [("block_count", 32, 4)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length is None

    def test_native_equals_context_when_uncapped(self, backend):
        """Both equal when no VRAM cap applied."""
        backend._context_length = 8192
        assert backend.native_context_length == backend.context_length

    def test_native_survives_parse_then_cap(self, tmp_path, backend):
        """Parse then set effective cap: native unchanged."""
        path = make_gguf(
            tmp_path,
            "llama",
            [
                ("context_length", 131072, 4),
                ("block_count", 32, 4),
                ("attention.head_count", 32, 4),
                ("attention.head_count_kv", 8, 4),
                ("embedding_length", 4096, 4),
            ],
        )
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 131072

        # Simulate VRAM capping via effective and max.
        backend._effective_context_length = 16384
        backend._max_context_length = 32768
        assert backend.native_context_length == 131072


# =====================================================================
# F. TestCrossPlatform -- binary I/O and serialization
# =====================================================================


class TestCrossPlatform:
    """Binary I/O and serialization correctness across platforms."""

    def test_le_uint32_context_length(self, tmp_path, backend):
        """Little-endian UINT32 parsed correctly."""
        path = make_gguf(tmp_path, "llama", [("context_length", 16384, 4)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 16384

    def test_le_uint64_context_length(self, tmp_path, backend):
        """Little-endian UINT64 parsed correctly."""
        path = make_gguf(tmp_path, "llama", [("context_length", 16384, 10)])
        backend._read_gguf_metadata(path)
        assert backend.native_context_length == 16384

    def test_gguf_magic_le_byte_order(self, tmp_path):
        """Magic 0x46554747 matches GGUF spec (little-endian 'GGUF')."""
        path = tmp_path / "magic_check.gguf"
        buf = io.BytesIO()
        buf.write(struct.pack("<I", 0x46554747))
        raw = buf.getvalue()
        # 'G' = 0x47, 'G' = 0x47, 'U' = 0x55, 'F' = 0x46
        assert raw == b"GGUF"

    def test_json_serialization_deterministic(self):
        """model_dump_json() is consistent across calls."""
        resp = LoadResponse(
            status = "loaded",
            model = "test",
            display_name = "Test",
            inference = {},
            native_context_length = 131072,
        )
        json1 = resp.model_dump_json()
        json2 = resp.model_dump_json()
        assert json1 == json2
        assert '"native_context_length":131072' in json1


def test_the_status_route_reports_what_a_self_sizing_load_asked_for():
    """The UI re-seeds its context pin from it, because the resolved window cannot say
    whether anyone chose that length; without it a pin is invisible after a refresh."""
    route_src = (Path(__file__).resolve().parents[1] / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    assert "requested_context_length = llama_backend.requested_n_ctx" in route_src
    # 0 is "size it yourself"; None is "records no request", and the UI tells them apart.
    assert (
        "requested_context_length = _nonnegative_int_or_none(\n"
        '                model_info.get("requested_context_length")\n'
        "            )," in route_src
    )
    # The parent cannot recompute the group once the worker holds the model: it mirrors all.
    src = (Path(__file__).resolve().parents[1] / "core/inference/worker.py").read_text("utf-8")
    assert '"native_context_length",\n                "max_context_length",' in src

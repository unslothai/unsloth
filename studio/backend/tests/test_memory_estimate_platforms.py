# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Estimate invariants hold on every platform and accelerator; only host seams move, nothing launches."""

from __future__ import annotations

import importlib.util as _ilu
import os
import platform as _platform
import sys
import types as _types
from pathlib import Path

import pytest

# Stub heavy deps before import; test_backend_tests_stub_heavy_imports.py enforces this.

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# Stub needs get_logger: process-wide setdefault, and freshness_flow calls it at import.
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

# Only stub httpx when missing; otherwise huggingface_hub.errors imports break.
try:
    import httpx as _httpx_real  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
    ):
        setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))

    class _FakeTimeout:
        def __init__(self, *a, **kw):
            pass

    _httpx_stub.Timeout = _FakeTimeout
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    sys.modules["httpx"] = _httpx_stub

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import core.inference.llama_cpp as llama_mod  # noqa: E402
import routes.inference as ri  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from models.inference import EstimateMemoryRequest  # noqa: E402

# Load shared fixtures by path, not copied; `tests` is not importable on every runner.
_TESTS_DIR = Path(__file__).resolve().parent

_kv_spec = _ilu.spec_from_file_location(
    "_kv_cache_estimation_for_platform_matrix", _TESTS_DIR / "test_kv_cache_estimation.py"
)
_kv_mod = _ilu.module_from_spec(_kv_spec)
_kv_spec.loader.exec_module(_kv_mod)
_make_gguf_bytes = _kv_mod._make_gguf_bytes

_plat_spec = _ilu.spec_from_file_location(
    "_llama_extra_args_platforms_for_memory_estimate",
    _TESTS_DIR / "test_llama_extra_args_platforms.py",
)
_plat_mod = _ilu.module_from_spec(_plat_spec)
_plat_spec.loader.exec_module(_plat_mod)

PLATFORMS = _plat_mod.PLATFORMS
# Deliberately does NOT move os.name; see _apply_cell.
_apply_platform = _plat_mod._apply_platform

_GIB = 1024**3

_SYSTEM = {"linux": "Linux", "wsl2": "Linux", "windows": "Windows", "macos": "Darwin"}


_UPSTREAM_ACCELERATORS = {
    label: (vulkan, memory) for label, vulkan, memory in _plat_mod.ACCELERATORS
}

ACCELERATORS = [
    ("nvidia-single", *_UPSTREAM_ACCELERATORS["nvidia-single"]),
    ("nvidia-multi", *_UPSTREAM_ACCELERATORS["nvidia-multi"]),
    ("amd-rocm", False, [(0, 12_000, 16_000)]),
    ("amd-vulkan", *_UPSTREAM_ACCELERATORS["amd-vulkan"]),
    ("apple-unified", False, [(0, 40_000, 65_536)]),
    ("cpu-only", *_UPSTREAM_ACCELERATORS["cpu-only"]),
]

_UNIFIED = "apple-unified"
_CPU_ONLY = "cpu-only"


def _reachable(platform_label: str, accelerator_label: str) -> bool:
    """Apple unified memory exists only under Darwin, so that pair is excluded; every other cell is kept."""
    return accelerator_label != _UNIFIED or platform_label == "macos"


MATRIX = [
    pytest.param(p, a, id = f"{p[0]}-{a[0]}")
    for p in PLATFORMS
    for a in ACCELERATORS
    if _reachable(p[0], a[0])
]

assert len(MATRIX) == 4 * 6 - 3 == 21


def _snapshot(memory) -> tuple:
    """Fills the real main._system_gpu_cache, keeping an empty probed list distinct from unfilled."""
    devices = [
        {"index": index, "memory_total_gb": round(total / 1024, 2), "vram_free_gb": free / 1024}
        for index, free, total in memory
    ]
    inference_gpu = {"available": bool(devices), "devices": devices}
    return (0.0, ({"available": bool(devices), "devices": devices}, inference_gpu))


def _apply_cell(monkeypatch, platform_row, accelerator_row) -> None:
    """Patch only host seams; never os.name, which swaps pathlib's flavour and breaks tmp_path files."""
    platform_label = platform_row[0]
    accelerator_label, vulkan, memory = accelerator_row

    _apply_platform(monkeypatch, platform_row)
    # Asserted: a `from sys import platform` in either module would silently void the matrix.
    assert ri.sys is sys and llama_mod.sys is sys

    apple = accelerator_label == _UNIFIED
    monkeypatch.setattr(_platform, "system", lambda: _SYSTEM[platform_label], raising = False)
    monkeypatch.setattr(
        _platform,
        "machine",
        lambda: "arm64" if apple else ("AMD64" if platform_label == "windows" else "x86_64"),
        raising = False,
    )

    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(_system_gpu_cache = _snapshot(memory)))

    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: vulkan)
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_effective_gpu_count",
        staticmethod(
            lambda gpu_indices = None: len(gpu_indices) if gpu_indices is not None else len(memory)
        ),
    )

    monkeypatch.setattr(
        "utils.hardware.hardware.IS_ROCM", accelerator_label == "amd-rocm", raising = False
    )
    for mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(mask, raising = False)
    # Inherited GGML/LLAMA_ARG_* would be read as user settings.
    for inherited in (
        "LLAMA_ARG_CTX_SIZE",
        "LLAMA_ARG_MMPROJ",
        "LLAMA_ARG_SPEC_DRAFT_MODEL",
        "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K",
        "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V",
        "LLAMA_ARG_SPLIT_MODE",
        "LLAMA_ARG_DEVICE",
    ):
        monkeypatch.delenv(inherited, raising = False)

    # Pin the binary: determinism, and shutil.which raises under simulated win32 on Linux.
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(
            lambda **kw: (
                "C:\\llama.cpp\\llama-server.exe"
                if platform_label == "windows"
                else "/opt/llama.cpp/llama-server"
            )
        ),
    )

    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(
            lambda cls, binary = None: {
                "found": True,
                "supports_mtp": True,
                "spec_draft_cache_k_flag": True,
                "spec_draft_cache_v_flag": True,
                "spec_draft_n_max_flag": "--spec-draft-n-max",
            }
        ),
    )


@pytest.fixture(autouse = True)
def _clear_estimate_caches():
    """Both module caches outlive a cell, so clear them each time or stale entries satisfy assertions."""
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()
    yield
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()


@pytest.fixture(autouse = True)
def _metal_budget_tripwire(monkeypatch):
    """Never reach _apple_metal_memory_budget_bytes: its bare mlx.core import aborts the process on
    macOS."""
    calls: list[str] = []

    def _tripwire() -> int:
        calls.append("reached")
        return 0

    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(_tripwire)
    )
    yield calls
    assert calls == [], (
        "the memory estimate reached LlamaCppBackend._apple_metal_memory_budget_bytes "
        f"{len(calls)} time(s). That function imports mlx.core, which aborts the process "
        "at the C level on macOS once torch is loaded, and this endpoint runs on every "
        "settings change."
    )


_GQA_FIELDS = {
    "context_length": 262144,
    "block_count": 32,
    "attention.head_count": 32,
    "attention.head_count_kv": 8,
    "attention.key_length": 128,
    "attention.value_length": 128,
    "embedding_length": 4096,
    "feed_forward_length": 12288,
    "vocab_size": 152064,
}

# Sparse 3 GiB pad so file sizes are not rounded to noise.
_WEIGHTS_BYTES = 3 * _GIB
_PROJECTOR_BYTES = 600 * 1024 * 1024
_DRAFTER_BYTES = 400 * 1024 * 1024
# NTFS zero-fills unless marked sparse first (ENOSPC on Windows); fallback stays GB-scale.
_DENSE_PAD_DIVISOR = 12


def _try_make_sparse(handle) -> bool:
    """Windows needs FSCTL_SET_SPARSE before extending the file, or the extension is committed in full."""
    if os.name != "nt" and not os.environ.get("FORCE_DENSE_PAD"):
        return True
    try:
        import ctypes
        import msvcrt

        _FSCTL_SET_SPARSE = 0x000900C4
        returned = ctypes.c_ulong(0)
        return bool(
            ctypes.windll.kernel32.DeviceIoControl(
                ctypes.c_void_p(msvcrt.get_osfhandle(handle.fileno())),
                _FSCTL_SET_SPARSE,
                None,
                0,
                None,
                0,
                ctypes.byref(returned),
                None,
            )
        )
    except Exception:
        return False


def _pad_to(path: Path, pad: int) -> None:
    """Extend `path` to `pad` bytes without paying for them where that is possible."""
    with open(path, "r+b") as handle:
        if not _try_make_sparse(handle):
            pad = max(len(handle.read()), pad // _DENSE_PAD_DIVISOR)
        handle.truncate(pad)


def _write_gguf(directory: Path, name: str, arch: str, fields: dict, *, pad: int) -> str:
    kv = {"general.architecture": arch}
    kv.update({f"{arch}.{k}": v for k, v in fields.items()})
    path = directory / name
    path.write_bytes(_make_gguf_bytes(arch, kv))
    if pad:
        _pad_to(path, pad)
    return str(path)


def _config(gguf_path: str, **overrides) -> SimpleNamespace:
    fields = dict(
        identifier = "local/model",
        gguf_file = gguf_path,
        is_gguf = True,
        gguf_variant = None,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        is_vision = False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture(scope = "module")
def shapes(tmp_path_factory):
    """Module scope: the ten sparse shapes are reused across cells, and nothing below mutates them."""
    root = tmp_path_factory.mktemp("platform-matrix-shapes")
    built: dict[str, tuple[str, SimpleNamespace]] = {}

    gqa = _write_gguf(root, "gqa.gguf", "qwen3", _GQA_FIELDS, pad = _WEIGHTS_BYTES)
    built["gqa"] = (gqa, _config(gqa))

    mla = _write_gguf(
        root,
        "mla.gguf",
        "deepseek2",
        {
            **_GQA_FIELDS,
            "attention.kv_lora_rank": 512,
            "attention.key_length_mla": 576,
            "attention.value_length_mla": 512,
        },
        pad = _WEIGHTS_BYTES,
    )
    built["mla"] = (mla, _config(mla))

    swa = _write_gguf(
        root,
        "swa.gguf",
        "gemma3",
        {**_GQA_FIELDS, "attention.sliding_window": 1024},
        pad = _WEIGHTS_BYTES,
    )
    built["swa"] = (swa, _config(swa))

    hybrid = _write_gguf(
        root,
        "hybrid.gguf",
        "granitehybrid",
        {
            **_GQA_FIELDS,
            "ssm.inner_size": 8192,
            "ssm.state_size": 128,
            "ssm.group_count": 8,
            "ssm.conv_kernel": 4,
            "full_attention_interval": 4,
        },
        pad = _WEIGHTS_BYTES,
    )
    built["hybrid_mamba"] = (hybrid, _config(hybrid))

    # Pure SSM loads with no attention dims in llama.cpp; the kv_estimable=False path.
    ssm = _write_gguf(
        root,
        "pure-ssm.gguf",
        "mamba",
        {
            "block_count": 48,
            "embedding_length": 2048,
            "context_length": 8192,
            "ssm.conv_kernel": 4,
            "ssm.inner_size": 4096,
            "ssm.state_size": 16,
            "ssm.time_step_rank": 128,
        },
        pad = _WEIGHTS_BYTES,
    )
    built["pure_ssm"] = (ssm, _config(ssm))

    nextn = _write_gguf(
        root,
        "nextn.gguf",
        "qwen3",
        {**_GQA_FIELDS, "nextn_predict_layers": 2},
        pad = _WEIGHTS_BYTES,
    )
    built["nextn_mtp"] = (nextn, _config(nextn))

    embedding = _write_gguf(
        root,
        "embedding.gguf",
        "bert",
        {**_GQA_FIELDS, "context_length": 8192, "pooling_type": 2},
        pad = _WEIGHTS_BYTES,
    )
    built["embedding"] = (embedding, _config(embedding, identifier = "local/embed-model"))

    vision = _write_gguf(root, "vision.gguf", "qwen3", _GQA_FIELDS, pad = _WEIGHTS_BYTES)
    projector = _write_gguf(
        root, "mmproj-vision.gguf", "clip", {"has_vision_encoder": 1}, pad = _PROJECTOR_BYTES
    )
    built["vision_projector"] = (
        vision,
        _config(vision, gguf_mmproj_file = projector, is_vision = True),
    )

    target = _write_gguf(root, "spec-target.gguf", "qwen3", _GQA_FIELDS, pad = _WEIGHTS_BYTES)
    drafter = _write_gguf(
        root,
        "mtp-draft.gguf",
        "qwen3",
        {**_GQA_FIELDS, "block_count": 2},
        pad = _DRAFTER_BYTES,
    )
    built["mtp_drafter"] = (target, _config(target, gguf_mtp_file = drafter))

    truncated = root / "truncated.gguf"
    truncated.write_bytes(b"GGUF\x03\x00\x00\x00 truncated, nothing further is readable")
    _pad_to(truncated, _WEIGHTS_BYTES)
    built["truncated_header"] = (str(truncated), _config(str(truncated)))

    return built


_SPEC_SHAPES = {"mtp_drafter": "mtp"}

SHAPE_NAMES = [
    "gqa",
    "mla",
    "swa",
    "hybrid_mamba",
    "pure_ssm",
    "nextn_mtp",
    "embedding",
    "vision_projector",
    "mtp_drafter",
    "truncated_header",
]

CONTEXTS = (4096, 32768, 131072)


def _price(shapes, shape_name: str, **kwargs):
    """Goes through the real route: the invariants are claims about the response, not the breakdown."""
    gguf_path, config = shapes[shape_name]
    spec = _SPEC_SHAPES.get(shape_name)
    if spec is not None:
        kwargs.setdefault("speculative_type", spec)
    request = EstimateMemoryRequest(model_path = gguf_path, **kwargs)

    original = ri._cached_estimate_config
    ri._cached_estimate_config = lambda *a, **kw: config
    try:
        return asyncio.run(
            ri.estimate_memory(request, fastapi_request = None, current_subject = "test")
        )
    finally:
        ri._cached_estimate_config = original


_ITEMS = (
    "weights_bytes",
    "kv_bytes",
    "compute_bytes",
    "drafter_runtime_bytes",
    "projector_runtime_bytes",
)

_NON_NEGATIVE = (*_ITEMS, "drafter_runtime_gpu_bytes", "total_bytes", "gpu_bytes", "n_ctx")


def _itemized_sum(response) -> int:
    """The itemization sums five terms: projector_runtime_bytes is a fifth line inside total_bytes."""
    return sum(getattr(response, item) for item in _ITEMS)


def _assert_core_invariants(
    response,
    *,
    cell: str,
    shape: str,
    note: str = "",
) -> None:
    """Everything that must hold of any answer, on any host, for any model."""
    where = f"[{cell}] {shape}{' ' + note if note else ''}"

    assert response.available is True, f"{where}: {response.reason}"

    for field in _NON_NEGATIVE:
        assert getattr(response, field) >= 0, f"{where}: {field} is {getattr(response, field)}"

    assert _itemized_sum(response) == response.total_bytes, (
        f"{where}: the itemization does not sum to the total. "
        f"weights={response.weights_bytes} kv={response.kv_bytes} "
        f"compute={response.compute_bytes} drafter={response.drafter_runtime_bytes} "
        f"projector={response.projector_runtime_bytes} "
        f"sum={_itemized_sum(response)} total={response.total_bytes} "
        f"(delta {response.total_bytes - _itemized_sum(response)})"
    )

    assert response.gpu_bytes <= response.total_bytes, (
        f"{where}: gpu_bytes {response.gpu_bytes} exceeds total_bytes "
        f"{response.total_bytes} -- the GPU share cannot be more than everything"
    )

    assert response.drafter_runtime_gpu_bytes <= response.drafter_runtime_bytes, (
        f"{where}: drafter_runtime_gpu_bytes {response.drafter_runtime_gpu_bytes} exceeds "
        f"drafter_runtime_bytes {response.drafter_runtime_bytes}"
    )

    if not response.kv_estimable:
        assert response.kv_bytes == 0, f"{where}: unsizable KV reported {response.kv_bytes} bytes"
        assert response.compute_bytes == 0, f"{where}: unsizable KV priced compute buffers"


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_every_shape_is_internally_consistent(
    monkeypatch, shapes, platform, accelerator
):
    """Asserts properties, never magnitudes: a sum off by one byte or a GPU share above total fails."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"

    for shape in SHAPE_NAMES:
        for n_ctx in CONTEXTS:
            ri._estimate_files_cache.clear()
            response = _price(shapes, shape, n_ctx = n_ctx)
            _assert_core_invariants(response, cell = cell, shape = shape, note = f"n_ctx={n_ctx}")


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_weights_do_not_move_with_the_context_slider(
    monkeypatch, shapes, platform, accelerator
):
    """weights_bytes is derived by subtracting the context term, so it must not move as n_ctx changes."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"

    for shape in SHAPE_NAMES:
        weights: list[int] = []
        kv: list[int] = []
        for n_ctx in CONTEXTS:
            ri._estimate_files_cache.clear()
            response = _price(shapes, shape, n_ctx = n_ctx)
            weights.append(response.weights_bytes)
            kv.append(response.kv_bytes)

        assert len(set(weights)) == 1, (
            f"[{cell}] {shape}: weights_bytes moved with the context slider: "
            f"{dict(zip(CONTEXTS, weights))}. The weights term is a subtraction; if it "
            f"moves, the term added and the term removed are no longer the same bytes."
        )
        # Non-decreasing: SWA caches plateau and unsizable ones stay 0.
        assert kv == sorted(
            kv
        ), f"[{cell}] {shape}: kv_bytes is not non-decreasing in n_ctx: {dict(zip(CONTEXTS, kv))}"


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_settings_do_not_break_the_itemization(
    monkeypatch, shapes, platform, accelerator
):
    """Each common setting reaches a different arm of the split, and none may break the itemization."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"

    settings = [
        ("f16 cache", dict(cache_type_kv = "f16")),
        ("q4_0 cache", dict(cache_type_kv = "q4_0")),
        ("4 slots", dict(n_parallel = 4)),
        ("no-kv-offload", dict(llama_extra_args = ["-nkvo"])),
        ("manual 0 layers", dict(gpu_memory_mode = "manual", gpu_layers = 0)),
        ("manual all layers", dict(gpu_memory_mode = "manual", gpu_layers = 999)),
        ("tensor split", dict(tensor_parallel = True, selected_gpu_ids = [0, 1])),
        ("checkpoints", dict(ctx_checkpoints = 8)),
        ("draft depth", dict(spec_draft_n_max = 8, spec_draft_cache_type = "q8_0")),
        ("vision off", dict(disable_vision = True)),
        ("native context", dict(n_ctx = 0)),
    ]

    for shape in ("gqa", "swa", "pure_ssm", "vision_projector", "mtp_drafter"):
        for note, kwargs in settings:
            ri._estimate_files_cache.clear()
            kwargs = {"n_ctx": 32768, **kwargs}
            response = _price(shapes, shape, **kwargs)
            _assert_core_invariants(response, cell = cell, shape = shape, note = note)


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_an_offloaded_byte_is_never_a_freed_byte(
    monkeypatch, shapes, platform, accelerator
):
    """Offloading moves bytes out of gpu_bytes, never total_bytes, since an offloaded byte is not freed."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"

    for shape in ("gqa", "vision_projector", "mtp_drafter", "nextn_mtp"):
        ri._estimate_files_cache.clear()
        resident = _price(shapes, shape, n_ctx = 32768)

        for note, kwargs in (
            ("manual 0 layers", dict(gpu_memory_mode = "manual", gpu_layers = 0)),
            ("no-kv-offload", dict(llama_extra_args = ["-nkvo"])),
            ("no-mmproj-offload", dict(llama_extra_args = ["--no-mmproj-offload"])),
            ("cpu device", dict(llama_extra_args = ["--device", "none"])),
        ):
            ri._estimate_files_cache.clear()
            offloaded = _price(shapes, shape, n_ctx = 32768, **kwargs)
            _assert_core_invariants(offloaded, cell = cell, shape = shape, note = note)

            assert offloaded.total_bytes == resident.total_bytes, (
                f"[{cell}] {shape} under {note}: total_bytes fell from "
                f"{resident.total_bytes} to {offloaded.total_bytes}. Offloading moves "
                f"bytes between pools; it does not free them, and on unified memory "
                f"there is only one pool to move them within."
            )
            assert offloaded.gpu_bytes <= resident.gpu_bytes, (
                f"[{cell}] {shape} under {note}: gpu_bytes ROSE from "
                f"{resident.gpu_bytes} to {offloaded.gpu_bytes}"
            )


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_a_probed_empty_inventory_shows_no_gpu_footprint(
    monkeypatch, shapes, platform, accelerator
):
    """gpu_bytes is zero only where a probe ran and found no GPU; an unfilled snapshot is not absence."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"
    cpu_only = accelerator[0] == _CPU_ONLY

    for shape in ("gqa", "pure_ssm", "vision_projector"):
        ri._estimate_files_cache.clear()
        response = _price(shapes, shape, n_ctx = 32768)
        _assert_core_invariants(response, cell = cell, shape = shape)

        if cpu_only:
            assert response.gpu_bytes == 0, (
                f"[{cell}] {shape}: a probed-empty inventory still reported "
                f"{response.gpu_bytes} GPU bytes"
            )
            assert response.drafter_runtime_gpu_bytes == 0, f"[{cell}] {shape}"
            assert response.total_bytes > 0, f"[{cell}] {shape}"
        else:
            assert response.gpu_bytes > 0, (
                f"[{cell}] {shape}: a probed inventory with {len(accelerator[2])} "
                f"device(s) reported no GPU footprint at all"
            )


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_an_unsizable_kv_still_carries_its_layer_count(
    monkeypatch, shapes, platform, accelerator
):
    """A pure-SSM header must still report block_count, or the offloaded layer fraction defaults to 1.0."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"

    ri._estimate_files_cache.clear()
    response = _price(shapes, "pure_ssm", n_ctx = 131072)
    assert response.kv_estimable is False, f"[{cell}]"
    assert response.layer_count == 48, (
        f"[{cell}] pure_ssm: kv_estimable is False and layer_count is "
        f"{response.layer_count}; without it the offload split has no denominator"
    )

    ri._estimate_files_cache.clear()
    pinned = _price(shapes, "pure_ssm", n_ctx = 131072, gpu_memory_mode = "manual", gpu_layers = 0)
    _assert_core_invariants(pinned, cell = cell, shape = "pure_ssm", note = "-ngl 0")
    assert (
        pinned.gpu_bytes == 0
    ), f"[{cell}] pure_ssm at --gpu-layers 0 reported {pinned.gpu_bytes} GPU bytes"


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_an_unreadable_header_answers_without_inventing_a_cache(
    monkeypatch, shapes, platform, accelerator
):
    """A truncated GGUF: a well-formed "cannot size this", never a partial number."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"

    ri._estimate_files_cache.clear()
    response = _price(shapes, "truncated_header", n_ctx = 131072)
    _assert_core_invariants(response, cell = cell, shape = "truncated_header")
    assert response.kv_estimable is False, f"[{cell}]"
    assert response.n_ctx == 0, f"[{cell}]: priced a context off a header it could not read"
    assert response.layer_count is None, f"[{cell}]"


def test_platform_matrix_the_itemization_is_five_terms_not_four(monkeypatch, shapes):
    """Four terms sum to the total only without a vision projector; projector_runtime_bytes is the fifth."""
    _apply_cell(monkeypatch, PLATFORMS[0], ACCELERATORS[0])

    ri._estimate_files_cache.clear()
    plain = _price(shapes, "gqa", n_ctx = 32768)
    four = plain.weights_bytes + plain.kv_bytes + plain.compute_bytes + plain.drafter_runtime_bytes
    assert plain.projector_runtime_bytes == 0
    assert four == plain.total_bytes

    ri._estimate_files_cache.clear()
    vision = _price(shapes, "vision_projector", n_ctx = 32768)
    four = (
        vision.weights_bytes + vision.kv_bytes + vision.compute_bytes + vision.drafter_runtime_bytes
    )
    assert vision.projector_runtime_bytes > 0
    assert four + vision.projector_runtime_bytes == vision.total_bytes
    assert four != vision.total_bytes, (
        "the projector term is no longer outside the four-item sum; if it has been "
        "folded into another line, this test and the PR body should both say four"
    )


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_the_metal_budget_is_never_reached(
    monkeypatch, shapes, platform, accelerator, _metal_budget_tripwire
):
    """The conftest pins _metal_device_is_paravirtual to False, masking Apple arms; restore it here."""
    _apply_cell(monkeypatch, platform, accelerator)
    apple = accelerator[0] == _UNIFIED

    if apple:
        # is_apple_silicon() is True here, the only gate before the mlx import.
        from utils.hardware import is_apple_silicon
        assert is_apple_silicon() is True

    for paravirtual in (False, True):
        if apple:
            monkeypatch.setattr(
                llama_mod, "_metal_device_is_paravirtual", lambda: paravirtual, raising = False
            )
            monkeypatch.setattr(
                ri, "_metal_device_is_paravirtual", lambda: paravirtual, raising = False
            )
        for shape in SHAPE_NAMES:
            for kwargs in (
                dict(n_ctx = 0),
                dict(n_ctx = 131072),
                dict(n_ctx = 0, llama_extra_args = ["-c", "0"]),
                dict(n_ctx = 131072, gpu_memory_mode = "manual", gpu_layers = 0),
            ):
                ri._estimate_files_cache.clear()
                _price(shapes, shape, **kwargs)
        if not apple:
            break

    assert (
        _metal_budget_tripwire == []
    ), f"[{platform[0]}-{accelerator[0]}] reached _apple_metal_memory_budget_bytes"


def test_platform_matrix_the_platform_label_alone_changes_nothing(monkeypatch, shapes):
    """The estimate reads the host, not the OS, so the platform label alone must change no answer."""
    answers = {}
    for platform_row in PLATFORMS:
        for accelerator_row in ACCELERATORS:
            if not _reachable(platform_row[0], accelerator_row[0]):
                continue
            with pytest.MonkeyPatch.context() as patcher:
                _apply_cell(patcher, platform_row, accelerator_row)
                ri._estimate_files_cache.clear()
                response = _price(shapes, "gqa", n_ctx = 32768)
                answers[(platform_row[0], accelerator_row[0])] = (
                    response.total_bytes,
                    response.gpu_bytes,
                    response.kv_bytes,
                    response.weights_bytes,
                )

    for accelerator_label in ("nvidia-single", "nvidia-multi", "amd-rocm", "amd-vulkan", _CPU_ONLY):
        by_platform = {
            platform_label: answer
            for (platform_label, acc), answer in answers.items()
            if acc == accelerator_label
        }
        assert (
            len(set(by_platform.values())) == 1
        ), f"{accelerator_label} priced differently per platform: {by_platform}"

    # Unified memory is a capacity fact; the frontend uses singleMemoryPool to read the row.
    assert answers[("macos", _UNIFIED)] == answers[("linux", "nvidia-single")]


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_the_probed_inventory_owns_the_split_on_a_vulkan_build(
    monkeypatch, shapes, platform, accelerator
):
    """A Vulkan build sees zero CUDA devices, so the split decision must trust the probed inventory."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"
    vulkan = accelerator[1]

    monkeypatch.setitem(
        sys.modules,
        "main",
        SimpleNamespace(_system_gpu_cache = _snapshot([(0, 12_000, 16_000), (1, 12_000, 16_000)])),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_effective_gpu_count",
        staticmethod(lambda gpu_indices = None: len(gpu_indices) if gpu_indices is not None else 0),
    )

    ri._estimate_files_cache.clear()
    unpinned = _price(shapes, "gqa", n_ctx = 32768, tensor_parallel = True)
    _assert_core_invariants(unpinned, cell = cell, shape = "gqa", note = "unpinned tensor")

    ri._estimate_files_cache.clear()
    pinned = _price(shapes, "gqa", n_ctx = 32768, tensor_parallel = True, selected_gpu_ids = [0, 1])
    _assert_core_invariants(pinned, cell = cell, shape = "gqa", note = "pinned tensor")

    if vulkan:
        assert unpinned.compute_bytes == pinned.compute_bytes, (
            f"[{cell}] a Vulkan build with two probed devices priced a single-device "
            f"load ({unpinned.compute_bytes}) where the same pinned request prices "
            f"{pinned.compute_bytes}; the CUDA count answered instead of the inventory"
        )
    else:
        assert (
            unpinned.compute_bytes < pinned.compute_bytes
        ), f"[{cell}] a CUDA-shaped build reporting zero devices still priced a two-device split"


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_the_tensor_latch_lookup_runs_on_every_cell(
    monkeypatch, shapes, platform, accelerator
):
    """_tensor_latches_allow_a_split swallows errors as allowed, so each cell must prove it ran."""
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"

    asked: list[tuple] = []
    real = LlamaCppBackend._tensor_quant_kv_unsupported_binary

    def _spy(
        cls,
        binary,
        cache_types = ("f16", "f16"),
    ):
        asked.append((binary, cache_types))
        return real.__func__(cls, binary, cache_types)

    monkeypatch.setattr(LlamaCppBackend, "_tensor_quant_kv_unsupported_binary", classmethod(_spy))

    ri._estimate_files_cache.clear()
    response = _price(
        shapes,
        "gqa",
        n_ctx = 32768,
        cache_type_kv = "q8_0",
        tensor_parallel = True,
        selected_gpu_ids = [0, 1],
    )
    _assert_core_invariants(response, cell = cell, shape = "gqa", note = "tensor + q8_0")

    assert asked, (
        f"[{cell}] the tensor-split latch lookup never reached the latches. Something "
        f"inside _tensor_latches_allow_a_split raised and its fail-open except arm "
        f"answered instead, so this cell priced a tensor split it never checked."
    )
    assert asked[0][1] == ("q8_0", "q8_0"), f"[{cell}]: latch asked about {asked[0][1]}"


# Explicit --gpu-layers 0 is knowable without a layer count.


def test_platform_matrix_a_manual_zero_offload_is_honoured_on_an_unreadable_header(
    monkeypatch, shapes
):
    _apply_cell(monkeypatch, PLATFORMS[0], ACCELERATORS[0])
    ri._estimate_files_cache.clear()
    manual = dict(gpu_memory_mode = "manual", gpu_layers = 0)
    pinned = _price(shapes, "truncated_header", n_ctx = 32768, **manual)
    assert pinned.layer_count is None
    assert pinned.gpu_bytes == 0, (
        f"--gpu-layers 0 on an unreadable header still reports {pinned.gpu_bytes} GPU "
        f"bytes out of a {pinned.total_bytes}-byte total"
    )

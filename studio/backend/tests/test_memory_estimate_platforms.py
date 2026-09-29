# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Load-Model memory estimate across every platform and accelerator Unsloth ships on.

``test_memory_estimate.py`` proves the arithmetic on one host; this file asks whether
the same number comes out, and stays internally consistent, when the host changes. The
estimate reads three host-shaped seams -- ``sys.platform``, the probed inference
inventory, and whether the llama.cpp build is Vulkan -- and each moves a different arm
of the placement split. The matrix is the four platforms by six accelerators, driven
through the real route on synthetic headers; nothing downloads, loads or launches.

Not "the number equals N", which a self-consistent wrong estimator also passes. The
properties such an estimator cannot satisfy:

* the itemization SUMS to the total exactly, in integers (see ``_itemized_sum``);
* ``gpu_bytes <= total_bytes``, every cell;
* ``weights_bytes`` does not move with the context slider. The load-bearing one: the
  weights term is the context term SUBTRACTED out of ``_estimate_gguf_required_gb``, so
  drift here means the two arms stopped naming the same bytes;
* ``kv_bytes`` non-decreasing in ``n_ctx``;
* ``drafter_runtime_gpu_bytes <= drafter_runtime_bytes``;
* no negative field;
* a probed-empty inventory reports ``gpu_bytes == 0``;
* moving bytes off the GPU never shrinks the TOTAL -- on unified memory that is the
  whole claim, since an offloaded byte is not a freed byte;
* ``layer_count`` survives an unsizable KV, or ``--gpu-layers 0`` reads fully resident.

Plus one tripwire per cell: nothing here may reach
``LlamaCppBackend._apple_metal_memory_budget_bytes``, whose bare ``import mlx.core``
aborts the process at the C level on macOS once torch is imported -- an abort no
``try``/``except`` catches. A panel firing on every slider tick must not be able to
take the server down, so it is checked rather than reasoned about.

The honest limit: this runs on Linux. ``sys.platform``, ``platform.system()`` and
``platform.machine()`` are moved; the kernel, filesystem semantics, the real
Metal/HIP/Vulkan runtimes and the real device probes are not. Branch coverage of the
estimator's host-shaped decisions, not a substitute for the per-OS CI matrix.
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import platform as _platform
import sys
import types as _types
from pathlib import Path

import pytest

# Stub heavy / unavailable deps before importing the module under test.
# Copied from tests/test_memory_estimate.py -- same reasons, and
# tests/test_backend_tests_stub_heavy_imports.py enforces that it is here.

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog. Carries get_logger because this stub is process-wide: whichever test
# module is imported first wins the setdefault, and utils/prebuilt/freshness_flow
# calls structlog.get_logger at import time.
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx -- only stub when the real library is missing. Unconditional stubbing shadows
# HTTPError/Response that huggingface_hub.errors imports at load time.
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

# The GGUF blob writer and the platform/accelerator tables are LOADED, not copied: a
# second copy of either drifts from the first, and then this file is asserting about
# bytes and hosts that no other test in the tree recognises. By path, because
# `tests` is not importable as a package name from every runner layout -- the idiom is
# test_llama_extra_args_platforms.py:26-33.
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

# (label, sys.platform, os.name, WSL-shaped release or None) -- linux / wsl2 / windows / macos.
PLATFORMS = _plat_mod.PLATFORMS
# Moves sys.platform and the WSL markers, and deliberately NOT os.name; see its docstring
# and the note on _apply_cell below.
_apply_platform = _plat_mod._apply_platform

_GIB = 1024**3

# platform.system() per platform label. The estimate reaches it through
# utils.hardware.is_apple_silicon, which is `system() == "Darwin" and machine() == "arm64"`.
_SYSTEM = {"linux": "Linux", "wsl2": "Linux", "windows": "Windows", "macos": "Darwin"}


# A. The accelerator table


# The upstream table by label, so a rename there fails here loudly instead of silently
# re-pointing a cell at the wrong hardware.
_UPSTREAM_ACCELERATORS = {
    label: (vulkan, memory) for label, vulkan, memory in _plat_mod.ACCELERATORS
}

# (label, vulkan, [(index, free_mib, total_mib)]). The four upstream entries reused
# verbatim, plus the two this file adds:
#
#   amd-rocm      -- a HIP build. NOT Vulkan: the ROCm prebuilt carries a hipBLAS ggml
#                    lib, so _is_vulkan_backend is False and the devices enumerate
#                    through the torch/CUDA-shaped path as NVIDIA's do.
#   apple-unified -- Metal, one device, which is what /api/system reports on Apple
#                    Silicon: get_backend_visible_gpu_info's MLX arm emits a single
#                    index-0 device whose "VRAM" is the machine's unified RAM. An empty
#                    inventory here would be a second cpu-only cell asserting nothing
#                    about unified memory.
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
    """Whether this (platform, accelerator) pair can physically exist.

    One exclusion, stated rather than dropped: apple-unified is Apple Silicon's Metal
    unified memory, which only exists under Darwin. (Asahi Linux runs on Apple Silicon,
    but Studio's is_apple_silicon() is `platform.system() == "Darwin"`, so there the
    estimate takes the Linux arm, already covered by linux/cpu-only.)

    Everything else is kept, including the three discrete-vendor cells on macOS --
    legacy shapes, an Intel Mac with an eGPU or a pre-10.14 CUDA build, and the
    estimator contains no code forbidding them. What they assert is "Darwin plus a
    non-empty probed inventory", which is the shape a Mac reports.
    """
    return accelerator_label != _UNIFIED or platform_label == "macos"


MATRIX = [
    pytest.param(p, a, id = f"{p[0]}-{a[0]}")
    for p in PLATFORMS
    for a in ACCELERATORS
    if _reachable(p[0], a[0])
]

# 4 platforms x 6 accelerators, minus apple-unified on the three non-Darwin platforms.
assert len(MATRIX) == 4 * 6 - 3 == 21


# B. Applying a cell


def _snapshot(memory) -> tuple:
    """A ``main._system_gpu_cache`` shaped exactly as main.py fills it.

    Patched at the SOURCE rather than over ``_cached_inference_devices``, so the cell
    exercises the real reader -- including the "not `or None`" rule that keeps an empty
    probed list distinct from an unfilled snapshot, which is the whole mechanism behind
    the CPU-only placement.
    """
    devices = [
        {"index": index, "memory_total_gb": round(total / 1024, 2), "vram_free_gb": free / 1024}
        for index, free, total in memory
    ]
    inference_gpu = {"available": bool(devices), "devices": devices}
    return (0.0, ({"available": bool(devices), "devices": devices}, inference_gpu))


def _apply_cell(monkeypatch, platform_row, accelerator_row) -> None:
    """Move every host-shaped seam this endpoint reads, and nothing else.

    NEVER ``os.name``: it swaps pathlib's flavour mid-run, the synthetic GGUFs on
    tmp_path stop resolving, and every byte asserted below becomes a claim about a file
    that was never opened. Documented at test_llama_extra_args_platforms.py:59-63 and
    test_slot_refit_platform_matrix.py:107. ``sys.platform``, ``platform.system()``,
    ``platform.machine()`` and WSL_DISTRO_NAME are the whole toolkit.
    """
    platform_label = platform_row[0]
    accelerator_label, vulkan, memory = accelerator_row

    # sys.platform + the WSL markers, through the harness that already owns them.
    _apply_platform(monkeypatch, platform_row)
    # The modules under test alias `sys` by `import sys`, so the object patched above is
    # the object they read. Asserted rather than assumed: a future `from sys import
    # platform` in either module would silently take this whole matrix off the branch it
    # believes it is testing, and nothing else would notice.
    assert ri.sys is sys and llama_mod.sys is sys

    apple = accelerator_label == _UNIFIED
    monkeypatch.setattr(_platform, "system", lambda: _SYSTEM[platform_label], raising = False)
    monkeypatch.setattr(
        _platform,
        "machine",
        lambda: "arm64" if apple else ("AMD64" if platform_label == "windows" else "x86_64"),
        raising = False,
    )

    # The probed inference inventory, through main.py's own snapshot shape.
    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(_system_gpu_cache = _snapshot(memory)))

    # The llama.cpp build flavour and the CUDA-visible count.
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

    # ROCm's own marker. Nothing on the estimate path branches on it today -- the
    # estimator sees a HIP host and a CUDA host identically, and only `vulkan` and the
    # device count move it -- so the amd-rocm cell's real job is to hold that true.
    monkeypatch.setattr(
        "utils.hardware.hardware.IS_ROCM", accelerator_label == "amd-rocm", raising = False
    )
    for mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(mask, raising = False)
    # A GGML/LLAMA_ARG_* value inherited from the runner's shell would be read as the
    # user's own setting and price a load nobody configured.
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

    # The llama-server binary, pinned for two reasons; the second is the interesting one:
    #
    #  * determinism. ``_tensor_latches_allow_a_split`` resolves the binary before asking
    #    two in-process latches about it, so unpinned, whether a cell consults them at
    #    all depends on whether the runner has llama.cpp installed.
    #  * the Windows cells could not reach them otherwise. ``_find_llama_server_binary``
    #    ends at ``shutil.which``, which calls
    #    ``_winapi.NeedCurrentDirectoryForExePath`` under ``sys.platform == "win32"``.
    #    On a Linux interpreter ``_winapi`` is None, so it raises AttributeError, the
    #    caller's fail-open ``except`` swallows it, and every Windows cell skipped the
    #    latch check while reporting a pass. An artifact of simulating win32 on Linux
    #    rather than a product bug, but exactly the quiet hole that makes a green matrix
    #    mean nothing, so the seam is moved above it.
    #    test_platform_matrix_the_tensor_latch_lookup_runs_on_every_cell holds it open.
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

    # The capability probe shells out to llama-server --help. Pinned so a cell's numbers
    # come from the header and the flags, not from whichever binary this runner has.
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
    """Both module caches are TTL'd, not per-request, so they leak across cells.

    ``_estimate_files_cache`` is keyed on the config and NOT on the context, which is
    the behaviour under test elsewhere -- so an entry left by another cell would satisfy
    an assertion here that the cell never actually computed.
    """
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()
    yield
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()


@pytest.fixture(autouse = True)
def _metal_budget_tripwire(monkeypatch):
    """NOTHING on this path may call ``_apple_metal_memory_budget_bytes``.

    Its body is a bare ``import mlx.core`` (core/inference/llama_cpp.py:7816). On macOS,
    importing MLX after torch has been imported aborts the process at the C level, and a
    C-level abort is not an exception: no ``try``/``except`` anywhere up the stack can
    catch it. ``/api/inference/estimate-memory`` fires on every tick of the context
    slider, so one reachable call there is a settings panel that kills the server.

    Recorded rather than raised, because a raise would be swallowed by one of the broad
    ``except Exception`` handlers on this path and the tripwire would report nothing.
    The recorder returns 0, which is the off-Apple-Silicon answer, so a cell that DOES
    reach it still completes and is still reported.
    """
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


# C. The model shapes


# Realistic geometry. The numbers matter: at 32k the KV cache here is larger than the
# weights, which is the regime the whole feature exists for and the one where a paired
# subtraction that has come unpaired is visible rather than lost in rounding.
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

# Sparse padding. _get_gguf_size_bytes is a stat(), so truncate() gives a 3 GiB weights
# file that costs no disk. Without it every file is a few hundred bytes, the GB-scale
# float in the files term rounds them to noise, and "weights do not move with the
# context" would pass on numbers too small to have moved.
_WEIGHTS_BYTES = 3 * _GIB
_PROJECTOR_BYTES = 600 * 1024 * 1024
_DRAFTER_BYTES = 400 * 1024 * 1024
# What "costs no disk" is worth on the runner that does not agree. POSIX gives a hole
# for free; NTFS allocates and zero-fills every byte unless the file is marked sparse
# FIRST, so twelve of these filled a Windows runner's disk and every cell in this file
# errored with ENOSPC. The flag is asked for below; this is what the fixture falls back
# to when it cannot be set, which keeps the shapes GB-scale without the 34 GB.
_DENSE_PAD_DIVISOR = 12


def _try_make_sparse(handle) -> bool:
    """Mark an open file sparse. True on any filesystem that needs no marking.

    Windows only: FSCTL_SET_SPARSE has to be issued before the file is extended, or
    the extension is already committed. Everything else holes-punches on truncate().
    """
    # FORCE_DENSE_PAD is how the fallback below gets exercised at all: it is the arm
    # only Windows takes, and a POSIX box would otherwise never run the code that this
    # file's own ENOSPC on a Windows runner is the reason for.
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
    """Every model shape the estimator has a distinct arm for, built once.

    Module scope because these are ten sparse files and twenty-one cells; per-test
    rebuild is pure overhead, and nothing below mutates them.
    """
    root = tmp_path_factory.mktemp("platform-matrix-shapes")
    built: dict[str, tuple[str, SimpleNamespace]] = {}

    gqa = _write_gguf(root, "gqa.gguf", "qwen3", _GQA_FIELDS, pad = _WEIGHTS_BYTES)
    built["gqa"] = (gqa, _config(gqa))

    # MLA: a compressed latent cache, priced off kv_lora_rank rather than the head dims.
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

    # SWA: the cache is capped by the window, so kv_bytes plateaus in n_ctx instead of
    # growing -- which is why the monotonicity invariant below is non-decreasing.
    swa = _write_gguf(
        root,
        "swa.gguf",
        "gemma3",
        {**_GQA_FIELDS, "attention.sliding_window": 1024},
        pad = _WEIGHTS_BYTES,
    )
    built["swa"] = (swa, _config(swa))

    # Hybrid Mamba: attention every full_attention_interval layers, recurrent state on
    # the rest. Both terms are charged, from different formulas.
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

    # Pure SSM: block_count and embedding_length, no attention dims. NOT a degenerate
    # stub -- llama.cpp reads the attention head counts with required=false
    # (src/llama-model.cpp:1306 region) while block_count is required, so every Mamba,
    # Mamba2 and RWKV model loads exactly like this. It is the kv_estimable=False path
    # for a whole family of real models.
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

    # NextN / embedded MTP: a drafter with no file, whose draft cache is still allocated.
    nextn = _write_gguf(
        root,
        "nextn.gguf",
        "qwen3",
        {**_GQA_FIELDS, "nextn_predict_layers": 2},
        pad = _WEIGHTS_BYTES,
    )
    built["nextn_mtp"] = (nextn, _config(nextn))

    # Embedding model: a pooling type llama-server launches with --embedding, which caps
    # the batch at the micro-batch and gives every slot an output buffer.
    embedding = _write_gguf(
        root,
        "embedding.gguf",
        "bert",
        {**_GQA_FIELDS, "context_length": 8192, "pooling_type": 2},
        pad = _WEIGHTS_BYTES,
    )
    built["embedding"] = (embedding, _config(embedding, identifier = "local/embed-model"))

    # Vision: a target plus a projector file, which is charged AND carries encoder
    # buffers of its own on top of the file.
    vision = _write_gguf(root, "vision.gguf", "qwen3", _GQA_FIELDS, pad = _WEIGHTS_BYTES)
    projector = _write_gguf(
        root, "mmproj-vision.gguf", "clip", {"has_vision_encoder": 1}, pad = _PROJECTOR_BYTES
    )
    built["vision_projector"] = (
        vision,
        _config(vision, gguf_mmproj_file = projector, is_vision = True),
    )

    # A separate drafter sidecar: its own file in the weights AND its own KV on top.
    target = _write_gguf(root, "spec-target.gguf", "qwen3", _GQA_FIELDS, pad = _WEIGHTS_BYTES)
    drafter = _write_gguf(
        root,
        "mtp-draft.gguf",
        "qwen3",
        {**_GQA_FIELDS, "block_count": 2},
        pad = _DRAFTER_BYTES,
    )
    built["mtp_drafter"] = (target, _config(target, gguf_mtp_file = drafter))

    # Truncated: a real file with an unreadable header. The parser raises, and the
    # caller must still produce a well-formed answer rather than a partial number.
    truncated = root / "truncated.gguf"
    truncated.write_bytes(b"GGUF\x03\x00\x00\x00 truncated, nothing further is readable")
    _pad_to(truncated, _WEIGHTS_BYTES)
    built["truncated_header"] = (str(truncated), _config(str(truncated)))

    return built


# Shapes that carry a drafter the mode has to be told about; everything else prices the
# same under the default (auto) mode.
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

# 4k / 32k / 128k. Three points, because two cannot distinguish "grew" from "grew then
# fell back", and the weights term has to hold still across all of them.
CONTEXTS = (4096, 32768, 131072)


# D. Driving the real route


def _price(shapes, shape_name: str, **kwargs):
    """The real ``POST /api/inference/estimate-memory`` handler, auth dependency aside.

    The route, not ``_gguf_memory_breakdown``, because the invariants below are claims
    about the RESPONSE: a breakdown that sums correctly and a route that drops one of
    its terms on the way into the model are different bugs, and only this call sees the
    second one.
    """
    gguf_path, config = shapes[shape_name]
    spec = _SPEC_SHAPES.get(shape_name)
    if spec is not None:
        kwargs.setdefault("speculative_type", spec)
    request = EstimateMemoryRequest(model_path = gguf_path, **kwargs)

    # Only the config resolution is replaced: it is the one step that can reach the
    # network, and it is not what any of this is about.
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
    """The itemization, summed. FIVE terms, not four.

    The PR body says "the four items now sum to Total exactly". That is true of every
    shape without a vision projector and false of every shape with one:
    ``projector_runtime_bytes`` is a fifth line, it is inside ``total_bytes``
    (routes/inference.py, ``runtime_bytes``), and the panel renders it as its own row
    (model-config-page.tsx:1103). Summing four here would have made the vision cell
    fail by exactly the projector's buffers, which is a wrong test rather than a found
    bug -- so the sum is five, and the four-term claim is checked separately, once, in
    test_platform_matrix_the_itemization_is_five_terms_not_four.
    """
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
        # kv_bytes == 0 here means UNKNOWN, and the flag is the only thing separating it
        # from a genuine zero.
        assert response.kv_bytes == 0, f"{where}: unsizable KV reported {response.kv_bytes} bytes"
        assert response.compute_bytes == 0, f"{where}: unsizable KV priced compute buffers"


# E. The matrix


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_every_shape_is_internally_consistent(
    monkeypatch, shapes, platform, accelerator
):
    """One cell, every model shape, every context: the answer holds together.

    This is the bulk of the matrix -- 10 shapes x 3 contexts per cell -- and it asserts
    only properties, never magnitudes. A sum that is off by one byte, a GPU share above
    the total, or a negative field is caught here on whichever host produced it.
    """
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
    """The load-bearing one.

    ``weights_bytes`` is not measured. It is derived by subtracting the context term
    back out of ``_estimate_gguf_required_gb``, so it is only correct while the term
    subtracted is the same term that was added. Any drift as ``n_ctx`` moves means the
    two arms have come unpaired, and the weights row silently absorbs part of the KV
    cache -- in either direction, on any host.
    """
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
        # Non-decreasing, not strictly increasing: a sliding-window cache is capped by
        # its window and legitimately plateaus, and an unsizable one stays at 0.
        assert kv == sorted(
            kv
        ), f"[{cell}] {shape}: kv_bytes is not non-decreasing in n_ctx: {dict(zip(CONTEXTS, kv))}"


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_settings_do_not_break_the_itemization(
    monkeypatch, shapes, platform, accelerator
):
    """The settings a user actually moves, on every host.

    Each of these reaches a different arm: the cache dtype rescales the KV term, slots
    rescale both KV and the compute buffers, ``--no-kv-offload`` moves the cache out of
    the GPU figure without leaving the total, manual layers rescale only the main
    weight, and a tensor split replicates the flat buffers per device. The invariants
    are the same ones; the point is that no setting breaks them.
    """
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
    """Moving bytes off the GPU changes where they are, never how many there are.

    ``total_bytes`` is aggregate memory: GPU, host RAM or a unified pool. So
    ``--gpu-layers 0`` and ``--no-kv-offload`` may only ever move bytes out of
    ``gpu_bytes``; a total that shrinks with them is the panel telling a user that
    offloading made memory disappear.

    On the apple-unified cell this is the entire claim rather than a nicety: there is
    one pool, so a byte moved off the GPU is still occupying the same RAM. A row that
    reported a smaller total under an offload would be advertising headroom that does
    not exist on the machine that has the least of it.
    """
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
    """``gpu_bytes == 0`` exactly on the cells that have no GPU, and only those.

    A probe that RAN and found nothing is the evidence; an unfilled snapshot is not
    absence, and the CUDA count is zero on every Vulkan and ROCm-via-Vulkan host that
    has plenty of GPU. Getting this wrong put a multi-gigabyte GPU figure against a
    capacity of zero on CPU-only Linux and Windows boxes.
    """
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
            # Still a real load, just not one on a card.
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
    """A pure-SSM header sizes no cache and must still report ``block_count``.

    llama.cpp reads the attention head counts with ``required = false`` while
    ``block_count`` is required, so every Mamba, Mamba2 and RWKV model reaches this path
    legitimately. Without the layer count ``_gguf_offloaded_layer_fraction`` has no
    denominator, answers 1.0, and a manual ``--gpu-layers 0`` on that whole family reads
    as a fully GPU-resident load -- the direction that calls an impossible load a fit.
    """
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"

    ri._estimate_files_cache.clear()
    response = _price(shapes, "pure_ssm", n_ctx = 131072)
    assert response.kv_estimable is False, f"[{cell}]"
    assert response.layer_count == 48, (
        f"[{cell}] pure_ssm: kv_estimable is False and layer_count is "
        f"{response.layer_count}; without it the offload split has no denominator"
    )

    # And the count is load-bearing, not decorative: it is what makes -ngl 0 read as a
    # CPU load rather than a GPU-resident one.
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
    # No block_count to recover: unlike the pure-SSM case above, this header carries
    # nothing at all. See the pinned gap below for what that costs.
    assert response.layer_count is None, f"[{cell}]"


# F. The claims this file exists to check independently


def test_platform_matrix_the_itemization_is_five_terms_not_four(monkeypatch, shapes):
    """The PR body's "the four items now sum to Total exactly" is one term short.

    Four terms do sum to the total on every shape without a vision projector. With one,
    ``projector_runtime_bytes`` is a fifth line -- inside ``total_bytes``, rendered as
    its own row by the panel -- and the four-term sum misses it by exactly the encoder's
    buffers. Pinned here so the wording and the arithmetic cannot drift apart again:
    the arithmetic is right, the sentence is stale.
    """
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
    """The tripwire, asserted per cell rather than reasoned about.

    Driven hard on purpose: every shape, the Apple-shaped context settings, and the
    paravirtual detector deliberately RESTORED. The backend conftest pins
    ``_metal_device_is_paravirtual`` to False on both ``core.inference.llama_cpp`` and
    ``routes.inference`` (tests/conftest.py:296-314) so the suite is host independent,
    which also masks whichever Apple arms consult it. On the apple-unified cell that
    would be assuming the answer, so it is put back -- both values -- and the tripwire
    is asked again with it live.
    """
    _apply_cell(monkeypatch, platform, accelerator)
    apple = accelerator[0] == _UNIFIED

    if apple:
        # is_apple_silicon() is already True on this cell (Darwin + arm64), which is the
        # ONLY gate in front of the mlx import. If any arm of the estimate consults the
        # budget, this cell is where it fires.
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

    # The autouse fixture asserts the recorder is empty at teardown; assert it here too
    # so the failure names this cell rather than a teardown error.
    assert (
        _metal_budget_tripwire == []
    ), f"[{platform[0]}-{accelerator[0]}] reached _apple_metal_memory_budget_bytes"


def test_platform_matrix_the_platform_label_alone_changes_nothing(monkeypatch, shapes):
    """Linux, WSL2, Windows and macOS price an identical load identically.

    Worth asserting rather than assuming, because it is the finding this matrix
    actually produced: the estimate reads the HOST, not the operating system. Sweeping
    all 21 cells collapses them to three behaviours -- one device, two devices, and no
    device -- and ``sys.platform`` is in none of them.

    That is the right design (bytes are bytes, and a GGUF header does not change shape
    on Windows), so this test is the net under it: the day someone adds a per-OS arm to
    this endpoint, it fails here and has to be justified, rather than shipping to three
    platforms that nobody could test.
    """
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

    # And macOS on Apple Silicon prices the same load as any other single-device host.
    # One pool is a CAPACITY fact, not a footprint one: the same bytes are allocated
    # either way, and which pool they come out of is the panel's question, answered
    # frontend-side from `singleMemoryPool` (memory-fit.ts). Nothing in the response
    # distinguishes them, which is why the row cannot be read without that flag.
    assert answers[("macos", _UNIFIED)] == answers[("linux", "nvidia-single")]


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_the_probed_inventory_owns_the_split_on_a_vulkan_build(
    monkeypatch, shapes, platform, accelerator
):
    """A Vulkan build enumerates no CUDA devices, and that is not evidence of no GPU.

    This is the one seam where vendor genuinely changes the answer, and it only shows
    up in the combination a real Vulkan host produces: two devices in the probed
    inventory and a CUDA count of ZERO, because torch cannot see a Vulkan card. If
    ``_tensor_split_possible`` asked the CUDA count there it would refuse the split on a
    machine with two cards, and price per-device compute buffers for a launch that
    replicates them -- gigabytes apart on the figure the fit verdict reads.

    Asserted on every cell, including the non-Vulkan ones, where the CUDA count is the
    right thing to ask and a zero really does mean no split.
    """
    _apply_cell(monkeypatch, platform, accelerator)
    cell = f"{platform[0]}-{accelerator[0]}"
    vulkan = accelerator[1]

    # Two devices in the probed inventory, nothing visible to torch: the Vulkan shape.
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

    # The same request with the cards named explicitly. A pin answers for itself on
    # every build, so it is the reference the probe-driven answer has to match on
    # Vulkan and to differ from where the CUDA count legitimately says "one card".
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
        # torch sees nothing, so there is no split to price, and the pin is the only
        # thing that can say otherwise.
        assert (
            unpinned.compute_bytes < pinned.compute_bytes
        ), f"[{cell}] a CUDA-shaped build reporting zero devices still priced a two-device split"


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_platform_matrix_the_tensor_latch_lookup_runs_on_every_cell(
    monkeypatch, shapes, platform, accelerator
):
    """Every cell really asks the tensor latches, rather than failing open past them.

    ``_tensor_latches_allow_a_split`` is wrapped in a fail-open ``except Exception``:
    anything that raises inside it answers "a split is allowed" and logs at debug. That
    is the right behaviour in production and a trap in a matrix, because a cell where
    the lookup EXPLODES is indistinguishable from a cell where it ran and said yes --
    both are green, and only one of them tested anything.

    It is not hypothetical. Before the binary was pinned in ``_apply_cell``, all six
    Windows cells took the exception arm on every single request, because a simulated
    ``sys.platform == "win32"`` sends ``shutil.which`` into ``_winapi`` and there is no
    ``_winapi`` on Linux. This test is what noticed, and it is what keeps it noticed.
    """
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
    # And it asked about the cache pair this launch actually plans, not the default.
    assert asked[0][1] == ("q8_0", "q8_0"), f"[{cell}]: latch asked about {asked[0][1]}"


# G. Regressions this matrix found
#
# This one was a strict xfail when the matrix first ran: a header that yields no dims
# left layer_count None, and _gguf_offloaded_layer_fraction answered 1.0 for every such
# request, so an explicit --gpu-layers 0 was reported as a fully GPU-resident load. The
# pure-SSM case was fixed by keeping block_count out of the header walk; this one has no
# count to keep. It does not need one -- zero layers on the GPU is knowable without
# knowing how many layers there are, and the fraction is a scale factor for a PARTIAL
# offload, which an explicit none is not.


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

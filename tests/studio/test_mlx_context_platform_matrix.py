# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Only an MLX load resolves the context triple; WSL rows mirror Linux rather than testing it."""

from __future__ import annotations

import ast
import importlib.util
import io
import platform
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

pytest.importorskip("torch", reason = "the studio backend imports torch at module scope")

# Imported before any fake torch, or the test would measure the spoof.
from core.inference.inference import runtime_context_length  # noqa: E402
from core.inference.mlx_inference import MLXInferenceBackend  # noqa: E402
from core.inference.orchestrator import _mirrored_model_entry  # noqa: E402
import routes.inference as routes_inference  # noqa: E402

WORKER_SOURCE = (STUDIO_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
HARDWARE_PACKAGE = STUDIO_BACKEND / "utils" / "hardware"
# Re-seated at the top of every spoof so a later cell does not build on the previous fake.
_REAL_TORCH = sys.modules.get("torch")


def _code_without_comments(path: Path) -> str:
    """Comments removed, string literals kept, so a WSL discriminator string stays visible."""
    text = path.read_text(encoding = "utf-8")
    return "".join(
        token.string if token.type != tokenize.COMMENT else ""
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
    )


def _load_sibling(name: str, path: Path):
    """Loads by path: tests/studio and studio/backend/tests are unpackaged, so neither imports the other."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: @dataclass resolves annotations via sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_OS_MATRIX = _load_sibling(
    "_mlx_ctx_os_matrix_7624",
    STUDIO_BACKEND / "tests" / "test_gpu_arch_gate_os_matrix_7624.py",
)
_DISPATCH = _load_sibling(
    "_mlx_ctx_hardware_dispatch",
    REPO_ROOT / "tests" / "studio" / "test_hardware_dispatch_matrix.py",
)

OS_KEYS = _OS_MATRIX.OS_KEYS
# "amd" is the ROCm wheel shape; the Radeon wheel without torch.version.hip is a row below.
VENDORS = ("nvidia", "amd", "cpu")
CELLS = [(os_key, vendor) for os_key in OS_KEYS for vendor in VENDORS]
CELL_IDS = [f"{os_key}-{vendor}" for os_key, vendor in CELLS]


@dataclass(frozen = True)
class Expectation:
    """What one cell must produce. ``real`` records whether the cell can be booted."""

    device: str
    is_rocm: bool
    mlx_selected: bool
    reports_triple: bool
    chat_only_reason: str | None
    real: bool
    note: str = ""


_MACHINE = {"windows": "x86_64", "linux": "x86_64", "wsl": "x86_64", "macos": "arm64"}

_NOT_A_REAL_CELL = (
    "Not a bootable host: macOS has shipped no CUDA driver since 10.13 and ROCm has no "
    "macOS build. Kept as an expectation about the detector's ordering, not a machine."
)

# A CPU-only wheel and an unusable CUDA build are not "no_gpu" (see _chat_only_reason).
_CPU_ONLY_REASONS = ("no_gpu", "torch_cpu_build", "torch_cuda_unavailable")

EXPECTED: dict[tuple[str, str], Expectation] = {
    ("windows", "nvidia"): Expectation(
        "CUDA",
        False,
        False,
        False,
        None,
        real = True,
    ),
    ("windows", "amd"): Expectation(
        "CUDA",
        True,
        False,
        False,
        None,
        real = True,
        note = "ROCm reuses torch.cuda over HIP; DeviceType stays CUDA, IS_ROCM flips.",
    ),
    ("windows", "cpu"): Expectation(
        "CPU",
        False,
        False,
        False,
        "no_gpu",
        real = True,
        note = "MLX stack present and healthy, and still CPU: the gate requires Darwin.",
    ),
    ("linux", "nvidia"): Expectation("CUDA", False, False, False, None, real = True),
    ("linux", "amd"): Expectation("CUDA", True, False, False, None, real = True),
    ("linux", "cpu"): Expectation("CPU", False, False, False, "no_gpu", real = True),
    ("wsl", "nvidia"): Expectation(
        "CUDA",
        False,
        False,
        False,
        None,
        real = True,
        note = "sys.platform is 'linux'; nothing in utils/hardware reads a WSL marker.",
    ),
    ("wsl", "amd"): Expectation("CUDA", True, False, False, None, real = True),
    ("wsl", "cpu"): Expectation("CPU", False, False, False, "no_gpu", real = True),
    ("macos", "nvidia"): Expectation(
        "CUDA",
        False,
        False,
        False,
        None,
        real = False,
        note = _NOT_A_REAL_CELL,
    ),
    ("macos", "amd"): Expectation(
        "CUDA",
        True,
        False,
        False,
        None,
        real = False,
        note = _NOT_A_REAL_CELL,
    ),
    ("macos", "cpu"): Expectation(
        "MLX",
        False,
        True,
        True,
        None,
        real = True,
        note = "The one cell that serves MLX: Darwin + arm64, no CUDA/XPU, healthy stack.",
    ),
}


def _devices_for(vendor: str) -> list:
    """The enumerated device list a vendor's torch reports."""
    if vendor == "cpu":
        return []
    if vendor == "nvidia":
        return [_OS_MATRIX._device(name = "NVIDIA GeForce RTX 4090", arch = "")]
    return [_OS_MATRIX._device(arch = "gfx1100", name = "AMD Radeon RX 7900 XTX")]


@pytest.fixture
def spoof_cell(monkeypatch, spoof_hardware):
    """Fake torch is installed last so it shadows the real one for the detector's import."""

    def _apply(
        os_key: str,
        vendor: str,
        *,
        machine: str | None = None,
        mlx: bool = True,
    ):
        machine = machine or _MACHINE[os_key]
        _, system_name = _OS_MATRIX._OS_CELLS[os_key]
        if _REAL_TORCH is not None:
            monkeypatch.setitem(sys.modules, "torch", _REAL_TORCH)
        spoof_hardware(
            _DISPATCH.HardwareProfile(
                name = f"{os_key}-{vendor}",
                system = system_name,
                machine = machine,
                cuda_available = vendor != "cpu",
                hip_version = "6.4" if vendor == "amd" else None,
                xpu_available = False,
                has_mlx = mlx,
                mps_available = system_name == "Darwin",
                expect_is_mlx = False,
                expect_device_type = "CPU",
                expect_is_rocm = vendor == "amd",
                expect_apple_silicon = system_name == "Darwin" and machine == "arm64",
            )
        )
        _OS_MATRIX._apply_os(monkeypatch, os_key, is_rocm = vendor == "amd")
        monkeypatch.setattr(platform, "machine", lambda: machine)
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _OS_MATRIX._fake_torch(_devices_for(vendor), vendor = vendor),
        )
        # An inherited ZE_AFFINITY_MASK plus a CPU-only torch would route the cell to XPU.
        for var in ("ZE_AFFINITY_MASK", "UNSLOTH_FORCE_XPU", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(var, raising = False)
        return _DISPATCH._import_studio_hardware_module()

    return _apply


@pytest.fixture
def spoof_hardware(monkeypatch):
    """``test_hardware_dispatch_matrix``'s fixture, bound to this module's monkeypatch."""
    return _DISPATCH.spoof_hardware.__wrapped__(monkeypatch)


@pytest.mark.parametrize(("os_key", "vendor"), CELLS, ids = CELL_IDS)
def test_detected_device_per_cell(os_key, vendor, spoof_cell):
    """Each cell resolves to the DeviceType recorded above, with a healthy MLX stack."""
    expected = EXPECTED[(os_key, vendor)]
    hw = spoof_cell(os_key, vendor)
    device = hw.detect_hardware()
    assert device == getattr(
        hw.DeviceType, expected.device
    ), f"{os_key}/{vendor}: expected {expected.device}, got {device!r}. {expected.note}"
    assert hw.IS_ROCM is expected.is_rocm, f"{os_key}/{vendor}: IS_ROCM"
    # Which CPU reason depends on whether the HOST has GPUs this torch cannot use.
    if expected.chat_only_reason == "no_gpu":
        assert hw.CHAT_ONLY_REASON in _CPU_ONLY_REASONS, f"{os_key}/{vendor}: chat-only reason"
    else:
        assert (
            hw.CHAT_ONLY_REASON == expected.chat_only_reason
        ), f"{os_key}/{vendor}: chat-only reason"


@pytest.mark.parametrize(("os_key", "vendor"), CELLS, ids = CELL_IDS)
def test_mlx_backend_selection_per_cell(os_key, vendor, spoof_cell):
    """Selection is the worker's own DEVICE == DeviceType.MLX check, pinned as the whole condition below."""
    expected = EXPECTED[(os_key, vendor)]
    hw = spoof_cell(os_key, vendor)
    hw.detect_hardware()
    selected = hw.DEVICE == hw.DeviceType.MLX
    assert selected is expected.mlx_selected, (
        f"{os_key}/{vendor}: MLXInferenceBackend selected={selected}, "
        f"expected {expected.mlx_selected}. {expected.note}"
    )


def test_worker_selects_mlx_on_device_type_alone():
    """Parsed with ast: a grep for DeviceType.MLX would match the import and a comment, not the guard."""
    tree = ast.parse(WORKER_SOURCE)
    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        constructed = any(
            isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == "MLXInferenceBackend"
            for inner in ast.walk(node)
        )
        if constructed:
            guards.append(ast.unparse(node.test))
    assert guards, "no if-statement in worker.py constructs MLXInferenceBackend"
    for guard in guards:
        assert "_hw.DEVICE == _hw.DeviceType.MLX" in guard, guard
        for forbidden in ("platform", "sys.platform", "is_apple_silicon", "machine"):
            assert forbidden not in guard, f"{forbidden!r} in the MLX guard: {guard}"


_MLX_MODEL = SimpleNamespace(args = SimpleNamespace(max_position_embeddings = 131072))
# Unsloth writes only the served length onto a transformers model.
_TORCH_MODEL = SimpleNamespace(max_seq_length = 4096)


def _model_info_for(mlx_selected: bool, requested: int) -> dict:
    """Both branches call the shipped resolvers, so the test cannot drift from what serving publishes."""
    if mlx_selected:
        served, native, ceiling = MLXInferenceBackend._resolve_context_lengths(
            None, _MLX_MODEL, requested
        )
        return {
            "is_mlx": True,
            "context_length": served,
            "native_context_length": native,
            "max_context_length": ceiling,
            "requested_context_length": requested or 0,
        }
    return {
        "is_mlx": False,
        "context_length": runtime_context_length(_TORCH_MODEL, requested),
    }


class _FakeOrchestrator:
    def __init__(self, name, entry):
        self.active_model_name = name
        self.models = {name: entry}
        self.context_length = None
        self.max_seq_length = None


@pytest.mark.parametrize(("os_key", "vendor"), CELLS, ids = CELL_IDS)
@pytest.mark.parametrize("requested", [0, 8192], ids = ["auto", "pinned"])
def test_context_triple_reported_per_cell(os_key, vendor, requested, spoof_cell, monkeypatch):
    """The triple reaches /v1/models only on the MLX cell; a loss at any of three seams looks identical."""
    expected = EXPECTED[(os_key, vendor)]
    hw = spoof_cell(os_key, vendor)
    hw.detect_hardware()
    mlx_selected = hw.DEVICE == hw.DeviceType.MLX
    assert mlx_selected is expected.mlx_selected

    model_info = _model_info_for(mlx_selected, requested)
    mirrored = _mirrored_model_entry(model_info, "some/model")

    if expected.reports_triple:
        assert mirrored["context_length"] == (requested or 131072)
        assert mirrored["native_context_length"] == 131072
        assert mirrored["max_context_length"] == 131072
        assert mirrored["requested_context_length"] == requested
    else:
        # 4096 under both requests: the request is only runtime_context_length's fallback, so
        # the attached length wins.
        assert mirrored["context_length"] == 4096
        assert mirrored["native_context_length"] is None
        assert mirrored["max_context_length"] is None

    monkeypatch.setattr(
        routes_inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False),
    )
    monkeypatch.setattr(
        routes_inference,
        "get_inference_backend",
        lambda: _FakeOrchestrator("some/model", mirrored),
    )
    monkeypatch.setattr(routes_inference, "_orchestrator_public_model_id", lambda _b: "some/model")
    (entry,) = routes_inference._openai_model_objects()
    assert entry["context_length"] == mirrored["context_length"]
    if expected.reports_triple:
        assert entry["native_context_length"] == 131072
        assert entry["max_context_length"] == 131072
    else:
        assert "native_context_length" not in entry
        assert "max_context_length" not in entry


def test_only_the_mlx_backend_resolves_a_native_window():
    """Only MLX resolves a native window; inference.py publishes context_length alone, so others
    withhold it."""
    assert runtime_context_length(_MLX_MODEL, 8192) == 8192
    served, native, ceiling = MLXInferenceBackend._resolve_context_lengths(None, _MLX_MODEL, 0)
    assert (served, native, ceiling) == (131072, 131072, 131072)
    assert set(_model_info_for(False, 0)) == {"is_mlx", "context_length"}


def test_wsl_is_indistinguishable_from_linux_in_the_detector():
    """Nothing in utils/hardware tells WSL from Linux; WSL discrimination lives in llama_cpp.py instead."""
    # Not the bare token "WSL": hardware.py mentions it in comments.
    markers = (
        "WSL_DISTRO_NAME",
        "WSLENV",
        "WSL_INTEROP",
        "/proc/version",
        "/proc/sys/kernel/osrelease",
        "microsoft-standard",
        "uname",
        "is_wsl",
    )
    for path in sorted(HARDWARE_PACKAGE.rglob("*.py")):
        code = _code_without_comments(path)
        for marker in markers:
            assert marker not in code, (
                f"{path.relative_to(REPO_ROOT)} names {marker!r}: WSL is no longer "
                "indistinguishable from Linux here, so the wsl rows in this file became "
                "real cells and their expectations must be re-derived."
            )
    assert "Microsoft" in _code_without_comments(HARDWARE_PACKAGE / "hardware.py")
    assert "_WINDOWS_DIRECTX_KEY" in (HARDWARE_PACKAGE / "hardware.py").read_text(encoding = "utf-8")
    llama_cpp = (STUDIO_BACKEND / "core" / "inference" / "llama_cpp.py").read_text(encoding = "utf-8")
    assert "_wsl_system_rocm_lib_dirs" in llama_cpp


@pytest.mark.parametrize("vendor", VENDORS)
def test_wsl_row_equals_the_linux_row(vendor, spoof_cell):
    """Measured, not merely argued: the two rows produce the same verdict."""
    hw = spoof_cell("linux", vendor)
    hw.detect_hardware()
    linux = (hw.DEVICE, hw.IS_ROCM, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON)
    hw = spoof_cell("wsl", vendor)
    hw.detect_hardware()
    assert (hw.DEVICE, hw.IS_ROCM, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON) == linux


def test_windows_on_arm_with_a_healthy_mlx_stack_is_still_cpu(spoof_cell):
    """is_apple_silicon ANDs Darwin with arm64, so Windows-on-ARM with MLX still lands on CPU."""
    hw = spoof_cell("windows", "cpu", machine = "arm64", mlx = True)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.is_apple_silicon() is False
    assert hw.CHAT_ONLY_REASON == "no_gpu"


def test_the_apple_silicon_gate_is_a_conjunction():
    """Source-level, because the runtime answer cannot distinguish AND from OR here."""
    source = (HARDWARE_PACKAGE / "hardware.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    (gate,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "is_apple_silicon"
    ]
    (returned,) = [node for node in ast.walk(gate) if isinstance(node, ast.Return)]
    expression = ast.unparse(returned.value)
    assert isinstance(returned.value, ast.BoolOp)
    assert isinstance(returned.value.op, ast.And), expression
    assert "'Darwin'" in expression and "'arm64'" in expression, expression


def test_apple_silicon_without_the_mlx_stack_falls_to_chat_only(spoof_cell):
    """The macos/cpu cell's other half: Darwin + arm64 with no usable stack is CPU."""
    hw = spoof_cell("macos", "cpu", mlx = False)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.is_apple_silicon() is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def test_intel_mac_is_not_an_mlx_host(spoof_cell):
    """x86_64 Darwin: the second impossible-on-macOS shape, and a real machine."""
    hw = spoof_cell("macos", "cpu", machine = "x86_64", mlx = True)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.is_apple_silicon() is False
    assert hw.CHAT_ONLY_REASON == "intel_mac"


def test_amd_sdk_wheel_reaches_is_rocm_without_version_hip(monkeypatch, spoof_hardware):
    """AMD SDK wheels leave torch.version.hip unset; IS_ROCM is read from torch.__version__ instead."""
    spoof_hardware(
        _DISPATCH.HardwareProfile(
            name = "windows-amd-sdk",
            system = "Windows",
            machine = "x86_64",
            cuda_available = True,
            hip_version = None,
            xpu_available = False,
            has_mlx = True,
            mps_available = False,
            expect_is_mlx = False,
            expect_device_type = "CUDA",
            expect_is_rocm = True,
            expect_apple_silicon = False,
        )
    )
    _OS_MATRIX._apply_os(monkeypatch, "windows", is_rocm = True)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _OS_MATRIX._fake_torch(_devices_for("amd"), vendor = "amd_sdk"),
    )
    for var in ("ZE_AFFINITY_MASK", "UNSLOTH_FORCE_XPU", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    hw = _DISPATCH._import_studio_hardware_module()
    assert hw.detect_hardware() == hw.DeviceType.CUDA
    assert hw.IS_ROCM is True


def test_every_cell_in_the_product_has_an_expectation():
    """No cell may be quietly dropped, and every impossible one must say so."""
    assert set(EXPECTED) == set(CELLS)
    unreal = {cell for cell, exp in EXPECTED.items() if not exp.real}
    assert unreal == {("macos", "nvidia"), ("macos", "amd")}
    for cell in unreal:
        assert EXPECTED[cell].note == _NOT_A_REAL_CELL
    assert {cell for cell, exp in EXPECTED.items() if exp.mlx_selected} == {("macos", "cpu")}
    assert {cell for cell, exp in EXPECTED.items() if exp.reports_triple} == {("macos", "cpu")}

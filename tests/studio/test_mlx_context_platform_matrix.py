# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""[Windows, Linux, WSL, macOS] x [NVIDIA, AMD/ROCm, CPU-only] for the MLX context report.

Per cell: the ``DeviceType`` ``detect_hardware()`` returns, whether ``worker.py`` would
construct ``MLXInferenceBackend`` (the selection is ``_hw.DEVICE == _hw.DeviceType.MLX``
and nothing else, asserted against the source too), and whether the context triple reaches
the API. Only an MLX load resolves ``native_context_length`` / ``max_context_length``, so
the other eleven cells withhold them through ``_mirrored_model_entry`` and ``/v1/models``.

Every cell is presented with a HEALTHY MLX stack, including the absurd ones: the gate, not
the missing package, is what must keep MLX off the other eleven. If the ordering in
``_detect_hardware_locked`` changed, only a matrix that installs mlx everywhere would see it.

What this CANNOT prove, recorded rather than skipped (tests at the bottom):

  * WSL is indistinguishable from Linux in ``utils/hardware/**``, so its row is asserted
    byte-identical to linux and the absence of any WSL discriminator is asserted structurally.
  * macOS x NVIDIA and macOS x AMD are not bootable cells; the rows describe the detector's
    ordering, not a machine.
  * Windows x MLX is impossible by construction: ``is_apple_silicon()`` ANDs Darwin with
    arm64, so Windows-on-ARM with a full MLX stack still lands on CPU.
  * No Apple Silicon, ROCm or AMD GPU exists on this host, so every non-CPU answer comes
    from the mocked torch shapes ``test_gpu_arch_gate_os_matrix_7624`` documents.

Machinery is reused from ``test_gpu_arch_gate_os_matrix_7624.py`` and
``test_hardware_dispatch_matrix.py``. It mutates ``hardware.py`` globals, so it is
registered in ``test_backend_ci_parallel_isolation.py::ISOLATED`` and in both halves of the
Backend CI pairing. Written without the workflow directory's literal path, which
``test_workflow_guards_run_unfiltered`` scans for.
"""

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

# The studio backend pulls in torch. A runner without it cannot answer any question this
# file asks, so skip the module rather than fail collection on it.
pytest.importorskip("torch", reason = "the studio backend imports torch at module scope")

# Imported eagerly, before any fake torch can be in place: these modules are the subject
# of the test, and importing them under a spoof would measure the spoof.
from core.inference.inference import runtime_context_length  # noqa: E402
from core.inference.mlx_inference import MLXInferenceBackend  # noqa: E402
from core.inference.orchestrator import _mirrored_model_entry  # noqa: E402
import routes.inference as routes_inference  # noqa: E402

WORKER_SOURCE = (STUDIO_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
HARDWARE_PACKAGE = STUDIO_BACKEND / "utils" / "hardware"
# The real torch, before anything here shadows it. Re-seated at the top of every spoof so
# a second cell in one test does not build its profile against the previous fake.
_REAL_TORCH = sys.modules.get("torch")


def _code_without_comments(path: Path) -> str:
    """Source with comments removed and string literals kept.

    Both halves matter for the WSL claim below: `WSL` appears in this package only in
    prose (two comments explaining that WSL is deliberately NOT special-cased), while a
    real discriminator would be a string -- ``os.environ.get("WSL_DISTRO_NAME")``,
    ``open("/proc/version")`` -- so stripping strings instead would hide exactly the thing
    being looked for.
    """
    text = path.read_text(encoding = "utf-8")
    return "".join(
        token.string if token.type != tokenize.COMMENT else ""
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
    )


def _load_sibling(name: str, path: Path):
    """Load a test module by path so its helpers can be reused verbatim.

    By path rather than by name: ``tests/studio`` and ``studio/backend/tests`` are both
    unpackaged, so neither is importable as ``tests.studio.x`` from the other.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for a module that is only half loaded.
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

# The four simulated hosts, exactly as #7624 spells them.
OS_KEYS = _OS_MATRIX.OS_KEYS
# The three GPU vendors. "amd" is the ROCm wheel shape (torch.version.hip set); the AMD
# SDK / Radeon wheel shape that leaves it unset is covered as an extra row below, because
# it is the one that reaches IS_ROCM through torch.__version__ instead.
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


# The machine a cell runs on. Darwin cells are arm64 (the only Apple Silicon shape);
# everything else is x86_64. Windows-on-ARM and Intel Mac get their own rows below.
_MACHINE = {"windows": "x86_64", "linux": "x86_64", "wsl": "x86_64", "macos": "arm64"}

_NOT_A_REAL_CELL = (
    "Not a bootable host: macOS has shipped no CUDA driver since 10.13 and ROCm has no "
    "macOS build. Kept as an expectation about the detector's ordering, not a machine."
)

# The reasons hardware.py groups as "no GPU this torch can use" (see its own tuple in
# _chat_only_reason): a CPU-only wheel and an unusable CUDA build are not "no_gpu".
_CPU_ONLY_REASONS = ("no_gpu", "torch_cpu_build", "torch_cuda_unavailable")

EXPECTED: dict[tuple[str, str], Expectation] = {
    # --- Windows -----------------------------------------------------------------
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
    # --- Linux -------------------------------------------------------------------
    ("linux", "nvidia"): Expectation("CUDA", False, False, False, None, real = True),
    ("linux", "amd"): Expectation("CUDA", True, False, False, None, real = True),
    ("linux", "cpu"): Expectation("CPU", False, False, False, "no_gpu", real = True),
    # --- WSL (indistinguishable from Linux; see the dedicated tests) --------------
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
    # --- macOS -------------------------------------------------------------------
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
    """Present one (OS, vendor, machine, mlx) host to ``detect_hardware()``.

    Layered rather than rewritten. ``spoof_hardware`` owns the MLX side (the fake
    ``mlx``/``mlx.core`` modules, the ``utils.mlx_repair`` stubs, and the meta-path finder
    that makes ``import mlx.core`` raise), ``_apply_os`` owns ``sys.platform`` /
    ``platform.system()``, and the fake torch is installed last so it shadows the real one
    for the ``import torch`` inside the detector.
    """

    def _apply(
        os_key: str,
        vendor: str,
        *,
        machine: str | None = None,
        mlx: bool = True,
    ):
        machine = machine or _MACHINE[os_key]
        _, system_name = _OS_MATRIX._OS_CELLS[os_key]
        # A test that presents two cells (the linux/wsl comparison) would otherwise build
        # the second profile against the first cell's fake torch, which has no .backends.
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
        # Neither hint may leak in from the host running this: an inherited
        # ZE_AFFINITY_MASK plus a CPU-only torch would route the cell to XPU.
        for var in ("ZE_AFFINITY_MASK", "UNSLOTH_FORCE_XPU", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(var, raising = False)
        return _DISPATCH._import_studio_hardware_module()

    return _apply


@pytest.fixture
def spoof_hardware(monkeypatch):
    """``test_hardware_dispatch_matrix``'s fixture, bound to this module's monkeypatch."""
    return _DISPATCH.spoof_hardware.__wrapped__(monkeypatch)


# ======================================================================================
# 1. Detection
# ======================================================================================


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
    # A CPU cell names one of the three reasons hardware.py itself groups: which one
    # depends on whether the HOST has GPUs this torch cannot use, so pinning a single
    # spelling would pass on a GPU-less runner and fail on a GPU box, and vice versa.
    if expected.chat_only_reason == "no_gpu":
        assert hw.CHAT_ONLY_REASON in _CPU_ONLY_REASONS, f"{os_key}/{vendor}: chat-only reason"
    else:
        assert hw.CHAT_ONLY_REASON == expected.chat_only_reason, (
            f"{os_key}/{vendor}: chat-only reason"
        )


# ======================================================================================
# 2. Backend selection
# ======================================================================================


@pytest.mark.parametrize(("os_key", "vendor"), CELLS, ids = CELL_IDS)
def test_mlx_backend_selection_per_cell(os_key, vendor, spoof_cell):
    """``MLXInferenceBackend`` is constructed on exactly one cell.

    The predicate is the worker's own: ``_hw.DEVICE == _hw.DeviceType.MLX``. The test
    below pins that this really is the whole condition, so evaluating it here is
    evaluating the selection rather than a paraphrase of it.
    """
    expected = EXPECTED[(os_key, vendor)]
    hw = spoof_cell(os_key, vendor)
    hw.detect_hardware()
    selected = hw.DEVICE == hw.DeviceType.MLX
    assert selected is expected.mlx_selected, (
        f"{os_key}/{vendor}: MLXInferenceBackend selected={selected}, "
        f"expected {expected.mlx_selected}. {expected.note}"
    )


def test_worker_selects_mlx_on_device_type_alone():
    """The construction site is guarded by the DEVICE comparison and nothing platform-ish.

    Read with ast, not a regex: the guard also carries the native-audio exclusion, and a
    grep for "DeviceType.MLX" would match the import line and the comment above it.
    """
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
        # A guard that also consulted the platform would make the DEVICE comparison a
        # partial answer, and every cell above would be measuring the wrong thing.
        for forbidden in ("platform", "sys.platform", "is_apple_silicon", "machine"):
            assert forbidden not in guard, f"{forbidden!r} in the MLX guard: {guard}"


# ======================================================================================
# 3. The context triple
# ======================================================================================

# A model config carrying a trained window, in the shape mlx-lm attaches it.
_MLX_MODEL = SimpleNamespace(args = SimpleNamespace(max_position_embeddings = 131072))
# What a transformers load attaches: Unsloth writes the served length onto the model and
# nothing else, so there is no native window to read back.
_TORCH_MODEL = SimpleNamespace(max_seq_length = 4096)


def _model_info_for(mlx_selected: bool, requested: int) -> dict:
    """The ``model_info`` the serving backend publishes for a load of ``requested``.

    Both branches call the shipped resolver rather than restating its answer: the MLX one
    is ``MLXInferenceBackend._resolve_context_lengths`` (which reads nothing off ``self``),
    the other is ``runtime_context_length``, which is the only context field
    ``core/inference/inference.py`` sets.
    """
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
    """The triple survives to ``/v1/models`` on the MLX cell and is withheld on the rest.

    Three seams, because a field can be lost at any of them and each loss looks identical
    from the last one: what the backend resolves, what the parent mirrors out of the
    subprocess (``_mirrored_model_entry``), and what the OpenAI listing publishes.
    """
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
        # A window is still reported -- transformers serves one -- but the model's own
        # length and the ceiling are unknown, and reporting a guess is what the PR's
        # frontend rule (loadedContextFields) reads as "this backend sized a window".
        #
        # 4096 under BOTH requests, and that is not a rounding of the pin: the request is
        # only runtime_context_length's FALLBACK, so whatever Unsloth attached to the
        # model wins and an 8192 pin does not show up in the report at all. The MLX rows
        # above are the contrast -- there the request is the served window.
        assert mirrored["context_length"] == 4096
        assert mirrored["native_context_length"] is None
        assert mirrored["max_context_length"] is None

    # /v1/models, through the real projection.
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
    """The asymmetry the matrix above turns on, stated once and directly.

    Only the MLX load resolves a triple. ``core/inference/inference.py`` publishes
    ``context_length`` and nothing else, whatever ``runtime_context_length`` itself can
    read, so "withheld on eleven cells" is a property of the serving path rather than of
    the eleven fixtures. Asserted on the published entry, not on the helper's own answer,
    which reads a declared window as well as the attached one.
    """
    assert runtime_context_length(_MLX_MODEL, 8192) == 8192
    served, native, ceiling = MLXInferenceBackend._resolve_context_lengths(None, _MLX_MODEL, 0)
    assert (served, native, ceiling) == (131072, 131072, 131072)
    assert set(_model_info_for(False, 0)) == {"is_mlx", "context_length"}


# ======================================================================================
# 4. The cells that are not measurements
# ======================================================================================


def test_wsl_is_indistinguishable_from_linux_in_the_detector():
    """No file under ``utils/hardware`` can tell WSL from Linux.

    So the three wsl rows above are not independent evidence, and this is what says so.
    ``llama_cpp.py`` does discriminate (``_wsl_system_rocm_lib_dirs``, and the #8403
    Windows free-VRAM cap deliberately does NOT engage under WSL) -- that is the point:
    the discrimination lives in the llama.cpp probe, not in device detection.
    """
    # Every way a Python process can learn it is under WSL. Not the bare token "WSL":
    # hardware.py carries two comments saying WSL is deliberately left alone, and a
    # comment is the opposite of a discriminator.
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
    # And the one string that IS a Windows-only lookup, named so this test cannot be read
    # as claiming the package never mentions Microsoft.
    assert "Microsoft" in _code_without_comments(HARDWARE_PACKAGE / "hardware.py")
    assert "_WINDOWS_DIRECTX_KEY" in (HARDWARE_PACKAGE / "hardware.py").read_text(encoding = "utf-8")
    # And the llama.cpp side, which does, so this stays an accurate statement of scope.
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
    """Windows x MLX is impossible by construction, not by the package being absent.

    arm64 alone is not enough, and this is the half of ``is_apple_silicon`` the ordinary
    Windows row cannot exercise (it is x86_64, so either conjunct would explain it).
    """
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
    """The AMD row's other wheel shape, which the vendor axis alone cannot carry.

    An AMD SDK / Radeon wheel leaves ``torch.version.hip`` unset, so IS_ROCM is reached
    through ``torch.__version__`` instead. Same verdict, different evidence.
    """
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
    # Exactly one cell serves MLX, and it is the only one that reports the triple.
    assert {cell for cell, exp in EXPECTED.items() if exp.mlx_selected} == {("macos", "cpu")}
    assert {cell for cell, exp in EXPECTED.items() if exp.reports_triple} == {("macos", "cpu")}

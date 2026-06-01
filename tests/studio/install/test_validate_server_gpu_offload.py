"""Tests for the GPU-offload validation added to install_llama_prebuilt.py.

Covers issue unslothai/unsloth#5807 (duplicates #5830 / #5106 / #5827): a
llama-server whose GPU backend fails to initialize still serves HTTP 200 from
CPU, so the old validate_server accepted it and Studio shipped a silently
CPU-only install. The hardened validate_server rejects such a binary when GPU
offload was requested, so the resolver / source build falls through to a
GPU-capable bundle instead of a silently CPU-only one.

Stdlib-only; the installer module is spec-loaded by absolute path (no
PYTHONPATH / heavy deps), matching test_install_llama_prebuilt_logic.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

server_log_shows_gpu_offload = M.server_log_shows_gpu_offload
resolve_smoke_test_install_kind = M.resolve_smoke_test_install_kind
smoke_test_server_binary = M.smoke_test_server_binary
validate_server = M.validate_server
PrebuiltFallback = M.PrebuiltFallback
HostInfo = M.HostInfo


# -- HostInfo factories ------------------------------------------------------


def nvidia_host(**overrides) -> HostInfo:
    defaults = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = "/usr/bin/nvidia-smi",
        driver_cuda_version = (13, 0),
        compute_caps = ["120"],
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
        has_rocm = False,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


def windows_nvidia_host(**overrides) -> HostInfo:
    return nvidia_host(
        system = "Windows",
        is_windows = True,
        is_linux = False,
        nvidia_smi = "nvidia-smi",
        **overrides,
    )


def rocm_host(**overrides) -> HostInfo:
    return nvidia_host(
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = True,
        **overrides,
    )


def macos_arm_host(**overrides) -> HostInfo:
    defaults = dict(
        system = "Darwin",
        machine = "arm64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


def cpu_host(**overrides) -> HostInfo:
    return nvidia_host(
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        **overrides,
    )


# -- Canned llama-server logs ------------------------------------------------
# Current llama.cpp (>= mid-2026) dropped the "model buffer size" lines and
# enumerates a "device_info:" block instead; older builds still print the
# buffer lines. Both formats must classify correctly.

CUDA_DEVICE_INFO_LOG = (
    "0.00.667 I log_info: verbosity = 3\n"
    "0.00.667 I device_info:\n"
    "0.01.101 I   - CUDA0   : NVIDIA GeForce RTX 5070 (12282 MiB, 11000 MiB free)\n"
    "0.01.101 I   - CPU     : AMD Ryzen 7 9700X (32000 MiB free)\n"
    "0.01.101 I system_info: n_threads = 8 | CUDA : ARCHS = 1200 | CPU : AVX2 = 1\n"
    "0.01.174 I srv  llama_server: model loaded\n"
)

CPU_ONLY_DEVICE_INFO_LOG = (
    "0.00.003 I log_info: verbosity = 3\n"
    "0.00.003 I device_info:\n"
    "0.00.003 I   - CPU     : AMD Ryzen 7 9700X (32000 MiB free)\n"
    "0.00.003 I system_info: n_threads = 8 | CPU : AVX2 = 1\n"
    "0.00.174 I srv  llama_server: model loaded\n"
)

CUDA_BUFFER_LOG = (
    "load_tensors: offloaded 33/33 layers to GPU\n"
    "load_tensors:        CUDA0 model buffer size = 21000.0 MiB\n"
    "load_tensors:   CPU_Mapped model buffer size =     0.6 MiB\n"
)

CPU_BUFFER_LOG = (
    "load_tensors:   CPU_Mapped model buffer size = 21000.0 MiB\n"
    "load_tensors:          CPU model buffer size =     0.6 MiB\n"
)

ROCM_BUFFER_LOG = "load_tensors:        ROCm0 model buffer size = 21000.0 MiB\n"
METAL_BUFFER_LOG = "load_tensors:       Metal model buffer size = 8000.0 MiB\n"
VULKAN_DEVICE_INFO_LOG = (
    "device_info:\n  - Vulkan0 : AMD Radeon (16000 MiB free)\n  - CPU : x86 (free)\n"
)
OPENCL_DEVICE_INFO_LOG = (
    "device_info:\n  - OpenCL0 : Adreno (4000 MiB free)\n  - CPU : arm (free)\n"
)
OFFLOADED_ZERO_LOG = "load_tensors: offloaded 0/33 layers to GPU\n"
NO_SIGNAL_LOG = "INFO [main] starting server\nload_tensors: file format = GGUF V3\n"
# system_info advertises a *compiled* backend; not an available device.
SYSTEM_INFO_ONLY_LOG = (
    "system_info: n_threads = 8 | CUDA : ARCHS = 1200 | CPU : AVX2 = 1\n"
)


# -- 1. Pure classifier ------------------------------------------------------


@pytest.mark.parametrize(
    "log_text,expected",
    [
        (CUDA_DEVICE_INFO_LOG, True),
        (CPU_ONLY_DEVICE_INFO_LOG, False),
        (CUDA_BUFFER_LOG, True),
        (CPU_BUFFER_LOG, False),
        (ROCM_BUFFER_LOG, True),
        (METAL_BUFFER_LOG, True),
        (VULKAN_DEVICE_INFO_LOG, True),
        (OPENCL_DEVICE_INFO_LOG, True),
        (OFFLOADED_ZERO_LOG, False),
        (NO_SIGNAL_LOG, None),
        ("", None),
        (SYSTEM_INFO_ONLY_LOG, None),
    ],
)
def test_server_log_shows_gpu_offload(log_text, expected):
    assert server_log_shows_gpu_offload(log_text) is expected


def test_opencl_device_in_prefixes():
    # Regression: OpenCL was matched by the device-row regex but missing from
    # the GPU prefix list, so an OpenCL-only device_info wrongly classified as
    # CPU-only and a working OpenCL build was rejected.
    assert "opencl" in M._GPU_DEVICE_PREFIXES
    assert server_log_shows_gpu_offload(OPENCL_DEVICE_INFO_LOG) is True


def test_system_info_not_mistaken_for_device():
    # A CUDA-compiled binary whose runtime failed still prints the system_info
    # CUDA backend, but enumerates only CPU under device_info -> must be False.
    log = (
        "device_info:\n  - CPU : x86 (free)\n"
        "system_info: | CUDA : ARCHS = 1200 | CPU : AVX2 = 1\n"
    )
    assert server_log_shows_gpu_offload(log) is False


def test_signal2_offloaded_zero_beats_device_info():
    # offloaded 0/33 (signal 2) is a definitive "nothing on GPU" and takes
    # priority over a device_info CUDA0 row -> False.
    log = (
        "load_tensors: offloaded 0/33 layers to GPU\n"
        "device_info:\n  - CUDA0 : NVIDIA (free)\n  - CPU : x (free)\n"
    )
    assert server_log_shows_gpu_offload(log) is False


def test_kv_buffer_on_gpu_with_cpu_model_is_not_offload():
    # A GPU KV/compute buffer while the model weights sit on CPU is still CPU
    # inference; only a GPU *model* buffer counts as offload.
    log = (
        "load_tensors:   CUDA0 KV buffer size = 100.0 MiB\n"
        "load_tensors:   CPU_Mapped model buffer size = 0.6 MiB\n"
    )
    assert server_log_shows_gpu_offload(log) is False


def test_offloading_zero_repeating_does_not_mask_zero_total():
    # Real CPU-only shape: a planning line says "offloading 0 repeating layers"
    # before the definitive "offloaded 0/33". The count must decide -> False.
    log = (
        "llm_load_tensors: offloading 0 repeating layers to GPU\n"
        "llm_load_tensors: offloaded 0/33 layers to GPU\n"
        "llm_load_tensors: CPU model buffer size = 7338.64 MiB\n"
    )
    assert server_log_shows_gpu_offload(log) is False


def test_uncounted_offloading_output_layer_is_gpu():
    # The uncounted "offloading output layer to GPU" phrasing (no number, no
    # later zero count) is a weak positive.
    log = "llm_load_tensors: offloading output layer to GPU\n"
    assert server_log_shows_gpu_offload(log) is True


def test_hip_musa_cann_model_buffer_is_gpu():
    for marker in ("HIP0", "MUSA0", "CANN0"):
        log = f"load_tensors:   {marker} model buffer size = 21000.0 MiB\n"
        assert server_log_shows_gpu_offload(log) is True, marker


def test_device_row_case_insensitive():
    log = "device_info:\n  - cuda0 : some gpu (free)\n  - CPU : x (free)\n"
    assert server_log_shows_gpu_offload(log) is True


def test_cuda_host_buffer_is_not_gpu_offload():
    # CUDA_Host is host-pinned CPU RAM, not device memory. A binary that pins
    # host memory but loads weights on CPU must not pass as GPU offload.
    log = (
        "load_tensors:   CUDA_Host model buffer size = 21000.0 MiB\n"
        "load_tensors:   CPU_Mapped model buffer size =     0.6 MiB\n"
    )
    assert server_log_shows_gpu_offload(log) is False
    # A real device buffer alongside a CUDA_Host line still reads as GPU.
    log_ok = (
        "load_tensors:   CUDA_Host model buffer size = 100.0 MiB\n"
        "load_tensors:   CUDA0 model buffer size = 21000.0 MiB\n"
    )
    assert server_log_shows_gpu_offload(log_ok) is True


def test_draft_offloaded_zero_before_main_offload_is_gpu():
    # Speculative decoding: a draft model logs "offloaded 0/2" before the main
    # model's "offloaded 33/33". The N>0 line must win.
    log = (
        "load_tensors: offloaded 0/2 layers to GPU\n"
        "load_tensors: offloaded 33/33 layers to GPU\n"
    )
    assert server_log_shows_gpu_offload(log) is True


def test_crlf_log_parses_identically():
    # Windows logs use CRLF; classification must not change.
    assert (
        server_log_shows_gpu_offload(CUDA_DEVICE_INFO_LOG.replace("\n", "\r\n")) is True
    )
    assert (
        server_log_shows_gpu_offload(CPU_ONLY_DEVICE_INFO_LOG.replace("\n", "\r\n"))
        is False
    )


# -- 2. resolve_smoke_test_install_kind --------------------------------------


@pytest.mark.parametrize(
    "host,expected",
    [
        (nvidia_host(), "linux-cuda"),
        (windows_nvidia_host(), "windows-cuda"),
        (rocm_host(), "linux-rocm"),
        (rocm_host(system = "Windows", is_windows = True, is_linux = False), "windows-hip"),
        (macos_arm_host(), "macos-arm64"),
        (cpu_host(), "linux-cpu"),
        (cpu_host(system = "Windows", is_windows = True, is_linux = False), "windows-cpu"),
    ],
)
def test_resolve_smoke_test_install_kind(host, expected):
    assert resolve_smoke_test_install_kind(host) == expected
    if expected.endswith("-cpu"):
        assert expected not in M._GPU_INSTALL_KINDS
    else:
        assert expected in M._GPU_INSTALL_KINDS


# -- 3. validate_server integration (mocked subprocess + HTTP) ---------------


class _FakeResponse:
    def __init__(self, status: int, body: bytes = b"{}"):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


class _FakeProc:
    """Stands in for the llama-server subprocess: writes a canned startup log
    to the handle it was given, then reports itself alive until terminated."""

    def __init__(self, log_text: str):
        self._log_text = log_text
        self._alive = True

    def __call__(self, command, *args, **kwargs):
        handle = kwargs.get("stdout")
        if handle is not None:
            handle.write(self._log_text)
            handle.flush()
        return self

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout = None):
        self._alive = False
        return 0

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False


@pytest.fixture
def patched_server(monkeypatch):
    """Patch validate_server's external touchpoints; return a configure()
    helper so each test picks the canned log it wants."""
    monkeypatch.setattr(M, "free_local_port", lambda: 18000)
    monkeypatch.setattr(M, "binary_env", lambda *a, **k: {})
    monkeypatch.setattr(M.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(200))

    def configure(log_text):
        monkeypatch.setattr(M.subprocess, "Popen", _FakeProc(log_text))

    return configure


def _run_validate(tmp_path, host, install_kind):
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF")
    validate_server(server, probe, host, tmp_path, install_kind = install_kind)


def test_gpu_intent_cpu_only_rejected(patched_server, tmp_path):
    # The #5807 bug: GPU requested, binary ran on CPU -> reject. The exception
    # must be GpuOffloadFailure specifically (not just the PrebuiltFallback
    # base) so the --smoke-test CLI can map it to EXIT_FALLBACK vs EXIT_ERROR.
    patched_server(CPU_ONLY_DEVICE_INFO_LOG)
    with pytest.raises(M.GpuOffloadFailure, match = "entirely on CPU"):
        _run_validate(tmp_path, windows_nvidia_host(), "windows-cuda")
    assert issubclass(M.GpuOffloadFailure, PrebuiltFallback)


def test_gpu_intent_gpu_offload_accepted(patched_server, tmp_path):
    patched_server(CUDA_DEVICE_INFO_LOG)
    _run_validate(tmp_path, windows_nvidia_host(), "windows-cuda")  # no raise


def test_cpu_kind_not_gpu_gated(patched_server, tmp_path):
    # A windows-cpu bundle is the intentional fallback; never GPU-gated even
    # if it (obviously) loads on CPU.
    patched_server(CPU_ONLY_DEVICE_INFO_LOG)
    _run_validate(tmp_path, windows_nvidia_host(), "windows-cpu")  # no raise


def test_gpu_intent_no_signal_accepted(patched_server, tmp_path):
    # No buffer/device signal -> conservative: do not reject on no evidence
    # (plain install validation; require_gpu_signal defaults False).
    patched_server(NO_SIGNAL_LOG)
    _run_validate(tmp_path, nvidia_host(), "linux-cuda")  # no raise


def test_smoke_test_no_signal_gpu_is_inconclusive(patched_server, tmp_path):
    # The smoke-test CLI sets require_gpu_signal, so a no-signal GPU log is
    # inconclusive (PrebuiltFallback -> EXIT_ERROR), not a silent pass.
    patched_server(NO_SIGNAL_LOG)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF")
    with pytest.raises(PrebuiltFallback):
        smoke_test_server_binary(
            str(server), nvidia_host(), install_dir = str(tmp_path), probe = str(probe)
        )


def test_rocm_cpu_only_rejected(patched_server, tmp_path):
    patched_server(CPU_ONLY_DEVICE_INFO_LOG)
    with pytest.raises(PrebuiltFallback):
        _run_validate(tmp_path, rocm_host(), "linux-rocm")


def test_macos_metal_not_in_offload_required_kinds():
    # macos-arm64 ships a GPU backend (Metal) and is launched with
    # --n-gpu-layers, but a CPU-only Metal load is an unfixable environment
    # limitation, so it must NOT be in the offload-required (reject) set.
    assert "macos-arm64" in M._GPU_INSTALL_KINDS
    assert "macos-arm64" not in M._GPU_OFFLOAD_REQUIRED_KINDS


def test_macos_metal_cpu_only_not_rejected(patched_server, tmp_path):
    # The macOS regression in #5858 CI: GitHub macOS runners have no usable
    # Metal, so the macos-arm64 prebuilt loads on CPU. It must be accepted, not
    # rejected into a source build that also runs on CPU and breaks the install.
    patched_server(CPU_ONLY_DEVICE_INFO_LOG)
    _run_validate(tmp_path, macos_arm_host(), "macos-arm64")  # no raise


def test_smoke_test_macos_metal_cpu_only_passes(patched_server, tmp_path):
    # The smoke-test CLI must also accept a CPU-only macOS Metal load (exit 0),
    # so setup.sh does not pointlessly retry a CPU source build on a Mac.
    patched_server(CPU_ONLY_DEVICE_INFO_LOG)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF")
    # No raise -> the CLI maps this to EXIT_SUCCESS.
    smoke_test_server_binary(
        str(server),
        macos_arm_host(),
        install_dir = str(tmp_path),
        probe = str(probe),
        install_kind = "macos-arm64",
    )


# -- 4. smoke_test_server_binary ---------------------------------------------


def test_smoke_test_missing_binary(tmp_path):
    with pytest.raises(PrebuiltFallback, match = "not found"):
        smoke_test_server_binary(
            str(tmp_path / "nope"), nvidia_host(), install_dir = str(tmp_path)
        )


def test_smoke_test_delegates_to_validate_server(monkeypatch, tmp_path):
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF")
    seen = {}

    def fake_validate(server_path, probe_path, host, install_dir, **kw):
        seen["install_kind"] = kw.get("install_kind")

    monkeypatch.setattr(M, "validate_server", fake_validate)
    kind = smoke_test_server_binary(
        str(server), nvidia_host(), install_dir = str(tmp_path), probe = str(probe)
    )
    assert kind == "linux-cuda"
    assert seen["install_kind"] == "linux-cuda"


def test_smoke_test_propagates_failure(monkeypatch, tmp_path):
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF")

    def fake_validate(*a, **k):
        raise PrebuiltFallback("loaded the model entirely on CPU")

    monkeypatch.setattr(M, "validate_server", fake_validate)
    with pytest.raises(PrebuiltFallback):
        smoke_test_server_binary(
            str(server),
            nvidia_host(),
            install_dir = str(tmp_path),
            probe = str(probe),
        )


# -- 5. --smoke-test CLI exit-code contract (the load-bearing fix contract) --
# setup.sh / setup.ps1 branch on: 2 = ran on CPU (rebuild CPU), 1 = inconclusive
# (keep GPU build), 0 = offload confirmed.


def _run_main_smoke(monkeypatch, smoke_impl):
    monkeypatch.setattr(M, "detect_host", lambda: nvidia_host())
    monkeypatch.setattr(M, "smoke_test_server_binary", smoke_impl)
    monkeypatch.setattr(
        sys, "argv", ["install_llama_prebuilt.py", "--smoke-test", "/x/llama-server"]
    )
    return M.main()


def test_main_smoke_exit_success(monkeypatch):
    rc = _run_main_smoke(monkeypatch, lambda *a, **k: "linux-cuda")
    assert rc == M.EXIT_SUCCESS


def test_main_smoke_exit_fallback_on_cpu_only(monkeypatch):
    def impl(*a, **k):
        raise M.GpuOffloadFailure("loaded the model entirely on CPU")

    rc = _run_main_smoke(monkeypatch, impl)
    assert rc == M.EXIT_FALLBACK  # 2 -> setup scripts rebuild CPU


def test_main_smoke_exit_error_on_inconclusive(monkeypatch):
    def impl(*a, **k):
        raise PrebuiltFallback("llama-server exited during startup")

    rc = _run_main_smoke(monkeypatch, impl)
    assert rc == M.EXIT_ERROR  # 1 -> setup scripts keep the GPU build


# -- 6. existing_gpu_install_offloads: re-validate a matching install (#5807) --


def _gpu_plan(install_kind = "linux-cuda"):
    choice = M.AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "b9001",
        name = "app-b9001-linux-x64-cuda13-newer.tar.gz",
        url = "https://example.com/x",
        source_label = "published",
        install_kind = install_kind,
    )
    return M.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "rel",
        attempts = [choice],
        approved_checksums = M.ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "rel",
            upstream_tag = "b9001",
            artifacts = {},
        ),
    )


def _with_server(tmp_path):
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF")
    return probe


def test_existing_cpu_kind_install_is_kept(tmp_path):
    # A non-GPU existing install is never offload-gated.
    probe = _with_server(tmp_path)
    assert (
        M.existing_gpu_install_offloads(
            tmp_path, nvidia_host(), _gpu_plan("linux-cpu"), probe
        )
        is True
    )


def test_existing_gpu_install_cpu_only_triggers_reinstall(monkeypatch, tmp_path):
    probe = _with_server(tmp_path)

    def fake_validate(*a, **k):
        raise M.GpuOffloadFailure("loaded the model entirely on CPU")

    monkeypatch.setattr(M, "validate_server", fake_validate)
    assert (
        M.existing_gpu_install_offloads(
            tmp_path, nvidia_host(), _gpu_plan("linux-cuda"), probe
        )
        is False
    )


def test_existing_gpu_install_offloading_is_kept(monkeypatch, tmp_path):
    probe = _with_server(tmp_path)
    monkeypatch.setattr(M, "validate_server", lambda *a, **k: None)
    assert (
        M.existing_gpu_install_offloads(
            tmp_path, nvidia_host(), _gpu_plan("linux-cuda"), probe
        )
        is True
    )


def test_existing_gpu_install_inconclusive_is_kept(monkeypatch, tmp_path):
    probe = _with_server(tmp_path)

    def fake_validate(*a, **k):
        raise PrebuiltFallback("llama-server exited during startup")

    monkeypatch.setattr(M, "validate_server", fake_validate)
    assert (
        M.existing_gpu_install_offloads(
            tmp_path, nvidia_host(), _gpu_plan("linux-cuda"), probe
        )
        is True
    )


def test_existing_gpu_install_no_binary_is_kept(tmp_path):
    # No llama-server present -> let the normal flow reinstall, don't crash.
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF")
    assert (
        M.existing_gpu_install_offloads(
            tmp_path, nvidia_host(), _gpu_plan("linux-cuda"), probe
        )
        is True
    )

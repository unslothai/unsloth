# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""vLLM on AMD GPUs: the ROCm profile, its support checks and its launch environment."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import engine_install as install
from core.inference import managed_engine

_LINUX = pytest.mark.skipif(sys.platform != "linux", reason = "local engine host is Linux only")


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(install, "gpu_platform", lambda: "rocm")


@pytest.fixture
def amd_host(rocm, monkeypatch, tmp_path):
    """A Linux host with ROCm 7.2.1 in /opt/rocm and one gfx1151."""
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install.platform, "libc_ver", lambda: ("glibc", "2.39"))
    monkeypatch.setattr(install, "ROCM_HOME", tmp_path)
    (tmp_path / ".info").mkdir()
    (tmp_path / ".info" / "version").write_text("7.2.1-81\n")
    (tmp_path / "lib").mkdir()
    for soname in install.ROCM_LIBRARIES:
        (tmp_path / "lib" / soname).touch()
    monkeypatch.setattr(install.os, "access", lambda *_: True)
    monkeypatch.delenv("CC", raising = False)
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    from utils.hardware import amd

    targets = {"value": ["gfx1151"]}
    monkeypatch.setattr(amd, "amd_kfd_gpu_gfx_targets", lambda: targets["value"])
    # What Studio's own torch presents; empty when it reports nothing, as on a CPU torch.
    arches = {}
    monkeypatch.setattr(install, "_rocm_gpu_arches", lambda: dict(arches))
    libraries = set(install.ROCM_SYSTEM_LIBRARIES)
    monkeypatch.setattr(amd, "_a_bare_soname_resolves", lambda soname, **_: soname in libraries)
    monkeypatch.setattr(amd, "amd_closed_nodes_block_the_runtime", lambda **_: False)
    return SimpleNamespace(rocm = tmp_path, targets = targets, libraries = libraries, arches = arches)


def test_amd_host_installs_vllms_rocm_build(rocm):
    chosen = install.profile("vllm")
    pins = install._pins("vllm")
    assert chosen["lock"] == "vllm-linux-rocm723" and chosen["cuda"] is None
    assert chosen["python"] == (3, 12)
    assert pins["vllm"][0] == "0.30.0+rocm723"
    assert pins["torch"][0].split("+")[0] == chosen["torch"]
    # TorchAO serves INT8 and FP8; the ROCm build has no bitsandbytes quantization to use.
    assert "torchao" in pins and "bitsandbytes" not in pins
    assert "int4" not in chosen["precisions"] and not chosen["bitsandbytes"]
    assert "--python-platform x86_64-manylinux_2_39" in install.requirements("vllm").read_text()
    assert install._compat_file("vllm").get("sizes"), "regenerate with engine_compat.py"
    assert install._index_arguments("vllm") == [
        "--index-url",
        "https://pypi.org/simple",
        "--extra-index-url",
        chosen["index"],
    ]


def test_nvidia_host_keeps_the_cuda_profile(monkeypatch):
    monkeypatch.setattr(install, "gpu_platform", lambda: "cuda")
    monkeypatch.setattr(install, "_studio_packages", lambda: {})
    assert install.profile("vllm")["lock"] == "vllm-linux-cu130-torch213"
    assert install._index_arguments("vllm") == ["--index-url", "https://pypi.org/simple"]
    smoke = install._smoke_source("vllm")
    assert "import bitsandbytes" in smoke and "torch.version.cuda == '13.0'" in smoke


def test_rocm_engine_never_shares_studios_torch(rocm, monkeypatch):
    # Studio's own ROCm torch comes from AMD's per-arch index, another build than vLLM's.
    monkeypatch.setattr(install, "_studio_packages", lambda: {"torch": "2.11.0+rocm7.13.0"})
    plan = install.install_plan("vllm")
    assert not plan["shared"] and not plan["provided"]
    assert install.download_bytes("vllm") == sum(install._compat_file("vllm")["sizes"].values())


def test_rocm_engine_stays_isolated_even_when_studio_matches_its_lock(rocm, monkeypatch):
    # The ROCm torch loads /opt/rocm instead of bundled libraries, so Studio's packages never stand in
    # for it, even on a Studio that happens to run the same Python with the very same builds.
    pins = {name: version for name, (version, _) in install._pins("vllm").items()}
    monkeypatch.setattr(install, "_studio_packages", lambda: dict(pins))
    monkeypatch.setattr(install, "_torch_runtime", lambda: {"torch"})
    monkeypatch.setattr(install, "_python", lambda engine: sys.version_info[:2])
    plan = install.install_plan("vllm")
    assert not plan["shared"] and not plan["provided"]


def test_windows_waits_for_hardware_detection_before_choosing_the_profile(monkeypatch):
    # IS_ROCM is False until detection settles; Windows has no KFD to fall back on.
    from utils.hardware import hardware

    monkeypatch.setattr(install.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "IS_ROCM", False)
    monkeypatch.setattr(hardware, "DETECTION_COMPLETE", threading.Event())

    def detect(*_a, **_k):
        hardware.IS_ROCM = True
        hardware.DETECTION_COMPLETE.set()

    monkeypatch.setattr(hardware, "ensure_hardware_detected", detect)
    assert install.gpu_platform() == "rocm"
    assert install.profile("vllm")["lock"] == "vllm-linux-rocm723"


def test_rocm_smoke_imports_the_hip_torch(rocm):
    smoke = install._smoke_source("vllm")
    assert "assert torch.version.hip" in smoke and "import torchao" in smoke
    assert "bitsandbytes" not in smoke and "'2.12.0'" in smoke
    compile(smoke, "smoke", "exec")


@pytest.mark.parametrize(
    ("version", "reason"),
    [
        (None, "Requires ROCm 7.2 or a newer 7.x release installed in /opt/rocm."),
        ("6.4.1", "Requires ROCm 7.2 or a newer 7.x release installed in /opt/rocm (found 6.4)."),
        ("7.1.0", "Requires ROCm 7.2 or a newer 7.x release installed in /opt/rocm (found 7.1)."),
        ("8.0.0", "Requires ROCm 7.2 or a newer 7.x release installed in /opt/rocm (found 8.0)."),
        ("7.2.1", None),
        ("7.13.0", None),
    ],
)
def test_amd_support_needs_a_rocm_7_runtime(amd_host, version, reason):
    if version is None:
        (amd_host.rocm / ".info" / "version").unlink()
    else:
        (amd_host.rocm / ".info" / "version").write_text(version)
    assert install.support_reason("vllm", wait = False) == reason


def test_amd_support_checks_the_selected_gpus_architecture(amd_host):
    amd_host.targets["value"] = ["gfx1030", "gfx1100"]
    assert install.support_reason("vllm", wait = False) is None
    assert install.support_reason("vllm", 1) is None
    assert install.support_reason("vllm", 0).endswith("(found gfx1030).")
    amd_host.targets["value"] = []
    assert "Ryzen AI Max" in install.support_reason("vllm", wait = False)


def test_amd_support_names_the_libraries_vllms_torch_links(amd_host):
    # A ROCm install from rocm-libs alone lacks the profiler libraries; torch's import would fail.
    for soname in ("librocprofiler-sdk.so.1", "libhsa-amd-aqlprofile64.so.1"):
        (amd_host.rocm / "lib" / soname).unlink()
    assert install.support_reason("vllm", wait = False) == (
        "The ROCm installation in /opt/rocm is missing librocprofiler-sdk.so.1, "
        "libhsa-amd-aqlprofile64.so.1. Install the full ROCm package (on Ubuntu: sudo apt install "
        "rocm), then retry."
    )
    for soname in ("librocprofiler-sdk.so.1", "libhsa-amd-aqlprofile64.so.1"):
        (amd_host.rocm / "lib" / soname).touch()
    amd_host.libraries.clear()
    assert install.support_reason("vllm", wait = False) == (
        "Requires libmpi.so.40, libmpi_cxx.so.40, libnuma.so.1, which vLLM's AMD build links. "
        "On Ubuntu: sudo apt install libopenmpi3t64 libnuma1"
    )
    # ROCm's own directory counts: the engine's torch finds it through its RUNPATH.
    (amd_host.rocm / "lib" / "libnuma.so.1").touch()
    amd_host.libraries.update(("libmpi.so.40", "libmpi_cxx.so.40"))
    assert install.support_reason("vllm", wait = False) is None


def test_amd_support_ignores_libraries_only_ld_library_path_provides(monkeypatch, tmp_path):
    # The engine starts without the caller's LD_LIBRARY_PATH, so only the system loader counts.
    from utils.hardware import amd

    custom = tmp_path / "openmpi"
    custom.mkdir()
    for soname in install.ROCM_SYSTEM_LIBRARIES:
        (custom / soname).touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(custom))
    monkeypatch.setattr(amd, "_ld_cache_sonames", lambda: frozenset())
    monkeypatch.setattr(amd, "_system_library_dirs", lambda: [])
    for soname in install.ROCM_SYSTEM_LIBRARIES:
        assert amd._a_bare_soname_resolves(soname)
        assert not amd._a_bare_soname_resolves(soname, with_ld_library_path = False)
    monkeypatch.setattr(amd, "_system_library_dirs", lambda: [str(custom)])
    assert amd._a_bare_soname_resolves("libnuma.so.1", with_ld_library_path = False)


def test_amd_support_needs_a_c_compiler(amd_host, monkeypatch):
    # Triton builds its HIP driver module with it the first time the engine compiles a kernel.
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    assert "C compiler" in install.support_reason("vllm", wait = False)
    monkeypatch.setenv("CC", "/opt/cc")
    assert install.support_reason("vllm", wait = False) is None


def test_amd_support_reads_the_target_the_gpu_presents(amd_host):
    # An RX 7600 (gfx1102) run as gfx1100 through HSA_OVERRIDE_GFX_VERSION, which the engine
    # inherits, is accepted; the kernel's own target is only the fallback.
    amd_host.targets["value"] = ["gfx1102"]
    assert install.support_reason("vllm", 0).endswith("(found gfx1102).")
    amd_host.arches.update({0: "gfx1100"})
    assert install.support_reason("vllm", 0) is None
    amd_host.targets["value"] = ["gfx1102", "gfx1030"]
    assert install.support_reason("vllm", 1).endswith("(found gfx1030).")
    # A GPU nothing names is left to vLLM, which refuses one it has no kernels for.
    assert install.support_reason("vllm", 2) is None


def test_rocm_environment_uses_a_managed_python_with_headers(rocm):
    # Ubuntu's python3.12 has no Python.h without python3.12-dev, and Triton compiles against it.
    assert install._venv_python_args("vllm") == [
        "--python",
        "3.12",
        "--python-preference",
        "only-managed",
    ]


def test_nvidia_environment_keeps_its_interpreter_choice(monkeypatch):
    monkeypatch.setattr(install, "gpu_platform", lambda: "cuda")
    monkeypatch.setattr(install, "_studio_packages", lambda: {})
    expected = sys.executable if sys.version_info[:2] == (3, 13) else "3.13"
    assert install._venv_python_args("vllm") == ["--python", expected]


def test_amd_support_reports_an_unopenable_kfd_and_old_glibc(amd_host, monkeypatch):
    monkeypatch.setattr(install.os, "access", lambda *_: False)
    assert "/dev/kfd" in install.support_reason("vllm", wait = False)
    monkeypatch.setattr(install.platform, "libc_ver", lambda: ("glibc", "2.35"))
    assert install.support_reason("vllm", wait = False) == "vLLM requires glibc 2.39 or newer."


def test_amd_support_needs_a_render_node_as_well_as_kfd(amd_host, monkeypatch):
    # A container given --device /dev/kfd without /dev/dri opens KFD and initialises no GPU.
    from utils.hardware import amd

    monkeypatch.setattr(amd, "amd_closed_nodes_block_the_runtime", lambda **_: True)
    monkeypatch.setattr(
        amd,
        "amd_node_permission_hint",
        lambda **_: "Recreate the container with --device /dev/dri.",
    )
    assert (
        install.support_reason("vllm", wait = False)
        == "Recreate the container with --device /dev/dri."
    )
    monkeypatch.setattr(amd, "amd_node_permission_hint", lambda **_: None)
    assert "/dev/dri/renderD*" in install.support_reason("vllm", wait = False)


def test_sdk_rocm_torch_still_maps_each_gpu_to_its_target(monkeypatch):
    # AMD's SDK / Radeon wheels leave torch.version.hip unset and carry "rocm" in the version.
    import torch
    from utils.hardware import hardware

    monkeypatch.setattr(install, "_rocm_arches", None)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.9.0+rocmsdk20251116")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    props = {0: SimpleNamespace(gcnArchName = "gfx1100"), 1: SimpleNamespace(gcnArchName = "gfx1030")}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props[i])
    monkeypatch.setattr(hardware, "_rocm_device_ordinal_active", lambda: False)
    monkeypatch.setattr(hardware, "_rocm_visibility_masks_are_stacked", lambda: False)
    monkeypatch.setattr(hardware, "_torch_ordinal_physical_ids", lambda count: [0, 1])
    assert install._rocm_gpu_arches() == {0: "gfx1100", 1: "gfx1030"}


def test_sglang_on_amd_points_to_vllm(amd_host):
    assert install.support_reason("sglang", wait = False) == (
        "SGLang requires an NVIDIA GPU. Use vLLM on AMD GPUs."
    )


def test_status_lists_the_precisions_the_build_loads(amd_host, monkeypatch, tmp_path):
    monkeypatch.setattr(install, "engine_root", lambda: tmp_path / "engines")
    status = install.status("vllm")
    assert status["unsupported_reason"] is None
    assert status["precisions"] == ["auto", "bf16", "fp16", "int8", "fp8"]
    monkeypatch.setattr(install, "gpu_platform", lambda: "cuda")
    monkeypatch.setattr(install, "support_reason", lambda *a, **k: None)
    assert "int4" in install.status("vllm")["precisions"]


def test_engine_port_stays_below_the_limit_when_the_os_offers_high_ports(monkeypatch):
    # Windows counts ephemeral ports up from 49152; past 55535 every OS-chosen port was refused and
    # the load failed with "Could not allocate an inference server port".
    taken = {30001}

    class Socket:
        def __init__(self, *a):
            self.port = None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def bind(self, address):
            if address[1] in taken:
                raise OSError(98, "Address already in use")
            self.port = address[1] or 60000

        def getsockname(self):
            return ("127.0.0.1", self.port)

    picks = iter([10001, 30001 - 20000, 41000 - 20000])
    monkeypatch.setattr(managed_engine.socket, "socket", Socket)
    monkeypatch.setattr(managed_engine.secrets, "randbelow", lambda n: next(picks) % n)
    assert managed_engine._free_port() == 41000
    monkeypatch.setattr(managed_engine.secrets, "randbelow", lambda n: 99999)
    with pytest.raises(RuntimeError, match = "allocate an inference server port"):
        managed_engine._free_port()


def test_rocm_mask_follows_an_inherited_rocr_mask(monkeypatch):
    # HIP numbers what ROCr left visible: under ROCR_VISIBLE_DEVICES=2,1 physical GPU 1 is HIP 1.
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,1")
    env = {"ROCR_VISIBLE_DEVICES": "2,1"}
    mask = managed_engine._rocm_visibility(env, [1])
    assert mask == {"HIP_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "1"}
    assert env["ROCR_VISIBLE_DEVICES"] == "2,1"
    # A GPU the ROCr mask hides cannot be named through it, so the mask goes.
    mask = managed_engine._rocm_visibility(env, [0])
    assert mask == {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_kfd_topology_names_each_gpu_in_hip_order(tmp_path):
    from utils.hardware.amd import amd_kfd_gpu_gfx_targets

    def node(number, text):
        (tmp_path / str(number)).mkdir()
        (tmp_path / str(number) / "properties").write_text(text)

    node(0, "simd_count 0\nvendor_id 0\ngfx_target_version 0\n")
    node(1, "simd_count 80\nvendor_id 4098\ngfx_target_version 110501\n")
    node(2, "simd_count 132\nvendor_id 4318\ngfx_target_version 0\n")
    node(10, "simd_count 440\nvendor_id 4098\ngfx_target_version 90010\n")
    assert amd_kfd_gpu_gfx_targets(str(tmp_path)) == ["gfx1151", "gfx90a"]
    assert amd_kfd_gpu_gfx_targets(str(tmp_path / "missing")) is None


def test_amd_refuses_4_bit_before_unloading(rocm, monkeypatch):
    from models.inference import LoadRequest

    info = {"path": "/env", "version": "0.30.0", "profile_digest": install.profile_digest("vllm")}
    monkeypatch.setattr(managed_engine, "installed", lambda _: info)
    monkeypatch.setattr(managed_engine, "support_reason", lambda *a: None)
    monkeypatch.setattr(managed_engine, "resolve_requested_gpu_ids", lambda ids: [0])
    request = LoadRequest(model_path = "m", engine = "vllm", engine_precision = "int4")
    with pytest.raises(ValueError, match = "^vLLM on this GPU cannot load weights as 4-bit"):
        managed_engine.validate_load("vllm", request)
    for precision in ("auto", "bf16", "fp16", "int8", "fp8"):
        managed_engine.validate_load(
            "vllm", request.model_copy(update = {"engine_precision": precision})
        )


@_LINUX
def test_a_rollback_never_crosses_gpu_platforms(rocm, monkeypatch, tmp_path):
    from models.inference import LoadRequest

    monkeypatch.setattr(install, "engine_root", lambda: tmp_path)
    monkeypatch.setattr(install, "support_reason", lambda *a, **k: None)
    monkeypatch.setattr(install, "_studio_packages", lambda: {})
    monkeypatch.setattr(install, "_jobs", {})
    monkeypatch.setattr(managed_engine, "support_reason", lambda *a: None)
    monkeypatch.setattr(managed_engine, "resolve_requested_gpu_ids", lambda ids: [0])
    for directory in ("env-cuda", "env-rocm"):
        (tmp_path / "vllm" / directory / "bin").mkdir(parents = True)
        (tmp_path / "vllm" / directory / "bin" / "python").touch()
    # A CUDA environment from before the NVIDIA card was swapped for an AMD one, then updated.
    cuda = {"directory": "env-cuda", "version": "0.30.0", "profile_digest": "cuda"}
    marker = tmp_path / "vllm" / "active.json"
    marker.write_text(
        json.dumps(
            {
                "directory": "env-rocm",
                "platform": "rocm",
                "profile_digest": install.profile_digest("vllm"),
                "previous_directory": "env-cuda",
                "previous": cuda,
            }
        )
    )
    assert install.status("vllm")["can_rollback"] is False
    with pytest.raises(RuntimeError, match = "another GPU platform"):
        install.rollback("vllm")
    # One restored before the swap is neither ready nor loaded, so the resident model stays.
    request = LoadRequest(model_path = "m", engine = "vllm")
    marker.write_text(json.dumps({**cuda, "restored": True}))
    assert install.status("vllm")["restored"] is False
    with pytest.raises(ValueError, match = "^Update vLLM"):
        managed_engine.validate_load("vllm", request)
    marker.write_text(json.dumps({**cuda, "platform": "rocm", "restored": True}))
    assert install.status("vllm")["restored"] is True
    managed_engine.validate_load("vllm", request)


def test_amd_refuses_bitsandbytes_checkpoints_and_skips_nvidia_smi(rocm, monkeypatch, tmp_path):
    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "bitsandbytes"}})
    )
    with pytest.raises(ValueError, match = "^vLLM on this GPU cannot load BitsAndBytes checkpoints"):
        managed_engine.validate_model(config, engine = "vllm")
    (tmp_path / "config.json").write_text(json.dumps({"num_attention_heads": 8}))

    def no_nvidia_smi(*_a, **_k):
        raise AssertionError("nvidia-smi is not an AMD tool")

    monkeypatch.setattr(managed_engine.subprocess, "run", no_nvidia_smi)
    options = managed_engine.validate_model(config, engine = "vllm", precision = "fp8")
    assert not options["disable_cuda_graph"]


@_LINUX
def test_device_memory_is_read_by_the_engines_torch(tmp_path):
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    # A warning printed after the measurement must not hide it.
    python.write_text(
        '#!/bin/sh\necho "torch banner"\necho "[[$HIP_VISIBLE_DEVICES, 2147483648]]"\n'
        'echo "UserWarning: Can\'t initialize amdsmi"\n'
    )
    python.chmod(0o755)
    rows = managed_engine._engine_memory_rows(
        {"path": str(tmp_path)},
        {"HIP_VISIBLE_DEVICES": "1073741824", "PATH": "/usr/bin:/bin"},
        [0],
    )
    assert rows == [(2048.0, 1024.0)]


@_LINUX
def test_rocm_engine_sees_one_device_mask_and_its_own_memory_budget(monkeypatch, tmp_path):
    import os
    import httpx
    from utils import vram_budget_settings

    monkeypatch.setattr(install, "engine_root", lambda: tmp_path)
    folder = tmp_path / "vllm" / "env-rocm"
    (folder / "bin").mkdir(parents = True)
    (folder / "bin" / "python").touch()
    (tmp_path / "vllm" / "active.json").write_text(
        json.dumps({"directory": "env-rocm", "version": "0.30.0", "platform": "rocm"})
    )
    measured = []

    def rows(info, env, gpu_ids):
        assert gpu_ids == [1]
        measured.append((info["path"], env["HIP_VISIBLE_DEVICES"], env["CUDA_VISIBLE_DEVICES"]))
        return [(65536.0, 60000.0)]

    monkeypatch.setattr(managed_engine, "_engine_memory_rows", rows)
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: 0.97)
    engine = managed_engine.ManagedEngine("vllm")
    budgets = []

    def command(python, model, port, key, context, memory, tensor_parallel_size):
        budgets.append(memory)
        code = (
            "import os\n"
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            " def do_GET(self):\n"
            "  self.send_response(200); self.end_headers()\n"
            "  self.wfile.write('|'.join(os.environ.get(k, '') for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')).encode())\n"
            " def log_message(self, *args): pass\n"
            f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n"
        )
        return [sys.executable, "-u", "-c", code]

    engine.adapter = SimpleNamespace(
        command = command,
        progress = lambda _: None,
        environment = lambda _: {},
        key_environment = lambda _: {},
    )
    errors = []

    def start():
        try:
            engine.start("model", 2048, [1], dict(os.environ, HIP_VISIBLE_DEVICES = "0,1"))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target = start)
    try:
        thread.start()
        thread.join(15)
        assert not thread.is_alive() and not errors
        assert httpx.get(engine.base_url, trust_env = False).text == "1|1"
        assert measured == [(str(folder), "1", "1")]
        # (60000 - max(3072, 6% of 65536)) / 65536, the most a ROCm card can give.
        assert budgets == [0.855]
    finally:
        assert engine.stop()
        thread.join(5)


@_LINUX
def test_a_startup_timeout_names_where_the_engine_stalled(monkeypatch, tmp_path):
    # vLLM went silent after "JIT kernel warmup" when its memory budget was too large; the error
    # used to say only that it timed out.
    import os

    monkeypatch.setattr(install, "engine_root", lambda: tmp_path)
    folder = tmp_path / "vllm" / "env-a"
    (folder / "bin").mkdir(parents = True)
    (folder / "bin" / "python").touch()
    (tmp_path / "vllm" / "active.json").write_text(json.dumps({"directory": "env-a"}))
    monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda *a: 0.5)
    monkeypatch.setattr(managed_engine, "STARTUP_STALL_S", 1.5)
    engine = managed_engine.ManagedEngine("vllm")
    code = "import time\nprint('JIT kernel warmup finished', flush=True)\ntime.sleep(60)\n"
    engine.adapter = SimpleNamespace(
        command = lambda *a, **k: [sys.executable, "-u", "-c", code],
        progress = lambda _: None,
        environment = lambda _: {},
        key_environment = lambda _: {},
    )
    with pytest.raises(RuntimeError, match = "(?s)timed out.*JIT kernel warmup finished"):
        engine.start("model", 2048, [0], dict(os.environ))
    assert not engine.alive()

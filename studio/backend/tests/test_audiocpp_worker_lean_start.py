# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An audio.cpp worker only proxies to audiocpp_server, so its startup never imports torch or transformers."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

_WORKER_DRIVER = r"""
import json, os, queue, sys

out, audio_cpp = sys.argv[1], sys.argv[2] == "1"
import core.inference.worker as w
import utils.hardware.hardware as hw
import utils.utils as uu

assert os.path.realpath(w.__file__).startswith(os.path.realpath(sys.argv[3])), w.__file__
calls = []


def spy(name, result = None, stop = False, owner = w):
    def fake(*args, **kwargs):
        calls.append(name)
        if stop:
            dump("stopped")
        return result(*args) if callable(result) else result
    setattr(owner, name, fake)


def dump(where):
    with open(out, "w") as f:
        json.dump({
            "where": where,
            "calls": calls,
            "torch": "torch" in sys.modules,
            "transformers": "transformers" in sys.modules,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        }, f)
    os._exit(0)


# Offline answer without the network; recording it proves the probe still runs.
spy("hf_dns_dead", True, owner = uu)
spy("detect_hardware", hw.DeviceType.CPU, owner = hw)
spy("_native_audio_security_targets_or_error", lambda model, *a: [model])
spy("_run_security_gates", True)
spy("_ensure_ssm_kernels", True)
# A non audio.cpp worker stops at activation, before the Unsloth import this test cannot afford.
spy("_activate_transformers_version", None, stop = not audio_cpp)
w._handle_load = lambda backend, config, resp_queue: dump(type(backend).__name__)

config = {
    "model_name": "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF" if audio_cpp else "org/some-llm",
    "hf_token": None,
    "device_backend": "cpu",
    "audio_device": None,
}
if audio_cpp:
    config["audio_cpp"] = True
w.run_inference_process(cmd_queue = queue.Queue(), resp_queue = queue.Queue(), cancel_event = None, config = config)
dump("returned")
"""


def _run_driver(tmp_path, *args):
    out = tmp_path / "result.json"
    env = {
        k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env["PYTHONPATH"] = str(_BACKEND)
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_DRIVER, str(out), *args, str(_BACKEND)],
        cwd = tmp_path,
        env = env,
        timeout = 240,
        capture_output = True,
        text = True,
    )
    assert out.exists(), proc.stderr[-4000:]
    return json.loads(out.read_text())


@pytest.mark.skipif(sys.platform == "darwin", reason = "macOS Apple Silicon takes the MLX branch")
def test_audio_cpp_worker_skips_torch_and_transformers_but_keeps_gates(tmp_path):
    result = _run_driver(tmp_path, "1")
    assert result["where"] == "AudioCppBackend", result
    assert result["torch"] is False, result
    assert result["transformers"] is False, result
    assert "detect_hardware" not in result["calls"]
    assert "_activate_transformers_version" not in result["calls"]
    # The offline probe, every security gate and the SSM step still run.
    assert result["hf_hub_offline"] == "1"
    for name in (
        "hf_dns_dead",
        "_native_audio_security_targets_or_error",
        "_run_security_gates",
        "_ensure_ssm_kernels",
    ):
        assert name in result["calls"], (name, result)


@pytest.mark.skipif(sys.platform == "darwin", reason = "macOS Apple Silicon takes the MLX branch")
def test_other_workers_still_detect_hardware_and_activate(tmp_path):
    result = _run_driver(tmp_path, "0")
    assert result["where"] == "stopped", result
    assert "detect_hardware" in result["calls"]
    assert "_activate_transformers_version" in result["calls"]
    assert result["hf_hub_offline"] == "1"


_GPU_MEMORY_DRIVER = r"""
import json, sys
import core.inference.worker as w

before = "torch" in sys.modules
cuda = w._worker_reclaimable_gpu_gb({"resolved_gpu_ids": [0, 3], "device_backend": "cuda"})
xpu = w._worker_reclaimable_gpu_gb({"resolved_gpu_ids": [1], "device_backend": "xpu"})
mlx = w._worker_reclaimable_gpu_gb({"resolved_gpu_ids": [0], "device_backend": "mlx"})
none = w._worker_reclaimable_gpu_gb({"resolved_gpu_ids": None, "device_backend": "cuda"})
print(json.dumps({"before": before, "after": "torch" in sys.modules, "cuda": cuda, "xpu": xpu, "mlx": mlx, "none": none}))
"""


def test_gpu_memory_reports_zero_without_importing_torch():
    proc = subprocess.run(
        [sys.executable, "-c", _GPU_MEMORY_DRIVER],
        cwd = _BACKEND,
        env = {**os.environ, "PYTHONPATH": str(_BACKEND)},
        timeout = 120,
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["before"] is False and result["after"] is False, result
    assert result["cuda"] == {"0": 0.0, "3": 0.0}
    assert result["xpu"] == {"1": 0.0}
    assert result["mlx"] is None and result["none"] is None


_APPLY_DRIVER = r"""
import json, os, sys
import utils.hardware.hardware as hw
import utils.hardware.mig as mig

mig.mask_for = lambda ids: None
hw.apply_gpu_ids([1], backend = "cuda", rocm_build = sys.argv[1] == "1")
print(json.dumps({
    "torch": "torch" in sys.modules,
    "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "hip": os.environ.get("HIP_VISIBLE_DEVICES"),
}))
"""


@pytest.mark.parametrize("rocm", [False, True])
def test_apply_gpu_ids_takes_the_parent_rocm_verdict_without_torch(rocm):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    }
    env["PYTHONPATH"] = str(_BACKEND)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    proc = subprocess.run(
        [sys.executable, "-c", _APPLY_DRIVER, "1" if rocm else "0"],
        cwd = _BACKEND,
        env = env,
        timeout = 120,
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["torch"] is False, result
    assert result["cvd"] == "1"
    assert result["hip"] == ("1" if rocm else None)

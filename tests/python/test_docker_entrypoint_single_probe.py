# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Entrypoint torch checks share one interpreter with the old output, order and exit codes (fake torch, no GPU)."""

import os
import shutil
import stat
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCKER = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "docker")
_CUDA = os.path.join(_DOCKER, "entrypoint.sh")
_ROCM = os.path.join(_DOCKER, "entrypoint-rocm.sh")

_posix_shell = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason = "POSIX shell required",
)


def _stub(path, body):
    with open(path, "w", encoding = "utf-8") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _fake_torch(tmp_path, init, cuda):
    pkg = tmp_path / "fake" / "torch"
    (pkg / "cuda").mkdir(parents = True)
    (pkg / "lib").mkdir()
    (pkg / "__init__.py").write_text("import warnings\n" + init + "from . import cuda\n")
    (pkg / "cuda" / "__init__.py").write_text("import warnings\n" + cuda)
    return tmp_path / "fake"


def _cuda_torch(
    tmp_path,
    caps,
    available = True,
    raise_in = None,
):
    body = f"CAPS = {caps!r}\n"
    body += f"def is_available(): return {available}\n"
    body += "def get_device_capability(i):\n"
    if raise_in == "get_device_capability":
        body += "    raise RuntimeError('CUDA error: system not yet initialized')\n"
    body += "    return CAPS[i]\n"
    body += "def get_device_name(i): return 'GPU%d' % i\n"
    body += "def device_count(): return len(CAPS)\n"
    body += f"def is_bf16_supported(): return {caps[0][0] >= 8}\n"
    return _fake_torch(tmp_path, "", body)


def _rocm_torch(
    tmp_path,
    arch = "gfx1100",
    available = True,
    warn = False,
):
    init = "__version__ = '2.11.0+rocm7.2'\nclass version:\n    hip = '7.2.53211'\n"
    body = f"class _P:\n    gcnArchName = {arch!r}\n"
    body += f"def is_available(): return {available}\n"
    body += "def device_count(): return 1\n"
    body += "def get_device_name(i): return 'AMD GPU'\n"
    body += "def get_device_properties(i):\n"
    if warn:
        body += "    warnings.warn('late HIP warning')\n"
    body += "    return _P()\n"
    body += "def is_bf16_supported(): return True\n"
    body += "def get_arch_list(): return ['gfx1100']\n"
    return _fake_torch(tmp_path, init, body)


def _run(
    tmp_path,
    script,
    fake,
    rocm = False,
):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls"
    _stub(
        str(bindir / "python"),
        f'echo x >> "{calls}"\nPYTHONPATH="{fake}" exec "{sys.executable}" "$@"\n',
    )
    env = {
        "PATH": str(bindir) + ":/usr/bin:/bin",
        "HOME": str(tmp_path),
        "UNSLOTH_STUDIO_HOME": str(tmp_path / "studio"),
    }
    if rocm:
        (tmp_path / "root" / "dev").mkdir(parents = True)
        (tmp_path / "root" / "dev" / "kfd").write_text("")
        _stub(str(bindir / "rocm-smi"), 'echo "GPU[0] : GPU ID: 0x1586"\n')
        info = tmp_path / "build-info"
        info.write_text("TORCH_INDEX_URL=x\nROCM_GFX=\nROCM_VERSION=7.2.4\n")
        topo = tmp_path / "topo" / "1"
        topo.mkdir(parents = True)
        (topo / "properties").write_text("vendor_id 4098\ngfx_target_version 110000\n")
        env.update(
            {
                "UNSLOTH_DEV_ROOT": str(tmp_path / "root"),
                "UNSLOTH_ROCM_BUILD_INFO": str(info),
                "UNSLOTH_KFD_TOPOLOGY": str(tmp_path / "topo"),
            }
        )
    else:
        _stub(
            str(bindir / "nvidia-smi"),
            'case "$*" in\n  *driver_version*) echo "580.10" ;;\n'
            '  *compute_cap*) echo "9.0" ;;\n  *) echo "GPU 0: NVIDIA H100 (UUID: GPU-abc)" ;;\nesac\n',
        )
    dump = tmp_path / "ran"
    proc = subprocess.run(
        [shutil.which("bash") or "/bin/bash", script, "bash", "-c", f"echo ran > {dump}"],
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    n = len(calls.read_text().split()) if calls.exists() else 0
    return proc.returncode, dump.exists(), proc.stderr, n


@_posix_shell
class TestCudaEntrypointOneProbe:
    def test_a_supported_gpu_is_reported_from_one_interpreter(self, tmp_path):
        rc, ran, err, n = _run(tmp_path, _CUDA, _cuda_torch(tmp_path, [(9, 0), (6, 1)]))
        assert rc == 0 and ran, err
        assert n == 1, f"{n} python processes"
        assert "Unsloth container: 2 GPU(s). Primary: GPU0  sm_90  bf16=True" in err, err
        assert "WARNING: GPU 1 (GPU1, sm_61) is below this image's sm_75 floor." in err, err

    def test_unavailable_cuda_stops_before_the_capability_report(self, tmp_path):
        rc, ran, err, _ = _run(tmp_path, _CUDA, _cuda_torch(tmp_path, [(9, 0)], available = False))
        assert rc == 1 and not ran
        assert "torch.cuda.is_available() is False despite nvidia-smi working." in err, err
        assert "Unsloth container:" not in err, err

    def test_the_arch_gate_and_the_turing_note_keep_their_exit_codes(self, tmp_path):
        rc, ran, err, _ = _run(tmp_path / "v", _CUDA, _cuda_torch(tmp_path / "v", [(7, 0)]))
        assert rc == 1 and not ran and "requires Turing or newer (sm_75+)" in err, err
        rc, ran, err, _ = _run(tmp_path / "t", _CUDA, _cuda_torch(tmp_path / "t", [(7, 5)]))
        assert rc == 0 and ran and "bfloat16 is not supported" in err, err

    def test_a_driver_error_after_the_availability_check_still_stops(self, tmp_path):
        fake = _cuda_torch(tmp_path, [(9, 0)], raise_in = "get_device_capability")
        rc, ran, err, _ = _run(tmp_path, _CUDA, fake)
        assert rc == 1 and not ran
        assert "RuntimeError: CUDA error: system not yet initialized" in err, err


@_posix_shell
class TestRocmEntrypointOneProbe:
    def test_checks_3_and_4_share_one_interpreter(self, tmp_path):
        rc, ran, err, n = _run(tmp_path, _ROCM, _rocm_torch(tmp_path), rocm = True)
        assert rc == 0 and ran, err
        assert n == 1, f"{n} python processes"
        assert "torch 2.11.0+rocm7.2  HIP=7.2.53211" in err and "arch=gfx1100" in err, err

    def test_check_3_output_stays_ahead_of_a_check_4_warning(self, tmp_path):
        """Check 3 used to end its own process, flushing its line before check 4 began."""
        rc, ran, err, _ = _run(tmp_path, _ROCM, _rocm_torch(tmp_path, warn = True), rocm = True)
        assert rc == 0 and ran, err
        assert err.index("HIP=7.2.53211") < err.index("late HIP warning"), err

    def test_unavailable_hip_stops_before_check_4(self, tmp_path):
        rc, ran, err, _ = _run(tmp_path, _ROCM, _rocm_torch(tmp_path, available = False), rocm = True)
        assert rc == 1 and not ran
        assert "torch.cuda.is_available() is False" in err and "GPU 0:" not in err, err

    def test_a_check_4_refusal_keeps_its_exit_code(self, tmp_path):
        rc, ran, err, _ = _run(tmp_path, _ROCM, _rocm_torch(tmp_path, arch = "gfx1033"), rocm = True)
        assert rc == 1 and not ran and "Van Gogh" in err, err

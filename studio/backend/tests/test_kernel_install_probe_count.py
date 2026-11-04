# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth install-kernels` spends no torch import on a check whose answer is already known."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import kernel_install  # noqa: E402

# Where each check's extension lives in a fake site: (relative path, source).
_PRESENT = {
    "xformers": [
        ("xformers/__init__.py", ""),
        ("xformers/_cpp_lib.py", "import torch\n\n\ndef _register_extensions():\n    pass\n"),
    ],
    "flash_attn": [("flash_attn_2_cuda.py", "")],
    "causal_conv1d": [("causal_conv1d_cuda.py", "")],
    "mamba_ssm": [("selective_scan_cuda.py", "")],
}


def _site(
    tmp_path,
    *,
    kernel = None,
    broken = False,
    torch_ok = True,
):
    """A PYTHONPATH with a fake torch that records its import, plus *kernel*'s extension."""
    site = tmp_path / "site"
    (site / "torch").mkdir(parents = True)
    marker = tmp_path / "torch_imported"
    (site / "torch" / "__init__.py").write_text(
        f"open({str(marker)!r}, 'a').write('x')\n"
        + ("" if torch_ok else "raise ImportError('no torch')\n")
    )
    files = _PRESENT.get(kernel, [])
    for i, (rel, source) in enumerate(files):
        path = site / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        # The extension itself (last file) fails to load, after it imported torch.
        last = i == len(files) - 1
        path.write_text(
            source + ("raise ImportError('undefined symbol')\n" if broken and last else "")
        )
    env = dict(os.environ, PYTHONPATH = str(site))
    return env, marker


def _run_check(name, env, tmp_path):
    return subprocess.run(
        # -S: only the fake site, so a real xformers / torch in this venv cannot answer.
        [sys.executable, "-S", "-c", kernel_install._load_check(name)],
        env = env,
        cwd = tmp_path,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    ).returncode


@pytest.mark.parametrize("name", list(kernel_install.KERNELS))
def test_an_absent_extension_fails_without_importing_torch(tmp_path, name):
    env, marker = _site(tmp_path)
    assert _run_check(name, env, tmp_path) != 0
    assert not marker.exists()


@pytest.mark.parametrize("name", list(kernel_install.KERNELS))
@pytest.mark.parametrize(
    "broken, torch_ok, loads",
    [(False, True, True), (True, True, False), (False, False, False)],
)
def test_a_present_extension_still_gets_the_full_check(tmp_path, name, broken, torch_ok, loads):
    env, marker = _site(tmp_path, kernel = name, broken = broken, torch_ok = torch_ok)
    assert (_run_check(name, env, tmp_path) == 0) is loads
    assert marker.exists()


def test_main_probes_torch_only_when_a_kernel_needs_a_wheel(monkeypatch):
    probes = []

    def probe(**kwargs):
        probes.append(kwargs)
        return None

    monkeypatch.setattr(kernel_install, "probe_torch_wheel_env", probe)
    monkeypatch.setattr(kernel_install, "_installed_version", lambda distribution: "1.0")
    monkeypatch.setattr(kernel_install, "_loads", lambda check, run: True)
    assert kernel_install.main([]) == 0
    assert probes == []

    # One kernel without a working copy: probed once, and later kernels reuse the answer.
    monkeypatch.setattr(kernel_install, "_installed_version", lambda distribution: None)
    monkeypatch.setattr(kernel_install, "package_installs_offline_env", lambda: False)
    assert kernel_install.main([]) == 0
    assert probes == [{"timeout": 120, "include_windows": True}]

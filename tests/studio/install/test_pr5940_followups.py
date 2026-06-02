# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the AMD-Windows installer follow-ups (PR #5940):

  * the huggingface_hub validation-model fetch + its urllib fallback,
  * run_capture's Windows-only amd-smi __COMPAT_LAYER=RunAsInvoker injection,
  * parity of the name->arch table between install.ps1 and setup.ps1.

Mock-only; no AMD hardware or network required.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# install_llama_prebuilt.py is self-contained (stdlib + optional filelock), so it
# loads without the studio backend on sys.path.
_PREBUILT_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
_SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_pr5940", _PREBUILT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
prebuilt = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = prebuilt
_SPEC.loader.exec_module(prebuilt)

_INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
_SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"


# ── _hf_resolve_url_parts ────────────────────────────────────────────────────

def test_hf_resolve_url_parts_valid():
    assert prebuilt._hf_resolve_url_parts(
        "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
    ) == ("ggml-org/models", "main", "tinyllamas/stories260K.gguf")


@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo/releases/download/x.gguf",  # not huggingface
    "https://huggingface.co/owner/repo",                       # no /resolve/<rev>/
    "https://huggingface.co/owner/repo/blob/main/x.gguf",      # /blob/ not /resolve/
    "not even a url",
])
def test_hf_resolve_url_parts_non_hf_returns_none(url):
    assert prebuilt._hf_resolve_url_parts(url) is None


# ── _fetch_validation_model_bytes ────────────────────────────────────────────

def test_fetch_validation_model_prefers_huggingface_hub(tmp_path):
    model = tmp_path / "stories260K.gguf"
    model.write_bytes(b"GGUF-via-hf")
    fake_hf = MagicMock(return_value=str(model))
    with patch.object(prebuilt, "validated_validation_model_bytes", side_effect=lambda b: b), \
         patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=fake_hf)}):
        assert prebuilt._fetch_validation_model_bytes() == b"GGUF-via-hf"
    assert fake_hf.called  # hf path was taken, urllib not needed


def test_fetch_validation_model_falls_back_to_urllib_on_hf_failure():
    fake_hf = MagicMock(side_effect=RuntimeError("hf unreachable"))
    with patch.object(prebuilt, "validated_validation_model_bytes", side_effect=lambda b: b), \
         patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=fake_hf)}), \
         patch.object(prebuilt, "download_bytes", return_value=b"GGUF-via-urllib") as dl:
        assert prebuilt._fetch_validation_model_bytes() == b"GGUF-via-urllib"
    assert dl.called  # fell back to the direct URL download


# ── run_capture amd-smi RunAsInvoker injection ───────────────────────────────

def _capture_env(command, system):
    captured = {"env": "sentinel"}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch.object(prebuilt.subprocess, "run", side_effect=fake_run), \
         patch.object(prebuilt.platform, "system", return_value=system):
        prebuilt.run_capture(command)
    return captured["env"]


def test_run_capture_injects_runasinvoker_for_amd_smi_on_windows():
    env = _capture_env(["amd-smi", "list"], "Windows")
    assert env is not None and env.get("__COMPAT_LAYER") == "RunAsInvoker"


def test_run_capture_injects_for_full_path_amd_smi_exe_on_windows():
    env = _capture_env(["amd-smi.exe", "version"], "Windows")
    assert env is not None and env.get("__COMPAT_LAYER") == "RunAsInvoker"


def test_run_capture_no_injection_for_non_amd_smi_on_windows():
    assert _capture_env(["rocminfo"], "Windows") is None


def test_run_capture_no_injection_on_linux():
    # amd-smi does not auto-elevate on Linux, so no env override is applied.
    assert _capture_env(["amd-smi", "list"], "Linux") is None


# ── name->arch table parity (install.ps1 vs setup.ps1) ───────────────────────

def _ps_name_arch_rows(text):
    return re.findall(r'@\{\s*P\s*=\s*"([^"]*)"\s*;\s*A\s*=\s*"(gfx[0-9a-z]+)"', text)


def test_ps_name_arch_tables_in_sync():
    t1 = _ps_name_arch_rows(_INSTALL_PS1.read_text(encoding="utf-8"))
    t2 = _ps_name_arch_rows(_SETUP_PS1.read_text(encoding="utf-8"))
    assert t1, "no nameArchTable found in install.ps1"
    assert t1 == t2, f"name->arch tables drifted:\ninstall.ps1={t1}\nsetup.ps1={t2}"


def test_rx_7700s_resolves_to_gfx1102_not_gfx1100():
    rows = _ps_name_arch_rows(_INSTALL_PS1.read_text(encoding="utf-8"))
    name = "AMD Radeon RX 7700S"
    matched = next((arch for pattern, arch in rows if re.search(pattern, name)), None)
    assert matched == "gfx1102", f"RX 7700S matched {matched!r}, expected gfx1102"


def test_radeon_8060s_resolves_to_gfx1151():
    rows = _ps_name_arch_rows(_INSTALL_PS1.read_text(encoding="utf-8"))
    name = "AMD Radeon(TM) 8060S Graphics"
    matched = next((arch for pattern, arch in rows if re.search(pattern, name)), None)
    assert matched == "gfx1151"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

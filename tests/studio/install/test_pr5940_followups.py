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
_SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt_pr5940", _PREBUILT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
prebuilt = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = prebuilt
_SPEC.loader.exec_module(prebuilt)

_INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
_SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
_INSTALL_SH = PACKAGE_ROOT / "install.sh"


# ── _hf_resolve_url_parts ────────────────────────────────────────────────────


def test_hf_resolve_url_parts_valid():
    assert prebuilt._hf_resolve_url_parts(
        "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
    ) == ("ggml-org/models", "main", "tinyllamas/stories260K.gguf")


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo/releases/download/x.gguf",  # not huggingface
        "https://huggingface.co/owner/repo",  # no /resolve/<rev>/
        "https://huggingface.co/owner/repo/blob/main/x.gguf",  # /blob/ not /resolve/
        "not even a url",
    ],
)
def test_hf_resolve_url_parts_non_hf_returns_none(url):
    assert prebuilt._hf_resolve_url_parts(url) is None


# ── _fetch_validation_model_bytes ────────────────────────────────────────────


def test_fetch_validation_model_prefers_huggingface_hub(tmp_path):
    model = tmp_path / "stories260K.gguf"
    model.write_bytes(b"GGUF-via-hf")
    fake_hf = MagicMock(return_value = str(model))
    with (
        patch.object(
            prebuilt, "validated_validation_model_bytes", side_effect = lambda b: b
        ),
        patch.dict(
            sys.modules, {"huggingface_hub": MagicMock(hf_hub_download = fake_hf)}
        ),
    ):
        assert prebuilt._fetch_validation_model_bytes() == b"GGUF-via-hf"
    assert fake_hf.called  # hf path was taken, urllib not needed


def test_fetch_validation_model_falls_back_to_urllib_on_hf_failure():
    fake_hf = MagicMock(side_effect = RuntimeError("hf unreachable"))
    with (
        patch.object(
            prebuilt, "validated_validation_model_bytes", side_effect = lambda b: b
        ),
        patch.dict(
            sys.modules, {"huggingface_hub": MagicMock(hf_hub_download = fake_hf)}
        ),
        patch.object(prebuilt, "download_bytes", return_value = b"GGUF-via-urllib") as dl,
    ):
        assert prebuilt._fetch_validation_model_bytes() == b"GGUF-via-urllib"
    assert dl.called  # fell back to the direct URL download


# ── run_capture amd-smi RunAsInvoker injection ───────────────────────────────


def _capture_env(command, system):
    captured = {"env": "sentinel"}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with (
        patch.object(prebuilt.subprocess, "run", side_effect = fake_run),
        patch.object(prebuilt.platform, "system", return_value = system),
    ):
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
    t1 = _ps_name_arch_rows(_INSTALL_PS1.read_text(encoding = "utf-8"))
    t2 = _ps_name_arch_rows(_SETUP_PS1.read_text(encoding = "utf-8"))
    assert t1, "no nameArchTable found in install.ps1"
    assert t1 == t2, f"name->arch tables drifted:\ninstall.ps1={t1}\nsetup.ps1={t2}"


def test_rx_7700s_resolves_to_gfx1102_not_gfx1100():
    rows = _ps_name_arch_rows(_INSTALL_PS1.read_text(encoding = "utf-8"))
    name = "AMD Radeon RX 7700S"
    matched = next((arch for pattern, arch in rows if re.search(pattern, name)), None)
    assert matched == "gfx1102", f"RX 7700S matched {matched!r}, expected gfx1102"


def test_radeon_8060s_resolves_to_gfx1151():
    rows = _ps_name_arch_rows(_INSTALL_PS1.read_text(encoding = "utf-8"))
    name = "AMD Radeon(TM) 8060S Graphics"
    matched = next((arch for pattern, arch in rows if re.search(pattern, name)), None)
    assert matched == "gfx1151"


def _sh_name_arch_rows(text):
    """Parse install.sh's `case "$_gpu_disp_mkt" in ... ) _gpu_disp_gfx="gfxNNNN"`
    name->arch table into [(substr_tokens, arch), ...] preserving order."""
    rows = []
    for line in text.splitlines():
        m = re.search(r'_gpu_disp_gfx="(gfx[0-9a-z]+)"', line)
        if not m or "*\"" not in line:
            continue
        tokens = re.findall(r'\*"([^"]+)"\*', line)
        if tokens:
            rows.append((tokens, m.group(1)))
    return rows


def _sh_resolve(rows, name):
    for tokens, arch in rows:
        if any(tok in name for tok in tokens):  # bash *"X"* == substring
            return arch
    return None


def test_install_sh_name_arch_agrees_with_ps_for_strix_and_non_amd():
    """The bash install.sh name->arch table must agree with the PowerShell
    source-of-truth for the Strix Halo (gfx1151) vs Strix Point (gfx1150)
    split, and must never misclassify NVIDIA/Intel as an AMD gfx."""
    sh_rows = _sh_name_arch_rows(_INSTALL_SH.read_text(encoding = "utf-8"))
    ps_rows = _ps_name_arch_rows(_INSTALL_PS1.read_text(encoding = "utf-8"))
    assert sh_rows, "no name->arch case table found in install.sh"
    cases = {
        "AMD Radeon(TM) 8060S Graphics": "gfx1151",  # Strix Halo
        "AMD Ryzen AI Max+ PRO 395 w/ Radeon 8060S": "gfx1151",
        "AMD Radeon 890M Graphics": "gfx1150",  # Strix Point (NOT gfx1151)
        "AMD Ryzen AI 9 HX 370 w/ Radeon 890M": "gfx1150",
        "AMD Radeon RX 7700S": "gfx1102",
        "NVIDIA GeForce RTX 4090": None,
        "Intel(R) Arc A770 Graphics": None,
    }
    for name, expect in cases.items():
        sh = _sh_resolve(sh_rows, name)
        assert sh == expect, f"install.sh: {name!r} -> {sh!r}, expected {expect!r}"
        if expect is not None:  # cross-check bash agrees with the PowerShell table
            ps = next((a for p, a in ps_rows if re.search(p, name)), None)
            assert sh == ps, f"install.sh/install.ps1 drift for {name!r}: {sh!r} vs {ps!r}"


# ── amd-smi gating (DiskPart UAC-prompt avoidance) ───────────────────────────
# On Windows amd-smi elevates a child at runtime on hosts without a working HIP
# runtime, popping a UAC/DiskPart prompt that __COMPAT_LAYER=RunAsInvoker cannot
# suppress (amd-smi's manifest is asInvoker). _amd_smi_allowed() therefore skips
# amd-smi by default on Windows-without-HIP-SDK; HIP-SDK hosts and an explicit
# opt-in keep it.


def _amd_smi_allowed_under(system, hipinfo_present, env):
    which = (
        (lambda name: r"C:\hip\bin\hipinfo.exe" if name == "hipinfo" else None)
        if hipinfo_present
        else (lambda name: None)
    )
    with (
        patch.object(prebuilt.platform, "system", return_value = system),
        patch.object(prebuilt.shutil, "which", side_effect = which),
        patch.dict(prebuilt.os.environ, env, clear = True),
    ):
        return prebuilt._amd_smi_allowed()


def test_amd_smi_allowed_on_linux_regardless():
    # Linux amd-smi does not elevate -> always allowed (no regression on Linux).
    assert _amd_smi_allowed_under("Linux", hipinfo_present = False, env = {}) is True


def test_amd_smi_skipped_on_windows_without_hip_sdk():
    # The DiskPart fix: no HIP SDK + no opt-in -> do not spawn amd-smi.
    assert _amd_smi_allowed_under("Windows", hipinfo_present = False, env = {}) is False


def test_amd_smi_allowed_on_windows_with_hip_sdk():
    # hipinfo present => amd-smi runs un-elevated, so it is allowed (no regression
    # for HIP-SDK Windows users, who never saw the prompt).
    assert _amd_smi_allowed_under("Windows", hipinfo_present = True, env = {}) is True


def test_amd_smi_opt_in_forces_on_windows_no_sdk():
    assert (
        _amd_smi_allowed_under(
            "Windows", hipinfo_present = False, env = {"UNSLOTH_ENABLE_AMD_SMI": "1"}
        )
        is True
    )


def test_amd_smi_opt_out_overrides_hip_sdk():
    assert (
        _amd_smi_allowed_under(
            "Windows", hipinfo_present = True, env = {"UNSLOTH_ENABLE_AMD_SMI": "0"}
        )
        is False
    )


def test_ps_installers_gate_amd_smi_on_windows():
    # Both PowerShell installers must gate the amd-smi probe behind HIP SDK
    # presence + the UNSLOTH_ENABLE_AMD_SMI opt-in, mirroring _amd_smi_allowed().
    for ps in (_INSTALL_PS1, _SETUP_PS1):
        text = ps.read_text(encoding = "utf-8")
        assert (
            "UNSLOTH_ENABLE_AMD_SMI" in text
        ), f"{ps.name} missing amd-smi opt-in gate"
        assert "amdSmiAllowed" in text, f"{ps.name} missing amd-smi gate variable"


def test_install_python_stack_gates_every_amd_smi_spawn():
    # Regression for the recurring DiskPart UAC prompt: EVERY function in
    # install_python_stack.py that both (a) names the `amd-smi` command and
    # (b) spawns a subprocess must also gate it behind _amd_smi_allowed().
    # The Windows "AMD GPU detected but ROCm torch missing" warning probe
    # spawned `amd-smi list` on Adrenalin-only hosts WITHOUT this gate, popping
    # the UAC/DiskPart prompt the rest of the PR avoids (RunAsInvoker cannot
    # suppress it, so not-spawning is the only fix).
    import ast

    src = (PACKAGE_ROOT / "studio" / "install_python_stack.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)

    def _names_amd_smi_command(node):
        # An EXACT "amd-smi"/"amd-smi.exe" string constant (command list or
        # shutil.which arg) -- not a substring inside a longer log message.
        return any(
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.lower() in ("amd-smi", "amd-smi.exe")
            for n in ast.walk(node)
        )

    def _spawns_subprocess(node):
        for n in ast.walk(node):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "subprocess"
            ):
                return True
        return False

    def _references_gate(node):
        return any(
            isinstance(n, ast.Name) and n.id == "_amd_smi_allowed"
            for n in ast.walk(node)
        )

    offenders = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _names_amd_smi_command(node)
        and _spawns_subprocess(node)
        and not _references_gate(node)
    ]
    assert not offenders, (
        "install_python_stack.py spawns amd-smi without an _amd_smi_allowed() "
        f"gate in: {offenders} -- this pops the Windows UAC/DiskPart prompt on "
        "Adrenalin-only (no HIP SDK) hosts."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

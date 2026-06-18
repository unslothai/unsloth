# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""AMD-Windows installer follow-ups (PR #5940): hf-hub validation-model fetch
+ urllib fallback, amd-smi RunAsInvoker injection, name->arch table parity.
Mock-only; no AMD hardware or network required."""

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# install_llama_prebuilt.py is self-contained, so it loads without the studio backend.
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
_AMD_PY = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
_PYSTACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"


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
        patch.object(prebuilt, "validated_validation_model_bytes", side_effect = lambda b: b),
        patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download = fake_hf)}),
    ):
        assert prebuilt._fetch_validation_model_bytes() == b"GGUF-via-hf"
    assert fake_hf.called  # hf path was taken, urllib not needed


def test_fetch_validation_model_falls_back_to_urllib_on_hf_failure():
    fake_hf = MagicMock(side_effect = RuntimeError("hf unreachable"))
    with (
        patch.object(prebuilt, "validated_validation_model_bytes", side_effect = lambda b: b),
        patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download = fake_hf)}),
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


def _sh_name_arch_rows(text, var = "_gpu_disp_gfx"):
    """Parse a bash name->arch case table into ordered [(substr_tokens, arch), ...]."""
    rows = []
    for line in text.splitlines():
        m = re.search(var + r'="(gfx[0-9a-z]+)"', line)
        if not m or '*"' not in line:
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
    """install.sh name->arch table must match the PowerShell source on the Strix
    Halo/Point split and never misclassify NVIDIA/Intel as an AMD gfx."""
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


def test_setup_sh_name_arch_table_in_sync_with_install_sh():
    """studio/setup.sh's name->arch table must stay row-for-row identical to
    install.sh's (order carries the RX 7700S -> gfx1102 rule)."""
    install_rows = _sh_name_arch_rows(_INSTALL_SH.read_text(encoding = "utf-8"))
    setup_rows = _sh_name_arch_rows(
        (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8"),
        var = "_setup_gfx",
    )
    assert setup_rows, "no name->arch case table found in studio/setup.sh"
    assert install_rows == setup_rows, (
        "bash name->arch tables drifted:\n"
        f"install.sh={install_rows}\nstudio/setup.sh={setup_rows}"
    )
    # Guards historical drift: Strix Point -> gfx1150, RX 7700S -> gfx1102 (before gfx1100).
    for name, expect in {
        "AMD Radeon 890M Graphics": "gfx1150",
        "AMD Ryzen AI 9 HX 370 w/ Radeon 890M": "gfx1150",
        "AMD Radeon(TM) 8060S Graphics": "gfx1151",
        "AMD Radeon RX 7700S": "gfx1102",
    }.items():
        got = _sh_resolve(setup_rows, name)
        assert got == expect, f"setup.sh: {name!r} -> {got!r}, expected {expect!r}"


# ── amd-smi gating (DiskPart UAC-prompt avoidance) ───────────────────────────
# On Windows w/o a HIP SDK, amd-smi pops a UAC/DiskPart prompt RunAsInvoker
# can't suppress, so _amd_smi_allowed() skips it unless HIP-SDK or opt-in.


def _amd_smi_allowed_under(system, hipinfo_present, env):
    # Build a real (temp) PATH so _external_hipinfo_on_path scans it like prod;
    # an external hipinfo.exe outside the pinned venv models a real HIP SDK.
    with tempfile.TemporaryDirectory() as tmp:
        venv_root = os.path.join(tmp, "venv")
        os.makedirs(venv_root)
        path_env = dict(env)
        if hipinfo_present:
            sdk_bin = os.path.join(tmp, "hipsdk", "bin")
            os.makedirs(sdk_bin)
            open(os.path.join(sdk_bin, "hipinfo.exe"), "w").close()
            path_env["PATH"] = sdk_bin + os.pathsep + path_env.get("PATH", "")
        with (
            patch.object(prebuilt.platform, "system", return_value = system),
            patch.object(prebuilt.sys, "prefix", venv_root),
            patch.dict(prebuilt.os.environ, path_env, clear = True),
        ):
            return prebuilt._amd_smi_allowed()


def test_amd_smi_allowed_on_linux_regardless():
    # Linux amd-smi does not elevate -> always allowed.
    assert _amd_smi_allowed_under("Linux", hipinfo_present = False, env = {}) is True


def test_amd_smi_skipped_on_windows_without_hip_sdk():
    # The DiskPart fix: no HIP SDK + no opt-in -> do not spawn amd-smi.
    assert _amd_smi_allowed_under("Windows", hipinfo_present = False, env = {}) is False


def test_amd_smi_allowed_on_windows_with_hip_sdk():
    # hipinfo present => amd-smi runs un-elevated, so it is allowed.
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
        _amd_smi_allowed_under("Windows", hipinfo_present = True, env = {"UNSLOTH_ENABLE_AMD_SMI": "0"})
        is False
    )


def test_amd_smi_skipped_when_hipinfo_is_venv_internal(tmp_path):
    # The AMD torch wheel ships hipInfo.exe inside the venv (Scripts/), and the
    # bnb fix prepends that dir to PATH. That venv-internal hipInfo is NOT a HIP
    # SDK and must NOT re-open the amd-smi gate -- otherwise amd-smi pops the
    # DiskPart UAC mid-install on Strix Halo boxes with no real HIP SDK (the bug
    # snapcast3r/UBER6 hit: arch came from the GPU-name table, hipInfo could not
    # run, so the probe fell through to `amd-smi list`).
    venv_root = tmp_path / "venv"
    venv_scripts = venv_root / "Scripts"
    venv_scripts.mkdir(parents = True)
    (venv_scripts / "hipinfo.exe").write_text("")
    with (
        patch.object(prebuilt.platform, "system", return_value = "Windows"),
        patch.object(prebuilt.sys, "prefix", str(venv_root)),
        patch.dict(prebuilt.os.environ, {"PATH": str(venv_scripts)}, clear = True),
    ):
        assert prebuilt._amd_smi_allowed() is False


def test_amd_smi_allowed_when_hipinfo_outside_venv(tmp_path):
    # A hipinfo from a real HIP SDK (outside the venv) still opens the gate, so
    # HIP-SDK Windows users keep amd-smi (no regression for the venv-exclusion).
    sdk_bin = tmp_path / "hipsdk" / "bin"
    sdk_bin.mkdir(parents = True)
    (sdk_bin / "hipinfo.exe").write_text("")
    with (
        patch.object(prebuilt.platform, "system", return_value = "Windows"),
        patch.object(prebuilt.sys, "prefix", str(tmp_path / "venv")),
        patch.dict(prebuilt.os.environ, {"PATH": str(sdk_bin)}, clear = True),
    ):
        assert prebuilt._amd_smi_allowed() is True


def test_amd_smi_allowed_when_external_hipinfo_shadowed_by_venv(tmp_path):
    # codex P2: the venv-internal hipInfo.exe is first on PATH (the bnb fix
    # prepends the venv Scripts dir), but a real HIP SDK's hipinfo sits later on
    # PATH with HIP_PATH/ROCM_PATH unset. A first-hit shutil.which/Get-Command
    # would stop at the venv copy and wrongly close the gate; scanning every PATH
    # entry must still find the external SDK and keep amd-smi available.
    venv_root = tmp_path / "venv"
    venv_scripts = venv_root / "Scripts"
    venv_scripts.mkdir(parents = True)
    (venv_scripts / "hipinfo.exe").write_text("")
    sdk_bin = tmp_path / "hipsdk" / "bin"
    sdk_bin.mkdir(parents = True)
    (sdk_bin / "hipinfo.exe").write_text("")
    path = str(venv_scripts) + os.pathsep + str(sdk_bin)  # venv copy first
    with (
        patch.object(prebuilt.platform, "system", return_value = "Windows"),
        patch.object(prebuilt.sys, "prefix", str(venv_root)),
        patch.dict(prebuilt.os.environ, {"PATH": path}, clear = True),
    ):
        assert prebuilt._amd_smi_allowed() is True


def test_amd_smi_skipped_when_env_root_hipinfo_is_venv_internal(tmp_path):
    # The venv exclusion must also cover the HIP_PATH/ROCM_PATH fallback: a hipinfo
    # under <venv>/_rocm_sdk_core/bin is still not a real HIP SDK.
    venv_root = tmp_path / "venv"
    hip_root = venv_root / "_rocm_sdk_core"
    (hip_root / "bin").mkdir(parents = True)
    (hip_root / "bin" / "hipinfo.exe").write_text("")
    with (
        patch.object(prebuilt.platform, "system", return_value = "Windows"),
        patch.object(prebuilt.sys, "prefix", str(venv_root)),
        patch.dict(prebuilt.os.environ, {"ROCM_PATH": str(hip_root)}, clear = True),
    ):
        assert prebuilt._amd_smi_allowed() is False


def test_amd_smi_allowed_when_env_root_hipinfo_outside_venv(tmp_path):
    # A real HIP SDK pointed to by HIP_PATH (outside the venv) still opens the
    # gate, so the env-root venv filter does not regress HIP-SDK Windows users.
    hip_root = tmp_path / "hipsdk"
    (hip_root / "bin").mkdir(parents = True)
    (hip_root / "bin" / "hipinfo.exe").write_text("")
    with (
        patch.object(prebuilt.platform, "system", return_value = "Windows"),
        patch.object(prebuilt.sys, "prefix", str(tmp_path / "venv")),
        patch.dict(prebuilt.os.environ, {"HIP_PATH": str(hip_root)}, clear = True),
    ):
        assert prebuilt._amd_smi_allowed() is True


def test_external_hipinfo_on_path_skips_venv_only(tmp_path):
    # The helper itself: a PATH holding only the venv-internal hipInfo must not
    # count as an external HIP SDK (the whole point of the venv filter).
    venv_root = tmp_path / "venv"
    venv_scripts = venv_root / "Scripts"
    venv_scripts.mkdir(parents = True)
    (venv_scripts / "hipinfo.exe").write_text("")
    with (
        patch.object(prebuilt.sys, "prefix", str(venv_root)),
        patch.dict(prebuilt.os.environ, {"PATH": str(venv_scripts)}, clear = True),
    ):
        assert prebuilt._external_hipinfo_on_path() is False


def test_python_hipinfo_gates_scan_all_path_entries():
    # All three copies of the amd-smi HIP-SDK probe must scan every PATH entry
    # (not just the first shutil.which hit) so the venv-internal hipInfo cannot
    # shadow a real SDK's hipinfo later on PATH.
    for src in (_PREBUILT_PATH, _AMD_PY, _PYSTACK_PY):
        text = src.read_text(encoding = "utf-8")
        assert (
            "_external_hipinfo_on_path" in text
        ), f"{src.name} must use the PATH-scanning hipinfo helper"


def test_path_inside_venv_resolves_symlinks(tmp_path):
    # realpath (not abspath): a venv reached through a symlink/junction must
    # still count as inside, else its hipInfo escapes the filter and amd-smi
    # pops the DiskPart UAC. abspath would leave the alias unresolved here.
    real = tmp_path / "real"
    (real / "Scripts").mkdir(parents = True)
    (real / "Scripts" / "hipInfo.exe").write_text("")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this runner")
    with patch.object(prebuilt.sys, "prefix", str(real)):
        assert prebuilt._path_inside_venv(str(link / "Scripts" / "hipInfo.exe")) is True


def test_ps_installers_gate_amd_smi_on_windows():
    # Both PowerShell installers must gate amd-smi like _amd_smi_allowed().
    for ps in (_INSTALL_PS1, _SETUP_PS1):
        text = ps.read_text(encoding = "utf-8")
        assert "UNSLOTH_ENABLE_AMD_SMI" in text, f"{ps.name} missing amd-smi opt-in gate"
        assert "amdSmiAllowed" in text, f"{ps.name} missing amd-smi gate variable"
        # The HIP-SDK probe must exclude the venv-internal hipInfo.exe (mirrors
        # _path_inside_venv()), else amd-smi can still pop the DiskPart UAC.
        assert (
            "Test-HipinfoIsVenvInternal" in text
        ), f"{ps.name} missing venv-internal hipinfo exclusion helper"
        # Get-Command returns only the first hipinfo; scan -All and skip the venv
        # copy so a real SDK hipinfo later on PATH is not shadowed (codex P2).
        assert (
            "Get-Command hipinfo -All" in text
        ), f"{ps.name} must enumerate all hipinfo on PATH (-All), not just the first"
        assert (
            "Test-HipinfoIsVenvInternal $_.Source" in text
        ), f"{ps.name} must run the venv exclusion while scanning hipinfo candidates"
        # The HIP_PATH/ROCM_PATH candidate must also be venv-filtered, else an env
        # var pointing into the venv reopens the gate.
        assert (
            "Test-HipinfoIsVenvInternal $hipinfoCandidate" in text
        ), f"{ps.name} must run the venv exclusion on the HIP_PATH/ROCM_PATH candidate"
        # VenvDir/VIRTUAL_ENV can be unset at probe time (the update flow), so the
        # venv root must also be derived from the setup python.
        assert (
            "UNSLOTH_SETUP_PYTHON" in text
        ), f"{ps.name} venv-internal check must seed the venv root from UNSLOTH_SETUP_PYTHON"
        # A custom Studio home moves the venv off the default path; it must be
        # seeded too or its hipInfo escapes the filter and reopens the gate.
        assert (
            "UNSLOTH_STUDIO_HOME" in text
        ), f"{ps.name} venv-internal check must seed the venv root from UNSLOTH_STUDIO_HOME"


@pytest.mark.parametrize("ps", [_INSTALL_PS1, _SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
def test_ps_venv_probe_expands_tilde_for_custom_studio_home(ps):
    # Both installers' venv-internal hipInfo probe seed the venv root from a
    # custom Studio home; a tilde form (~\studio) must be expanded to USERPROFILE
    # like the canonical resolver, else [IO.Path]::GetFullPath keeps the literal
    # ~ (relative to cwd) and the custom-home hipInfo escapes the filter.
    text = ps.read_text(encoding = "utf-8")
    i = text.find("$studioHomeEnv = ")
    j = text.find('Join-Path $studioHomeEnv "unsloth_studio"', i)
    assert i != -1 and j != -1, f"{ps.name}: studioHomeEnv venv-root seed not found"
    block = text[i:j]
    assert "USERPROFILE" in block and ".Substring(1)" in block, (
        f"{ps.name}: the venv-internal probe must expand a leading ~ in the custom "
        "Studio home before seeding the venv root (mirroring the canonical resolver)"
    )


def _ps_floor_map(text, prefix):
    # {gfx -> spec} for entries like "gfx1151" = "torchvision>=0.26.0,<0.27.0".
    return dict(re.findall(r'"(gfx[0-9a-z]+)"\s*=\s*"(' + re.escape(prefix) + r'[^"]*)"', text))


def test_install_setup_ps_rocm_torch_floors_in_sync():
    # install.ps1 and setup.ps1 pull from AMD's per-arch index; their torch and
    # companion (torchvision/torchaudio) floor maps must match so both resolve
    # the same ABI-consistent trio. install.ps1 once left the companions bare,
    # which can resolve an incompatible set and fall back to CPU.
    it = _INSTALL_PS1.read_text(encoding = "utf-8")
    st = _SETUP_PS1.read_text(encoding = "utf-8")
    for prefix in ("torch>=", "torchvision>=", "torchaudio>="):
        i_map = _ps_floor_map(it, prefix)
        s_map = _ps_floor_map(st, prefix)
        assert i_map, f"install.ps1 has no {prefix!r} floor map"
        assert (
            i_map == s_map
        ), f"{prefix!r} floor map drift:\ninstall.ps1={i_map}\nsetup.ps1={s_map}"
    # Strix Halo (the field case) must be pinned, not bare.
    assert _ps_floor_map(it, "torchvision>=").get("gfx1151") == "torchvision>=0.26.0,<0.27.0"
    # The ROCm install must pass the pinned companion specs, not bare names.
    assert (
        "$torchSpec $visionSpec $audioSpec" in it
    ), "install.ps1 ROCm install must use the pinned companion specs"


def test_install_ps1_rocm_cpu_fallback_uses_retry():
    # The ROCm->CPU fallback is the recovery path most likely to hit a transient
    # index issue, yet it once used the non-retrying helper. It must retry like
    # every other torch install in the file.
    text = _INSTALL_PS1.read_text(encoding = "utf-8")
    i = text.find("ROCm PyTorch install failed")
    assert i != -1, "ROCm->CPU fallback block not found in install.ps1"
    window = text[i : i + 600]
    assert (
        "Invoke-InstallCommandRetry" in window
    ), "the ROCm->CPU fallback torch install must use Invoke-InstallCommandRetry"


def test_install_python_stack_gates_every_amd_smi_spawn():
    # Regression for the DiskPart UAC prompt: every function naming `amd-smi`
    # AND spawning a subprocess must gate it behind _amd_smi_allowed().
    import ast

    src = (PACKAGE_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)

    def _names_amd_smi_command(node):
        # Exact "amd-smi"/"amd-smi.exe" constant, not a substring in a log.
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
        return any(isinstance(n, ast.Name) and n.id == "_amd_smi_allowed" for n in ast.walk(node))

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


def test_install_ps1_installs_rocm_torch_for_known_arch():
    # A known AMD arch (even name-inferred, $HasROCm false) must select the ROCm
    # torch index directly instead of laying down a CPU base that setup.ps1 then
    # force-reinstalls as ROCm (wasted download). The bundled-runtime
    # repo.amd.com wheels need no HIP SDK, so the gate is $ROCmGfxArch-based.
    text = _INSTALL_PS1.read_text(encoding = "utf-8")
    gates = [
        ln
        for ln in text.splitlines()
        if '$TorchIndexUrl -like "*/cpu"' in ln and ln.lstrip().startswith("if ")
    ]
    assert gates, "ROCm-index selection gate not found in install.ps1"
    assert any("$ROCmGfxArch" in ln for ln in gates), (
        "install.ps1 gates ROCm torch only on probe-verified ROCm ($HasROCm); a "
        "known/name-inferred arch should install ROCm directly to avoid a wasted "
        "CPU PyTorch base that setup.ps1 immediately force-reinstalls."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

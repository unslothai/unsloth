# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsupported-AMD advice must not offer HIP SDK or UNSLOTH_ROCM_GFX_ARCH for cards ROCm lacks."""

import ast
import contextlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_INSTALL_SH = PACKAGE_ROOT / "install.sh"
_INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
_SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
_SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
_STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"

# PowerShell resolves a bare VAR=value as a command, so a .ps1 must never print it (#8458).
_POSIX_ASSIGNMENT = "UNSLOTH_LLAMA_CPP_BACKEND=vulkan"
# `export`, or the variable is invisible to the installer the user runs next.
_POSIX_SETTER = f"export {_POSIX_ASSIGNMENT}"
_PWSH_SETTER = '$env:UNSLOTH_LLAMA_CPP_BACKEND = "vulkan"'
_SETTER = {
    "install.sh": _POSIX_SETTER,
    "setup.sh": _POSIX_SETTER,
    "install.ps1": _PWSH_SETTER,
    "setup.ps1": _PWSH_SETTER,
    "install_python_stack.py": _PWSH_SETTER,  # _detect_windows_gfx_arch is Windows-only
}


def _load_stack_module():
    spec = importlib.util.spec_from_file_location("studio_install_python_stack_rdna1", _STACK_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()


# WMI reports the marketing name; lspci reports the chip plus slash-joined board names.
_RDNA1_NAMES = [
    ("AMD Radeon RX 5700 XT", "gfx1010"),
    ("AMD Radeon RX 5700", "gfx1010"),
    ("AMD Radeon RX 5600 XT", "gfx1010"),
    ("AMD Radeon Pro 5600 XT", "gfx1010"),
    ("Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT]", "gfx1010"),
    ("AMD Radeon Pro V520", "gfx1011"),
    ("AMD Radeon Pro 5600M", "gfx1011"),
    ("AMD Radeon RX 5500 XT", "gfx1012"),
    ("Navi 14 [Radeon RX 5500/5500M / Pro 5500M]", "gfx1012"),
    # Pro boards LLVM's table omits; die confirmed from libdrm amdgpu.ids and pci.ids.
    ("AMD Radeon Pro W5700", "gfx1010"),
    ("Navi 10 [Radeon Pro W5700X]", "gfx1010"),
    ("AMD Radeon Pro W5500", "gfx1012"),
    ("AMD Radeon Pro W5500M", "gfx1012"),
    ("Navi 14 [Radeon Pro W5300M]", "gfx1012"),
    ("AMD Radeon RX 5300", "gfx1012"),
    ("AMD Radeon RX 5300M", "gfx1012"),
    # Mac Pro MPX boards (pci.ids 7319/731b): Navi 10 names lacking 'RX 5700' and a W prefix.
    ("Navi 10 [Radeon Pro 5700 XT]", "gfx1010"),
    ("Navi 10 [Radeon Pro 5700]", "gfx1010"),
    ("AMD Radeon Pro 5700 XT", "gfx1010"),
]

# Cards the supported table owns, plus a non-AMD one.
_NOT_RDNA1_NAMES = [
    "AMD Radeon RX 9070 XT",
    "AMD Radeon RX 9060 XT",
    "AMD Radeon RX 7900 XTX",
    "AMD Radeon RX 6800 XT",
    "AMD Radeon 8060S Graphics",
    "NVIDIA GeForce RTX 4090",
    # 'W5700' must not be read out of 'W7500', nor 'W5500' out of 'W6500'.
    "AMD Radeon PRO W7500",
    "AMD Radeon PRO W7900",
    "AMD Radeon PRO W6500",
    "AMD Radeon PRO W6400",
]


class TestUnsupportedNameLookup:
    @pytest.mark.parametrize("name,expected", _RDNA1_NAMES)
    def test_rdna1_names_resolve_in_the_supported_table_on_windows(self, name, expected):
        """The behavioural half, inverted since #11614: RDNA 1 routes on Windows, so the
        Windows name table owns it and the messaging table must not claim it."""
        assert stack_mod._gfx_arch_from_gpu_name(name) == expected
        assert stack_mod._unsupported_gfx_arch_from_gpu_name(name) is None

    @pytest.mark.parametrize("name", _NOT_RDNA1_NAMES)
    def test_supported_and_non_amd_names_are_not_claimed(self, name):
        assert stack_mod._unsupported_gfx_arch_from_gpu_name(name) is None

    def test_empty_name_is_not_claimed(self):
        assert stack_mod._unsupported_gfx_arch_from_gpu_name("") is None

    def test_no_unsupported_arch_can_reach_a_wheel_index(self):
        """The scope guard. An arch in this table with an index-family entry would
        turn a messaging row into an installation change."""
        families = stack_mod._GFX_TO_AMD_INDEX_ARCH
        for _pat, arch in stack_mod._UNSUPPORTED_GPU_NAME_ARCH_TABLE:
            assert arch not in families, f"{arch} is routable; it must not be in this table"

    def test_the_two_tables_share_no_arch(self):
        supported = {arch for _p, arch in stack_mod._WIN_GPU_NAME_ARCH_TABLE}
        unsupported = {arch for _p, arch in stack_mod._UNSUPPORTED_GPU_NAME_ARCH_TABLE}
        assert not (supported & unsupported)


def _wmi_detect(names, arm64 = False):
    """Drives _detect_windows_gfx_arch with names and no hipinfo/amd-smi; arm64 pins the host arch."""
    ps_result = MagicMock()
    ps_result.returncode = 0
    amd = [n for n in names if re.search(r"AMD|Radeon", n, re.IGNORECASE)]
    ps_result.stdout = ("\r\n".join(amd) + "\r\n").encode()

    def _run(cmd, **kwargs):
        if cmd and "powershell.exe" in str(cmd[0]).lower():
            return ps_result
        raise FileNotFoundError(cmd[0])

    buf = io.StringIO()
    with patch.dict(os.environ, {}, clear = False):
        for _v in (
            "HIP_PATH",
            "ROCM_PATH",
            "UNSLOTH_ROCM_GFX_ARCH",
            "UNSLOTH_ENABLE_AMD_SMI",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            os.environ.pop(_v, None)
        with contextlib.redirect_stdout(buf):
            with patch("shutil.which", return_value = None):
                with patch("os.path.isfile", return_value = False):
                    with patch("subprocess.run", side_effect = _run):
                        with patch.object(stack_mod, "_is_windows_arm64", return_value = arm64):
                            result = stack_mod._detect_windows_gfx_arch()
    return result, buf.getvalue()


class TestExplicitIndexPinIsHonoured:
    """A pinned UNSLOTH_TORCH_INDEX_URL or _FAMILY reaches the ROCm path, so CPU-only advice is
    wrong then."""

    _CPU_CLAIM = "torch will be CPU-only"

    def test_the_python_warning_drops_the_cpu_claim_when_pinned(self):
        with patch.dict(os.environ, {"UNSLOTH_TORCH_INDEX_URL": "https://example/gfx1010"}):
            _arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert "gfx803" in out, "the card is still named"
        assert self._CPU_CLAIM not in out, f"a pinned index still gets the CPU-only verdict:\n{out}"

    @pytest.mark.parametrize(
        "env",
        [
            {"UNSLOTH_TORCH_INDEX_URL": "   "},
            {"UNSLOTH_TORCH_INDEX_FAMILY": "\t\n "},
            {"UNSLOTH_TORCH_INDEX_URL": "", "UNSLOTH_TORCH_INDEX_FAMILY": " "},
        ],
        ids = ["url-spaces", "family-blank", "both-blank"],
    )
    def test_a_blank_pin_is_not_a_pin(self, env):
        """get_torch_index_url trims both variables and treats a blank one as unset, so a
        blank value must not suppress the CPU-only verdict here either. Dropping the
        .strip() from the read passes every other test in this file."""
        with patch.dict(os.environ, env, clear = False):
            for _k in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
                if _k not in env:
                    os.environ.pop(_k, None)
            _arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert (
            self._CPU_CLAIM in out
        ), f"a blank pin ({env}) was read as a pin, dropping the CPU-only warning:\n{out}"

    def test_the_claim_is_there_without_a_pin(self):
        """The positive control: without it the test above passes on any wording."""
        with patch.dict(os.environ, {}, clear = False):
            for _v in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
                os.environ.pop(_v, None)
            _arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert self._CPU_CLAIM in out

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1, _SETUP_SH], ids = lambda p: p.name)
    def test_each_shell_arm_reads_the_pin(self, path):
        """The shell sources cannot be called from here, so require the arm to READ the
        pin: the wording is only correct because it is conditional on it."""
        lines = _normalised(path).splitlines()
        hits = [
            i
            for i, line in enumerate(lines)
            if "so torch stays" in line and not line.lstrip().startswith("#")
        ] or [
            i
            for i, line in enumerate(lines)
            if "torch stays CPU-only" in line and not line.lstrip().startswith("#")
        ]
        assert hits, f"{path.name}: the CPU-only claim was not found"
        for i in hits:
            # Bounded so an unrelated mention further up cannot satisfy it.
            window = "\n".join(lines[max(i - 10, 0) : i])
            assert "UNSLOTH_TORCH_INDEX_URL" in window, (
                f"{path.name}:{i + 1}: claims CPU-only unconditionally, which a pinned "
                f"index makes false:\n{window}"
            )


class TestWindowsArm64GetsNoVulkanAdvice:
    """studio/setup.ps1 THROWS when the Vulkan variable is set on Windows ARM64 (no
    bundle is published there), so telling an ARM64 user to set it and re-run aborts
    the update instead of enabling GGUF acceleration."""

    def test_the_throw_this_depends_on_is_still_there(self):
        src = _normalised(_SETUP_PS1)
        assert (
            "no Windows ARM64 Vulkan bundle is published" in src
        ), "the ARM64 guard this test is built on was renamed; re-read setup.ps1"

    # setup.ps1 runs the Python stack before its own throw, so its setter is the last advice seen.
    @pytest.mark.parametrize(
        "path,guard",
        [
            (_INSTALL_PS1, "Get-HostMachineArch"),
            (_SETUP_PS1, "Get-HostMachineArch"),
            (_STACK_PY, "_is_windows_arm64"),
        ],
        ids = lambda p: getattr(p, "name", p),
    )
    def test_every_vulkan_offer_is_behind_an_arch_check(self, path, guard):
        lines = _normalised(path).splitlines()
        offers = [
            i
            for i, line in enumerate(lines)
            if _SETTER[path.name] in line and not line.lstrip().startswith("#")
        ]
        assert offers, f"{path.name}: no Vulkan offer found"
        for i in offers:
            # Check the resolver itself: a hardcoded $unsupArm64 = $false mutant survived a name check.
            back = "\n".join(lines[max(i - 20, 0) : i])
            assert guard in back, (
                f"{path.name}:{i + 1}: offers the Vulkan variable without checking for "
                f"Windows ARM64, where setting it throws:\n{back}"
            )


class TestPythonStackWindowsArm64:
    """The Windows WMI path prints the same Vulkan advice as install.ps1, and the same
    ARM64 throw applies to it: setup.ps1 rejects the variable there."""

    def test_arm64_gets_a_source_build_note_instead_of_the_setter(self):
        _arch, out = _wmi_detect(["AMD Radeon RX 580"], arm64 = True)
        assert "gfx803" in out, "the card is still named"
        assert (
            "UNSLOTH_LLAMA_CPP_BACKEND" not in out
        ), f"ARM64 is still told to set the variable setup.ps1 throws on:\n{out}"
        assert "ARM64" in out and "source" in out

    def test_x64_still_gets_the_setter(self):
        """Positive control: the guard must not silence the advice everywhere."""
        _arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert "UNSLOTH_LLAMA_CPP_BACKEND" in out

    @pytest.mark.parametrize(
        "registry,env,expected",
        [
            ("ARM64", {"PROCESSOR_ARCHITECTURE": "ARM64"}, True),
            # x64 emulation on ARM64: only the machine-scope registry value tells the truth.
            ("ARM64", {"PROCESSOR_ARCHITECTURE": "AMD64"}, True),
            # ARCHITEW6432 still counts on the builds that do set it.
            ("", {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": "ARM64"}, True),
            ("AMD64", {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": ""}, False),
            # Unreadable registry falls back to the per-process signals, not to True.
            ("", {"PROCESSOR_ARCHITECTURE": "AMD64"}, False),
        ],
        ids = [
            "native-arm64",
            "emulated-x64-on-arm64",
            "architew6432-set",
            "real-x64",
            "no-registry-x64",
        ],
    )
    def test_the_arch_probe_reads_the_machine_not_the_process(self, registry, env, expected):
        """PROCESSOR_ARCHITECTURE reflects the process, so the arch must come from the machine registry."""
        with patch.object(stack_mod, "IS_WINDOWS", True):
            with patch.object(stack_mod, "_machine_arch_from_registry", return_value = registry):
                with patch.object(stack_mod.platform, "machine", return_value = "AMD64"):
                    with patch.dict(os.environ, env, clear = False):
                        for _k in ("PROCESSOR_ARCHITEW6432", "PROCESSOR_ARCHITECTURE"):
                            if _k not in env:
                                os.environ.pop(_k, None)
                        assert stack_mod._is_windows_arm64() is expected

    def test_it_is_false_off_windows(self):
        with patch.object(stack_mod, "IS_WINDOWS", False):
            with patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "ARM64"}):
                assert stack_mod._is_windows_arm64() is False


class TestWindowsWmiMessage:
    def test_rdna1_adapter_now_yields_its_arch(self):
        """Since #11614 the reporter's card routes: the WMI path infers gfx1010 and the
        multi-arch index takes it from there. No "not covered" message for it."""
        arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert arch == "gfx1010"
        assert "does not cover" not in out

    def test_polaris_adapter_still_yields_no_arch(self):
        """CPU fallback unchanged where nothing routes. This is the assertion that keeps
        the wording fix honest."""
        arch, _out = _wmi_detect(["AMD Radeon RX 580"])
        assert arch is None

    def test_polaris_adapter_is_named_with_its_arch(self):
        _arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert "gfx803" in out
        assert "AMD Radeon RX 580" in out

    def test_polaris_adapter_is_not_told_to_set_the_override(self):
        """The defect proper. Setting UNSLOTH_ROCM_GFX_ARCH=gfx803 lands on the
        unmapped-arch path and returns CPU anyway, so instructing it is an errand
        with no ending. The variable may still be NAMED, to say it cannot help."""
        _arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert "Set UNSLOTH_ROCM_GFX_ARCH to your GPU's arch" not in out
        assert "gfx1200" not in out
        assert "no UNSLOTH_ROCM_GFX_ARCH value changes that" in out

    def test_an_actually_unknown_adapter_keeps_the_override_advice(self):
        """The other half of the branch. A card we simply do not recognise is a
        different situation and the override really can rescue it."""
        arch, out = _wmi_detect(["AMD Radeon Graphics"])
        assert arch is None
        assert "Set UNSLOTH_ROCM_GFX_ARCH to your GPU's arch" in out

    def test_a_supported_adapter_is_unaffected(self):
        arch, out = _wmi_detect(["AMD Radeon RX 9070 XT"])
        assert arch == "gfx1201"
        assert "does not cover" not in out


def _sh_function_body(source: str, name: str) -> str:
    needle = f"{name}() {{"
    start = source.find(needle)
    assert start != -1, f"{name}() not found"
    depth = 0
    i = start + len(needle) - 1
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated {name}()")


def _ps_block(source: str, header: str) -> str:
    start = source.find(header)
    assert start != -1, f"{header} not found"
    i = source.find("(", start)
    depth = 0
    while i < len(source):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated {header}")


def _sh_rows(body: str) -> "list[tuple[list[str], str]]":
    rows = []
    for line in body.splitlines():
        m = re.match(r"\s*(\*.*?)\)\s*echo\s+(gfx[0-9a-z]+)\s*;;", line.split("#", 1)[0])
        if m:
            rows.append(([p.strip() for p in m.group(1).split("|")], m.group(2)))
    return rows


def _ps_rows(block: str) -> "list[tuple[str, str]]":
    return re.findall(r'@\{\s*P\s*=\s*"([^"]+)"\s*;\s*A\s*=\s*"(gfx[0-9a-z]+)"\s*\}', block)


def _match_sh(rows, gpu_name):
    import fnmatch
    for patterns, arch in rows:
        for pattern in patterns:
            if fnmatch.fnmatchcase(gpu_name, pattern.replace('"', "")):
                return arch
    return None


def _match_ps(rows, gpu_name):
    for pattern, arch in rows:
        if re.search(pattern, gpu_name, re.IGNORECASE):
            return arch
    return None


def _all_copies():
    """Every copy of the unsupported table, as a resolver taking a GPU name."""
    sh_rows = _sh_rows(
        _sh_function_body(
            _INSTALL_SH.read_text(encoding = "utf-8"),
            "_infer_unsupported_amd_gfx_arch_from_gpu_name",
        )
    )
    setup_sh_rows = _sh_rows(
        _sh_function_body(_SETUP_SH.read_text(encoding = "utf-8"), "_setup_unsupported_gfx_from_name")
    )
    install_ps1_rows = _ps_rows(
        _ps_block(_INSTALL_PS1.read_text(encoding = "utf-8"), "$unsupportedNameArchTable = @(")
    )
    setup_ps1_rows = _ps_rows(
        _ps_block(_SETUP_PS1.read_text(encoding = "utf-8"), "$unsupportedNameArchTable = @(")
    )
    return {
        "install.sh": lambda n: _match_sh(sh_rows, n),
        "studio/setup.sh": lambda n: _match_sh(setup_sh_rows, n),
        "install.ps1": lambda n: _match_ps(install_ps1_rows, n),
        "studio/setup.ps1": lambda n: _match_ps(setup_ps1_rows, n),
        "studio/install_python_stack.py": stack_mod._unsupported_gfx_arch_from_gpu_name,
    }


class TestUnsupportedTableParity:
    """The same drift guard the supported tables already carry. Five hand-copied
    tables is how #7264 / #7277 / #7293 each shipped half-applied."""

    def test_every_copy_parses_non_empty(self):
        rows = {
            "install.sh": _sh_rows(
                _sh_function_body(
                    _INSTALL_SH.read_text(encoding = "utf-8"),
                    "_infer_unsupported_amd_gfx_arch_from_gpu_name",
                )
            ),
            "studio/setup.sh": _sh_rows(
                _sh_function_body(
                    _SETUP_SH.read_text(encoding = "utf-8"), "_setup_unsupported_gfx_from_name"
                )
            ),
            "install.ps1": _ps_rows(
                _ps_block(
                    _INSTALL_PS1.read_text(encoding = "utf-8"), "$unsupportedNameArchTable = @("
                )
            ),
            "studio/setup.ps1": _ps_rows(
                _ps_block(_SETUP_PS1.read_text(encoding = "utf-8"), "$unsupportedNameArchTable = @(")
            ),
        }
        for where, parsed in rows.items():
            assert parsed, f"{where}: parsed an empty unsupported table (moved or renamed?)"

    _LINUX_COPIES = ("install.sh", "studio/setup.sh")

    @pytest.mark.parametrize("name,expected", _RDNA1_NAMES)
    def test_linux_copies_still_name_rdna1(self, name, expected):
        """Linux keeps the message: the multi-arch route is Windows-only until it has
        been run on bare-metal Linux (#11614)."""
        answers = {w: fn(name) for w, fn in _all_copies().items() if w in self._LINUX_COPIES}
        assert set(answers.values()) == {expected}, f"{name!r} resolves inconsistently: {answers}"

    @pytest.mark.parametrize("name,_expected", _RDNA1_NAMES)
    def test_windows_copies_no_longer_claim_rdna1(self, name, _expected):
        """The Windows copies route RDNA 1, so a claim here would print "not covered"
        at a card that is about to get wheels."""
        answers = {w: fn(name) for w, fn in _all_copies().items() if w not in self._LINUX_COPIES}
        assert set(answers.values()) == {None}, f"{name!r} is still called unsupported: {answers}"

    @pytest.mark.parametrize("name", _NOT_RDNA1_NAMES)
    def test_no_copy_claims_a_supported_card(self, name):
        answers = {where: fn(name) for where, fn in _all_copies().items()}
        assert set(answers.values()) == {None}, f"{name!r} was claimed as unsupported: {answers}"


def _normalised(path: Path) -> str:
    """CRLF-normalised source text. install.ps1 / setup.ps1 ship CRLF, so a
    substring spanning a line break never matches without this."""
    return path.read_text(encoding = "utf-8").replace("\r\n", "\n")


class TestAdviceIsNotEmittedForRdna1:
    """Each installer's unsupported arm must come BEFORE its "arch unknown" arm,
    and must not repeat the advice that arm gives."""

    @pytest.mark.parametrize(
        "path,unsupported_marker,unknown_marker",
        [
            (
                _INSTALL_PS1,
                "elseif ($ROCmUnsupportedGfxArch) {\n        # Detected, identified",
                'step "gpu" "AMD GPU detected -- arch unknown"',
            ),
            (
                _SETUP_PS1,
                "elseif ($script:ROCmUnsupportedGfxArch) {\n    # Detected, identified",
                'step "gpu" "AMD GPU detected -- arch unknown"',
            ),
        ],
    )
    def test_unsupported_arm_precedes_the_arch_unknown_arm(
        self, path, unsupported_marker, unknown_marker
    ):
        src = _normalised(path)
        # Without the found-guard, both finds are -1 and the ordering check passes vacuously.
        assert unsupported_marker in src, f"{path.name}: unsupported arm not found"
        assert unknown_marker in src, f"{path.name}: arch-unknown arm not found"
        assert src.index(unsupported_marker) < src.index(unknown_marker)

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1])
    def test_the_unsupported_arm_says_the_override_cannot_help(self, path):
        src = _normalised(path)
        # Scoped to the card, not the host: see _HOST_WIDE_CLAIMS below for why.
        needle = "setting UNSLOTH_ROCM_GFX_ARCH will not change that for it."
        assert needle in src, f"{path.name}: the override disclaimer is missing"

    def test_install_sh_cpu_note_keeps_the_sdk_advice_only_for_unknown_cards(self):
        """install.sh emits the same wording at two sites. Pin the WHOLE line at each
        one: a shared needle matches the other site and passes a deleted branch."""
        src = _normalised(_INSTALL_SH)
        unsupported = (
            'substep "AMD GPU detected ($_unsup_disp_gfx) -- Unsloth has no ROCm PyTorch '
            'wheels for that arch, installing CPU PyTorch." "$C_WARN"'
        )
        sdk_advice = "Install the ROCm/HIP SDK and re-run this installer for GPU PyTorch."
        assert unsupported in src, "install.sh: unsupported arm of the CPU note not found"
        assert sdk_advice in src, "install.sh: SDK advice not found (branch renamed?)"
        assert src.index(unsupported) < src.index(sdk_advice)

    def test_install_sh_index_selection_does_not_send_users_to_repair_rocminfo(self):
        src = _normalised(_INSTALL_SH)
        unsupported = (
            'echo "[WARN] AMD GPU detected ($_amd_unsup_gfx) -- Unsloth has no ROCm PyTorch '
            'wheels for that arch, installing CPU PyTorch." >&2'
        )
        repair_advice = "install or repair rocminfo/amd-smi"
        assert unsupported in src, "install.sh: unsupported arm of the index selector not found"
        assert repair_advice in src, "install.sh: rocminfo advice not found (branch renamed?)"
        assert src.index(unsupported) < src.index(repair_advice)

    def test_install_sh_unsupported_lookup_is_wired_to_both_sites(self):
        """The lookup must be CALLED, not merely defined; the assertions above read
        strings that a dead branch would still contain."""
        src = _normalised(_INSTALL_SH)
        assert src.count("_infer_linux_unsupported_amd_gfx_arch 2>/dev/null") == 2

    # An installed HIP SDK is the symptom (the old advice said to install it), so arms must guard it.
    _HIPSDK_ARMS = [
        (_INSTALL_PS1, "$HipSdkInstalled -and $ROCmGpuLabel", " -and -not $ROCmUnsupportedGfxArch"),
        (_INSTALL_PS1, "$HipSdkInstalled -and -not $HasROCm", " -and -not $ROCmUnsupportedGfxArch"),
        (
            _SETUP_PS1,
            "$HipSdkInstalled -and $ROCmGpuLabel",
            " -and -not $script:ROCmUnsupportedGfxArch",
        ),
    ]

    @pytest.mark.parametrize(
        "path,condition,guard",
        _HIPSDK_ARMS,
        ids = [f"{p.name}:{c[:34]}" for p, c, _g in _HIPSDK_ARMS],
    )
    def test_the_hip_sdk_arm_does_not_outrank_the_unsupported_arm(self, path, condition, guard):
        """Stated as a ban on the UNGUARDED condition, not as a search for the guarded
        one: asserting only that the guarded text exists stays green if someone adds a
        second, unguarded copy of the arm, and both spellings would then be present."""
        src = _normalised(path)
        assert condition + guard in src, (
            f"{path.name}: the {condition!r} arm has lost its unsupported-arch guard, so an "
            f"RDNA 1 user who already installed the HIP SDK never sees the new message"
        )
        assert condition + ")" not in src, (
            f"{path.name}: an unguarded {condition!r} arm is present and precedes the "
            f"unsupported arm"
        )

    # Without CUDA or XPU unsloth raises NotImplementedError at import, so never claim CPU training.
    # Scoped per arm: install.ps1's $ROCmGfxArch hint makes the claim for a different card.
    _TRAINING_ARMS = [
        (_INSTALL_PS1, "Unsloth installs no ROCm PyTorch wheels for $ROCmUnsupportedGfxArch"),
        (
            _INSTALL_PS1,
            "Installing CPU PyTorch -- Unsloth has no ROCm PyTorch wheels for "
            "$ROCmUnsupportedGfxArch.",
        ),
        (_SETUP_PS1, "Unsloth installs no ROCm PyTorch wheels for $script:ROCmUnsupportedGfxArch"),
        (_SETUP_SH, "no ROCm PyTorch wheels Unsloth installs"),
    ]

    @pytest.mark.parametrize(
        "path,anchor", _TRAINING_ARMS, ids = [f"{p.name}:{a[:34]}" for p, a in _TRAINING_ARMS]
    )
    def test_no_unsupported_arm_promises_cpu_training(self, path, anchor):
        lines = _normalised(path).splitlines()
        hits = [
            i
            for i, line in enumerate(lines)
            if anchor in line and not line.lstrip().startswith("#")
        ]
        assert len(hits) == 1, f"{path.name}: expected one arm anchored on {anchor!r}, got {hits}"
        window = "\n".join(
            line
            for line in _arm_window(lines, hits[0])
            if not line.lstrip().startswith(("#", "//"))
        )
        assert "runs on CPU on this GPU" not in window, (
            f"{path.name}:{hits[0] + 1}: promises CPU training, which raises "
            f"NotImplementedError at `import unsloth` on a host with no CUDA/XPU "
            f"accelerator:\n{window}"
        )
        assert "training and GPU inference are unavailable" in window, (
            f"{path.name}:{hits[0] + 1}: never says training is unavailable on this "
            f"GPU:\n{window}"
        )

    def test_readme_does_not_sweep_in_every_pre_rdna2_amd_gpu(self):
        """README must not sweep all AMD GPUs older than RDNA 2 into Vulkan; Vega 20 (gfx906) has a
        ROCm path."""
        src = _normalised(PACKAGE_ROOT / "README.md")
        # Any spelling of the cutoff: an inexact phrasing slipped past an exact-string ban.
        blanket = re.search(r"AMD GPUs? older than RDNA ?2", src, re.IGNORECASE)
        assert not blanket, (
            f"README: {blanket.group(0)!r} claims ROCm PyTorch covers nothing older "
            "than RDNA 2, which is wrong for gfx906"
        )
        assert "rocm6.3" in _normalised(_INSTALL_SH), "install.sh: no gfx906 ROCm index left"
        describes_group = re.search(r"no ROCm PyTorch wheels|Polaris|RDNA ?1", src, re.IGNORECASE)
        if not describes_group:
            return
        # Describe the group completely: name its members and cut the covered one back out.
        for _member in ("Polaris", "RDNA 1"):
            assert _member in src, (
                f"README describes the uncovered AMD group ({describes_group.group(0)!r}) "
                f"but never names {_member} as part of it"
            )
        assert "gfx906" in src, "README: never carves Vega 20 out of the unsupported group"

    def test_setup_sh_names_the_arch_instead_of_claiming_rocm(self):
        src = _normalised(_SETUP_SH)
        needle = 'step "gpu" "AMD GPU detected ($_setup_unsup_gfx) -- no ROCm PyTorch wheels Unsloth installs"'
        fallthrough = 'step "gpu" "AMD ROCm"'
        assert needle in src, "studio/setup.sh: unsupported arm not found"
        assert fallthrough in src, "studio/setup.sh: plain AMD ROCm arm not found"
        assert src.index(needle) < src.index(fallthrough)


def _run_setup_kfd_lookup(gpu_name: str, lspci_lines: "list[str] | None", tmp_path) -> str:
    """Runs setup.sh's KFD lookup with scripted lspci output; lspci_lines None means lspci is absent."""
    src = _SETUP_SH.read_text(encoding = "utf-8")
    body = "\n".join(
        _sh_function_body(src, name)
        for name in ("_setup_unsupported_gfx_from_name", "_setup_unsupported_gfx_any")
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # PATH is bin_dir and nothing else, so "lspci absent" really is absent rather than the host's own lspci answering.
    real_sh = shutil.which("sh")
    assert real_sh, "no POSIX sh on this host"
    (bin_dir / "sh").symlink_to(real_sh)
    for _tool in ("grep", "cat"):
        _found = shutil.which(_tool)
        assert _found, f"no {_tool} on this host"
        (bin_dir / _tool).symlink_to(_found)
    if lspci_lines is not None:
        fake = bin_dir / "lspci"
        printed = "\n".join(lspci_lines)
        fake.write_text(f'#!/bin/sh\ncat <<"LSPCI_EOF"\n{printed}\nLSPCI_EOF\n', encoding = "utf-8")
        fake.chmod(0o755)
    env = dict(os.environ, PATH = str(bin_dir))
    out = subprocess.run(
        ["sh", "-c", f'{body}\n_setup_unsupported_gfx_any "$1" || true\n', "sh", gpu_name],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 30,
        env = env,
    )
    return out.stdout.strip()


_KFD_NAVI10 = [
    "0a:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
    "Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT] [1002:731f] (rev c1)"
]
_KFD_POLARIS = [
    "01:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
    "Ellesmere [Radeon RX 470/480/570/570X/580/580X/590] [1002:67df] (rev e7)"
]
_KFD_NAVI31 = [
    "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
    "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] [1002:744c] (rev cc)"
]


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shell only")
class TestSetupShKfdOnlyHost:
    """The KFD sysfs fallback marks the host AMD without rocminfo or amd-smi, so it
    leaves _setup_mkt empty -- and a host with no ROCm userspace is exactly the one
    #8529 and #8458 describe. Handed "", the lookup used to fall through to the plain
    "AMD ROCm" report on a machine that has no ROCm."""

    def test_the_kfd_host_is_diagnosed_from_lspci(self, tmp_path):
        assert _run_setup_kfd_lookup("", _KFD_NAVI10, tmp_path) == "gfx1010"

    def test_polaris_on_the_kfd_host_is_diagnosed_too(self, tmp_path):
        assert _run_setup_kfd_lookup("", _KFD_POLARIS, tmp_path) == "gfx803"

    def test_a_covered_card_is_never_claimed_from_lspci(self, tmp_path):
        """The scope guard: an RX 7900 with no ROCm installed is a missing runtime,
        not an uncovered generation, and must keep the plain report."""
        assert _run_setup_kfd_lookup("", _KFD_NAVI31, tmp_path) == ""

    def test_no_lspci_is_not_an_error(self, tmp_path):
        assert _run_setup_kfd_lookup("", None, tmp_path) == ""

    def test_a_reported_name_still_decides(self, tmp_path):
        """rocminfo/amd-smi named the card; lspci is not consulted behind their back."""
        assert _run_setup_kfd_lookup("AMD Radeon RX 5500 XT", _KFD_NAVI10, tmp_path) == "gfx1012"

    def test_an_unmapped_reported_name_falls_through_to_lspci(self, tmp_path):
        """An unmapped reported name must fall through to lspci, or a generic 'AMD Radeon Graphics'
        hides it."""
        assert _run_setup_kfd_lookup("AMD Radeon Graphics", _KFD_NAVI10, tmp_path) == "gfx1010"

    def test_an_unmapped_name_over_a_covered_card_still_claims_nothing(self, tmp_path):
        assert _run_setup_kfd_lookup("AMD Radeon Graphics", _KFD_NAVI31, tmp_path) == ""

    def test_the_report_site_uses_the_lspci_aware_lookup(self):
        src = _normalised(_SETUP_SH)
        assert (
            'elif _setup_unsup_gfx=$(_setup_unsupported_gfx_any "$_setup_mkt"); then' in src
        ), "studio/setup.sh: the gpu report no longer goes through the lspci-aware lookup"

    def test_the_lspci_name_never_reaches_the_routing_table(self):
        """Routing must stay byte-identical. The supported inference table keys on
        _setup_mkt, which feeds --rocm-gfx into the prebuilt and whisper commands, so
        the lspci read must not be written back into it."""
        src = _normalised(_SETUP_SH)
        assigns = re.findall(r"^\s*_setup_mkt=(.*)$", src, re.MULTILINE)
        assert assigns, "studio/setup.sh: no _setup_mkt assignment found"
        for rhs in assigns:
            assert "lspci" not in rhs, f"_setup_mkt fed from lspci: {rhs!r}"


def test_the_unsupported_arch_variable_is_declared_outside_the_amd_block():
    """$ROCmUnsupportedGfxArch must be declared outside the AMD block, or StrictMode fails NVIDIA hosts."""
    src = _normalised(_INSTALL_PS1)
    m = re.search(
        r"^    \$ROCmGfxArch = \$null\n(?P<between>(?:.*\n)*?)    if \(-not \$HasNvidiaSmi\) \{",
        src,
        re.MULTILINE,
    )
    assert m, "install.ps1: the AMD declaration block was restructured; re-check this test"
    assert "$ROCmUnsupportedGfxArch = $null" in m.group("between"), (
        "install.ps1: $ROCmUnsupportedGfxArch is declared inside the -not $HasNvidiaSmi "
        "block, so an NVIDIA host reaches its readers with the variable unset"
    )


def test_setup_ps1_hoists_the_unsupported_arch_variable_too():
    """Same property, expressed as setup.ps1 writes it: at script scope, column 0,
    so no block can gate the declaration away from the summary that reads it."""
    src = _normalised(_SETUP_PS1)
    decl = "$script:ROCmUnsupportedGfxArch = $null"
    assert decl in src, "setup.ps1: the unsupported-arch declaration is gone"
    assert any(line == decl for line in src.split("\n")), (
        "setup.ps1: the unsupported-arch declaration is indented, so it now sits "
        "inside a block an NVIDIA host skips"
    )


_ROCM_ARM = {
    "install.ps1": ("} elseif ($HasROCm", "$ROCmUnsupportedGfxArch"),
    "setup.ps1": ("} elseif ($HasROCm", "$script:ROCmUnsupportedGfxArch"),
}


@pytest.mark.parametrize("name", sorted(_ROCM_ARM))
def test_the_generic_rocm_arm_yields_to_an_identified_uncovered_card(name):
    """Generic ROCm arm must yield to an identified uncovered card, since amd-smi may omit the gfx token."""
    source_path = _INSTALL_PS1 if name == "install.ps1" else _SETUP_PS1
    opener, var = _ROCM_ARM[name]
    src = _normalised(source_path)
    arm = next((ln for ln in src.split("\n") if ln.strip().startswith(opener)), None)
    assert arm is not None, f"{name}: the $HasROCm arm was renamed"
    assert f"-not {var}" in arm, (
        f"{name}: the generic ROCm arm outranks the identified-uncovered-card arm, so a "
        f"card we already named is reported as ordinary ROCm:\n{arm}"
    )


def test_the_rocm_summary_chain_yields_to_an_identified_uncovered_card():
    """The summary's ROCm arm must yield to the uncovered-card arm, which would otherwise never run."""
    src = _normalised(_SETUP_PS1)
    lines = src.split("\n")
    opener = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.strip().startswith("if ($HasROCm")
            and "$rocmVerLabel" in "\n".join(lines[i : i + 3])
        ),
        None,
    )
    assert opener is not None, "setup.ps1: the ROCm summary chain was renamed"
    chain = "\n".join(lines[opener : opener + 20])
    assert (
        "$script:ROCmUnsupportedGfxArch" in chain
    ), "setup.ps1: the summary chain no longer has an uncovered-card arm"
    assert "-not $script:ROCmUnsupportedGfxArch" in lines[opener], (
        "setup.ps1: the ROCm summary reports an uncovered card as ordinary ROCm, so the "
        f"same run says 'ROCm' here and 'no wheels' below:\n{lines[opener]}"
    )


# A host is not one GPU: masking to another card and pinning its arch can install wheels.
# Runtime detection misfires on Ryzen Vega iGPUs and misses Instinct/V620 parts.

_ALL_SOURCES = [_INSTALL_SH, _SETUP_SH, _INSTALL_PS1, _SETUP_PS1, _STACK_PY]

_HOST_WIDE_CLAIMS = [
    "will not enable ROCm PyTorch.",
    "Installing the ROCm/HIP SDK will not change this.",
    "no UNSLOTH_ROCM_GFX_ARCH value changes that.",
    "UNSLOTH_ROCM_GFX_ARCH will not change that.",
    "can enable ROCm here.",
]

_SCOPED_CLAIMS = {
    "install.sh": [
        "will not give it ROCm PyTorch.",
        "Installing the ROCm/HIP SDK will not give this GPU ROCm PyTorch.",
    ],
    "setup.sh": [
        "no UNSLOTH_ROCM_GFX_ARCH value gives this GPU one.",
    ],
    "install.ps1": [
        "will not change that for it.",
        "can give this GPU ROCm.",
    ],
    "setup.ps1": [
        "will not change that for it.",
    ],
    "install_python_stack.py": [
        "changes that on this GPU.",
    ],
}


@pytest.mark.parametrize("source_path", _ALL_SOURCES, ids = [p.name for p in _ALL_SOURCES])
def test_no_advice_arm_speaks_for_the_whole_host(source_path):
    src = _normalised(source_path)
    for line in src.split("\n"):
        if line.lstrip().startswith("#"):
            continue
        for claim in _HOST_WIDE_CLAIMS:
            assert claim not in line, (
                f"{source_path.name}: this sentence claims the HOST has no ROCm path, "
                f"which is false beside a covered card:\n{line}"
            )


@pytest.mark.parametrize("name,claims", sorted(_SCOPED_CLAIMS.items()))
def test_the_scoped_wording_is_the_one_that_ships(name, claims):
    """The other half: dropping the sentence entirely would also pass the ban above,
    and would take the answer with it. Each arm still has to say ROCm cannot reach
    the card it just named."""
    source_path = next(p for p in _ALL_SOURCES if p.name == name)
    src = _normalised(source_path)
    emitted = [ln for ln in src.split("\n") if not ln.lstrip().startswith("#")]
    for claim in claims:
        assert any(claim in ln for ln in emitted), (
            f"{source_path.name}: the scoped verdict {claim!r} is gone; an arm that "
            f"names an uncovered arch has to say ROCm does not reach it"
        )


def _run_sh_lookup(source_path: Path, fn_name: str, gpu_name: str) -> str:
    body = _sh_function_body(source_path.read_text(encoding = "utf-8"), fn_name)
    script = f'{body}\n{fn_name} "$1" || true\n'
    out = subprocess.run(
        ["sh", "-c", script, "sh", gpu_name],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 30,
    )
    return out.stdout.strip()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shell only")
class TestShellLookupsRun:
    """Parsing a case table and evaluating it in Python is not the same as the
    shell evaluating it; run the real thing on the reporter's card."""

    @pytest.mark.parametrize(
        "path,fn",
        [
            (_INSTALL_SH, "_infer_unsupported_amd_gfx_arch_from_gpu_name"),
            (_SETUP_SH, "_setup_unsupported_gfx_from_name"),
        ],
    )
    def test_rx_5700_xt_resolves_to_gfx1010(self, path, fn):
        assert _run_sh_lookup(path, fn, "AMD Radeon RX 5700 XT") == "gfx1010"

    @pytest.mark.parametrize(
        "path,fn",
        [
            (_INSTALL_SH, "_infer_unsupported_amd_gfx_arch_from_gpu_name"),
            (_SETUP_SH, "_setup_unsupported_gfx_from_name"),
        ],
    )
    def test_rx_9070_xt_is_not_claimed(self, path, fn):
        assert _run_sh_lookup(path, fn, "AMD Radeon RX 9070 XT") == ""

    @pytest.mark.parametrize(
        "path,fn",
        [
            (_INSTALL_SH, "_infer_unsupported_amd_gfx_arch_from_gpu_name"),
            (_SETUP_SH, "_setup_unsupported_gfx_from_name"),
        ],
    )
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("AMD Radeon RX 580", "gfx803"),
            ("AMD Radeon RX 5700 XT", "gfx1010"),
            ("AMD Radeon RX 5500 XT", "gfx1012"),
        ],
    )
    def test_polaris_does_not_shadow_rdna1_in_the_real_shell(self, path, fn, name, expected):
        """`case` has no negative lookahead, so the *"RX 570"* arm is only safe
        because it comes last. Evaluate it in a real shell rather than trusting
        the ordering by inspection."""
        assert _run_sh_lookup(path, fn, name) == expected


def _arm_window(lines: "list[str]", start: int) -> "list[str]":
    """Returns the branch at start, ending at the first dedent past its anchor, with a length cap."""
    indent = len(lines[start]) - len(lines[start].lstrip())
    # One step out: the Vulkan offer is a sibling branch of the anchored claim.
    floor = max(indent - 4, 0)
    out = [lines[start]]
    for line in lines[start + 1 : start + 25]:
        if line.strip() and (len(line) - len(line.lstrip())) < floor:
            break
        out.append(line)
    return out


class TestVulkanAdvice:
    """Advice names UNSLOTH_LLAMA_CPP_BACKEND, not legacy UNSLOTH_FORCE_VULKAN; it applies at install."""

    # The Python copy joins string fragments; live-output tests cover it instead.
    _SHELL_SOURCES = [_INSTALL_PS1, _SETUP_PS1, _INSTALL_SH, _SETUP_SH]

    # Asserted on EMITTED text only: the same phrases appear in comments explaining the branch.
    _REQUIRED = [
        # The offer must survive, not just the variable name: "no GPU acceleration is available" followed by a GPU
        # backend's name is worse than either half alone.
        ("through Vulkan", "the affirmative Vulkan offer"),
    ]

    @staticmethod
    def _emitted_text(path: Path) -> str:
        """Only the lines that PRINT, so comments explaining the branch cannot
        satisfy an assertion about what the user is told."""
        emitters = ("substep", "echo ", "_safe_print", "step ", "Write-StudioLine")
        return "\n".join(
            line
            for line in _normalised(path).splitlines()
            if any(e in line for e in emitters) and not line.lstrip().startswith(("#", "//"))
        )

    @pytest.mark.parametrize("path", _SHELL_SOURCES, ids = lambda p: p.name)
    @pytest.mark.parametrize("needle,what", _REQUIRED, ids = lambda v: v if " " not in v else None)
    def test_the_emitted_advice_carries_every_part(self, path, needle, what):
        assert needle in self._emitted_text(
            path
        ), f"{path.name}: the message a user actually sees is missing {what} ({needle!r})"

    @pytest.mark.parametrize("path", _SHELL_SOURCES, ids = lambda p: p.name)
    def test_the_emitted_advice_uses_the_right_shell_syntax(self, path):
        """The setter has to be pasteable into the shell that reads this file."""
        emitted = self._emitted_text(path)
        assert _SETTER[path.name] in emitted, (
            f"{path.name}: the printed advice never gives the setter in this shell's "
            f"syntax ({_SETTER[path.name]!r})"
        )

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1, _STACK_PY], ids = lambda p: p.name)
    def test_no_windows_source_teaches_the_posix_setter(self, path):
        """No Windows emitter may print the POSIX env-var setter; comments may quote it."""
        offenders = [
            line.strip()
            for line in _normalised(path).splitlines()
            if _POSIX_ASSIGNMENT in line
            and any(e in line for e in ("substep", "_safe_print", "step ", "Write-StudioLine"))
            and not line.lstrip().startswith("#")
        ]
        assert (
            not offenders
        ), f"{path.name}: prints a POSIX assignment PowerShell cannot parse: {offenders}"

    # install.ps1's second anchor names $ROCmUnsupportedGfxArch: the earlier sentence is for a supported card.
    _ADVICE_SITES = [
        (_INSTALL_SH, "Unsloth has no ROCm PyTorch wheels for that arch", 2),
        (_INSTALL_PS1, "Unsloth installs no ROCm PyTorch wheels for $ROCmUnsupportedGfxArch", 1),
        (
            _INSTALL_PS1,
            "Installing CPU PyTorch -- Unsloth has no ROCm PyTorch wheels for "
            "$ROCmUnsupportedGfxArch.",
            1,
        ),
        (
            _SETUP_PS1,
            "Unsloth installs no ROCm PyTorch wheels for $script:ROCmUnsupportedGfxArch",
            1,
        ),
        (_SETUP_SH, "no ROCm PyTorch wheels Unsloth installs", 1),
    ]

    @pytest.mark.parametrize(
        "path,anchor,count",
        _ADVICE_SITES,
        ids = [f"{p.name}:{a[:34]}" for p, a, _c in _ADVICE_SITES],
    )
    def test_each_advisory_arm_offers_vulkan(self, path, anchor, count):
        """Each advisory arm must offer Vulkan, checked by real line number so a deleted arm fails."""
        lines = _normalised(path).splitlines()
        hits = [
            i
            for i, line in enumerate(lines)
            if anchor in line and not line.lstrip().startswith("#")
        ]
        assert len(hits) == count, (
            f"{path.name}: expected {count} advisory arm(s) anchored on {anchor!r}, found "
            f"{len(hits)} at lines {[i + 1 for i in hits]}. An arm was removed, renamed, or "
            f"duplicated; the advice must follow it either way."
        )
        for i in hits:
            # Strip comments from the window: the phrases also appear in the explaining comment.
            window = "\n".join(
                line for line in _arm_window(lines, i) if not line.lstrip().startswith(("#", "//"))
            )
            # Both halves are required: GGUF chat and the Vulkan backend, with Vulkan matched case-sensitively.
            assert "GGUF chat" in window and "Vulkan" in window, (
                f"{path.name}:{i + 1}: this arm dead-ends without offering GPU GGUF chat "
                f"through Vulkan:\n{window}"
            )
            assert _SETTER[path.name] in window, (
                f"{path.name}:{i + 1}: this arm offers Vulkan without naming the variable "
                f"that selects it, in this shell's syntax:\n{window}"
            )
            assert "install time" in window or "at launch" in window, (
                f"{path.name}:{i + 1}: this arm names the variable but never says the bundle "
                f"is chosen at install time, which is the mistake #8458 made:\n{window}"
            )

    @pytest.mark.parametrize("path", _SHELL_SOURCES, ids = lambda p: p.name)
    def test_every_site_that_names_the_variable_also_says_when(self, path):
        """Every site naming the Vulkan selector must also say when it applies (install time)."""
        emitted = self._emitted_text(path).splitlines()
        mentions = [i for i, line in enumerate(emitted) if _SETTER[path.name] in line]
        assert mentions, f"{path.name}: no site names the Vulkan variable"
        for i in mentions:
            window = "\n".join(emitted[i : i + 4])
            assert "install time" in window, (
                f"{path.name}: the advice at emitted line {i + 1} names the variable but "
                f"never says the bundle is chosen at install time:\n{window}"
            )

    @pytest.mark.parametrize("path", _SHELL_SOURCES + [_STACK_PY], ids = lambda p: p.name)
    def test_the_legacy_spelling_is_not_taught(self, path):
        """Printed text must not teach legacy UNSLOTH_FORCE_VULKAN, which loses to
        UNSLOTH_LLAMA_CPP_BACKEND."""
        emitters = ("substep", "echo", "_safe_print", "step ", "Write-StudioLine")
        offenders = [
            line.strip()
            for line in _normalised(path).splitlines()
            if "UNSLOTH_FORCE_VULKAN" in line and any(e in line for e in emitters)
        ]
        assert not offenders, f"{path.name}: teaches the legacy variable: {offenders}"

    @pytest.mark.parametrize("needle,what", _REQUIRED, ids = lambda v: v if " " not in v else None)
    def test_the_vulkan_advice_is_reached_by_the_rdna1_wmi_path(self, needle, what):
        """Source-text assertions cannot tell a live branch from a dead one, and the
        Python copy's message is not readable line by line. Drive the real Windows
        path and read what it actually printed."""
        arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert arch is None, "routing must be unchanged; this is a wording fix"
        assert needle in out, f"the printed advice is missing {what} ({needle!r})"

    def test_the_printed_advice_uses_powershell_syntax(self):
        """This branch is Windows-only, so its setter has to be pasteable into
        PowerShell. Read the live output, not the source: the message is built from
        implicitly-joined fragments and no single source line carries it."""
        _arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert _PWSH_SETTER in out, f"the printed advice is not pasteable into PowerShell:\n{out}"
        assert (
            _POSIX_ASSIGNMENT not in out
        ), f"the printed advice gives a POSIX assignment PowerShell cannot parse:\n{out}"

    def test_the_printed_advice_says_when_to_set_it(self):
        """Python builds the message from implicit string fragments, so only live output can show
        the advice."""
        _arch, out = _wmi_detect(["AMD Radeon RX 580"])
        assert "install time" in out, (
            f"the printed advice names the Vulkan variable but never says when to "
            f"set it:\n{out}"
        )

    def test_an_unknown_amd_card_gets_no_vulkan_advice(self):
        """Scope. A card we simply failed to recognise may well have ROCm wheels,
        so it keeps the override advice and must not be pushed onto Vulkan."""
        _arch, out = _wmi_detect(["AMD Radeon Graphics"])
        assert "UNSLOTH_LLAMA_CPP_BACKEND" not in out

    def test_a_supported_card_gets_no_vulkan_advice(self):
        _arch, out = _wmi_detect(["AMD Radeon RX 9070 XT"])
        assert "UNSLOTH_LLAMA_CPP_BACKEND" not in out

    def test_readme_copy_paste_blocks_use_the_current_spelling(self):
        """Fenced README command blocks, which users copy, must use the current variable spelling."""
        src = _normalised(PACKAGE_ROOT / "README.md")
        blocks = re.findall(r"```(?:bash|powershell)\n(.*?)```", src, re.DOTALL)
        setters = [
            line.strip()
            for block in blocks
            for line in block.splitlines()
            if "VULKAN" in line.upper() or "LLAMA_CPP_BACKEND" in line
        ]
        for line in setters:
            assert (
                "UNSLOTH_LLAMA_CPP_BACKEND" in line
            ), f"README teaches the legacy spelling in a copy-paste block: {line!r}"

    def test_forcing_vulkan_on_macos_says_so_instead_of_going_quiet(self):
        """macOS has no Vulkan bundle, so a forced Vulkan request must be logged as ignored, not silent."""
        prebuilt = (PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py").read_text(
            encoding = "utf-8"
        )
        # Scoped to the routing function: `if host.is_macos:` appears many times.
        routing = next(
            (
                ast.get_source_segment(prebuilt, node)
                for node in ast.parse(prebuilt).body
                if isinstance(node, ast.FunctionDef) and node.name == "_route_to_vulkan_prebuilt"
            ),
            None,
        )
        assert routing, "_route_to_vulkan_prebuilt was renamed or moved"
        branch = re.search(
            r"if host\.is_macos:\n(?P<body>(?:[ \t]+.*\n|\n)+?)[ \t]{8}return ", routing
        )
        assert branch, "the macOS branch in _route_to_vulkan_prebuilt was restructured"
        body = branch.group("body")
        assert "ignored on macOS" in body and "Metal" in body, (
            "the macOS branch no longer says the forced backend was ignored and Metal "
            f"used, so the request fails silently there:\n{body}"
        )
        assert "if forced:" in body, (
            "the ignore notice is no longer gated on an explicit request, so every macOS "
            f"install logs it:\n{body}"
        )


_POLARIS_NAMES = [
    ("AMD Radeon RX 580", "gfx803"),
    ("AMD Radeon RX 580 Series", "gfx803"),
    ("AMD Radeon RX 570", "gfx803"),
    ("AMD Radeon RX 590", "gfx803"),
    ("AMD Radeon RX 480", "gfx803"),
    ("AMD Radeon RX 470", "gfx803"),
    ("Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]", "gfx803"),
    # Polaris 10 workstation boards: pci.ids groups them on Ellesmere, with no RX number.
    ("Ellesmere [Radeon Pro WX 7100 / WX 7100 Mobile / WX 5100 / V7300X / V7350x2]", "gfx803"),
    ("AMD Radeon Pro WX 7100", "gfx803"),
    ("AMD Radeon Pro WX 5100", "gfx803"),
]

# Polaris 11/12 deliberately excluded: a different die, and the table must never guess.
_POLARIS_11_12_NAMES = [
    "AMD Radeon RX 560",
    "AMD Radeon RX 550",
    "AMD Radeon RX 460",
]


class TestPolarisRow:
    @pytest.mark.parametrize("name,expected", _POLARIS_NAMES)
    def test_polaris_names_resolve_to_gfx803(self, name, expected):
        assert stack_mod._unsupported_gfx_arch_from_gpu_name(name) == expected

    @pytest.mark.parametrize("name,_expected", _POLARIS_NAMES)
    def test_polaris_still_gets_no_supported_arch(self, name, _expected):
        """The behavioural half again: CPU torch must remain the outcome."""
        assert stack_mod._gfx_arch_from_gpu_name(name) is None

    @pytest.mark.parametrize("name", _POLARIS_11_12_NAMES)
    def test_polaris_11_12_is_not_claimed(self, name):
        assert stack_mod._unsupported_gfx_arch_from_gpu_name(name) is None

    @pytest.mark.parametrize("name,_expected", _RDNA1_NAMES)
    def test_the_polaris_pattern_is_correct_on_its_own(self, name, _expected):
        """Table order hides a missing (?!0) guard, so the Polaris pattern is matched alone to test
        the guard."""
        pattern = next(
            p for p, arch in stack_mod._UNSUPPORTED_GPU_NAME_ARCH_TABLE if arch == "gfx803"
        )
        assert (
            re.search(pattern, name, re.IGNORECASE) is None
        ), f"the Polaris pattern claims the RDNA 1 name {name!r} when matched on its own"

    @pytest.mark.parametrize("name,expected", _POLARIS_NAMES)
    def test_all_copies_agree_on_polaris(self, name, expected):
        answers = {where: fn(name) for where, fn in _all_copies().items()}
        assert set(answers.values()) == {expected}, f"{name!r} resolves inconsistently: {answers}"

    @pytest.mark.parametrize("name", _POLARIS_11_12_NAMES)
    def test_no_copy_claims_polaris_11_12(self, name):
        answers = {where: fn(name) for where, fn in _all_copies().items()}
        assert set(answers.values()) == {None}, f"{name!r} was claimed: {answers}"

    @pytest.mark.parametrize("name,_expected", _RDNA1_NAMES)
    def test_the_polaris_row_is_correct_alone_in_every_regex_copy(self, name, _expected):
        """Row order masks a missing (?!0) guard in the .ps1 tables, so each Polaris row is checked
        alone."""
        for where, rows in (
            (
                "install.ps1",
                _ps_rows(_ps_block(_normalised(_INSTALL_PS1), "$unsupportedNameArchTable = @(")),
            ),
            (
                "studio/setup.ps1",
                _ps_rows(_ps_block(_normalised(_SETUP_PS1), "$unsupportedNameArchTable = @(")),
            ),
        ):
            pattern = next(p for p, arch in rows if arch == "gfx803")
            assert (
                re.search(pattern, name, re.IGNORECASE) is None
            ), f"{where}: the Polaris pattern claims the RDNA 1 name {name!r} when matched alone"

    @pytest.mark.parametrize(
        "path,fn",
        [
            (_INSTALL_SH, "_infer_unsupported_amd_gfx_arch_from_gpu_name"),
            (_SETUP_SH, "_setup_unsupported_gfx_from_name"),
        ],
        ids = ["install.sh", "studio/setup.sh"],
    )
    def test_the_shell_case_arms_keep_polaris_last(self, path, fn):
        """Shell case globs have no negative lookahead, so the Polaris arm is safe only if it stays last."""
        rows = _sh_rows(_sh_function_body(path.read_text(encoding = "utf-8"), fn))
        arches = [arch for _patterns, arch in rows]
        assert "gfx803" in arches, f"{path.name}: no Polaris arm"
        assert arches[-1] == "gfx803", (
            f"{path.name}: the Polaris arm must stay last, after every RDNA 1 arm; "
            f"arm order is {arches}"
        )

    def test_gfx803_is_not_routable(self):
        """The scope guard, restated for the new arch: gfx803 must stay absent from
        the wheel-index map, or a messaging row becomes an install change."""
        assert "gfx803" not in stack_mod._GFX_TO_AMD_INDEX_ARCH

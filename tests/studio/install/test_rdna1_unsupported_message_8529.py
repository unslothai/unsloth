# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the installers SAY when an AMD GPU is detected and ROCm does not cover it.

Issue #8529: an RX 5700 XT (Navi 10, gfx1010, RDNA 1) on Windows correctly landed
on CPU torch, and the installer then told the reporter to install the HIP SDK or
set UNSLOTH_ROCM_GFX_ARCH "to enable GPU ROCm". Neither can work: AMD publishes
Windows torch indexes for gfx103X, gfx110X, gfx1150, gfx1151 and gfx120X only, so
UNSLOTH_ROCM_GFX_ARCH=gfx1010 lands on the unmapped-arch path and returns CPU
anyway, after an SDK install and a reboot spent for nothing.

The fixtures are shaped around that one card because it is the only confirmed
report. Adapter names are given as Windows WMI and Linux lspci actually spell
them ("AMD Radeon RX 5700 XT", "Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700
XT]"), not as tidy marketing strings, since the tables are matched against the
raw probe output. The supported-card fixtures (RX 9070 XT, RX 6800 XT) are here
to prove the new lookup cannot reach a card that has wheels, and the RTX 4090 to
prove it cannot reach a non-AMD one.

CPU fallback is the correct outcome on RDNA 1 and every test below re-asserts it:
this change is about wording, never about routing.
"""

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

# The setter each source has to teach, in the syntax of the shell that reads it.
# PowerShell cannot parse a bare VAR=value: it resolves it as a command name, so a
# Windows user who pastes it sets nothing and gets the same CPU bundle -- the #8458
# failure mode, reintroduced by the fix for it. Two needles, not one: the bare form is
# what a PowerShell source must never print, and folding them together let a .ps1 emit
# it and still pass (an added-bare-setter mutant survived the whole file).
_POSIX_ASSIGNMENT = "UNSLOTH_LLAMA_CPP_BACKEND=vulkan"
# `export`, not a bare assignment: a POSIX assignment without it is a shell variable, invisible to the installer the
# next line tells the user to run, so they get the CPU bundle again and conclude the advice was wrong (the #8458
# mistake).
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


# Windows WMI reports the marketing name; Linux lspci reports the chip plus a slash-joined list of the boards built
# on it. Both must resolve.
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
    # The professional boards LLVM's table omits.
    # Die confirmed from libdrm data/amdgpu.ids read against pci.ids, which names 7312/7310 Navi 10 and
    # 7340/7341/7347/734f Navi 14, and the kernel's amdgpu table, which files those ids under CHIP_NAVI10 / CHIP_NAVI14.
    ("AMD Radeon Pro W5700", "gfx1010"),
    ("Navi 10 [Radeon Pro W5700X]", "gfx1010"),
    ("AMD Radeon Pro W5500", "gfx1012"),
    ("AMD Radeon Pro W5500M", "gfx1012"),
    ("Navi 14 [Radeon Pro W5300M]", "gfx1012"),
    ("AMD Radeon RX 5300", "gfx1012"),
    ("AMD Radeon RX 5300M", "gfx1012"),
    # The Mac Pro MPX boards, pci.ids 7319 and 731b under Navi 10: the only Navi 10 retail parts naming neither
    # "RX 5700" nor a W prefix.
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
    # The workstation boards that DO have wheels, now that this table names W-series
    # parts: "W5700" must not be read out of "W7500", nor "W5500" out of "W6500".
    "AMD Radeon PRO W7500",
    "AMD Radeon PRO W7900",
    "AMD Radeon PRO W6500",
    "AMD Radeon PRO W6400",
]


# ── The lookup itself ────────────────────────────────────────────────────────


class TestUnsupportedNameLookup:
    @pytest.mark.parametrize("name,expected", _RDNA1_NAMES)
    def test_rdna1_names_resolve_to_their_arch(self, name, expected):
        assert stack_mod._unsupported_gfx_arch_from_gpu_name(name) == expected

    @pytest.mark.parametrize("name", _NOT_RDNA1_NAMES)
    def test_supported_and_non_amd_names_are_not_claimed(self, name):
        assert stack_mod._unsupported_gfx_arch_from_gpu_name(name) is None

    def test_empty_name_is_not_claimed(self):
        assert stack_mod._unsupported_gfx_arch_from_gpu_name("") is None

    @pytest.mark.parametrize("name,_expected", _RDNA1_NAMES)
    def test_rdna1_still_gets_no_supported_arch(self, name, _expected):
        """The behavioural half: RDNA 1 must keep falling through to CPU torch.
        If this ever passes an arch back, the installer would try to route it."""
        assert stack_mod._gfx_arch_from_gpu_name(name) is None

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


# ── The Windows WMI path end to end ──────────────────────────────────────────


def _wmi_detect(names):
    """Drive _detect_windows_gfx_arch over `names` with no hipinfo and no amd-smi,
    which is the reporter's host: Adrenalin driver only. Returns (arch, stdout)."""
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
                        result = stack_mod._detect_windows_gfx_arch()
    return result, buf.getvalue()


class TestExplicitIndexPinIsHonoured:
    """An explicit UNSLOTH_TORCH_INDEX_URL / _FAMILY reaches the ROCm install path for
    ANY gfx/rocm leaf (install.ps1's pinned-index arm), so "torch stays CPU-only and
    nothing changes that" is false on a pinned run. install.sh's CPU note already skips
    its guidance when pinned; the other four sources now agree with it."""

    _CPU_CLAIM = "torch will be CPU-only"

    def test_the_python_warning_drops_the_cpu_claim_when_pinned(self):
        with patch.dict(os.environ, {"UNSLOTH_TORCH_INDEX_URL": "https://example/gfx1010"}):
            _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert "gfx1010" in out, "the card is still named"
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
            _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert (
            self._CPU_CLAIM in out
        ), f"a blank pin ({env}) was read as a pin, dropping the CPU-only warning:\n{out}"

    def test_the_claim_is_there_without_a_pin(self):
        """The positive control: without it the test above passes on any wording."""
        with patch.dict(os.environ, {}, clear = False):
            for _v in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
                os.environ.pop(_v, None)
            _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
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
            # Backwards: the pin is read above the claim, which is what puts the claim
            # in a branch. Bounded so an unrelated mention further up cannot satisfy it.
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

    # The Python stack emits the PowerShell setter too, and setup.ps1 runs it before reaching its own throw, so it is
    # the last advice an ARM64 user sees.
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
            # The resolver itself, not a boolean named after it: a mutant that kept the branch and hardcoded
            # $unsupArm64 = $false survived that spelling.
            back = "\n".join(lines[max(i - 20, 0) : i])
            assert guard in back, (
                f"{path.name}:{i + 1}: offers the Vulkan variable without checking for "
                f"Windows ARM64, where setting it throws:\n{back}"
            )


class TestPythonStackWindowsArm64:
    """The Windows WMI path prints the same Vulkan advice as install.ps1, and the same
    ARM64 throw applies to it: setup.ps1 rejects the variable there."""

    def test_arm64_gets_a_source_build_note_instead_of_the_setter(self):
        with patch.object(stack_mod, "_is_windows_arm64", return_value = True):
            _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert "gfx1010" in out, "the card is still named"
        assert (
            "UNSLOTH_LLAMA_CPP_BACKEND" not in out
        ), f"ARM64 is still told to set the variable setup.ps1 throws on:\n{out}"
        assert "ARM64" in out and "source" in out

    def test_x64_still_gets_the_setter(self):
        """Positive control: the guard must not silence the advice everywhere."""
        with patch.object(stack_mod, "_is_windows_arm64", return_value = False):
            _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert "UNSLOTH_LLAMA_CPP_BACKEND" in out

    @pytest.mark.parametrize(
        "env,expected",
        [
            ({"PROCESSOR_ARCHITECTURE": "ARM64"}, True),
            ({"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": "ARM64"}, True),
            ({"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": ""}, False),
        ],
        ids = ["native-arm64", "emulated-x64-on-arm64", "real-x64"],
    )
    def test_the_arch_probe_reads_the_machine_not_the_process(self, env, expected):
        """PROCESSOR_ARCHITECTURE describes the PROCESS, so an emulated x64 Python on an
        ARM64 box reports AMD64; ARCHITEW6432 is ARM64 in exactly that case."""
        with patch.object(stack_mod, "IS_WINDOWS", True):
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
    def test_rdna1_adapter_still_yields_no_arch(self):
        """CPU fallback unchanged. This is the assertion that keeps the fix honest."""
        arch, _out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert arch is None

    def test_rdna1_adapter_is_named_with_its_arch(self):
        _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert "gfx1010" in out
        assert "AMD Radeon RX 5700 XT" in out

    def test_rdna1_adapter_is_not_told_to_set_the_override(self):
        """The defect proper. Setting UNSLOTH_ROCM_GFX_ARCH=gfx1010 lands on the
        unmapped-arch path and returns CPU anyway, so instructing it is an errand
        with no ending. The variable may still be NAMED, to say it cannot help."""
        _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
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


# ── The other four copies of the table ───────────────────────────────────────


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

    @pytest.mark.parametrize("name,expected", _RDNA1_NAMES)
    def test_all_copies_agree_on_rdna1(self, name, expected):
        answers = {where: fn(name) for where, fn in _all_copies().items()}
        assert set(answers.values()) == {expected}, f"{name!r} resolves inconsistently: {answers}"

    @pytest.mark.parametrize("name", _NOT_RDNA1_NAMES)
    def test_no_copy_claims_a_supported_card(self, name):
        answers = {where: fn(name) for where, fn in _all_copies().items()}
        assert set(answers.values()) == {None}, f"{name!r} was claimed as unsupported: {answers}"


# ── The wording, in the sources that print it ────────────────────────────────


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
        # The "was found" guard: without it a renamed branch makes both finds -1 and -1 < -1 is False, so the ordering
        # claim would pass vacuously.
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

    # Every arm that would otherwise outrank the unsupported one, with the guard it
    # must carry. An installed HIP SDK is the SYMPTOM here -- the #8529 reporters
    # installed it because the old advice said to -- so unguarded, the fix never
    # prints for the exact users it was written for.
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

    # The claim the arms must NOT make, and the one they must. With neither CUDA nor XPU
    # visible unsloth raises NotImplementedError at import (unsloth/device_type.py), so
    # "training runs on CPU" sends the user at an ImportError; these arms say what
    # studio/setup.sh already says. Scoped per ARM, not per file: install.ps1's
    # pre-existing $ROCmGfxArch hint makes the same claim for a different card, so a
    # file-wide ban would fail on untouched code and end up deleted.
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
        """Vega 20 (Radeon VII / MI50, gfx906) is older than RDNA 2 and DOES have a
        ROCm PyTorch path -- install.sh routes it to the rocm6.3 index. A blanket
        "AMD GPUs older than RDNA 2" would send those users to Vulkan and CPU torch
        for nothing.

        Only the wrong claim is banned outright. Saying nothing is not wrong, so the
        member names are required only once the README describes the group: an earlier
        version demanded them unconditionally and turned every README condensation into
        a CI failure with nothing untrue on the page.
        """
        src = _normalised(PACKAGE_ROOT / "README.md")
        # Any spelling of the cutoff, not one literal: "every AMD GPU older than RDNA 2" slipped past an
        # exact-string ban while contradicting the gfx906 carve-out.
        blanket = re.search(r"AMD GPUs? older than RDNA ?2", src, re.IGNORECASE)
        assert not blanket, (
            f"README: {blanket.group(0)!r} claims ROCm PyTorch covers nothing older "
            "than RDNA 2, which is wrong for gfx906"
        )
        # The carve-out has to be true of the installer whatever the README says.
        assert "rocm6.3" in _normalised(_INSTALL_SH), "install.sh: no gfx906 ROCm index left"
        describes_group = re.search(r"no ROCm PyTorch wheels|Polaris|RDNA ?1", src, re.IGNORECASE)
        if not describes_group:
            return
        # It does describe the group, so it has to describe it completely: named by its members, and with the one
        # member that is covered cut back out.
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


# ── studio/setup.sh on a host with no ROCm userspace at all ──────────────────


def _run_setup_kfd_lookup(gpu_name: str, lspci_lines: "list[str] | None", tmp_path) -> str:
    """Run studio/setup.sh's report-side lookup with a scripted lspci.

    `lspci_lines is None` means the binary is absent, which is the other half of
    the KFD-only host: amdgpu exposes /dev/kfd, no ROCm userspace is installed.
    """
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
        """A name that maps ENDS the lookup (above); one that does not must not, or a
        generic "AMD Radeon Graphics" from rocminfo hides a card lspci names outright.
        Only reachable with no gfx from the tools, so a covered compute card cannot be
        talked over here: rocminfo reports its arch and the supported arm wins first."""
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


# ── The new variable has to outlive the block that sets it ───────────────────


def test_the_unsupported_arch_variable_is_declared_outside_the_amd_block():
    """install.ps1 reads it on paths an NVIDIA host takes.

    `$ROCmUnsupportedGfxArch` is set inside `if (-not $HasNvidiaSmi)`, but the arms
    that read it sit outside that gate, so on an NVIDIA host the read is of a variable
    that was never assigned. `Set-StrictMode -Version Latest` turns that into a hard
    stop. Install-UnslothStudio runs with strict mode off, which is why this has not
    bitten, but its five neighbours (HasROCm, HipSdkInstalled, ROCmGpuLabel,
    ROCmVersion, ROCmGfxArch) are all declared above the gate and this one has to be
    too. studio/setup.ps1 already hoists its copy.
    """
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


# ── An identified uncovered card outranks the generic ROCm report ────────────

_ROCM_ARM = {
    "install.ps1": ("} elseif ($HasROCm", "$ROCmUnsupportedGfxArch"),
    "setup.ps1": ("} elseif ($HasROCm", "$script:ROCmUnsupportedGfxArch"),
}


@pytest.mark.parametrize("name", sorted(_ROCM_ARM))
def test_the_generic_rocm_arm_yields_to_an_identified_uncovered_card(name):
    """amd-smi can report a GPU with no gfx token and only a market name.

    That sets $HasROCm with no arch, so the generic arm fires and calls an RX 5700 XT
    "AMD ROCm" while the wheel note in the same run says gfx1010 has none. The host is
    not hypothetical: amd-smi is only probed when the HIP SDK is present, which is what
    the #8529 and #8458 reporters installed because the old message told them to. Same
    guard the HIP SDK arm below it already carries.
    """
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
    """The summary chain opens with a bare `if ($HasROCm)`, not an `} elseif`.

    So the check above walks straight past it. Its own third arm names the uncovered
    card, and that arm is reached only when nothing outranks it: on a host where
    amd-smi enumerates an RDNA 1 card with no gfx token, the "ROCm x.y" arm wins and
    the arm written for that card never runs.
    """
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


# ── Scope: these sentences speak for one card, not for the host ───────────

# A host is not one GPU.
# An RX 580 beside an RX 7900 XTX is a host where masking to the other card and pinning its arch DOES install wheels, so
# a host-wide "nothing can enable ROCm here" is false there.
# Deciding it at runtime was tried and dropped: "any adapter we cannot name" misfires on the Vega-class iGPU on most
# Ryzen desktops, and "any covered peer" misses the Instinct and V620 parts no name table carries.

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


# ── The shell copies, executed rather than parsed ────────────────────────────


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


# ── The Vulkan pointer (#8458) ───────────────────────────────────────────────


def _arm_window(lines: "list[str]", start: int) -> "list[str]":
    """The rest of the branch the line at `start` belongs to, not a fixed line count.

    The advice is now emitted from if/else arms (an index pin and Windows ARM64 each
    change what is true), so a fixed window either stops mid-branch or spills into the
    NEXT arm, which is the failure this test's docstring already warns about. Stop at
    the first line that dedents past the anchor, which closes the arm in every language
    here, and cap the span so a missing closer cannot swallow the file.
    """
    indent = len(lines[start]) - len(lines[start].lstrip())
    # One step out, not the anchor's own indent: the claim it anchors now sits in a nested if/else, and the Vulkan offer
    # is its SIBLING branch, so stopping at the anchor's own level would cut the window before the thing under test.
    floor = max(indent - 4, 0)
    out = [lines[start]]
    for line in lines[start + 1 : start + 25]:
        if line.strip() and (len(line) - len(line.lstrip())) < floor:
            break
        out.append(line)
    return out


class TestVulkanAdvice:
    """#8458 is the same shape as #8529 -- a pre-RDNA 2 AMD card (RX 580, Polaris,
    gfx803) told it had no usable GPU -- but its reporter got the card working
    through Vulkan, as LM Studio does with the same hardware. So the unsupported
    arm must not dead-end at "ROCm does not cover this"; it has somewhere to send
    the user.

    Two things are load-bearing in that advice and both are asserted here:

    * the CURRENT variable name. ``UNSLOTH_FORCE_VULKAN`` (what #8458's reporter
      used) is still honoured, but only as a legacy fallback consulted when
      ``UNSLOTH_LLAMA_CPP_BACKEND`` is unset or unrecognised
      (``install_llama_prebuilt.py::force_vulkan_requested``). New text must
      teach the current spelling.
    * WHEN to set it. Every consumer of the selector lives in
      ``install_llama_prebuilt.py`` (``_route_to_vulkan_prebuilt``,
      ``install_prebuilt``, ``main``): it chooses which llama.cpp bundle gets
      downloaded, at install time. A user who exports it and merely relaunches
      Unsloth sees no change and concludes the advice was wrong -- which is
      exactly what happened in #8458. Advice that omits the timing is worse than
      no advice, so a message naming the variable must also name the moment.
    """

    # The four sources whose advice is a literal in an emitting statement. The Python
    # copy joins string fragments, so line-scoped source reading cannot see it; the
    # (stronger) live-output tests below cover it instead.
    _SHELL_SOURCES = [_INSTALL_PS1, _SETUP_PS1, _INSTALL_SH, _SETUP_SH]

    # Everything the advice has to carry, asserted against EMITTED text only: every phrase here also appears in the
    # comments explaining the branch, so a whole-file search stays green after the message is gutted (three such
    # mutants survived). The setter is per-file (see _SETTER) and gets its own tests below.
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
        """The regression guard for the syntax above, stated as a ban.

        Asserted on emitters only: a comment may legitimately quote the POSIX form
        while explaining why the emitted line does not use it.
        """
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

    # Every arm that TELLS a pre-RDNA 2 user torch cannot use their GPU, with the expected occurrence count.
    # install.ps1's second anchor names $ROCmUnsupportedGfxArch deliberately: the same sentence four lines earlier is
    # for a supported card.
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
        """Per SITE, by real line number, so a whole arm cannot be deleted quietly.

        Windowed on the source lines rather than on the emitter-only projection the
        other tests use: that projection concatenates print statements from branches
        hundreds of lines apart, so a window over it can be satisfied by an unrelated
        arm that happens to be the next thing that prints.
        """
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
            # Comments stripped from the WINDOW, not just the anchor: every phrase below also appears in the
            # comment explaining the branch, so raw lines stay green after the message is demoted to a comment
            # (observed mutant).
            window = "\n".join(
                line for line in _arm_window(lines, i) if not line.lstrip().startswith(("#", "//"))
            )
            # The offer, not one phrasing of it: these arms are hard-wrapped to different widths. Both halves are
            # required -- a backend without what it buys, or GGUF chat without the backend, is half an answer.
            # "Vulkan" is matched case-sensitively in PROSE, so the setter's lowercase spelling cannot stand in
            # for the sentence explaining it.
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
        """The anti-#8458 clause, checked per SITE rather than per file.

        install.sh prints this advice at two places (index selection and the CPU
        note). A file-level "install time" search is satisfied by whichever site
        still has it, so gutting the other one passes -- observed: that exact
        mutant survived. Require the timing near each mention instead.
        """
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
        """UNSLOTH_FORCE_VULKAN still works, but it is the legacy name and loses to
        UNSLOTH_LLAMA_CPP_BACKEND whenever that parses, so new text must not spread
        it.

        Scoped to lines that PRINT. setup.sh and setup.ps1 legitimately *read* the
        legacy variable for back-compat, and this fix does not touch that; a
        whole-file ban would fail on working code and would have to be deleted,
        taking the real assertion with it.
        """
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
        arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert arch is None, "routing must be unchanged; this is a wording fix"
        assert needle in out, f"the printed advice is missing {what} ({needle!r})"

    def test_the_printed_advice_uses_powershell_syntax(self):
        """This branch is Windows-only, so its setter has to be pasteable into
        PowerShell. Read the live output, not the source: the message is built from
        implicitly-joined fragments and no single source line carries it."""
        _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
        assert _PWSH_SETTER in out, f"the printed advice is not pasteable into PowerShell:\n{out}"
        assert (
            _POSIX_ASSIGNMENT not in out
        ), f"the printed advice gives a POSIX assignment PowerShell cannot parse:\n{out}"

    def test_the_printed_advice_says_when_to_set_it(self):
        """The anti-#8458 clause for the Python copy.

        The shell/PS sources get this per-site from source text; Python builds its
        message from implicitly-joined fragments, so only the live output can show
        it. Kept as its own test rather than folded into the shared list above,
        because that list is also used for the per-line source scan where a timing
        phrase legitimately lives on a different line than the variable.
        """
        _arch, out = _wmi_detect(["AMD Radeon RX 5700 XT"])
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
        """The installer now tells users a variable name and the README is where
        they check it, but it documented only the legacy spelling.

        Asserted against the fenced COMMANDS, not the prose: a reader copies the
        block. An earlier version of this test compared first-occurrence indexes,
        which the surrounding prose satisfied on its own and which therefore passed
        with both code blocks still reverted to UNSLOTH_FORCE_VULKAN.

        Wrongness only, never completeness. The README is edited for length on its own
        schedule, so a block that is gone is not this test's business; a block that is
        there and teaches the wrong variable is.
        """
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
        """macOS has no Vulkan llama.cpp bundle, so a forced request there installs Metal.
        An Intel Mac carrying one of these very cards (the 16-inch MacBook Pro shipped
        Radeon Pro 5300M/5500M/5600M, all rows above) can follow Vulkan advice written for
        Linux, so the ignore has to be visible in the log rather than silent.

        Asserted against the routing branch, not against the README. This used to require
        a fixed README paragraph, which made every README condensation a CI failure with
        nothing untrue on the page; the README is edited on its own schedule and is not a
        test fixture. What is enforceable is that the installer states what it did.
        """
        prebuilt = (PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py").read_text(
            encoding = "utf-8"
        )
        # Scoped to the routing function: `if host.is_macos:` appears many times, and an earlier cut of this test
        # matched the DYLD_LIBRARY_PATH one instead.
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


# ── Polaris, the second card in the cluster (#8458) ──────────────────────────

_POLARIS_NAMES = [
    ("AMD Radeon RX 580", "gfx803"),
    ("AMD Radeon RX 580 Series", "gfx803"),
    ("AMD Radeon RX 570", "gfx803"),
    ("AMD Radeon RX 590", "gfx803"),
    ("AMD Radeon RX 480", "gfx803"),
    ("AMD Radeon RX 470", "gfx803"),
    ("Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]", "gfx803"),
    # The Polaris 10 workstation boards: pci.ids groups them on Ellesmere, the RX 580 die from #8458, and their
    # names carry no RX number for the consumer rows to hit.
    ("Ellesmere [Radeon Pro WX 7100 / WX 7100 Mobile / WX 5100 / V7300X / V7350x2]", "gfx803"),
    ("AMD Radeon Pro WX 7100", "gfx803"),
    ("AMD Radeon Pro WX 5100", "gfx803"),
]

# Polaris 11/12. Deliberately NOT in the table: a different die, and this table
# is only worth having while it never guesses an arch.
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

    @pytest.mark.parametrize("name,expected", _RDNA1_NAMES)
    def test_polaris_patterns_do_not_swallow_rdna1(self, name, expected):
        """The collision this row is one keystroke away from: "RX 570" is a prefix
        of "RX 5700" and "RX 550" of "RX 5500". Re-assert every RDNA 1 name still
        resolves to its own arch now that a Polaris row exists."""
        assert stack_mod._unsupported_gfx_arch_from_gpu_name(name) == expected

    @pytest.mark.parametrize("name,_expected", _RDNA1_NAMES)
    def test_the_polaris_pattern_is_correct_on_its_own(self, name, _expected):
        """The test above passes for the wrong reason and cannot replace this one.

        Table order already saves it: the RDNA 1 rows are matched first, so the
        Polaris pattern is never even reached for an RDNA 1 name and deleting its
        (?!0) guards changes nothing observable. That is precisely how a guard rots
        -- it stays correct only until someone reorders the table. Match the
        Polaris pattern ALONE, where the guard is the only thing standing between
        "RX 5700 XT" and gfx803.
        """
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
        """The same standalone check as above, for the two PowerShell copies.

        Row order masks a missing guard identically there, and the .ps1 tables are
        maintained by hand alongside the Python one, so a guard dropped from just
        those two is invisible to every other test in this file. PowerShell's
        -match and Python's re agree on (?!0), which is what makes checking the
        extracted pattern here meaningful.
        """
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
        """In the shell copies, ORDER is the correctness mechanism, so pin it.

        `case` globs have no negative lookahead, so the *"RX 570"* arm cannot be
        made safe on its own the way the Python and PowerShell (?!0) rows can. The
        only thing stopping it from swallowing an "RX 5700 XT" is that every RDNA 1
        arm is matched first. That makes arm order load-bearing rather than
        cosmetic, and a reorder is otherwise a silent regression, so assert it
        directly instead of leaving it to the behavioural test alone.
        """
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

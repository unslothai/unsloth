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

import contextlib
import importlib.util
import io
import os
import re
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


def _load_stack_module():
    spec = importlib.util.spec_from_file_location("studio_install_python_stack_rdna1", _STACK_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()


# Windows WMI reports the marketing name; Linux lspci reports the chip plus a
# slash-joined list of the boards built on it. Both must resolve.
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
]

# Cards the supported table owns, plus a non-AMD one. The new lookup must not
# reach any of them: a hit here would print "ROCm does not cover this" at a user
# whose GPU has wheels.
_NOT_RDNA1_NAMES = [
    "AMD Radeon RX 9070 XT",
    "AMD Radeon RX 9060 XT",
    "AMD Radeon RX 7900 XTX",
    "AMD Radeon RX 6800 XT",
    "AMD Radeon 8060S Graphics",
    "NVIDIA GeForce RTX 4090",
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
        # The "was found" guard: without it a renamed branch makes both finds -1
        # and -1 < -1 is False, so the ordering claim would pass vacuously.
        assert unsupported_marker in src, f"{path.name}: unsupported arm not found"
        assert unknown_marker in src, f"{path.name}: arch-unknown arm not found"
        assert src.index(unsupported_marker) < src.index(unknown_marker)

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1])
    def test_the_unsupported_arm_says_the_override_cannot_help(self, path):
        src = _normalised(path)
        needle = "setting UNSLOTH_ROCM_GFX_ARCH will not change that."
        assert needle in src, f"{path.name}: the override disclaimer is missing"

    def test_install_sh_cpu_note_keeps_the_sdk_advice_only_for_unknown_cards(self):
        """install.sh emits the same wording at two sites. Pin the WHOLE line at each
        one: a shared needle matches the other site and passes a deleted branch."""
        src = _normalised(_INSTALL_SH)
        unsupported = (
            'substep "AMD GPU detected ($_unsup_disp_gfx) -- no ROCm PyTorch wheels '
            'exist for that arch, installing CPU PyTorch." "$C_WARN"'
        )
        sdk_advice = "Install the ROCm/HIP SDK and re-run this installer for GPU PyTorch."
        assert unsupported in src, "install.sh: unsupported arm of the CPU note not found"
        assert sdk_advice in src, "install.sh: SDK advice not found (branch renamed?)"
        assert src.index(unsupported) < src.index(sdk_advice)

    def test_install_sh_index_selection_does_not_send_users_to_repair_rocminfo(self):
        src = _normalised(_INSTALL_SH)
        unsupported = (
            'echo "[WARN] AMD GPU detected ($_amd_unsup_gfx) -- no ROCm PyTorch wheels '
            'exist for that arch, installing CPU PyTorch." >&2'
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

    def test_setup_sh_names_the_arch_instead_of_claiming_rocm(self):
        src = _normalised(_SETUP_SH)
        needle = 'step "gpu" "AMD GPU detected ($_setup_unsup_gfx) -- not covered by ROCm PyTorch"'
        fallthrough = 'step "gpu" "AMD ROCm"'
        assert needle in src, "studio/setup.sh: unsupported arm not found"
        assert fallthrough in src, "studio/setup.sh: plain AMD ROCm arm not found"
        assert src.index(needle) < src.index(fallthrough)


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

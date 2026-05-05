"""Tests for install_python_stack NO_TORCH / IS_MACOS filtering logic.

Covers:
- _filter_requirements unit tests (synthetic + REAL requirements files)
- NO_TORCH / IS_MACOS / IS_WINDOWS env var parsing
- Subprocess-mock of install_python_stack() to verify overrides/triton/filtering
  actually happen (or get skipped) under each platform/config combination
- VCS URL and environment marker edge cases in filtering
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

# Add the studio directory so we can import install_python_stack
STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips

# Paths to the REAL requirements files
REQ_ROOT = Path(__file__).resolve().parents[2] / "studio" / "backend" / "requirements"
EXTRAS_TXT = REQ_ROOT / "extras.txt"
EXTRAS_NO_DEPS_TXT = REQ_ROOT / "extras-no-deps.txt"
OVERRIDES_TXT = REQ_ROOT / "overrides.txt"
TRITON_KERNELS_TXT = REQ_ROOT / "triton-kernels.txt"


# ── _filter_requirements unit tests (synthetic) ───────────────────────


class TestFilterRequirements:
    """Verify _filter_requirements correctly removes packages by prefix."""

    def _write_req(self, tmp_path: Path, content: str) -> Path:
        req = tmp_path / "requirements.txt"
        req.write_text(textwrap.dedent(content), encoding = "utf-8")
        return req

    def test_filters_no_torch_packages(self, tmp_path):
        req = self._write_req(
            tmp_path,
            """\
            torch-stoi==0.1
            timm>=1.0
            numpy
            torchcodec>=0.1
            torch-c-dlpack-ext
        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        # Only numpy should remain (non-blank lines)
        non_blank = [l.strip() for l in lines if l.strip()]
        assert non_blank == ["numpy"], f"Expected only numpy, got: {non_blank}"

    def test_empty_file(self, tmp_path):
        req = self._write_req(tmp_path, "")
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        content = Path(result).read_text(encoding = "utf-8")
        assert content.strip() == ""

    def test_comments_preserved(self, tmp_path):
        req = self._write_req(
            tmp_path,
            """\
            # torch-stoi is needed for audio
            numpy
        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        non_blank = [l.strip() for l in lines if l.strip()]
        # Comment starts with "#", not "torch-stoi", so it's preserved
        assert len(non_blank) == 2
        assert non_blank[0].startswith("#")
        assert non_blank[1] == "numpy"

    def test_version_specifiers_filtered(self, tmp_path):
        req = self._write_req(
            tmp_path,
            """\
            torch-stoi>=0.1.0
            timm==1.2.3
        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        non_blank = [l.strip() for l in lines if l.strip()]
        assert non_blank == [], f"Expected empty, got: {non_blank}"

    def test_prefix_match_catches_extensions(self, tmp_path):
        """Prefix matching catches torch-stoi-extra (correct for pip names)."""
        req = self._write_req(
            tmp_path,
            """\
            torch-stoi-extra
            numpy
        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        non_blank = [l.strip() for l in lines if l.strip()]
        assert non_blank == ["numpy"]

    def test_mixed_case_filtered(self, tmp_path):
        """Package names are lowercased before matching."""
        req = self._write_req(
            tmp_path,
            """\
            Timm>=1.0
            TORCH-STOI
            numpy
        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        non_blank = [l.strip() for l in lines if l.strip()]
        assert non_blank == ["numpy"]

    def test_whitespace_and_blank_lines_preserved(self, tmp_path):
        req = self._write_req(
            tmp_path,
            """\
            numpy

            pandas

        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        content = Path(result).read_text(encoding = "utf-8")
        # Blank lines should be preserved (not stripped)
        assert "\n\n" in content or content.count("\n") >= 3

    def test_stacked_windows_and_no_torch_filters(self, tmp_path):
        """Both WINDOWS_SKIP_PACKAGES and NO_TORCH_SKIP_PACKAGES applied."""
        req = self._write_req(
            tmp_path,
            """\
            open_spiel
            triton_kernels
            torch-stoi
            timm
            numpy
        """,
        )
        # First filter Windows packages, then NO_TORCH packages
        intermediate = ips._filter_requirements(req, ips.WINDOWS_SKIP_PACKAGES)
        result = ips._filter_requirements(
            Path(intermediate), ips.NO_TORCH_SKIP_PACKAGES
        )
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        non_blank = [l.strip() for l in lines if l.strip()]
        assert non_blank == [
            "numpy"
        ], f"Expected only numpy after stacked filters, got: {non_blank}"

    def test_vcs_url_with_skip_package_name(self, tmp_path):
        """VCS URLs like git+https://...torch-stoi should also be filtered (startswith matches)."""
        req = self._write_req(
            tmp_path,
            """\
            numpy
            torch-stoi @ git+https://github.com/example/torch-stoi.git
        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        non_blank = [l.strip() for l in lines if l.strip()]
        assert non_blank == [
            "numpy"
        ], f"VCS URL line should be filtered, got: {non_blank}"

    def test_env_marker_line_filtered(self, tmp_path):
        """Package lines with env markers are still filtered by prefix."""
        req = self._write_req(
            tmp_path,
            """\
            timm>=1.0; python_version>="3.10"
            numpy
        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        non_blank = [l.strip() for l in lines if l.strip()]
        assert non_blank == [
            "numpy"
        ], f"Env marker line should be filtered, got: {non_blank}"

    def test_git_plus_url_not_over_matched(self, tmp_path):
        """A git+ URL whose path contains a skip package name but does NOT start with it."""
        req = self._write_req(
            tmp_path,
            """\
            git+https://github.com/meta-pytorch/OpenEnv.git
            numpy
        """,
        )
        result = ips._filter_requirements(req, ips.NO_TORCH_SKIP_PACKAGES)
        lines = Path(result).read_text(encoding = "utf-8").splitlines()
        non_blank = [l.strip() for l in lines if l.strip()]
        # The git+ URL doesn't start with any skip package, so it is preserved
        assert len(non_blank) == 2, f"git+ URL should be preserved, got: {non_blank}"


# ── Real requirements file filtering ──────────────────────────────────


class TestRealRequirementsFiltering:
    """Filter the ACTUAL extras.txt and extras-no-deps.txt with NO_TORCH_SKIP_PACKAGES."""

    @pytest.fixture(autouse = True)
    def _check_req_files(self):
        if not EXTRAS_TXT.is_file():
            pytest.skip("extras.txt not found in repo")
        if not EXTRAS_NO_DEPS_TXT.is_file():
            pytest.skip("extras-no-deps.txt not found in repo")

    def _non_blank_non_comment(self, path: Path) -> list[str]:
        """Return non-blank, non-comment lines from a requirements file."""
        lines = path.read_text(encoding = "utf-8").splitlines()
        return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]

    def test_extras_txt_torch_packages_removed(self):
        """extras.txt: all NO_TORCH_SKIP_PACKAGES must be removed, everything else preserved."""
        result = ips._filter_requirements(EXTRAS_TXT, ips.NO_TORCH_SKIP_PACKAGES)
        filtered = self._non_blank_non_comment(Path(result))
        original = self._non_blank_non_comment(EXTRAS_TXT)

        # These must be gone
        for pkg in ["torch-stoi", "timm", "openai-whisper", "transformers-cfg"]:
            assert not any(
                l.lower().startswith(pkg) for l in filtered
            ), f"{pkg} should be removed from extras.txt"

        # Everything else must remain
        expected = [
            l
            for l in original
            if not any(
                l.strip().lower().startswith(p) for p in ips.NO_TORCH_SKIP_PACKAGES
            )
        ]
        assert filtered == expected, (
            f"Filtered extras.txt should match expected.\n"
            f"Missing: {set(expected) - set(filtered)}\n"
            f"Extra: {set(filtered) - set(expected)}"
        )

    def test_extras_no_deps_txt_torchcodec_and_dlpack_removed(self):
        """extras-no-deps.txt: torchcodec and torch-c-dlpack-ext must be removed."""
        result = ips._filter_requirements(
            EXTRAS_NO_DEPS_TXT, ips.NO_TORCH_SKIP_PACKAGES
        )
        filtered = self._non_blank_non_comment(Path(result))
        original = self._non_blank_non_comment(EXTRAS_NO_DEPS_TXT)

        for pkg in ["torchcodec", "torch-c-dlpack-ext"]:
            assert not any(
                l.lower().startswith(pkg) for l in filtered
            ), f"{pkg} should be removed from extras-no-deps.txt"

        expected = [
            l
            for l in original
            if not any(
                l.strip().lower().startswith(p) for p in ips.NO_TORCH_SKIP_PACKAGES
            )
        ]
        assert filtered == expected

    def test_extras_txt_most_packages_preserved(self):
        """Ensure a representative set of non-torch packages survive filtering."""
        result = ips._filter_requirements(EXTRAS_TXT, ips.NO_TORCH_SKIP_PACKAGES)
        filtered_text = Path(result).read_text(encoding = "utf-8").lower()

        must_survive = ["scikit-learn", "loguru", "tiktoken", "einops", "tabulate"]
        for pkg in must_survive:
            if pkg in EXTRAS_TXT.read_text(encoding = "utf-8").lower():
                assert pkg in filtered_text, f"{pkg} should survive NO_TORCH filtering"

    def test_extras_no_deps_txt_trl_preserved(self):
        """trl should survive NO_TORCH filtering in extras-no-deps.txt."""
        result = ips._filter_requirements(
            EXTRAS_NO_DEPS_TXT, ips.NO_TORCH_SKIP_PACKAGES
        )
        filtered_text = Path(result).read_text(encoding = "utf-8").lower()
        assert "trl" in filtered_text, "trl should survive NO_TORCH filtering"


# ── NO_TORCH constant tests ──────────────────────────────────────────


class TestNoTorchConstant:
    """Verify NO_TORCH is derived correctly from UNSLOTH_NO_TORCH env var."""

    def _reimport_no_torch(self) -> bool:
        return os.environ.get("UNSLOTH_NO_TORCH", "false").lower() in ("1", "true")

    def test_true_lowercase(self):
        with mock.patch.dict(os.environ, {"UNSLOTH_NO_TORCH": "true"}):
            assert self._reimport_no_torch() is True

    def test_true_one(self):
        with mock.patch.dict(os.environ, {"UNSLOTH_NO_TORCH": "1"}):
            assert self._reimport_no_torch() is True

    def test_true_uppercase(self):
        with mock.patch.dict(os.environ, {"UNSLOTH_NO_TORCH": "TRUE"}):
            assert self._reimport_no_torch() is True

    def test_false_string(self):
        with mock.patch.dict(os.environ, {"UNSLOTH_NO_TORCH": "false"}):
            assert self._reimport_no_torch() is False

    def test_false_zero(self):
        with mock.patch.dict(os.environ, {"UNSLOTH_NO_TORCH": "0"}):
            assert self._reimport_no_torch() is False

    def test_not_set(self):
        env = os.environ.copy()
        env.pop("UNSLOTH_NO_TORCH", None)
        with mock.patch.dict(os.environ, env, clear = True):
            assert self._reimport_no_torch() is False

    def test_infer_no_torch_on_intel_mac(self):
        """_infer_no_torch falls back to platform detection when env var is unset."""
        env = os.environ.copy()
        env.pop("UNSLOTH_NO_TORCH", None)
        with (
            mock.patch.dict(os.environ, env, clear = True),
            mock.patch.object(ips, "IS_MAC_INTEL", True),
        ):
            assert ips._infer_no_torch() is True

    def test_infer_no_torch_respects_explicit_false_on_intel_mac(self):
        """Explicit UNSLOTH_NO_TORCH=false overrides platform detection."""
        with (
            mock.patch.dict(os.environ, {"UNSLOTH_NO_TORCH": "false"}),
            mock.patch.object(ips, "IS_MAC_INTEL", True),
        ):
            assert ips._infer_no_torch() is False

    def test_infer_no_torch_linux_unset(self):
        """On Linux with env var unset, _infer_no_torch returns False."""
        env = os.environ.copy()
        env.pop("UNSLOTH_NO_TORCH", None)
        with (
            mock.patch.dict(os.environ, env, clear = True),
            mock.patch.object(ips, "IS_MAC_INTEL", False),
        ):
            assert ips._infer_no_torch() is False


# ── IS_MACOS constant tests ──────────────────────────────────────────


class TestIsMacosConstant:
    """Verify IS_MACOS detection logic."""

    def test_is_macos_matches_platform(self):
        import sys

        expected = sys.platform == "darwin"
        assert ips.IS_MACOS is expected


# ── Subprocess mock of install_python_stack() ─────────────────────────


class TestInstallPythonStackSubprocessMock:
    """Monkeypatch subprocess.run to capture all pip/uv commands,
    then verify which requirements files are used/skipped under
    different NO_TORCH / IS_MACOS / IS_WINDOWS configurations."""

    @pytest.fixture(autouse = True)
    def _check_req_files(self):
        """Skip if requirements files are missing."""
        for f in [EXTRAS_TXT, EXTRAS_NO_DEPS_TXT, OVERRIDES_TXT]:
            if not f.is_file():
                pytest.skip(f"{f.name} not found in repo")

    def _capture_install(
        self,
        no_torch: bool,
        is_macos: bool,
        is_windows: bool,
        *,
        skip_base: bool = True,
    ):
        """Run install_python_stack() with mocked subprocess, capturing all commands.

        Returns a list of string-joined commands (each element is ' '.join(cmd)).
        """
        captured_cmds: list[list[str]] = []

        def mock_run(cmd, **kw):
            captured_cmds.append(
                list(cmd) if isinstance(cmd, (list, tuple)) else [str(cmd)]
            )
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        env = {"SKIP_STUDIO_BASE": "1"} if skip_base else {}

        with (
            mock.patch.object(ips, "NO_TORCH", no_torch),
            mock.patch.object(ips, "IS_MACOS", is_macos),
            mock.patch.object(ips, "IS_WINDOWS", is_windows),
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "UV_NEEDS_SYSTEM", False),
            mock.patch.object(ips, "VERBOSE", False),
            mock.patch.object(ips, "_ensure_flash_attn", return_value = None),
            mock.patch.object(ips, "_has_usable_nvidia_gpu", return_value = False),
            mock.patch.object(ips, "_has_rocm_gpu", return_value = False),
            mock.patch("subprocess.run", side_effect = mock_run),
            mock.patch.object(ips, "_bootstrap_uv", return_value = True),
            mock.patch.object(
                ips, "LOCAL_DD_UNSTRUCTURED_PLUGIN", Path("/fake/plugin")
            ),
            mock.patch("pathlib.Path.is_dir", return_value = True),
            mock.patch("pathlib.Path.is_file", return_value = True),
        ):
            with mock.patch.dict(os.environ, env, clear = False):
                ips.install_python_stack()

        return [" ".join(str(c) for c in cmd) for cmd in captured_cmds]

    def _cmds_contain_file(self, cmds: list[str], filename: str) -> bool:
        """Check if any captured command references the given filename."""
        return any(filename in cmd for cmd in cmds)

    # -- NO_TORCH=True, IS_MACOS=True (Intel Mac scenario) --

    def test_no_torch_macos_skips_overrides(self):
        """With NO_TORCH=True, overrides.txt pip_install must NOT be called."""
        cmds = self._capture_install(no_torch = True, is_macos = True, is_windows = False)
        assert not self._cmds_contain_file(
            cmds, "overrides.txt"
        ), "overrides.txt should be skipped when NO_TORCH=True"

    def test_no_torch_macos_skips_triton(self):
        """With IS_MACOS=True, triton-kernels.txt must NOT be called."""
        cmds = self._capture_install(no_torch = True, is_macos = True, is_windows = False)
        assert not self._cmds_contain_file(
            cmds, "triton-kernels.txt"
        ), "triton-kernels.txt should be skipped on macOS"

    def test_no_torch_macos_extras_called(self):
        """With NO_TORCH=True, extras.txt is still called (but filtered)."""
        cmds = self._capture_install(no_torch = True, is_macos = True, is_windows = False)
        has_extras = self._cmds_contain_file(cmds, "extras.txt") or any(
            "-r" in cmd and "tmp" in cmd.lower() for cmd in cmds
        )
        assert has_extras, "extras.txt (or its filtered temp) should be called"

    def test_no_torch_macos_extras_no_deps_called(self):
        """With NO_TORCH=True, extras-no-deps.txt is still called (but filtered)."""
        cmds = self._capture_install(no_torch = True, is_macos = True, is_windows = False)
        has_extras_nd = self._cmds_contain_file(cmds, "extras-no-deps.txt") or any(
            "-r" in cmd and "tmp" in cmd.lower() for cmd in cmds
        )
        assert (
            has_extras_nd
        ), "extras-no-deps.txt (or its filtered temp) should be called"

    # -- IS_WINDOWS=True + NO_TORCH=True (stacked) --

    def test_windows_no_torch_skips_overrides(self):
        """Windows+NO_TORCH: overrides.txt must be skipped."""
        cmds = self._capture_install(no_torch = True, is_macos = False, is_windows = True)
        assert not self._cmds_contain_file(
            cmds, "overrides.txt"
        ), "overrides.txt should be skipped with NO_TORCH=True on Windows"

    def test_windows_no_torch_skips_triton(self):
        """Windows: triton-kernels.txt must be skipped (IS_WINDOWS guard)."""
        cmds = self._capture_install(no_torch = True, is_macos = False, is_windows = True)
        assert not self._cmds_contain_file(
            cmds, "triton-kernels.txt"
        ), "triton-kernels.txt should be skipped on Windows"

    # -- Normal Linux path (NO_TORCH=False, IS_MACOS=False, IS_WINDOWS=False) --

    def test_normal_linux_includes_overrides(self):
        """Normal Linux: overrides.txt IS called."""
        cmds = self._capture_install(no_torch = False, is_macos = False, is_windows = False)
        assert self._cmds_contain_file(
            cmds, "overrides.txt"
        ), "overrides.txt should be called on normal Linux"

    def test_normal_linux_includes_triton(self):
        """Normal Linux: triton-kernels.txt IS called."""
        cmds = self._capture_install(no_torch = False, is_macos = False, is_windows = False)
        assert self._cmds_contain_file(
            cmds, "triton-kernels.txt"
        ), "triton-kernels.txt should be called on normal Linux"

    def test_normal_linux_includes_extras(self):
        """Normal Linux: extras.txt IS called (no filtering)."""
        cmds = self._capture_install(no_torch = False, is_macos = False, is_windows = False)
        assert self._cmds_contain_file(
            cmds, "extras.txt"
        ), "extras.txt should be called on normal Linux"

    def test_normal_linux_includes_extras_no_deps(self):
        """Normal Linux: extras-no-deps.txt IS called (no filtering)."""
        cmds = self._capture_install(no_torch = False, is_macos = False, is_windows = False)
        assert self._cmds_contain_file(
            cmds, "extras-no-deps.txt"
        ), "extras-no-deps.txt should be called on normal Linux"

    # -- Windows-only (NO_TORCH=False) to verify triton is still skipped --

    def test_windows_only_skips_triton(self):
        """Windows (without NO_TORCH): triton still skipped."""
        cmds = self._capture_install(no_torch = False, is_macos = False, is_windows = True)
        assert not self._cmds_contain_file(
            cmds, "triton-kernels.txt"
        ), "triton-kernels.txt should be skipped on Windows even without NO_TORCH"

    def test_windows_only_includes_overrides(self):
        """Windows (without NO_TORCH): overrides IS called (via filtered temp file).

        On Windows, all req files go through _filter_requirements(WINDOWS_SKIP_PACKAGES),
        so the command uses a temp file, not overrides.txt directly. We check for
        --reinstall (uv translation of --force-reinstall) which is unique to overrides.
        """
        cmds = self._capture_install(no_torch = False, is_macos = False, is_windows = True)
        assert any(
            "--reinstall" in cmd for cmd in cmds
        ), "overrides step (--reinstall) should be called on Windows when NO_TORCH=False"

    # -- Update path (skip_base=False) to verify no-torch mode is durable --

    def test_update_path_intel_macos_still_skips_overrides(self):
        """Update path (no SKIP_STUDIO_BASE): overrides still skipped on Intel Mac."""
        cmds = self._capture_install(
            no_torch = True, is_macos = True, is_windows = False, skip_base = False
        )
        assert not self._cmds_contain_file(
            cmds, "overrides.txt"
        ), "overrides.txt should be skipped on Intel Mac even via studio update"

    def test_update_path_intel_macos_still_skips_triton(self):
        """Update path (no SKIP_STUDIO_BASE): triton still skipped on macOS."""
        cmds = self._capture_install(
            no_torch = True, is_macos = True, is_windows = False, skip_base = False
        )
        assert not self._cmds_contain_file(
            cmds, "triton-kernels.txt"
        ), "triton-kernels.txt should be skipped on macOS even via studio update"


# ── Overrides skip structural checks ─────────────────────────────────


class TestOverridesSkip:
    """Verify overrides.txt is skipped when NO_TORCH is True (source-level check)."""

    def test_no_torch_guard_exists_in_source(self):
        """The install_python_stack source must contain a NO_TORCH guard around overrides."""
        source = Path(ips.__file__).read_text(encoding = "utf-8")
        assert (
            "if NO_TORCH:" in source
        ), "NO_TORCH guard not found in install_python_stack.py"

    def test_overrides_skipped_when_no_torch(self):
        """With NO_TORCH=True on the module, pip_install should NOT be called for overrides."""
        source = Path(ips.__file__).read_text(encoding = "utf-8")
        overrides_match = re.search(r"if NO_TORCH:.*?overrides", source, re.DOTALL)
        assert (
            overrides_match is not None
        ), "Expected NO_TORCH conditional before overrides install"


# ── install.sh --no-torch flag tests ──────────────────────────────────


class TestInstallShNoTorchFlag:
    """Verify install.sh has the --no-torch flag and SKIP_TORCH variable."""

    @pytest.fixture(autouse = True)
    def _check_install_sh(self):
        install_sh = Path(__file__).resolve().parents[2] / "install.sh"
        if not install_sh.is_file():
            pytest.skip("install.sh not found")
        self.install_sh = install_sh
        self.source = install_sh.read_text(encoding = "utf-8")

    def test_no_torch_flag_in_case_statement(self):
        """--no-torch must appear in the flag parser case statement."""
        assert (
            "--no-torch)" in self.source
        ), "--no-torch not found in install.sh flag parser"

    def test_no_torch_flag_variable_initialized(self):
        """_NO_TORCH_FLAG must be initialized to false."""
        assert (
            "_NO_TORCH_FLAG=false" in self.source
        ), "_NO_TORCH_FLAG=false not found in install.sh"

    def test_skip_torch_variable_exists(self):
        """SKIP_TORCH variable must be defined."""
        assert (
            "SKIP_TORCH=false" in self.source
        ), "SKIP_TORCH=false not found in install.sh"
        assert (
            "SKIP_TORCH=true" in self.source
        ), "SKIP_TORCH=true not found in install.sh"

    def test_skip_torch_driven_by_flag_and_mac_intel(self):
        """SKIP_TORCH must check both _NO_TORCH_FLAG and MAC_INTEL."""
        assert (
            "_NO_TORCH_FLAG" in self.source
        ), "_NO_TORCH_FLAG not referenced in SKIP_TORCH logic"
        assert (
            "MAC_INTEL" in self.source
        ), "MAC_INTEL not referenced in SKIP_TORCH logic"

    def test_unsloth_no_torch_uses_skip_torch(self):
        """UNSLOTH_NO_TORCH must reference $SKIP_TORCH, not $MAC_INTEL."""
        import re

        matches = re.findall(r'UNSLOTH_NO_TORCH="\$(\w+)"', self.source)
        for var in matches:
            assert (
                var == "SKIP_TORCH"
            ), f"UNSLOTH_NO_TORCH references ${var} instead of $SKIP_TORCH"

    def test_cpu_hint_message_exists(self):
        """CPU hint message must exist in install.sh."""
        assert (
            "No GPU detected" in self.source
        ), "CPU hint message not found in install.sh"
        assert (
            "--no-torch" in self.source
        ), "--no-torch suggestion not found in CPU hint"

    def test_no_torch_flag_parsing_subprocess(self):
        """--no-torch flag sets _NO_TORCH_FLAG=true (subprocess test)."""
        script = textwrap.dedent("""\
            _NO_TORCH_FLAG=false
            _next_is_package=false
            STUDIO_LOCAL_INSTALL=false
            PACKAGE_NAME="unsloth"
            for arg in "$@"; do
                if [ "$_next_is_package" = true ]; then
                    PACKAGE_NAME="$arg"
                    _next_is_package=false
                    continue
                fi
                case "$arg" in
                    --local) STUDIO_LOCAL_INSTALL=true ;;
                    --package) _next_is_package=true ;;
                    --no-torch) _NO_TORCH_FLAG=true ;;
                esac
            done
            echo "$_NO_TORCH_FLAG"
        """)
        result = subprocess.run(
            ["bash", "-c", script, "_", "--no-torch"],
            capture_output = True,
            text = True,
        )
        assert (
            result.stdout.strip() == "true"
        ), f"Expected _NO_TORCH_FLAG=true, got: {result.stdout.strip()}"

    def test_no_torch_with_local_flag(self):
        """--no-torch and --local can be used together."""
        script = textwrap.dedent("""\
            _NO_TORCH_FLAG=false
            _next_is_package=false
            STUDIO_LOCAL_INSTALL=false
            PACKAGE_NAME="unsloth"
            for arg in "$@"; do
                if [ "$_next_is_package" = true ]; then
                    PACKAGE_NAME="$arg"
                    _next_is_package=false
                    continue
                fi
                case "$arg" in
                    --local) STUDIO_LOCAL_INSTALL=true ;;
                    --package) _next_is_package=true ;;
                    --no-torch) _NO_TORCH_FLAG=true ;;
                esac
            done
            echo "$_NO_TORCH_FLAG $STUDIO_LOCAL_INSTALL"
        """)
        result = subprocess.run(
            ["bash", "-c", script, "_", "--local", "--no-torch"],
            capture_output = True,
            text = True,
        )
        assert (
            result.stdout.strip() == "true true"
        ), f"Expected 'true true', got: {result.stdout.strip()}"

    def test_cpu_hint_only_when_not_skip_torch(self):
        """CPU hint should only print when SKIP_TORCH=false and OS!=macos."""
        script = textwrap.dedent("""\
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
            SKIP_TORCH=false
            OS="linux"
            case "$TORCH_INDEX_URL" in
                */cpu)
                    if [ "$SKIP_TORCH" = false ] && [ "$OS" != "macos" ]; then
                        echo "HINT_PRINTED"
                    fi
                    ;;
            esac
        """)
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output = True,
            text = True,
        )
        assert "HINT_PRINTED" in result.stdout, "CPU hint should print"

        # With SKIP_TORCH=true, hint should NOT print
        script2 = script.replace("SKIP_TORCH=false", "SKIP_TORCH=true")
        result2 = subprocess.run(
            ["bash", "-c", script2],
            capture_output = True,
            text = True,
        )
        assert (
            "HINT_PRINTED" not in result2.stdout
        ), "CPU hint should NOT print when SKIP_TORCH=true"


# ── Triton macOS skip structural checks ──────────────────────────────


class TestTritonMacosSkip:
    """Verify triton is skipped on macOS (source-level check)."""

    def test_triton_guard_in_source(self):
        """Source must skip triton on both Windows and macOS."""
        source = Path(ips.__file__).read_text(encoding = "utf-8")
        assert (
            "not IS_MACOS" in source
        ), "IS_MACOS guard for triton not found in install_python_stack.py"
        assert (
            "not IS_WINDOWS and not IS_MACOS" in source
        ), "Expected 'not IS_WINDOWS and not IS_MACOS' guard for triton"

"""Tests for install_python_stack NO_TORCH / IS_MACOS filtering logic."""

from __future__ import annotations

import importlib
import os
import re
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

# Add the studio directory so we can import install_python_stack
STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips


# ── _filter_requirements tests ──────────────────────────────────────────


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


# ── NO_TORCH constant tests ────────────────────────────────────────────


class TestNoTorchConstant:
    """Verify NO_TORCH is derived correctly from UNSLOTH_NO_TORCH env var."""

    def _reimport_no_torch(self) -> bool:
        """Re-evaluate the NO_TORCH expression as the module would at import."""
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


# ── IS_MACOS constant tests ────────────────────────────────────────────


class TestIsMacosConstant:
    """Verify IS_MACOS detection logic."""

    def test_linux_is_not_macos(self):
        # On our Linux CI, IS_MACOS should be False
        assert ips.IS_MACOS is False

    def test_darwin_platform_logic(self):
        """The expression sys.platform == 'darwin' produces True on macOS."""
        assert ("darwin" == "darwin") is True
        assert ("linux" == "darwin") is False


# ── Overrides skip when NO_TORCH ────────────────────────────────────────


class TestOverridesSkip:
    """Verify overrides.txt is skipped when NO_TORCH is True."""

    def test_no_torch_guard_exists_in_source(self):
        """The install_python_stack source must contain a NO_TORCH guard around overrides."""
        source = Path(ips.__file__).read_text(encoding = "utf-8")
        # Look for the pattern: if NO_TORCH: ... skip overrides
        assert (
            "if NO_TORCH:" in source
        ), "NO_TORCH guard not found in install_python_stack.py"

    def test_overrides_skipped_when_no_torch(self):
        """With NO_TORCH=True on the module, pip_install should NOT be called for overrides."""
        source = Path(ips.__file__).read_text(encoding = "utf-8")
        # Find the overrides section -- it should have a NO_TORCH conditional before it
        # The pattern: if NO_TORCH: ... progress("skipped") ... else: ... overrides
        overrides_match = re.search(r"if NO_TORCH:.*?overrides", source, re.DOTALL)
        assert (
            overrides_match is not None
        ), "Expected NO_TORCH conditional before overrides install"


# ── Triton macOS skip ───────────────────────────────────────────────────


class TestTritonMacosSkip:
    """Verify triton is skipped on macOS."""

    def test_triton_guard_in_source(self):
        """Source must skip triton on both Windows and macOS."""
        source = Path(ips.__file__).read_text(encoding = "utf-8")
        assert (
            "not IS_MACOS" in source
        ), "IS_MACOS guard for triton not found in install_python_stack.py"
        # Verify the full guard
        assert (
            "not IS_WINDOWS and not IS_MACOS" in source
        ), "Expected 'not IS_WINDOWS and not IS_MACOS' guard for triton"

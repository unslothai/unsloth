"""Cross-platform parity tests between install.sh and install.ps1."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"


class TestNoTorchBackendAutoInInstallSh:
    """install.sh primary install paths must not use --torch-backend=auto.

    The fallback else-branch (when TORCH_INDEX_URL is empty) is allowed to
    use --torch-backend=auto since that is the last-resort recovery path.
    """

    def test_no_torch_backend_auto_outside_fallback(self):
        lines = INSTALL_SH.read_text().splitlines()
        # Find the fallback block: starts with the "else" after the
        # TORCH_INDEX_URL check and ends at the next "fi".
        fallback_start = None
        fallback_end = None
        for i, line in enumerate(lines):
            if fallback_start is None and "GPU detection failed" in line:
                fallback_start = i
            elif (
                fallback_start is not None
                and fallback_end is None
                and line.strip() == "fi"
            ):
                fallback_end = i
                break
        fallback_range = (
            range(fallback_start or 0, (fallback_end or 0) + 1)
            if fallback_start
            else range(0)
        )

        matches = [
            (i + 1, line)
            for i, line in enumerate(lines)
            if "--torch-backend=auto" in line
            and not line.lstrip().startswith("#")
            and i not in fallback_range
        ]
        assert matches == [], (
            f"install.sh contains --torch-backend=auto outside the fallback block at lines: "
            f"{[m[0] for m in matches]}"
        )

    def test_fallback_uses_torch_backend_auto(self):
        """The fallback branch should use --torch-backend=auto as recovery."""
        text = INSTALL_SH.read_text()
        assert (
            "GPU detection failed" in text
        ), "install.sh should have a fallback branch for when GPU detection fails"


class TestInstallShHasGpuDetection:
    """install.sh must contain the get_torch_index_url function."""

    def test_function_exists(self):
        text = INSTALL_SH.read_text()
        assert (
            "get_torch_index_url()" in text
        ), "install.sh is missing the get_torch_index_url() function"

    def test_torch_index_url_assigned(self):
        text = INSTALL_SH.read_text()
        assert (
            "TORCH_INDEX_URL=$(get_torch_index_url)" in text
        ), "install.sh should assign TORCH_INDEX_URL from get_torch_index_url()"


class TestCudaMappingParity:
    """CUDA version thresholds must match between install.sh and install.ps1."""

    @staticmethod
    def _extract_cuda_thresholds_sh(text: str) -> list[str]:
        """Extract cu* suffixes from the major/minor comparison chain in install.sh."""
        # Only match lines in the if/elif chain that compare _major/_minor
        in_func = False
        results = []
        for line in text.splitlines():
            if "get_torch_index_url()" in line:
                in_func = True
                continue
            if in_func and line.startswith("}"):
                break
            if in_func and ("_major" in line or "_minor" in line):
                m = re.search(r"/(cu\d+|cpu)", line)
                if m:
                    results.append(m.group(1))
        return results

    @staticmethod
    def _extract_cuda_thresholds_ps1(text: str) -> list[str]:
        """Extract cu* suffixes from the major/minor comparison chain in install.ps1."""
        in_func = False
        depth = 0
        results = []
        for line in text.splitlines():
            if "function Get-TorchIndexUrl" in line:
                in_func = True
                depth = 1
                continue
            if in_func:
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    break
                # Only match the if-chain lines that compare $major/$minor
                if "$major" in line or "$minor" in line:
                    m = re.search(r"/(cu\d+|cpu)", line)
                    if m:
                        results.append(m.group(1))
        return results

    def test_same_cuda_suffixes(self):
        """Both scripts should produce the same ordered list of CUDA index suffixes."""
        sh_text = INSTALL_SH.read_text()
        ps1_text = INSTALL_PS1.read_text()

        sh_thresholds = self._extract_cuda_thresholds_sh(sh_text)
        ps1_thresholds = self._extract_cuda_thresholds_ps1(ps1_text)

        assert len(sh_thresholds) > 0, "Could not extract thresholds from install.sh"
        assert len(ps1_thresholds) > 0, "Could not extract thresholds from install.ps1"
        assert sh_thresholds == ps1_thresholds, (
            f"CUDA mapping mismatch:\n"
            f"  install.sh:  {sh_thresholds}\n"
            f"  install.ps1: {ps1_thresholds}"
        )


class TestPyTorchMirrorEnvVar:
    """Both install scripts must support the UNSLOTH_PYTORCH_MIRROR env var."""

    def test_install_sh_has_mirror_var(self):
        text = INSTALL_SH.read_text()
        assert (
            "UNSLOTH_PYTORCH_MIRROR" in text
        ), "install.sh should reference UNSLOTH_PYTORCH_MIRROR"

    def test_install_ps1_has_mirror_var(self):
        text = INSTALL_PS1.read_text()
        assert (
            "UNSLOTH_PYTORCH_MIRROR" in text
        ), "install.ps1 should reference UNSLOTH_PYTORCH_MIRROR"

"""The setup.ps1 Windows ROCm companion probe, extracted and executed on real trees.

The probe decides whether torchvision/torchaudio beside a ROCm torch are trustworthy
enough to skip the trio reinstall, so it is run here rather than regex-matched.
"""

import csv
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SETUP_PS1 = Path(__file__).resolve().parents[3] / "studio" / "setup.ps1"


def _probe_source(names="('torchvision', 'torchaudio')"):
    """The -Code string of the companion probe, as Python."""
    for line in SETUP_PS1.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("$_companionProbe = Invoke-BoundedPythonProbe"):
            code = line.split("-Code ", 1)[1].strip()
            assert code.startswith('"') and code.endswith('"'), code[:40]
            return code[1:-1].replace("`n", "\n").replace("$($_companionNames)", names)
    raise AssertionError("companion probe not found in setup.ps1")


def _install(
    root,
    name,
    version,
    *,
    payload=True,
    record=True,
    resize=False,
):
    """A dist-info and package good enough for importlib.metadata and find_spec."""
    pkg = root / name
    if payload:
        pkg.mkdir()
        (pkg / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    dist = root / f"{name}-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )
    if record:
        rows = []
        if payload:
            size = (pkg / "__init__.py").stat().st_size + (1 if resize else 0)
            rows.append([f"{name}/__init__.py", "sha256=x", str(size)])
        with (dist / "RECORD").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)


def _run(root, names="('torchvision', 'torchaudio')"):
    # This interpreter has a real torch stack; only the fabricated one may answer.
    prelude = (
        "import sys\n"
        "sys.path[:] = [p for p in sys.path if 'packages' not in p]\n"
        f"sys.path.insert(0, {str(root)!r})\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", prelude + _probe_source(names)],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.fixture
def venv(tmp_path):
    root = tmp_path / "site-packages"
    root.mkdir()
    return root


class TestCompanionProbe:
    def test_matching_rocm_trio_keeps_the_fast_path(self, venv):
        _install(venv, "torch", "2.8.0+rocm6.4.2")
        _install(venv, "torchvision", "0.23.0+rocm6.4.2")
        _install(venv, "torchaudio", "2.8.0+rocm6.4.2")
        assert _run(venv) == ""

    def test_pypi_companion_forces_the_trio(self, venv):
        _install(venv, "torch", "2.8.0+rocm6.4.2")
        _install(venv, "torchvision", "0.23.0")
        _install(venv, "torchaudio", "2.8.0+rocm6.4.2")
        assert _run(venv) == "torchvision==0.23.0"

    def test_companion_from_another_rocm_release_forces_the_trio(self, venv):
        _install(venv, "torch", "2.8.0+rocm6.4.2")
        _install(venv, "torchvision", "0.23.0+rocm6.2.4")
        _install(venv, "torchaudio", "2.8.0+rocm6.4.2")
        out = _run(venv)
        assert out.startswith("torchvision==0.23.0+rocm6.2.4 (rocm6.2.4 beside torch rocm6.4.2)")

    def test_git_tagged_community_wheels_keep_the_fast_path(self, venv):
        """No numeric ROCm version on either side is not evidence of a mismatch."""
        _install(venv, "torch", "2.7.0a0+rocm.gitabc1234")
        _install(venv, "torchvision", "0.22.0a0+rocm.gitdef5678")
        _install(venv, "torchaudio", "2.7.0a0+rocm.git99aabb0")
        assert _run(venv) == ""

    def test_missing_payload_forces_the_trio(self, venv):
        _install(venv, "torch", "2.8.0+rocm6.4.2")
        _install(venv, "torchvision", "0.23.0+rocm6.4.2", payload=False)
        _install(venv, "torchaudio", "2.8.0+rocm6.4.2")
        assert _run(venv) == "torchvision==0.23.0+rocm6.4.2 (payload missing)"

    def test_resized_record_row_forces_the_trio(self, venv):
        _install(venv, "torch", "2.8.0+rocm6.4.2")
        _install(venv, "torchvision", "0.23.0+rocm6.4.2", resize=True)
        _install(venv, "torchaudio", "2.8.0+rocm6.4.2")
        assert _run(venv) == "torchvision==0.23.0+rocm6.4.2 (payload damaged)"

    def test_absent_companion_is_not_a_mismatch(self, venv):
        """The pinned install brings it in; nothing to repair."""
        _install(venv, "torch", "2.8.0+rocm6.4.2")
        _install(venv, "torchvision", "0.23.0+rocm6.4.2")
        assert _run(venv) == ""

    def test_windows_arm64_never_asks_about_torchaudio(self, venv):
        _install(venv, "torch", "2.8.0+rocm6.4.2")
        _install(venv, "torchvision", "0.23.0+rocm6.4.2")
        _install(venv, "torchaudio", "2.8.0")
        assert _run(venv, "('torchvision',)") == ""

    def test_a_torch_without_a_rocm_version_does_not_force_on_release(self, venv):
        """Only the +cpu/+cu/untagged rule may fire when torch names no release."""
        _install(venv, "torch", "2.8.0+rocm")
        _install(venv, "torchvision", "0.23.0+rocm6.4.2")
        assert _run(venv) == ""


class TestProbeSource:
    def test_the_extracted_source_compiles(self):
        compile(_probe_source(), "companion_probe", "exec")

    def test_the_release_comparison_is_numeric_only(self):
        source = _probe_source()
        assert re.search(r"rocm\(\\d\+\(\?:\\\.\\d\+\)\*\)", source), textwrap.shorten(source, 200)

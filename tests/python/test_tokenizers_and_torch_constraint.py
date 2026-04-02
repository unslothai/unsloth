"""
Tests for two install fixes:
  1. tokenizers added to no-torch-runtime.txt  (prevents AutoConfig crash)
  2. TORCH_CONSTRAINT variable in install.sh    (arm64 macOS + py313+ -> torch>=2.6)
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import textwrap

import pytest

# ── Locate source files relative to this test ──────────────────────────
_TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent  # tests/
_REPO_ROOT = _TESTS_DIR.parent  # unsloth/
_INSTALL_SH = _REPO_ROOT / "install.sh"
_INSTALL_PS1 = _REPO_ROOT / "install.ps1"
_NO_TORCH_RT = (
    _REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"
)


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding = "utf-8")


def _lines(path: pathlib.Path) -> list[str]:
    """Return non-comment, non-blank lines stripped."""
    return [
        ln.strip()
        for ln in _read(path).splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


# ======================================================================
# Group 1 -- Structural checks (no network, instant)
# ======================================================================
class TestStructuralTokenizers:
    """Verify tokenizers presence and ordering in no-torch-runtime.txt."""

    def test_tokenizers_present(self):
        """tokenizers must be a standalone package line."""
        pkgs = _lines(_NO_TORCH_RT)
        bare_names = [
            p.split(">")[0].split("<")[0].split("!")[0].split("=")[0] for p in pkgs
        ]
        assert "tokenizers" in bare_names

    def test_tokenizers_before_transformers(self):
        """tokenizers should appear before transformers (install order intent)."""
        pkgs = _lines(_NO_TORCH_RT)
        bare_names = [
            p.split(">")[0].split("<")[0].split("!")[0].split("=")[0] for p in pkgs
        ]
        idx_tok = bare_names.index("tokenizers")
        idx_tf = bare_names.index("transformers")
        assert idx_tok < idx_tf, (
            f"tokenizers at index {idx_tok} should appear before "
            f"transformers at index {idx_tf}"
        )

    def test_torch_not_in_no_torch_file(self):
        """torch itself must NOT be listed in the no-torch requirements."""
        pkgs = _lines(_NO_TORCH_RT)
        bare_names = [
            p.split(">")[0].split("<")[0].split("!")[0].split("=")[0] for p in pkgs
        ]
        assert "torch" not in bare_names


class TestStructuralTorchConstraint:
    """Verify TORCH_CONSTRAINT wiring in install.sh."""

    _sh = _read(_INSTALL_SH)

    def test_default_assignment_exists(self):
        assert 'TORCH_CONSTRAINT="torch>=2.4,<2.11.0"' in self._sh

    def test_tightened_assignment_exists(self):
        assert 'TORCH_CONSTRAINT="torch>=2.6,<2.11.0"' in self._sh

    def test_variable_used_in_pip_install(self):
        """$TORCH_CONSTRAINT must appear in a uv pip install line."""
        assert '"$TORCH_CONSTRAINT"' in self._sh

    def test_hardcoded_torch_constraint_only_once(self):
        """The hard-coded torch>=2.4,<2.11.0 string should appear exactly once
        in install.sh (the default assignment), not in pip install lines."""
        count = self._sh.count('"torch>=2.4,<2.11.0"')
        assert count == 1, f"Expected 1, found {count}"

    def test_tightening_guarded_by_skip_torch(self):
        """The block must check SKIP_TORCH=false."""
        # Find the tightening if-block
        m = re.search(
            r"if\s.*SKIP_TORCH.*=\s*false.*&&.*OS.*=.*macos.*&&.*_ARCH.*=.*arm64",
            self._sh,
        )
        assert m is not None, "Guard not found: SKIP_TORCH + macos + arm64"

    def test_tightening_guarded_by_arch(self):
        m = re.search(r"_ARCH.*=.*arm64", self._sh)
        assert m is not None

    def test_tightening_guarded_by_os(self):
        m = re.search(r"OS.*=.*macos", self._sh)
        assert m is not None


class TestStructuralInstallPs1Unchanged:
    """install.ps1 should NOT have TORCH_CONSTRAINT variable."""

    _ps1 = _read(_INSTALL_PS1)

    def test_no_torch_constraint_variable(self):
        assert "TORCH_CONSTRAINT" not in self._ps1
        assert "$TorchConstraint" not in self._ps1

    def test_hardcoded_torch_constraint_present(self):
        assert '"torch>=2.4,<2.11.0"' in self._ps1


# ======================================================================
# Group 2 -- Shell snippet tests (bash subprocess, mocked python)
# ======================================================================
class TestTorchConstraintShell:
    """Test the TORCH_CONSTRAINT block using bash subprocesses with
    mocked python binaries that return controlled minor versions."""

    # The extracted snippet we test in isolation.  We override OS, _ARCH,
    # SKIP_TORCH, and provide a mock python at $VENV_DIR/bin/python.
    _SNIPPET_TEMPLATE = textwrap.dedent(r"""
        #!/bin/bash
        set -e
        SKIP_TORCH={skip_torch}
        OS="{os}"
        _ARCH="{arch}"
        VENV_DIR="{venv_dir}"

        TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
        if [ "$SKIP_TORCH" = false ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
            _PY_MINOR=$("$VENV_DIR/bin/python" -c \
                "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
            if [ "$_PY_MINOR" -ge 13 ] 2>/dev/null; then
                TORCH_CONSTRAINT="torch>=2.6,<2.11.0"
            fi
        fi
        echo "$TORCH_CONSTRAINT"
    """).strip()

    @staticmethod
    def _make_mock_python(tmp_path: pathlib.Path, minor: int) -> pathlib.Path:
        """Create a mock python that prints a controlled minor version."""
        venv = tmp_path / "venv"
        bin_dir = venv / "bin"
        bin_dir.mkdir(parents = True, exist_ok = True)
        mock_py = bin_dir / "python"
        mock_py.write_text(
            textwrap.dedent(f"""\
            #!/bin/bash
            # Mock python: always report minor={minor}
            if echo "$@" | grep -q "sys.version_info.minor"; then
                echo "{minor}"
            else
                echo "0"
            fi
        """)
        )
        mock_py.chmod(0o755)
        return venv

    def _run(
        self,
        tmp_path: pathlib.Path,
        *,
        py_minor: int = 12,
        os_val: str = "macos",
        arch: str = "arm64",
        skip_torch: str = "false",
    ) -> str:
        venv = self._make_mock_python(tmp_path, py_minor)
        script = self._SNIPPET_TEMPLATE.format(
            skip_torch = skip_torch,
            os = os_val,
            arch = arch,
            venv_dir = str(venv),
        )
        script_file = tmp_path / "test_snippet.sh"
        script_file.write_text(script)
        script_file.chmod(0o755)
        result = subprocess.run(
            ["bash", str(script_file)],
            capture_output = True,
            text = True,
            timeout = 10,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        return result.stdout.strip()

    # -- arm64 macOS tightening cases --

    def test_arm64_macos_py313_tightened(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.6,<2.11.0"

    def test_arm64_macos_py314_tightened(self, tmp_path):
        out = self._run(tmp_path, py_minor = 14, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.6,<2.11.0"

    # -- arm64 macOS default (older python) --

    def test_arm64_macos_py312_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 12, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.4,<2.11.0"

    def test_arm64_macos_py311_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 11, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.4,<2.11.0"

    # -- Linux (unaffected) --

    def test_linux_x86_py313_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "linux", arch = "x86_64")
        assert out == "torch>=2.4,<2.11.0"

    def test_linux_aarch64_py313_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "linux", arch = "aarch64")
        assert out == "torch>=2.4,<2.11.0"

    # -- Intel Mac (arch mismatch) --

    def test_intel_mac_x86_py313_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "macos", arch = "x86_64")
        assert out == "torch>=2.4,<2.11.0"

    # -- SKIP_TORCH bypass --

    def test_skip_torch_arm64_macos_py313_default(self, tmp_path):
        out = self._run(
            tmp_path,
            py_minor = 13,
            os_val = "macos",
            arch = "arm64",
            skip_torch = "true",
        )
        assert out == "torch>=2.4,<2.11.0"

    # -- WSL --

    def test_wsl_py313_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "wsl", arch = "x86_64")
        assert out == "torch>=2.4,<2.11.0"

    # -- Edge cases --

    def test_py_minor_0_fallback_default(self, tmp_path):
        """If python query fails (returns 0), should stay at default."""
        out = self._run(tmp_path, py_minor = 0, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.4,<2.11.0"

    def test_boundary_py_minor_12_not_tightened(self, tmp_path):
        out = self._run(tmp_path, py_minor = 12, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.4,<2.11.0"

    def test_boundary_py_minor_13_tightened(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.6,<2.11.0"

    def test_mock_uv_receives_correct_constraint(self, tmp_path):
        """Verify a mock uv would receive the correct constraint string."""
        venv = self._make_mock_python(tmp_path, minor = 13)

        # Create a mock uv that logs its arguments
        mock_uv = tmp_path / "mock_uv"
        log_file = tmp_path / "uv_log.txt"
        mock_uv.write_text(
            textwrap.dedent(f"""\
            #!/bin/bash
            echo "$@" >> {log_file}
        """)
        )
        mock_uv.chmod(0o755)

        script = textwrap.dedent(f"""\
            #!/bin/bash
            set -e
            SKIP_TORCH=false
            OS="macos"
            _ARCH="arm64"
            VENV_DIR="{venv}"

            TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
            if [ "$SKIP_TORCH" = false ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
                _PY_MINOR=$("$VENV_DIR/bin/python" -c \\
                    "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
                if [ "$_PY_MINOR" -ge 13 ] 2>/dev/null; then
                    TORCH_CONSTRAINT="torch>=2.6,<2.11.0"
                fi
            fi
            # Simulate the uv pip install line
            {mock_uv} pip install --python "$VENV_DIR/bin/python" "$TORCH_CONSTRAINT" torchvision torchaudio
        """)
        script_file = tmp_path / "test_uv.sh"
        script_file.write_text(script)
        script_file.chmod(0o755)

        result = subprocess.run(
            ["bash", str(script_file)],
            capture_output = True,
            text = True,
            timeout = 10,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        logged = log_file.read_text()
        assert "torch>=2.6,<2.11.0" in logged, f"uv log: {logged}"

    def test_mock_uv_receives_default_constraint(self, tmp_path):
        """On py3.12 arm64 macOS, uv should receive the default constraint."""
        venv = self._make_mock_python(tmp_path, minor = 12)
        mock_uv = tmp_path / "mock_uv"
        log_file = tmp_path / "uv_log.txt"
        mock_uv.write_text(
            textwrap.dedent(f"""\
            #!/bin/bash
            echo "$@" >> {log_file}
        """)
        )
        mock_uv.chmod(0o755)

        script = textwrap.dedent(f"""\
            #!/bin/bash
            set -e
            SKIP_TORCH=false
            OS="macos"
            _ARCH="arm64"
            VENV_DIR="{venv}"

            TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
            if [ "$SKIP_TORCH" = false ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
                _PY_MINOR=$("$VENV_DIR/bin/python" -c \\
                    "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
                if [ "$_PY_MINOR" -ge 13 ] 2>/dev/null; then
                    TORCH_CONSTRAINT="torch>=2.6,<2.11.0"
                fi
            fi
            {mock_uv} pip install --python "$VENV_DIR/bin/python" "$TORCH_CONSTRAINT" torchvision torchaudio
        """)
        script_file = tmp_path / "test_uv.sh"
        script_file.write_text(script)
        script_file.chmod(0o755)

        result = subprocess.run(
            ["bash", str(script_file)],
            capture_output = True,
            text = True,
            timeout = 10,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        logged = log_file.read_text()
        assert "torch>=2.4,<2.11.0" in logged, f"uv log: {logged}"


# ======================================================================
# Group 3 -- E2E tokenizers fix (requires network, ~2-5 min)
# ======================================================================
@pytest.mark.e2e
class TestE2ETokenizersFix:
    """Creates real uv venvs to verify tokenizers + transformers work
    without torch installed."""

    @staticmethod
    def _create_venv(tmp_path: pathlib.Path, name: str, py: str) -> pathlib.Path:
        venv = tmp_path / name
        result = subprocess.run(
            ["uv", "venv", str(venv), "--python", py],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        if result.returncode != 0:
            pytest.skip(f"uv venv creation failed for {py}: {result.stderr}")
        return venv

    @staticmethod
    def _pip_install(venv: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
        py = str(venv / "bin" / "python")
        cmd = ["uv", "pip", "install", "--python", py, *args]
        return subprocess.run(cmd, capture_output = True, text = True, timeout = 300)

    @staticmethod
    def _run_python(venv: pathlib.Path, code: str) -> subprocess.CompletedProcess:
        py = str(venv / "bin" / "python")
        return subprocess.run(
            [py, "-c", code],
            capture_output = True,
            text = True,
            timeout = 60,
        )

    @pytest.mark.parametrize("py_version", ["3.12", "3.13"])
    def test_autoconfig_works_with_no_torch_runtime(self, tmp_path, py_version):
        """Install from no-torch-runtime.txt with --no-deps (matching the
        real install.sh path), then verify AutoConfig imports successfully."""
        venv = self._create_venv(tmp_path, f"tok-{py_version}", py_version)
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"

        result = self._run_python(
            venv, "from transformers import AutoConfig; print('OK')"
        )
        assert (
            result.returncode == 0
        ), f"AutoConfig import failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert "OK" in result.stdout

    @pytest.mark.parametrize("py_version", ["3.12", "3.13"])
    def test_tokenizers_directly_importable(self, tmp_path, py_version):
        venv = self._create_venv(tmp_path, f"tok-imp-{py_version}", py_version)
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "import tokenizers; print('OK')")
        assert result.returncode == 0, f"Failed: {result.stderr}"

    @pytest.mark.parametrize("py_version", ["3.12", "3.13"])
    def test_torch_not_importable(self, tmp_path, py_version):
        """In the no-torch scenario, torch should not be available."""
        venv = self._create_venv(tmp_path, f"no-torch-{py_version}", py_version)
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "import torch")
        assert result.returncode != 0, "torch should NOT be importable"

    def test_negative_control_no_tokenizers(self, tmp_path):
        """Without tokenizers, AutoConfig should fail. We create a copy of
        no-torch-runtime.txt with the tokenizers line removed."""
        venv = self._create_venv(tmp_path, "neg-ctrl", "3.12")
        req_no_tokenizers = tmp_path / "no-tokenizers.txt"
        req_no_tokenizers.write_text(
            "\n".join(
                line
                for line in _read(_NO_TORCH_RT).splitlines()
                if line.strip() != "tokenizers"
            ),
            encoding = "utf-8",
        )
        r = self._pip_install(venv, "--no-deps", "-r", str(req_no_tokenizers))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "from transformers import AutoConfig")
        assert (
            result.returncode != 0
        ), "AutoConfig should fail without tokenizers installed"
        assert (
            "tokenizers" in result.stderr.lower()
            or "ModuleNotFoundError" in result.stderr
        )


# ======================================================================
# Group 4 -- Integration: install.sh reads no-torch-runtime.txt correctly
# ======================================================================
class TestInstallShNoTorchIntegration:
    """Verify install.sh has the correct no-torch-runtime.txt wiring."""

    _sh = _read(_INSTALL_SH)

    def test_find_no_torch_runtime_exists(self):
        assert "_find_no_torch_runtime()" in self._sh

    def test_no_deps_invocation_for_migrated(self):
        """Migrated path should use --no-deps -r."""
        assert '--no-deps -r "$_NO_TORCH_RT"' in self._sh

    def test_no_deps_invocation_for_fresh(self):
        """Fresh install path should also use --no-deps -r."""
        # Count occurrences of the no-deps -r pattern
        count = self._sh.count('--no-deps -r "$_NO_TORCH_RT"')
        assert count >= 2, f"Expected >=2 no-deps -r invocations, found {count}"

    def test_mock_uv_skip_torch_reads_requirements(self, tmp_path):
        """When SKIP_TORCH=true, the _find_no_torch_runtime path should be used."""
        # We test this structurally: verify the SKIP_TORCH=true blocks contain
        # _find_no_torch_runtime calls
        skip_blocks = re.findall(
            r'if \[ "\$SKIP_TORCH" = true \].*?(?=\n    (?:else|elif|fi))',
            self._sh,
            re.DOTALL,
        )
        found = any("_find_no_torch_runtime" in block for block in skip_blocks)
        assert found, "SKIP_TORCH=true block should call _find_no_torch_runtime"


# ======================================================================
# Group 5 -- Full no-torch sandbox (requires network, ~5 min)
# ======================================================================
@pytest.mark.e2e
class TestE2EFullNoTorchSandbox:
    """Creates venvs and installs the actual no-torch-runtime.txt."""

    @staticmethod
    def _create_venv(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
        venv = tmp_path / name
        result = subprocess.run(
            ["uv", "venv", str(venv), "--python", "3.12"],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        if result.returncode != 0:
            pytest.skip(f"uv venv creation failed: {result.stderr}")
        return venv

    @staticmethod
    def _pip_install(venv: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
        py = str(venv / "bin" / "python")
        cmd = ["uv", "pip", "install", "--python", py, *args]
        return subprocess.run(cmd, capture_output = True, text = True, timeout = 600)

    @staticmethod
    def _run_python(venv: pathlib.Path, code: str) -> subprocess.CompletedProcess:
        py = str(venv / "bin" / "python")
        return subprocess.run(
            [py, "-c", code],
            capture_output = True,
            text = True,
            timeout = 60,
        )

    def test_autoconfig_succeeds(self, tmp_path):
        """The real bug fix: install with --no-deps (matching install.sh)
        and verify from transformers import AutoConfig works."""
        venv = self._create_venv(tmp_path, "full-no-torch")
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(
            venv, "from transformers import AutoConfig; print('OK')"
        )
        assert (
            result.returncode == 0
        ), f"AutoConfig failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_torch_not_importable(self, tmp_path):
        """With --no-deps (as install.sh uses), torch must not be pulled in."""
        venv = self._create_venv(tmp_path, "no-torch-check")
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "import torch")
        assert result.returncode != 0, "torch should NOT be importable"

    def test_tokenizers_importable(self, tmp_path):
        venv = self._create_venv(tmp_path, "tok-check")
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "import tokenizers; print('OK')")
        assert result.returncode == 0, f"tokenizers import failed: {result.stderr}"

    def test_safetensors_importable(self, tmp_path):
        venv = self._create_venv(tmp_path, "st-check")
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "import safetensors; print('OK')")
        assert result.returncode == 0, f"safetensors import failed: {result.stderr}"

    def test_huggingface_hub_importable(self, tmp_path):
        venv = self._create_venv(tmp_path, "hfhub-check")
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "import huggingface_hub; print('OK')")
        assert result.returncode == 0, f"huggingface_hub import failed: {result.stderr}"

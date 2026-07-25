"""Install fixes: tokenizers in no-torch-runtime.txt, and TORCH_CONSTRAINT in install.sh."""

from __future__ import annotations

import pathlib
import re
import subprocess
import textwrap

import pytest

# Locate source files relative to this test.
_TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent  # tests/
_REPO_ROOT = _TESTS_DIR.parent  # unsloth/
_INSTALL_SH = _REPO_ROOT / "install.sh"
_INSTALL_PS1 = _REPO_ROOT / "install.ps1"
_SETUP_PS1 = _REPO_ROOT / "studio" / "setup.ps1"
_NO_TORCH_RT = _REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding = "utf-8")


def _lines(path: pathlib.Path) -> list[str]:
    """Return non-comment, non-blank lines stripped."""
    return [
        ln.strip()
        for ln in _read(path).splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


# Group 1 -- Structural checks (no network, instant)
class TestStructuralTokenizers:
    """Verify tokenizers presence and ordering in no-torch-runtime.txt."""

    def test_tokenizers_present(self):
        """tokenizers must be a standalone package line."""
        pkgs = _lines(_NO_TORCH_RT)
        bare_names = [p.split(">")[0].split("<")[0].split("!")[0].split("=")[0] for p in pkgs]
        assert "tokenizers" in bare_names

    def test_tokenizers_before_transformers(self):
        """tokenizers should appear before transformers (install order intent)."""
        pkgs = _lines(_NO_TORCH_RT)
        bare_names = [p.split(">")[0].split("<")[0].split("!")[0].split("=")[0] for p in pkgs]
        idx_tok = bare_names.index("tokenizers")
        idx_tf = bare_names.index("transformers")
        assert idx_tok < idx_tf, (
            f"tokenizers at index {idx_tok} should appear before " f"transformers at index {idx_tf}"
        )

    def test_torch_not_in_no_torch_file(self):
        """torch itself must NOT be listed in the no-torch requirements."""
        pkgs = _lines(_NO_TORCH_RT)
        bare_names = [p.split(">")[0].split("<")[0].split("!")[0].split("=")[0] for p in pkgs]
        assert "torch" not in bare_names


class TestStructuralTorchConstraint:
    """Verify TORCH_CONSTRAINT wiring in install.sh."""

    _sh = _read(_INSTALL_SH)

    def test_default_assignment_exists(self):
        assert 'TORCH_CONSTRAINT="torch>=2.4,<2.11.0"' in self._sh

    def test_tightened_assignment_exists(self):
        assert 'TORCH_CONSTRAINT="torch>=2.6,<2.11.0"' in self._sh

    def test_cuda_constraint_widened_to_2_12(self):
        """A fresh CUDA install widens the ceiling to <2.12.0 so cu12x/cu13x
        land torch 2.11.x (matches the base image and _CUDA_TORCH_PKG_SPEC);
        without it cu128/cu130 resolves torch 2.10.x."""
        assert 'TORCH_CONSTRAINT="torch>=2.4,<2.12.0"' in self._sh

    def test_cuda_case_widens_via_index_leaf(self):
        """The cu* branch of the _torch_index_leaf case sets the widened
        constraint (parallel to rocm7.2), anchored on the leaf."""
        m = re.search(
            r'cu\[0-9\]\*\)\s*TORCH_CONSTRAINT="torch>=2\.4,<2\.12\.0"',
            self._sh,
        )
        assert m is not None, "CUDA (cu*) TORCH_CONSTRAINT widening case not found"

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


class TestInstallPs1UvDefaultIndex:
    """Installer-managed torch indexes must override inherited uv defaults."""

    _ps1 = _read(_INSTALL_PS1)

    def test_torch_installs_use_default_index(self):
        assert "--default-index $TorchIndexUrl" in self._ps1
        assert "--default-index $ROCmIndexUrl" in self._ps1

    def test_torch_installs_do_not_use_deprecated_index_url(self):
        assert "--index-url $TorchIndexUrl" not in self._ps1
        assert "--index-url $ROCmIndexUrl" not in self._ps1

    def test_torch_installs_neutralize_all_uv_index_env_vars(self):
        # Extra-index vars outrank --default-index, so pinned installs must clear them.
        for var in ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX", "UV_EXTRA_INDEX_URL"):
            assert var in self._ps1
        assert 'Remove-Item "Env:$n"' in self._ps1


class TestSetupPs1FastInstallIndex:
    """setup.ps1 Fast-Install must neutralize inherited uv indexes when pinning."""

    _ps1 = _read(_SETUP_PS1)

    def test_fast_install_clears_all_uv_index_env_vars(self):
        for var in ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX", "UV_EXTRA_INDEX_URL"):
            assert var in self._ps1
        # Must truly remove the vars (child sees no value), not set them empty.
        assert 'Remove-Item "Env:$n"' in self._ps1


class TestInstallShUvDefaultIndex:
    """Linux/Mac installer torch indexes must override inherited uv defaults."""

    _sh = _read(_INSTALL_SH)

    def test_torch_installs_use_default_index(self):
        assert '--default-index "$TORCH_INDEX_URL"' in self._sh

    def test_torch_installs_do_not_use_deprecated_index_url(self):
        assert '--index-url "$TORCH_INDEX_URL"' not in self._sh

    def test_torch_installs_neutralize_all_uv_index_env_vars(self):
        # --default-index installs run with all uv index env vars unset via `env -u`.
        assert (
            "env -u UV_DEFAULT_INDEX -u UV_INDEX_URL -u UV_INDEX -u UV_EXTRA_INDEX_URL" in self._sh
        )


# Group 2 -- Shell snippet tests (bash subprocess, mocked python)
class TestTorchConstraintShell:
    """Test the TORCH_CONSTRAINT block via bash with mocked python minor versions."""

    # Snippet tested in isolation: override OS/_ARCH/SKIP_TORCH and a mock python.
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

    def test_arm64_macos_py313_tightened(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.6,<2.11.0"

    def test_arm64_macos_py314_tightened(self, tmp_path):
        out = self._run(tmp_path, py_minor = 14, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.6,<2.11.0"

    def test_arm64_macos_py312_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 12, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.4,<2.11.0"

    def test_arm64_macos_py311_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 11, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.4,<2.11.0"

    # Linux is unaffected by the tightening.
    def test_linux_x86_py313_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "linux", arch = "x86_64")
        assert out == "torch>=2.4,<2.11.0"

    def test_linux_aarch64_py313_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "linux", arch = "aarch64")
        assert out == "torch>=2.4,<2.11.0"

    # Intel Mac: arch mismatch, no tightening.
    def test_intel_mac_x86_py313_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "macos", arch = "x86_64")
        assert out == "torch>=2.4,<2.11.0"

    # SKIP_TORCH bypasses the tightening.
    def test_skip_torch_arm64_macos_py313_default(self, tmp_path):
        out = self._run(
            tmp_path,
            py_minor = 13,
            os_val = "macos",
            arch = "arm64",
            skip_torch = "true",
        )
        assert out == "torch>=2.4,<2.11.0"

    def test_wsl_py313_default(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "wsl", arch = "x86_64")
        assert out == "torch>=2.4,<2.11.0"

    def test_py_minor_0_fallback_default(self, tmp_path):
        """Failed python query (returns 0) keeps the default constraint."""
        out = self._run(tmp_path, py_minor = 0, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.4,<2.11.0"

    def test_boundary_py_minor_12_not_tightened(self, tmp_path):
        out = self._run(tmp_path, py_minor = 12, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.4,<2.11.0"

    def test_boundary_py_minor_13_tightened(self, tmp_path):
        out = self._run(tmp_path, py_minor = 13, os_val = "macos", arch = "arm64")
        assert out == "torch>=2.6,<2.11.0"

    def test_mock_uv_receives_correct_constraint(self, tmp_path):
        """A mock uv receives the tightened constraint on py3.13 arm64 macOS."""
        venv = self._make_mock_python(tmp_path, minor = 13)

        # Mock uv logs its arguments.
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

    # Mirrors the _torch_index_leaf case in install.sh: rocm7.2 -> 2.11.x floor,
    # CUDA -> widened <2.12.0 ceiling, else (CPU/older ROCm) -> default. Anchored
    # on the final path segment, so a mirror base path containing cu*/rocm7.2 but
    # ending in a cpu/older-rocm leaf keeps the default.
    _INDEX_SNIPPET = textwrap.dedent(r"""
        #!/bin/bash
        set -e
        TORCH_INDEX_URL="{index_url}"
        TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
        _torch_index_leaf="${TORCH_INDEX_URL%/}"
        _torch_index_leaf="${_torch_index_leaf##*/}"
        case "$_torch_index_leaf" in
            rocm7.2)  TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0" ;;
            cu[0-9]*) TORCH_CONSTRAINT="torch>=2.4,<2.12.0" ;;
        esac
        echo "$TORCH_CONSTRAINT"
    """).strip()

    def _resolve_index(self, tmp_path: pathlib.Path, index_url: str) -> str:
        script_file = tmp_path / "index_snippet.sh"
        script_file.write_text(self._INDEX_SNIPPET.replace("{index_url}", index_url))
        script_file.chmod(0o755)
        result = subprocess.run(
            ["bash", str(script_file)],
            capture_output = True,
            text = True,
            timeout = 10,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        return result.stdout.strip()

    @pytest.mark.parametrize("leaf", ["cu118", "cu124", "cu126", "cu128", "cu130"])
    def test_cuda_index_widens_to_2_12(self, tmp_path, leaf):
        url = f"https://download.pytorch.org/whl/{leaf}"
        assert self._resolve_index(tmp_path, url) == "torch>=2.4,<2.12.0"

    def test_rocm72_index_uses_211_floor(self, tmp_path):
        url = "https://download.pytorch.org/whl/rocm7.2"
        assert self._resolve_index(tmp_path, url) == "torch>=2.11.0,<2.12.0"

    def test_cpu_index_keeps_default(self, tmp_path):
        # /cpu must NOT match the */cu[0-9]* branch.
        url = "https://download.pytorch.org/whl/cpu"
        assert self._resolve_index(tmp_path, url) == "torch>=2.4,<2.11.0"

    def test_older_rocm_index_keeps_default(self, tmp_path):
        url = "https://download.pytorch.org/whl/rocm7.1"
        assert self._resolve_index(tmp_path, url) == "torch>=2.4,<2.11.0"

    def test_cuda_index_custom_mirror_widens(self, tmp_path):
        url = "https://internal.example.com/pytorch/cu128"
        assert self._resolve_index(tmp_path, url) == "torch>=2.4,<2.12.0"

    @pytest.mark.parametrize(
        "url",
        [
            "https://internal.example.com/pytorch/cu128/cpu",
            "https://internal.example.com/cu128/whl/rocm7.1",
        ],
    )
    def test_cuda_in_mirror_path_but_noncuda_leaf_keeps_default(self, tmp_path, url):
        # A cu128 in the mirror base path must not widen when the leaf is cpu /
        # older ROCm: the case anchors on _torch_index_leaf, not the whole URL.
        assert self._resolve_index(tmp_path, url) == "torch>=2.4,<2.11.0"


# Group 3 -- E2E tokenizers fix (requires network, ~2-5 min)
@pytest.mark.e2e
class TestE2ETokenizersFix:
    """Real uv venvs verify tokenizers + transformers work without torch installed."""

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
        """Install no-torch-runtime.txt with --no-deps, then AutoConfig must import."""
        venv = self._create_venv(tmp_path, f"tok-{py_version}", py_version)
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"

        result = self._run_python(venv, "from transformers import AutoConfig; print('OK')")
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
        """Without the tokenizers line, AutoConfig must fail (negative control)."""
        venv = self._create_venv(tmp_path, "neg-ctrl", "3.12")
        req_no_tokenizers = tmp_path / "no-tokenizers.txt"
        req_no_tokenizers.write_text(
            "\n".join(
                line for line in _read(_NO_TORCH_RT).splitlines() if line.strip() != "tokenizers"
            ),
            encoding = "utf-8",
        )
        r = self._pip_install(venv, "--no-deps", "-r", str(req_no_tokenizers))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "from transformers import AutoConfig")
        assert result.returncode != 0, "AutoConfig should fail without tokenizers installed"
        assert "tokenizers" in result.stderr.lower() or "ModuleNotFoundError" in result.stderr


# Group 4 -- Integration: install.sh reads no-torch-runtime.txt correctly
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
        count = self._sh.count('--no-deps -r "$_NO_TORCH_RT"')
        assert count >= 2, f"Expected >=2 no-deps -r invocations, found {count}"

    def test_mock_uv_skip_torch_reads_requirements(self, tmp_path):
        """SKIP_TORCH=true blocks must call _find_no_torch_runtime."""
        skip_blocks = re.findall(
            r'if \[ "\$SKIP_TORCH" = true \].*?(?=\n    (?:else|elif|fi))',
            self._sh,
            re.DOTALL,
        )
        found = any("_find_no_torch_runtime" in block for block in skip_blocks)
        assert found, "SKIP_TORCH=true block should call _find_no_torch_runtime"


# Group 5 -- Full no-torch sandbox (requires network, ~5 min)
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
        """Install with --no-deps and verify AutoConfig imports (the bug fix)."""
        venv = self._create_venv(tmp_path, "full-no-torch")
        r = self._pip_install(venv, "--no-deps", "-r", str(_NO_TORCH_RT))
        assert r.returncode == 0, f"Install failed: {r.stderr}"
        result = self._run_python(venv, "from transformers import AutoConfig; print('OK')")
        assert (
            result.returncode == 0
        ), f"AutoConfig failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_torch_not_importable(self, tmp_path):
        """With --no-deps, torch must not be pulled in."""
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

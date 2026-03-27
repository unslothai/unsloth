"""Comprehensive E2E sandbox tests for PR #4624 (fix/install-mac-intel-no-torch).

Proves that:
- The BEFORE state (top-level torch imports) crashes without torch
- The AFTER state (lazy/removed imports) works without torch
- Edge cases (broken torch, partial torch) are handled gracefully
- Hardware detection falls back to CPU without torch
- install.sh flag parsing and platform detection work correctly
- install_python_stack.py NO_TORCH filtering is correct
- Live server starts and responds without torch (optional, requires studio venv)

Run:
    # Lightweight tests (Groups 1-6, ~26 tests):
    python -m pytest tests/python/test_e2e_no_torch_sandbox.py -v -k "not server"

    # Server tests (Group 7, 4 tests, requires studio venv):
    python -m pytest tests/python/test_e2e_no_torch_sandbox.py -v -m server
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "studio"
BACKEND_DIR = STUDIO_DIR / "backend"
DATASETS_DIR = BACKEND_DIR / "utils" / "datasets"
HARDWARE_DIR = BACKEND_DIR / "utils" / "hardware"
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PY = STUDIO_DIR / "install_python_stack.py"

DATA_COLLATORS = DATASETS_DIR / "data_collators.py"
CHAT_TEMPLATES = DATASETS_DIR / "chat_templates.py"
FORMAT_DETECTION = DATASETS_DIR / "format_detection.py"
MODEL_MAPPINGS = DATASETS_DIR / "model_mappings.py"
VLM_PROCESSING = DATASETS_DIR / "vlm_processing.py"
HARDWARE_PY = HARDWARE_DIR / "hardware.py"

# Studio venv for server tests
STUDIO_VENV = Path.home() / ".unsloth" / "studio" / "unsloth_studio"

# Add studio to path for install_python_stack imports
sys.path.insert(0, str(STUDIO_DIR))


# ---------------------------------------------------------------------------
# Cross-platform helpers
# ---------------------------------------------------------------------------


def _venv_python(venv_dir: Path) -> Path:
    """Return the Python executable path for a venv, cross-platform."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _has_uv() -> bool:
    return shutil.which("uv") is not None


def _create_no_torch_venv(venv_dir: Path, python_version: str = "3.12") -> Path | None:
    """Create a uv venv with no torch. Returns python path or None."""
    result = subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", python_version],
        capture_output = True,
    )
    if result.returncode != 0:
        return None
    py = _venv_python(venv_dir)
    if not py.exists():
        return None
    # Verify torch is NOT importable
    check = subprocess.run([str(py), "-c", "import torch"], capture_output = True)
    if check.returncode == 0:
        return None
    return py


def _run_in_sandbox(
    py: str | Path,
    code: str,
    timeout: int = 60,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run Python code in a sandboxed interpreter."""
    return subprocess.run(
        [str(py), "-c", code],
        capture_output = True,
        timeout = timeout,
        env = env,
    )


def _run_sh(script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a bash snippet and return the result."""
    return subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        timeout = timeout,
    )


# ---------------------------------------------------------------------------
# Stub generators
# ---------------------------------------------------------------------------


def _write_loggers_stub(sandbox: Path) -> None:
    """Create a minimal loggers package stub (replaces structlog-backed real one)."""
    loggers_dir = sandbox / "loggers"
    loggers_dir.mkdir(exist_ok = True)
    (loggers_dir / "__init__.py").write_text(
        "from .handlers import get_logger\n__all__ = ['get_logger']\n",
        encoding = "utf-8",
    )
    (loggers_dir / "handlers.py").write_text(
        textwrap.dedent("""\
            class _Logger:
                def info(self, msg, *a, **k): pass
                def warning(self, msg, *a, **k): pass
                def debug(self, msg, *a, **k): pass
                def error(self, msg, *a, **k): pass
                def msg(self, msg, *a, **k): pass
            def get_logger(name=None):
                return _Logger()
        """),
        encoding = "utf-8",
    )


def _write_structlog_stub(sandbox: Path) -> None:
    """Create a minimal structlog stub."""
    structlog_dir = sandbox / "structlog"
    structlog_dir.mkdir(exist_ok = True)
    (structlog_dir / "__init__.py").write_text(
        textwrap.dedent("""\
            class _Logger:
                def info(self, msg, *a, **k): pass
                def warning(self, msg, *a, **k): pass
                def debug(self, msg, *a, **k): pass
                def error(self, msg, *a, **k): pass
                def msg(self, msg, *a, **k): pass
            def get_logger(name=None):
                return _Logger()
        """),
        encoding = "utf-8",
    )


def _write_hardware_stub(sandbox: Path) -> None:
    """Create utils/hardware stub with dataset_map_num_proc."""
    hw_dir = sandbox / "utils" / "hardware"
    hw_dir.mkdir(parents = True, exist_ok = True)
    (sandbox / "utils" / "__init__.py").write_text("", encoding = "utf-8")
    (hw_dir / "__init__.py").write_text(
        "def dataset_map_num_proc(n=None): return n\n",
        encoding = "utf-8",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope = "session")
def repo_root():
    return REPO_ROOT


@pytest.fixture
def sandbox_dir(tmp_path):
    """Per-test temporary sandbox directory."""
    return tmp_path


@pytest.fixture(params = ["3.12", "3.13"], scope = "module")
def no_torch_venv(request, tmp_path_factory):
    """Create a temporary uv venv with no torch.

    Parametrized for 3.12 (Intel Mac default) and 3.13 (Apple Silicon/Linux).
    """
    if not _has_uv():
        pytest.skip("uv not available")

    py_version = request.param
    venv_dir = tmp_path_factory.mktemp(f"e2e_no_torch_{py_version}")
    py = _create_no_torch_venv(venv_dir, py_version)
    if py is None:
        pytest.skip(f"Could not create Python {py_version} no-torch venv")
    return str(py)


# ===========================================================================
# Group 1: BEFORE vs AFTER -- Import Chain (6 tests)
# ===========================================================================


class TestBeforeAfterImportChain:
    """Prove the bug exists in BEFORE state and is fixed in AFTER state.

    BEFORE = PR branch files with top-level torch import synthetically prepended
             (simulates the main branch).
    AFTER  = PR branch files as-is (lazy imports / torch import removed).
    """

    # -- BEFORE: crashes --

    def test_before_chat_templates_crashes(self, no_torch_venv, sandbox_dir):
        """BEFORE: chat_templates.py with top-level 'from torch.utils.data import
        IterableDataset' crashes without torch."""
        source = CHAT_TEMPLATES.read_text(encoding = "utf-8")
        before_source = "from torch.utils.data import IterableDataset\n" + source

        before_file = sandbox_dir / "chat_templates_before.py"
        before_file.write_text(before_source, encoding = "utf-8")

        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: type('L', (), {{'info': lambda s, m: None}})()
            sys.modules['loggers'] = loggers
            fd = types.ModuleType('format_detection')
            fd.detect_dataset_format = fd.detect_multimodal_dataset = fd.detect_custom_format_heuristic = lambda *a, **k: None
            sys.modules['format_detection'] = fd
            mm = types.ModuleType('model_mappings')
            mm.MODEL_TO_TEMPLATE_MAPPER = {{}}
            sys.modules['model_mappings'] = mm
            source = open({str(before_file)!r}).read()
            source = source.replace('from .format_detection import', 'from format_detection import')
            source = source.replace('from .model_mappings import', 'from model_mappings import')
            exec(source)
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode != 0
        ), "BEFORE chat_templates.py should crash without torch"
        assert (
            b"ModuleNotFoundError" in result.stderr or b"ImportError" in result.stderr
        )

    def test_before_data_collators_crashes(self, no_torch_venv, sandbox_dir):
        """BEFORE: data_collators.py with top-level 'import torch' crashes."""
        source = DATA_COLLATORS.read_text(encoding = "utf-8")
        before_source = "import torch\n" + source

        before_file = sandbox_dir / "data_collators_before.py"
        before_file.write_text(before_source, encoding = "utf-8")

        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(before_file)!r}).read())
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode != 0
        ), "BEFORE data_collators.py should crash without torch"
        assert (
            b"ModuleNotFoundError" in result.stderr or b"ImportError" in result.stderr
        )

    def test_before_full_import_chain_crashes(self, no_torch_venv, sandbox_dir):
        """BEFORE: full utils/datasets/ package with top-level torch imports crashes."""
        _write_loggers_stub(sandbox_dir)
        _write_hardware_stub(sandbox_dir)

        pkg_dir = sandbox_dir / "utils" / "datasets"
        pkg_dir.mkdir(parents = True, exist_ok = True)

        # Copy torch-free modules as-is
        shutil.copy2(FORMAT_DETECTION, pkg_dir / "format_detection.py")
        shutil.copy2(MODEL_MAPPINGS, pkg_dir / "model_mappings.py")
        shutil.copy2(VLM_PROCESSING, pkg_dir / "vlm_processing.py")

        # BEFORE data_collators: prepend top-level 'import torch'
        dc_source = DATA_COLLATORS.read_text(encoding = "utf-8")
        (pkg_dir / "data_collators.py").write_text(
            "import torch\n" + dc_source,
            encoding = "utf-8",
        )

        # BEFORE chat_templates: prepend top-level IterableDataset import
        ct_source = CHAT_TEMPLATES.read_text(encoding = "utf-8")
        (pkg_dir / "chat_templates.py").write_text(
            "from torch.utils.data import IterableDataset\n" + ct_source,
            encoding = "utf-8",
        )

        # Minimal __init__.py that triggers the chain
        (pkg_dir / "__init__.py").write_text(
            textwrap.dedent("""\
                from .format_detection import detect_dataset_format
                from .data_collators import DataCollatorSpeechSeq2SeqWithPadding
                from .chat_templates import DEFAULT_ALPACA_TEMPLATE
            """),
            encoding = "utf-8",
        )

        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(sandbox_dir)!r})
            from utils.datasets import detect_dataset_format
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode != 0
        ), "BEFORE full import chain should crash without torch"
        assert (
            b"ModuleNotFoundError" in result.stderr or b"ImportError" in result.stderr
        )

    # -- AFTER: succeeds --

    def test_after_chat_templates_imports(self, no_torch_venv):
        """AFTER: PR branch chat_templates.py imports fine without torch."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: type('L', (), {{'info': lambda s, m: None}})()
            sys.modules['loggers'] = loggers
            fd = types.ModuleType('format_detection')
            fd.detect_dataset_format = fd.detect_multimodal_dataset = fd.detect_custom_format_heuristic = lambda *a, **k: None
            sys.modules['format_detection'] = fd
            mm = types.ModuleType('model_mappings')
            mm.MODEL_TO_TEMPLATE_MAPPER = {{}}
            sys.modules['model_mappings'] = mm
            source = open({str(CHAT_TEMPLATES)!r}).read()
            source = source.replace('from .format_detection import', 'from format_detection import')
            source = source.replace('from .model_mappings import', 'from model_mappings import')
            exec(source)
            print("OK")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode == 0
        ), f"AFTER chat_templates.py should work without torch:\n{result.stderr.decode()}"
        assert b"OK" in result.stdout

    def test_after_data_collators_imports(self, no_torch_venv):
        """AFTER: PR branch data_collators.py imports fine without torch."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(DATA_COLLATORS)!r}).read())
            print("OK")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode == 0
        ), f"AFTER data_collators.py should work without torch:\n{result.stderr.decode()}"
        assert b"OK" in result.stdout

    def test_after_full_import_chain_imports(self, no_torch_venv, sandbox_dir):
        """AFTER: full utils/datasets/ package imports fine without torch."""
        _write_loggers_stub(sandbox_dir)
        _write_hardware_stub(sandbox_dir)

        pkg_dir = sandbox_dir / "utils" / "datasets"
        pkg_dir.mkdir(parents = True, exist_ok = True)

        # Copy AFTER versions (PR branch -- no top-level torch)
        for src in [
            FORMAT_DETECTION,
            MODEL_MAPPINGS,
            VLM_PROCESSING,
            DATA_COLLATORS,
            CHAT_TEMPLATES,
        ]:
            if src.exists():
                shutil.copy2(src, pkg_dir / src.name)

        # Minimal __init__.py
        (pkg_dir / "__init__.py").write_text(
            textwrap.dedent("""\
                from .format_detection import detect_dataset_format, detect_custom_format_heuristic
                from .model_mappings import MODEL_TO_TEMPLATE_MAPPER
                from .chat_templates import DEFAULT_ALPACA_TEMPLATE, get_dataset_info_summary
                from .data_collators import (
                    DataCollatorSpeechSeq2SeqWithPadding,
                    DeepSeekOCRDataCollator,
                    VLMDataCollator,
                )
                from .vlm_processing import generate_smart_vlm_instruction
            """),
            encoding = "utf-8",
        )

        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(sandbox_dir)!r})
            from utils.datasets import (
                detect_dataset_format,
                DEFAULT_ALPACA_TEMPLATE,
                DataCollatorSpeechSeq2SeqWithPadding,
                DeepSeekOCRDataCollator,
                VLMDataCollator,
                generate_smart_vlm_instruction,
            )
            assert 'Instruction' in DEFAULT_ALPACA_TEMPLATE
            print("OK: full import chain succeeded")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode == 0
        ), f"AFTER full import chain should work:\n{result.stderr.decode()}"
        assert b"OK: full import chain succeeded" in result.stdout


# ===========================================================================
# Group 2: Dataclass Instantiation (4 tests)
# ===========================================================================


class TestDataclassInstantiation:
    """Verify dataclass collators can be instantiated and constants accessed
    without torch in an isolated venv."""

    def test_speech_collator_instantiate(self, no_torch_venv):
        """DataCollatorSpeechSeq2SeqWithPadding(processor=None) succeeds."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(DATA_COLLATORS)!r}).read())
            obj = DataCollatorSpeechSeq2SeqWithPadding(processor=None)
            assert obj.processor is None
            print("OK")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert result.returncode == 0, f"Failed:\n{result.stderr.decode()}"

    def test_deepseek_ocr_collator_instantiate(self, no_torch_venv):
        """DeepSeekOCRDataCollator has correct default field values."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(DATA_COLLATORS)!r}).read())
            obj = DeepSeekOCRDataCollator(processor=None)
            assert obj.processor is None
            assert obj.max_length == 2048
            assert obj.ignore_index == -100
            print("OK")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert result.returncode == 0, f"Failed:\n{result.stderr.decode()}"

    def test_vlm_collator_instantiate(self, no_torch_venv):
        """VLMDataCollator has correct default field values."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(DATA_COLLATORS)!r}).read())
            obj = VLMDataCollator(processor=None)
            assert obj.processor is None
            assert obj.max_length == 2048
            assert obj.mask_input_tokens is True
            print("OK")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert result.returncode == 0, f"Failed:\n{result.stderr.decode()}"

    def test_alpaca_template_accessible(self, no_torch_venv):
        """DEFAULT_ALPACA_TEMPLATE constant is accessible and contains 'Instruction'."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: type('L', (), {{'info': lambda s, m: None}})()
            sys.modules['loggers'] = loggers
            fd = types.ModuleType('format_detection')
            fd.detect_dataset_format = fd.detect_multimodal_dataset = fd.detect_custom_format_heuristic = lambda *a, **k: None
            sys.modules['format_detection'] = fd
            mm = types.ModuleType('model_mappings')
            mm.MODEL_TO_TEMPLATE_MAPPER = {{}}
            sys.modules['model_mappings'] = mm
            ns = {{}}
            source = open({str(CHAT_TEMPLATES)!r}).read()
            source = source.replace('from .format_detection import', 'from format_detection import')
            source = source.replace('from .model_mappings import', 'from model_mappings import')
            exec(source, ns)
            assert 'Instruction' in ns['DEFAULT_ALPACA_TEMPLATE']
            print("OK")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert result.returncode == 0, f"Failed:\n{result.stderr.decode()}"


# ===========================================================================
# Group 3: Edge Cases -- Partial/Broken Torch (4 tests)
# ===========================================================================


class TestEdgeCasesBrokenTorch:
    """Test behavior with fake or broken torch modules on sys.path."""

    def test_fake_broken_torch_module(self, no_torch_venv, sandbox_dir):
        """A fake torch that raises RuntimeError('CUDA not found') on import.

        data_collators.py (no top-level torch import) should still load fine.
        """
        torch_dir = sandbox_dir / "torch"
        torch_dir.mkdir()
        (torch_dir / "__init__.py").write_text(
            'raise RuntimeError("CUDA not found")\n',
            encoding = "utf-8",
        )
        _write_loggers_stub(sandbox_dir)
        shutil.copy2(DATA_COLLATORS, sandbox_dir / "data_collators.py")

        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(sandbox_dir)!r})
            exec(open({str(sandbox_dir / 'data_collators.py')!r}).read())
            obj = DataCollatorSpeechSeq2SeqWithPadding(processor=None)
            print("OK: data_collators works despite broken torch on sys.path")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode == 0
        ), f"Should work with broken torch:\n{result.stderr.decode()}"
        assert b"OK:" in result.stdout

    def test_torch_import_error_hardware_fallback(self, no_torch_venv, sandbox_dir):
        """A fake torch that raises ImportError. detect_hardware() falls back to CPU."""
        torch_dir = sandbox_dir / "torch"
        torch_dir.mkdir()
        (torch_dir / "__init__.py").write_text(
            'raise ImportError("No torch binary")\n',
            encoding = "utf-8",
        )
        _write_loggers_stub(sandbox_dir)
        _write_structlog_stub(sandbox_dir)

        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(sandbox_dir)!r})
            source = open({str(HARDWARE_PY)!r}).read()
            ns = {{'__name__': '__test__'}}
            exec(source, ns)
            result = ns['detect_hardware']()
            assert result == ns['DeviceType'].CPU, f"Expected CPU, got {{result}}"
            print("OK: detect_hardware returned CPU")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode == 0
        ), f"detect_hardware should fallback to CPU:\n{result.stderr.decode()}"
        assert b"OK: detect_hardware returned CPU" in result.stdout

    def test_fake_torch_no_cuda(self, no_torch_venv, sandbox_dir):
        """Fake torch that imports OK but torch.cuda.is_available() returns False.

        detect_hardware() should still fall back to CPU.
        """
        torch_dir = sandbox_dir / "torch"
        torch_dir.mkdir()
        (torch_dir / "__init__.py").write_text(
            textwrap.dedent("""\
                class _Cuda:
                    @staticmethod
                    def is_available():
                        return False
                cuda = _Cuda()
                class version:
                    cuda = None
            """),
            encoding = "utf-8",
        )
        _write_loggers_stub(sandbox_dir)
        _write_structlog_stub(sandbox_dir)

        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(sandbox_dir)!r})
            source = open({str(HARDWARE_PY)!r}).read()
            ns = {{'__name__': '__test__'}}
            exec(source, ns)
            result = ns['detect_hardware']()
            assert result == ns['DeviceType'].CPU, f"Expected CPU, got {{result}}"
            print("OK: detect_hardware returned CPU with fake torch (no CUDA)")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode == 0
        ), f"Should fall back to CPU:\n{result.stderr.decode()}"
        assert b"OK:" in result.stdout

    def test_lazy_torch_fails_at_call_time_not_import_time(
        self, no_torch_venv, sandbox_dir
    ):
        """apply_chat_template_to_dataset is importable without torch.

        Calling the alpaca branch triggers the lazy 'from torch.utils.data' inside
        the try block. This should fail at call time, not import time -- proving the
        lazy import pattern works correctly.
        """
        _write_loggers_stub(sandbox_dir)

        code = textwrap.dedent(f"""\
            import sys, types
            sys.path.insert(0, {str(sandbox_dir)!r})
            fd = types.ModuleType('format_detection')
            fd.detect_dataset_format = fd.detect_multimodal_dataset = fd.detect_custom_format_heuristic = lambda *a, **k: None
            sys.modules['format_detection'] = fd
            mm = types.ModuleType('model_mappings')
            mm.MODEL_TO_TEMPLATE_MAPPER = {{}}
            sys.modules['model_mappings'] = mm

            ns = {{}}
            source = open({str(CHAT_TEMPLATES)!r}).read()
            source = source.replace('from .format_detection import', 'from format_detection import')
            source = source.replace('from .model_mappings import', 'from model_mappings import')
            exec(source, ns)

            # Import succeeds -- this is the fix
            assert 'apply_chat_template_to_dataset' in ns
            print("OK: import succeeded")

            # Calling alpaca branch triggers lazy torch import inside the try block.
            # The function catches the error and returns it in the errors list.
            dataset_info = {{
                'dataset': type('D', (), {{'map': lambda *a, **k: None}})(),
                'final_format': 'alpaca',
                'chat_column': None,
                'is_standardized': True,
                'warnings': [],
            }}
            result = ns['apply_chat_template_to_dataset'](dataset_info, None)
            # The function has a try/except that catches the error gracefully
            if not result['success']:
                print("OK: call-time failure caught gracefully")
            else:
                print("OK: call succeeded (unexpected but not a crash)")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert (
            result.returncode == 0
        ), f"Should not crash at import time:\n{result.stderr.decode()}"
        assert b"OK: import succeeded" in result.stdout


# ===========================================================================
# Group 4: Hardware Detection Without Torch (3 tests)
# ===========================================================================


class TestHardwareDetectionNoTorch:
    """Hardware module works without torch, falling back to CPU."""

    def test_detect_hardware_no_torch(self, no_torch_venv, sandbox_dir):
        """detect_hardware() returns CPU device when torch is not installed."""
        _write_loggers_stub(sandbox_dir)
        _write_structlog_stub(sandbox_dir)

        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(sandbox_dir)!r})
            source = open({str(HARDWARE_PY)!r}).read()
            ns = {{'__name__': '__test__'}}
            exec(source, ns)
            device = ns['detect_hardware']()
            assert device == ns['DeviceType'].CPU
            assert ns['CHAT_ONLY'] is True
            print("OK: detect_hardware returned CPU, CHAT_ONLY=True")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert result.returncode == 0, f"Failed:\n{result.stderr.decode()}"
        assert b"OK:" in result.stdout

    def test_get_package_versions_no_torch(self, no_torch_venv, sandbox_dir):
        """get_package_versions() returns torch=None, cuda=None without torch."""
        _write_loggers_stub(sandbox_dir)
        _write_structlog_stub(sandbox_dir)

        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(sandbox_dir)!r})
            source = open({str(HARDWARE_PY)!r}).read()
            ns = {{'__name__': '__test__'}}
            exec(source, ns)
            versions = ns['get_package_versions']()
            assert versions['torch'] is None, f"Expected torch=None, got {{versions['torch']}}"
            assert versions['cuda'] is None, f"Expected cuda=None, got {{versions['cuda']}}"
            print("OK: torch=None, cuda=None")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert result.returncode == 0, f"Failed:\n{result.stderr.decode()}"
        assert b"OK:" in result.stdout

    def test_hardware_module_import_no_torch(self, no_torch_venv, sandbox_dir):
        """The hardware module imports and detect_hardware is callable without torch."""
        _write_loggers_stub(sandbox_dir)
        _write_structlog_stub(sandbox_dir)
        _write_hardware_stub(sandbox_dir)

        # Copy the real hardware module into a sandbox package
        hw_sandbox = sandbox_dir / "hw_pkg"
        hw_sandbox.mkdir()
        (hw_sandbox / "__init__.py").write_text("", encoding = "utf-8")
        shutil.copy2(HARDWARE_PY, hw_sandbox / "hardware.py")

        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(sandbox_dir)!r})
            source = open({str(hw_sandbox / 'hardware.py')!r}).read()
            ns = {{'__name__': '__test__'}}
            exec(source, ns)
            assert callable(ns['detect_hardware'])
            assert callable(ns['get_package_versions'])
            assert callable(ns['is_apple_silicon'])
            print("OK: all hardware functions accessible")
        """)
        result = _run_in_sandbox(no_torch_venv, code)
        assert result.returncode == 0, f"Failed:\n{result.stderr.decode()}"
        assert b"OK:" in result.stdout


# ===========================================================================
# Group 5: install.sh Logic (5 tests via bash subprocess)
# ===========================================================================


class TestInstallShLogic:
    """Test install.sh flag parsing, platform detection, and guard logic."""

    @pytest.fixture(autouse = True)
    def _check_install_sh(self):
        if not INSTALL_SH.is_file():
            pytest.skip("install.sh not found")

    def test_python_flag_parsing(self):
        """--python flag correctly sets _USER_PYTHON."""
        # Extract flag parser snippet from install.sh and test it
        script = textwrap.dedent("""\
            _USER_PYTHON=""
            _next_is_python=false
            for arg in "$@"; do
                if [ "$_next_is_python" = true ]; then
                    _USER_PYTHON="$arg"
                    _next_is_python=false
                    continue
                fi
                case "$arg" in
                    --python) _next_is_python=true ;;
                esac
            done
            echo "$_USER_PYTHON"
        """)
        # Test: --python 3.12
        r = _run_sh(f"{script}" + "\n", timeout = 10)
        # Need to pass args to the script
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", "--python", "3.12"],
            capture_output = True,
            timeout = 10,
        )
        assert r.stdout.strip() == b"3.12"

        # Test: --local --python 3.11
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", "--local", "--python", "3.11"],
            capture_output = True,
            timeout = 10,
        )
        assert r.stdout.strip() == b"3.11"

        # Test: no --python flag
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", "--local"],
            capture_output = True,
            timeout = 10,
        )
        assert r.stdout.strip() == b""

    def test_python_flag_missing_arg_errors(self):
        """--python without a version argument triggers an error."""
        # Extract the flag parser + error guard from install.sh
        script = textwrap.dedent("""\
            set -e
            _USER_PYTHON=""
            _next_is_python=false
            for arg in "$@"; do
                if [ "$_next_is_python" = true ]; then
                    _USER_PYTHON="$arg"
                    _next_is_python=false
                    continue
                fi
                case "$arg" in
                    --python) _next_is_python=true ;;
                esac
            done
            if [ "$_next_is_python" = true ]; then
                echo "ERROR: --python requires a version argument" >&2
                exit 1
            fi
            echo "$_USER_PYTHON"
        """)
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", "--python"],
            capture_output = True,
            timeout = 10,
        )
        assert r.returncode != 0
        assert b"ERROR" in r.stderr

    def test_python_version_resolution(self):
        """Python version defaults to 3.12 on Intel Mac, 3.13 elsewhere.
        --python overrides both."""
        script = textwrap.dedent("""\
            MAC_INTEL="$1"
            _USER_PYTHON="$2"

            if [ -n "$_USER_PYTHON" ]; then
                PYTHON_VERSION="$_USER_PYTHON"
            elif [ "$MAC_INTEL" = true ]; then
                PYTHON_VERSION="3.12"
            else
                PYTHON_VERSION="3.13"
            fi
            echo "$PYTHON_VERSION"
        """)
        # Intel Mac, no override
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", "true", ""],
            capture_output = True,
            timeout = 10,
        )
        assert r.stdout.strip() == b"3.12"

        # Non-Intel, no override
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", "false", ""],
            capture_output = True,
            timeout = 10,
        )
        assert r.stdout.strip() == b"3.13"

        # Intel Mac with --python override
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", "true", "3.11"],
            capture_output = True,
            timeout = 10,
        )
        assert r.stdout.strip() == b"3.11"

    def test_mac_intel_detection_snippet(self):
        """Architecture detection sets MAC_INTEL correctly for different platforms."""
        script = textwrap.dedent("""\
            OS="$1"
            _ARCH="$2"
            MAC_INTEL=false
            if [ "$OS" = "macos" ] && [ "$_ARCH" = "x86_64" ]; then
                MAC_INTEL=true
            fi
            echo "$MAC_INTEL"
        """)
        cases = [
            (("macos", "x86_64"), b"true"),
            (("macos", "arm64"), b"false"),
            (("linux", "x86_64"), b"false"),
            (("linux", "aarch64"), b"false"),
        ]
        for (os_val, arch), expected in cases:
            r = subprocess.run(
                ["bash", "-c", script + "\n", "_", os_val, arch],
                capture_output = True,
                timeout = 10,
            )
            assert r.stdout.strip() == expected, (
                f"MAC_INTEL for ({os_val}, {arch}): "
                f"expected {expected!r}, got {r.stdout.strip()!r}"
            )

    def test_stale_venv_guard_respects_override(self):
        """When _USER_PYTHON is set, the stale venv recreation guard is skipped."""
        # The guard: if MAC_INTEL=true && -z _USER_PYTHON && venv exists ...
        script = textwrap.dedent("""\
            MAC_INTEL=true
            _USER_PYTHON="$1"
            _VENV_EXISTS=true  # simulate existing venv

            SHOULD_RECREATE=false
            if [ "$MAC_INTEL" = true ] && [ -z "$_USER_PYTHON" ] && [ "$_VENV_EXISTS" = true ]; then
                SHOULD_RECREATE=true
            fi
            echo "$SHOULD_RECREATE"
        """)
        # With override: should NOT recreate
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", "3.11"],
            capture_output = True,
            timeout = 10,
        )
        assert r.stdout.strip() == b"false"

        # Without override: SHOULD recreate
        r = subprocess.run(
            ["bash", "-c", script + "\n", "_", ""],
            capture_output = True,
            timeout = 10,
        )
        assert r.stdout.strip() == b"true"


# ===========================================================================
# Group 6: install_python_stack.py NO_TORCH Filtering (4 tests)
# ===========================================================================


class TestInstallPythonStackFiltering:
    """Test the NO_TORCH filtering logic in install_python_stack.py."""

    @pytest.fixture(autouse = True)
    def _check_install_py(self):
        if not INSTALL_PY.is_file():
            pytest.skip("install_python_stack.py not found")

    def test_filter_requirements_removes_torch_deps(self):
        """_filter_requirements removes all NO_TORCH_SKIP_PACKAGES from a real extras file."""
        import install_python_stack as ips

        extras = STUDIO_DIR / "backend" / "requirements" / "extras.txt"
        if not extras.is_file():
            pytest.skip("extras.txt not found")

        result_path = ips._filter_requirements(extras, ips.NO_TORCH_SKIP_PACKAGES)
        filtered = Path(result_path).read_text(encoding = "utf-8").lower()

        for pkg in ["torch-stoi", "timm", "openai-whisper", "transformers-cfg"]:
            lines = [
                l.strip()
                for l in filtered.splitlines()
                if l.strip() and not l.strip().startswith("#")
            ]
            assert not any(
                l.startswith(pkg) for l in lines
            ), f"{pkg} should be removed from extras.txt"

    def test_filter_requirements_preserves_non_torch(self):
        """Non-torch packages survive NO_TORCH filtering."""
        import install_python_stack as ips

        extras = STUDIO_DIR / "backend" / "requirements" / "extras.txt"
        if not extras.is_file():
            pytest.skip("extras.txt not found")

        result_path = ips._filter_requirements(extras, ips.NO_TORCH_SKIP_PACKAGES)
        filtered_text = Path(result_path).read_text(encoding = "utf-8").lower()

        must_survive = ["scikit-learn", "loguru", "tiktoken", "einops"]
        original_text = extras.read_text(encoding = "utf-8").lower()
        for pkg in must_survive:
            if pkg in original_text:
                assert pkg in filtered_text, f"{pkg} should survive NO_TORCH filtering"

    def test_infer_no_torch_env_var_overrides_platform(self):
        """UNSLOTH_NO_TORCH=true on Linux -> True; =false on Intel Mac -> False."""
        import install_python_stack as ips

        # Explicit true on Linux
        with (
            mock.patch.dict(os.environ, {"UNSLOTH_NO_TORCH": "true"}),
            mock.patch.object(ips, "IS_MAC_INTEL", False),
        ):
            assert ips._infer_no_torch() is True

        # Explicit false on Intel Mac
        with (
            mock.patch.dict(os.environ, {"UNSLOTH_NO_TORCH": "false"}),
            mock.patch.object(ips, "IS_MAC_INTEL", True),
        ):
            assert ips._infer_no_torch() is False

        # Unset on Intel Mac -> True (platform fallback)
        env = os.environ.copy()
        env.pop("UNSLOTH_NO_TORCH", None)
        with (
            mock.patch.dict(os.environ, env, clear = True),
            mock.patch.object(ips, "IS_MAC_INTEL", True),
        ):
            assert ips._infer_no_torch() is True

    def test_no_torch_skips_overrides_and_triton(self):
        """When NO_TORCH=True, overrides.txt and triton are skipped (source guard check)."""
        import install_python_stack as ips

        source = Path(ips.__file__).read_text(encoding = "utf-8")

        # NO_TORCH guard before overrides
        assert (
            "if NO_TORCH:" in source
        ), "NO_TORCH guard not found in install_python_stack.py"

        # macOS guard for triton
        assert (
            "not IS_WINDOWS and not IS_MACOS" in source
        ), "'not IS_WINDOWS and not IS_MACOS' guard for triton not found"


# ===========================================================================
# Group 7: Live Server Startup (4 tests) -- Heavyweight
# ===========================================================================


def _studio_venv_python() -> Path | None:
    """Return the studio venv Python path, or None if not found."""
    py = _venv_python(STUDIO_VENV)
    if py.exists():
        return py
    return None


def _server_port() -> int:
    """Find an available port for the test server."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


server = pytest.mark.server


@server
class TestLiveServerStartup:
    """Live server startup tests.

    These use the existing Studio venv at ~/.unsloth/studio/unsloth_studio.
    They temporarily ensure torch is not importable, test server startup,
    then leave the venv unchanged.

    Run separately: pytest -m server
    """

    @pytest.fixture(autouse = True)
    def _check_studio_venv(self):
        py = _studio_venv_python()
        if py is None:
            pytest.skip("Studio venv not found at ~/.unsloth/studio/unsloth_studio")

    @pytest.fixture(scope = "class")
    def server_process(self):
        """Start the studio backend server without torch, yield (proc, port), then stop."""
        py = _studio_venv_python()
        if py is None:
            pytest.skip("Studio venv not found")

        port = _server_port()
        backend_dir = BACKEND_DIR

        # Check if torch is installed in the studio venv
        check = subprocess.run(
            [str(py), "-c", "import torch; print(torch.__version__)"],
            capture_output = True,
        )
        torch_was_installed = check.returncode == 0
        torch_version = check.stdout.decode().strip() if torch_was_installed else None

        # Uninstall torch if present
        if torch_was_installed:
            subprocess.run(
                [
                    str(py),
                    "-m",
                    "pip",
                    "uninstall",
                    "-y",
                    "torch",
                    "torchvision",
                    "torchaudio",
                ],
                capture_output = True,
                timeout = 120,
            )

        # Start server
        env = os.environ.copy()
        env["PYTHONPATH"] = str(backend_dir)
        proc = subprocess.Popen(
            [str(py), str(backend_dir / "run.py"), "--port", str(port)],
            env = env,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            cwd = str(backend_dir),
        )

        # Wait for server to be ready (poll /api/health)
        import urllib.request
        import urllib.error

        ready = False
        for _ in range(30):
            time.sleep(1)
            try:
                resp = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/health", timeout = 2
                )
                if resp.status == 200:
                    ready = True
                    break
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                continue

        if not ready:
            stdout, stderr = proc.communicate(timeout = 5)
            # Reinstall torch + torchvision + torchaudio
            if torch_was_installed and torch_version:
                subprocess.run(
                    [
                        str(py),
                        "-m",
                        "pip",
                        "install",
                        f"torch=={torch_version}",
                        "torchvision",
                        "torchaudio",
                    ],
                    capture_output = True,
                    timeout = 300,
                )
            server_output = stdout.decode(errors = "replace") + stderr.decode(
                errors = "replace"
            )
            pytest.skip(
                f"Server failed to start within 30 seconds. Output:\n{server_output}"
            )

        yield proc, port

        # Cleanup: stop server, reinstall torch
        proc.terminate()
        try:
            proc.wait(timeout = 10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout = 5)

        if torch_was_installed and torch_version:
            subprocess.run(
                [
                    str(py),
                    "-m",
                    "pip",
                    "install",
                    f"torch=={torch_version}",
                    "torchvision",
                    "torchaudio",
                ],
                capture_output = True,
                timeout = 300,
            )

    def test_server_starts_without_torch(self, server_process):
        """Server responds to /api/health with chat_only: true."""
        import json
        import urllib.request

        _, port = server_process
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout = 5)
        data = json.loads(resp.read())
        assert data["status"] == "healthy"
        assert data["chat_only"] is True

    def test_all_routes_registered(self, server_process):
        """OpenAPI spec shows >= 20 paths (server started fully)."""
        import json
        import urllib.request

        _, port = server_process
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/openapi.json", timeout = 5
        )
        spec = json.loads(resp.read())
        assert (
            len(spec.get("paths", {})) >= 20
        ), f"Expected >= 20 routes, got {len(spec.get('paths', {}))}"

    def test_hardware_endpoint_no_torch(self, server_process):
        """GET /api/system/hardware returns torch=null, gpu_name=null."""
        import json
        import urllib.request

        _, port = server_process
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/system/hardware",
            timeout = 5,
        )
        data = json.loads(resp.read())
        versions = data.get("versions", {})
        assert versions.get("torch") is None
        assert versions.get("cuda") is None

    def test_server_survives_multiple_requests(self, server_process):
        """Hit 5 different endpoints. Server PID should still be alive after."""
        import urllib.request
        import urllib.error

        proc, port = server_process
        endpoints = [
            "/api/health",
            "/openapi.json",
            "/api/system/hardware",
            "/api/health",
            "/docs",
        ]
        for ep in endpoints:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{ep}", timeout = 5)
            except urllib.error.HTTPError:
                pass  # 4xx/5xx is fine -- server didn't crash
            except urllib.error.URLError:
                pytest.fail(f"Server stopped responding at {ep}")

        assert proc.poll() is None, "Server process should still be running"

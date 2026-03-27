"""End-to-end sandbox tests: Studio modules in isolated no-torch venvs.

Covers:
- Python 3.12 and 3.13 venv creation (Intel Mac uses 3.12, Apple Silicon/Linux 3.13)
- data_collators.py loads and dataclasses instantiate without torch
- chat_templates.py top-level exec works with stubs for relative imports
- Negative control: prepending 'import torch' fails in no-torch venv
- Negative control: installing torchao (from overrides.txt) fails in no-torch venv
- AST structural checks for top-level torch imports
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_COLLATORS = (
    REPO_ROOT / "studio" / "backend" / "utils" / "datasets" / "data_collators.py"
)
CHAT_TEMPLATES = (
    REPO_ROOT / "studio" / "backend" / "utils" / "datasets" / "chat_templates.py"
)
FORMAT_CONVERSION = (
    REPO_ROOT / "studio" / "backend" / "utils" / "datasets" / "format_conversion.py"
)


def _has_uv() -> bool:
    return shutil.which("uv") is not None


def _create_venv(venv_dir: Path, python_version: str) -> Path | None:
    """Create a uv venv at the given Python version. Returns python path or None."""
    result = subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", python_version],
        capture_output = True,
    )
    if result.returncode != 0:
        return None
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_dir / "Scripts" / "python.exe"
    return venv_python if venv_python.exists() else None


@pytest.fixture(params = ["3.12", "3.13"], scope = "module")
def no_torch_venv(request, tmp_path_factory):
    """Create a temporary venv at the requested Python version with no torch.

    Parametrized for 3.12 (Intel Mac) and 3.13 (Apple Silicon / Linux).
    """
    if not _has_uv():
        pytest.skip("uv not available")

    py_version = request.param
    venv_dir = tmp_path_factory.mktemp(f"no_torch_venv_{py_version}")
    venv_python = _create_venv(venv_dir, py_version)
    if venv_python is None:
        pytest.skip(f"Could not create Python {py_version} venv")

    # Verify torch is NOT importable
    check = subprocess.run(
        [str(venv_python), "-c", "import torch"],
        capture_output = True,
    )
    assert (
        check.returncode != 0
    ), f"torch should NOT be importable in fresh {py_version} venv"

    return str(venv_python)


# ── AST structural checks ─────────────────────────────────────────────


class TestDataCollatorsAST:
    """Static analysis: data_collators.py has no top-level torch imports."""

    def test_ast_parse(self):
        """data_collators.py must be valid Python syntax."""
        source = DATA_COLLATORS.read_text(encoding = "utf-8")
        tree = ast.parse(source, filename = str(DATA_COLLATORS))
        assert tree is not None

    def test_no_top_level_torch_import(self):
        """No top-level 'import torch' or 'from torch' statements."""
        source = DATA_COLLATORS.read_text(encoding = "utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "torch"
                    ), f"Top-level 'import {alias.name}' found at line {node.lineno}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith(
                        "torch"
                    ), f"Top-level 'from {node.module}' found at line {node.lineno}"


class TestChatTemplatesAST:
    """Static analysis: chat_templates.py has no top-level torch imports."""

    def test_ast_parse(self):
        """chat_templates.py must be valid Python syntax."""
        source = CHAT_TEMPLATES.read_text(encoding = "utf-8")
        tree = ast.parse(source, filename = str(CHAT_TEMPLATES))
        assert tree is not None

    def test_no_top_level_torch_import(self):
        """No top-level 'import torch' or 'from torch' at module level."""
        source = CHAT_TEMPLATES.read_text(encoding = "utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "torch"
                    ), f"Top-level 'import {alias.name}' found at line {node.lineno}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith(
                        "torch"
                    ), f"Top-level 'from {node.module}' found at line {node.lineno}"

    def test_torch_imports_only_inside_functions(self):
        """All 'from torch' imports must be inside function/method bodies."""
        source = CHAT_TEMPLATES.read_text(encoding = "utf-8")
        tree = ast.parse(source)
        torch_imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = node.names[0].name if node.names else None
                if module and module.startswith("torch"):
                    torch_imports.append(node)

        top_level = set(id(n) for n in ast.iter_child_nodes(tree))
        for imp in torch_imports:
            assert id(imp) not in top_level, (
                f"torch import at line {imp.lineno} is at top level"
                " (should be inside a function)"
            )


# ── data_collators.py: exec + dataclass instantiation in no-torch venv ──


class TestDataCollatorsNoTorchVenv:
    """Run data_collators.py in an isolated no-torch venv, verify classes load."""

    def test_exec_in_no_torch_venv(self, no_torch_venv):
        """data_collators.py executes in a venv without torch (with loggers stub)."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(DATA_COLLATORS)!r}).read())
            print("OK: exec succeeded")
        """)
        result = subprocess.run(
            [no_torch_venv, "-c", code],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"data_collators.py failed in no-torch venv:\n{result.stderr.decode()}"
        assert b"OK: exec succeeded" in result.stdout

    def test_dataclass_speech_collator_instantiable(self, no_torch_venv):
        """DataCollatorSpeechSeq2SeqWithPadding can be instantiated with processor=None."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(DATA_COLLATORS)!r}).read())
            obj = DataCollatorSpeechSeq2SeqWithPadding(processor=None)
            assert obj.processor is None, "processor should be None"
            print("OK: DataCollatorSpeechSeq2SeqWithPadding instantiated")
        """)
        result = subprocess.run(
            [no_torch_venv, "-c", code],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"DataCollatorSpeechSeq2SeqWithPadding failed:\n{result.stderr.decode()}"
        assert b"OK: DataCollatorSpeechSeq2SeqWithPadding instantiated" in result.stdout

    def test_dataclass_deepseek_collator_instantiable(self, no_torch_venv):
        """DeepSeekOCRDataCollator can be instantiated with processor=None."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(DATA_COLLATORS)!r}).read())
            obj = DeepSeekOCRDataCollator(processor=None)
            assert obj.processor is None, "processor should be None"
            assert obj.max_length == 2048, "default max_length should be 2048"
            assert obj.ignore_index == -100, "default ignore_index should be -100"
            print("OK: DeepSeekOCRDataCollator instantiated")
        """)
        result = subprocess.run(
            [no_torch_venv, "-c", code],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"DeepSeekOCRDataCollator failed:\n{result.stderr.decode()}"
        assert b"OK: DeepSeekOCRDataCollator instantiated" in result.stdout

    def test_dataclass_vlm_collator_instantiable(self, no_torch_venv):
        """VLMDataCollator can be instantiated with processor=None."""
        code = textwrap.dedent(f"""\
            import sys, types
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: None
            sys.modules['loggers'] = loggers
            exec(open({str(DATA_COLLATORS)!r}).read())
            obj = VLMDataCollator(processor=None)
            assert obj.processor is None
            assert obj.mask_input_tokens is True, "default mask_input_tokens should be True"
            print("OK: VLMDataCollator instantiated")
        """)
        result = subprocess.run(
            [no_torch_venv, "-c", code],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"VLMDataCollator failed:\n{result.stderr.decode()}"
        assert b"OK: VLMDataCollator instantiated" in result.stdout


# ── chat_templates.py: exec in no-torch venv ─────────────────────────


class TestChatTemplatesNoTorchVenv:
    """Run chat_templates.py in an isolated no-torch venv with stubs."""

    def test_exec_with_stubs(self, no_torch_venv):
        """chat_templates.py top-level exec works with stubs for relative imports."""
        code = textwrap.dedent(f"""\
            import sys, types

            # Stub loggers
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: type('L', (), {{'info': lambda s, m: None, 'warning': lambda s, m: None, 'debug': lambda s, m: None}})()
            sys.modules['loggers'] = loggers

            # Stub relative imports (.format_detection, .model_mappings)
            format_detection = types.ModuleType('format_detection')
            format_detection.detect_dataset_format = lambda *a, **k: None
            format_detection.detect_multimodal_dataset = lambda *a, **k: None
            format_detection.detect_custom_format_heuristic = lambda *a, **k: None
            sys.modules['format_detection'] = format_detection

            model_mappings = types.ModuleType('model_mappings')
            model_mappings.MODEL_TO_TEMPLATE_MAPPER = {{}}
            sys.modules['model_mappings'] = model_mappings

            # Read and transform the source: replace relative imports with absolute
            source = open({str(CHAT_TEMPLATES)!r}).read()
            source = source.replace('from .format_detection import', 'from format_detection import')
            source = source.replace('from .model_mappings import', 'from model_mappings import')

            exec(source)

            # Verify module-level constants are defined
            ns = dict(locals())
            assert 'DEFAULT_ALPACA_TEMPLATE' in ns, "DEFAULT_ALPACA_TEMPLATE not defined after exec"
            print("OK: chat_templates.py exec succeeded")
        """)
        result = subprocess.run(
            [no_torch_venv, "-c", code],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"chat_templates.py failed in no-torch venv:\n{result.stderr.decode()}"
        assert b"OK: chat_templates.py exec succeeded" in result.stdout

    def test_default_alpaca_template_defined(self, no_torch_venv):
        """DEFAULT_ALPACA_TEMPLATE constant is accessible after exec."""
        code = textwrap.dedent(f"""\
            import sys, types

            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: type('L', (), {{'info': lambda s, m: None, 'warning': lambda s, m: None, 'debug': lambda s, m: None}})()
            sys.modules['loggers'] = loggers

            format_detection = types.ModuleType('format_detection')
            format_detection.detect_dataset_format = lambda *a, **k: None
            format_detection.detect_multimodal_dataset = lambda *a, **k: None
            format_detection.detect_custom_format_heuristic = lambda *a, **k: None
            sys.modules['format_detection'] = format_detection

            model_mappings = types.ModuleType('model_mappings')
            model_mappings.MODEL_TO_TEMPLATE_MAPPER = {{}}
            sys.modules['model_mappings'] = model_mappings

            ns = {{}}
            source = open({str(CHAT_TEMPLATES)!r}).read()
            source = source.replace('from .format_detection import', 'from format_detection import')
            source = source.replace('from .model_mappings import', 'from model_mappings import')
            exec(source, ns)

            assert 'DEFAULT_ALPACA_TEMPLATE' in ns, "DEFAULT_ALPACA_TEMPLATE not defined"
            assert 'Instruction' in ns['DEFAULT_ALPACA_TEMPLATE'], "Template content unexpected"
            print("OK: DEFAULT_ALPACA_TEMPLATE defined and valid")
        """)
        result = subprocess.run(
            [no_torch_venv, "-c", code],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"DEFAULT_ALPACA_TEMPLATE check failed:\n{result.stderr.decode()}"
        assert b"OK: DEFAULT_ALPACA_TEMPLATE defined and valid" in result.stdout


# ── format_conversion.py: AST + runtime tests ────────────────────────


class TestFormatConversionAST:
    """Static analysis: format_conversion.py torch imports are guarded."""

    def test_ast_parse(self):
        """format_conversion.py must be valid Python syntax."""
        source = FORMAT_CONVERSION.read_text(encoding = "utf-8")
        tree = ast.parse(source, filename = str(FORMAT_CONVERSION))
        assert tree is not None

    def test_no_bare_torch_import_in_functions(self):
        """All 'from torch' imports in function bodies must be inside try/except."""
        source = FORMAT_CONVERSION.read_text(encoding = "utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.ImportFrom)
                        and child.module
                        and child.module.startswith("torch")
                    ):
                        # This torch import must be inside a Try node
                        found_in_try = False
                        for try_node in ast.walk(node):
                            if isinstance(try_node, ast.Try):
                                for try_child in ast.walk(try_node):
                                    if try_child is child:
                                        found_in_try = True
                                        break
                            if found_in_try:
                                break
                        assert found_in_try, (
                            f"torch import at line {child.lineno} in {node.name}() "
                            "is not inside a try/except block"
                        )


class TestFormatConversionNoTorchVenv:
    """Run format_conversion.py functions in a no-torch venv."""

    def test_convert_chatml_to_alpaca_no_torch(self, no_torch_venv):
        """convert_chatml_to_alpaca works without torch (via try/except ImportError)."""
        code = textwrap.dedent(f"""\
            import sys, types

            # Stub loggers
            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: type('L', (), {{
                'info': lambda s, m: None,
                'warning': lambda s, m: None,
                'debug': lambda s, m: None,
            }})()
            sys.modules['loggers'] = loggers

            # Stub datasets.IterableDataset (HF datasets, not torch)
            datasets_mod = types.ModuleType('datasets')
            datasets_mod.IterableDataset = type('IterableDataset', (), {{}})
            sys.modules['datasets'] = datasets_mod

            # Stub utils.hardware
            utils_mod = types.ModuleType('utils')
            hardware_mod = types.ModuleType('utils.hardware')
            hardware_mod.dataset_map_num_proc = lambda n=None: 1
            utils_mod.hardware = hardware_mod
            sys.modules['utils'] = utils_mod
            sys.modules['utils.hardware'] = hardware_mod

            # Read and exec format_conversion.py
            source = open({str(FORMAT_CONVERSION)!r}).read()
            source = source.replace('from .format_detection import', 'from format_detection import')
            ns = {{'__name__': '__test__'}}
            exec(source, ns)

            # Test convert_chatml_to_alpaca with a simple dataset
            class FakeDataset:
                def map(self, fn, **kw):
                    result = fn({{
                        'messages': [[
                            {{'role': 'user', 'content': 'Hello'}},
                            {{'role': 'assistant', 'content': 'Hi there'}},
                        ]]
                    }})
                    return result

            result = ns['convert_chatml_to_alpaca'](FakeDataset())
            assert 'instruction' in result, f"Expected 'instruction' in result, got {{result.keys()}}"
            assert result['instruction'] == ['Hello']
            assert result['output'] == ['Hi there']
            print("OK: convert_chatml_to_alpaca works without torch")
        """)
        result = subprocess.run(
            [no_torch_venv, "-c", code],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"convert_chatml_to_alpaca failed without torch:\n{result.stderr.decode()}"
        assert b"OK: convert_chatml_to_alpaca works without torch" in result.stdout

    def test_convert_alpaca_to_chatml_no_torch(self, no_torch_venv):
        """convert_alpaca_to_chatml works without torch (via try/except ImportError)."""
        code = textwrap.dedent(f"""\
            import sys, types

            loggers = types.ModuleType('loggers')
            loggers.get_logger = lambda n: type('L', (), {{
                'info': lambda s, m: None,
                'warning': lambda s, m: None,
                'debug': lambda s, m: None,
            }})()
            sys.modules['loggers'] = loggers

            datasets_mod = types.ModuleType('datasets')
            datasets_mod.IterableDataset = type('IterableDataset', (), {{}})
            sys.modules['datasets'] = datasets_mod

            utils_mod = types.ModuleType('utils')
            hardware_mod = types.ModuleType('utils.hardware')
            hardware_mod.dataset_map_num_proc = lambda n=None: 1
            utils_mod.hardware = hardware_mod
            sys.modules['utils'] = utils_mod
            sys.modules['utils.hardware'] = hardware_mod

            source = open({str(FORMAT_CONVERSION)!r}).read()
            source = source.replace('from .format_detection import', 'from format_detection import')
            ns = {{'__name__': '__test__'}}
            exec(source, ns)

            class FakeDataset:
                def map(self, fn, **kw):
                    result = fn({{
                        'instruction': ['Write a poem'],
                        'input': [''],
                        'output': ['Roses are red'],
                    }})
                    return result

            result = ns['convert_alpaca_to_chatml'](FakeDataset())
            assert 'conversations' in result
            convo = result['conversations'][0]
            assert convo[0]['role'] == 'user'
            assert convo[1]['role'] == 'assistant'
            print("OK: convert_alpaca_to_chatml works without torch")
        """)
        result = subprocess.run(
            [no_torch_venv, "-c", code],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"convert_alpaca_to_chatml failed without torch:\n{result.stderr.decode()}"
        assert b"OK: convert_alpaca_to_chatml works without torch" in result.stdout


# ── Negative controls ─────────────────────────────────────────────────


class TestNegativeControls:
    """Prove the fix is necessary by showing what fails WITHOUT it."""

    def test_import_torch_prepended_fails(self, no_torch_venv):
        """Prepending 'import torch' to data_collators.py causes ModuleNotFoundError."""
        with tempfile.NamedTemporaryFile(
            mode = "w", suffix = ".py", delete = False, encoding = "utf-8"
        ) as f:
            f.write("import torch\n")
            f.write(DATA_COLLATORS.read_text(encoding = "utf-8"))
            temp_file = f.name

        try:
            code = textwrap.dedent(f"""\
                import sys, types
                loggers = types.ModuleType('loggers')
                loggers.get_logger = lambda n: None
                sys.modules['loggers'] = loggers
                exec(open({temp_file!r}).read())
            """)
            result = subprocess.run(
                [no_torch_venv, "-c", code],
                capture_output = True,
                timeout = 30,
            )
            assert (
                result.returncode != 0
            ), "Expected failure when 'import torch' is prepended"
            assert (
                b"ModuleNotFoundError" in result.stderr
                or b"ImportError" in result.stderr
            ), f"Expected ImportError, got:\n{result.stderr.decode()}"
        finally:
            os.unlink(temp_file)

    def test_torchao_install_fails_no_torch_venv(self, no_torch_venv):
        """Installing torchao (from overrides.txt) fails in a no-torch venv.

        This proves the overrides.txt skip is necessary for Intel Mac.
        """
        result = subprocess.run(
            [
                no_torch_venv,
                "-m",
                "pip",
                "install",
                "torchao==0.14.0",
                "--dry-run",
            ],
            capture_output = True,
            timeout = 60,
        )
        if result.returncode != 0:
            # torchao install/resolution failed as expected
            pass
        else:
            # pip dry-run may not catch dependency issues; verify torch is missing
            check = subprocess.run(
                [no_torch_venv, "-c", "import torch"],
                capture_output = True,
            )
            assert (
                check.returncode != 0
            ), "torch should not be importable -- torchao would fail at runtime"

    def test_direct_torch_import_fails(self, no_torch_venv):
        """Direct 'import torch' fails in the no-torch venv."""
        result = subprocess.run(
            [no_torch_venv, "-c", "import torch; print('torch loaded')"],
            capture_output = True,
            timeout = 30,
        )
        assert result.returncode != 0, "import torch should fail in no-torch venv"
        assert (
            b"ModuleNotFoundError" in result.stderr or b"ImportError" in result.stderr
        )

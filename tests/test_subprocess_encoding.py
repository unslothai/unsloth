import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def _load_subprocess_encoding_module():
    # Loading through `import unsloth...` would execute the package initializer
    # whose GPU import behavior this helper is intended to protect.
    path = Path(__file__).resolve().parents[1] / "unsloth" / "_subprocess_encoding.py"
    spec = importlib.util.spec_from_file_location("_unsloth_subprocess_encoding_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invalid_utf8_command():
    return [
        sys.executable,
        "-c",
        "import os; os.write(1, b'\\xf8GPU')",
    ]


def test_replaces_invalid_text_output_and_restores_popen():
    module = _load_subprocess_encoding_module()
    original_popen = subprocess.Popen

    with pytest.raises(UnicodeDecodeError):
        subprocess.check_output(_invalid_utf8_command(), encoding = "utf-8")

    with module.replace_invalid_subprocess_text():
        output = subprocess.check_output(_invalid_utf8_command(), encoding = "utf-8")
        assert output == "\ufffdGPU"

    assert subprocess.Popen is original_popen


def test_preserves_explicit_strict_decoding_policy():
    module = _load_subprocess_encoding_module()

    with module.replace_invalid_subprocess_text():
        with pytest.raises(UnicodeDecodeError):
            subprocess.check_output(
                _invalid_utf8_command(),
                encoding = "utf-8",
                errors = "strict",
            )


def test_restores_popen_when_context_body_raises():
    module = _load_subprocess_encoding_module()
    original_popen = subprocess.Popen

    with pytest.raises(RuntimeError, match = "backend import failed"):
        with module.replace_invalid_subprocess_text():
            assert subprocess.Popen is not original_popen
            raise RuntimeError("backend import failed")

    assert subprocess.Popen is original_popen

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def _load_subprocess_encoding_module():
    module_path = Path(__file__).resolve().parents[1] / "unsloth" / "_subprocess_encoding.py"
    spec = importlib.util.spec_from_file_location(
        "unsloth_subprocess_encoding_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invalid_utf8_command():
    return [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(bytes([0xf8]))",
    ]


def test_replaces_invalid_bytes_for_text_subprocesses():
    helper = _load_subprocess_encoding_module()

    with pytest.raises(UnicodeDecodeError):
        subprocess.run(
            _invalid_utf8_command(),
            stdout = subprocess.PIPE,
            check = True,
            encoding = "utf-8",
        )

    with helper.replace_subprocess_decode_errors():
        result = subprocess.run(
            _invalid_utf8_command(),
            stdout = subprocess.PIPE,
            check = True,
            encoding = "utf-8",
        )

    assert result.stdout == "\ufffd"


def test_preserves_binary_subprocess_output():
    helper = _load_subprocess_encoding_module()

    with helper.replace_subprocess_decode_errors():
        result = subprocess.run(
            _invalid_utf8_command(),
            stdout = subprocess.PIPE,
            check = True,
        )

    assert result.stdout == bytes([0xF8])


def test_restores_popen_after_success_and_error():
    helper = _load_subprocess_encoding_module()
    original_popen = subprocess.Popen

    with helper.replace_subprocess_decode_errors():
        assert subprocess.Popen is not original_popen
    assert subprocess.Popen is original_popen

    with pytest.raises(RuntimeError):
        with helper.replace_subprocess_decode_errors():
            assert subprocess.Popen is not original_popen
            raise RuntimeError("stop initialization")

    assert subprocess.Popen is original_popen

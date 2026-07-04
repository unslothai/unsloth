import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock

# Add the module path to sys.path to import from unsloth
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_mock_run(make_returncode = 1, cmake_returncode = 0):
    """Return a mock for subprocess.run that records all calls."""
    captured = []
    call_index = [0]

    def mock_run(args, **kwargs):
        captured.append(args)
        result = MagicMock()
        # First call is `make clean`; subsequent calls are cmake
        result.returncode = make_returncode if call_index[0] == 0 else cmake_returncode
        call_index[0] += 1
        return result

    return mock_run, captured


def test_make_clean_uses_list_args():
    """
    Security invariant: the `make clean` subprocess.run call must use a list of
    arguments (not a shell string) to prevent shell injection.
    """
    pytest.importorskip("unsloth.save")
    import unsloth.save as save_module

    mock_run, captured = _make_mock_run(make_returncode = 0)

    with patch.object(save_module.subprocess, "run", side_effect = mock_run):
        with patch.object(save_module.subprocess, "Popen", return_value = MagicMock()):
            try:
                save_module.install_llama_cpp_make_non_blocking()
            except Exception:
                pass  # Errors unrelated to our assertion are acceptable

    assert len(captured) > 0, "subprocess.run was never called"

    for args in captured:
        assert isinstance(args, list), (
            f"subprocess.run called with a string instead of a list: {args!r}\n"
            "Passing a string (especially with shell=True) enables shell injection."
        )
        for arg in args:
            assert ";" not in arg, f"Shell metacharacter ';' in argument: {arg!r}"
            assert "||" not in arg, f"Shell metacharacter '||' in argument: {arg!r}"
            assert "&&" not in arg, f"Shell metacharacter '&&' in argument: {arg!r}"
            assert "`" not in arg, f"Shell metacharacter '`' in argument: {arg!r}"
            assert "$(" not in arg, f"Shell metacharacter '$(' in argument: {arg!r}"


def test_cmake_configure_uses_list_args():
    """
    Security invariant: the cmake configure subprocess.run call (triggered when
    `make clean` returns non-zero) must also use a list of arguments.
    """
    pytest.importorskip("unsloth.save")
    import unsloth.save as save_module

    # make returns 1 → triggers cmake fallback path
    mock_run, captured = _make_mock_run(make_returncode = 1, cmake_returncode = 0)

    with patch.object(save_module.subprocess, "run", side_effect = mock_run):
        with patch.object(save_module.subprocess, "Popen", return_value = MagicMock()):
            try:
                save_module.install_llama_cpp_make_non_blocking()
            except Exception:
                pass

    assert (
        len(captured) >= 2
    ), f"Expected at least 2 subprocess.run calls (make + cmake), got {len(captured)}"

    for args in captured:
        assert isinstance(
            args, list
        ), f"subprocess.run called with a string instead of a list: {args!r}"
        for arg in args:
            assert ";" not in arg, f"Shell metacharacter ';' in argument: {arg!r}"
            assert "||" not in arg, f"Shell metacharacter '||' in argument: {arg!r}"
            assert "&&" not in arg, f"Shell metacharacter '&&' in argument: {arg!r}"
            assert "`" not in arg, f"Shell metacharacter '`' in argument: {arg!r}"
            assert "$(" not in arg, f"Shell metacharacter '$(' in argument: {arg!r}"


def test_make_not_found_falls_through_to_cmake():
    """
    When `make` is not installed (FileNotFoundError), the code must fall through
    to the cmake path rather than crashing with an unhandled exception.
    """
    pytest.importorskip("unsloth.save")
    import unsloth.save as save_module

    call_index = [0]

    def mock_run(args, **kwargs):
        call_index[0] += 1
        if call_index[0] == 1:
            raise FileNotFoundError("make not found")
        result = MagicMock()
        result.returncode = 0  # cmake succeeds
        return result

    with patch.object(save_module.subprocess, "run", side_effect = mock_run):
        with patch.object(save_module.subprocess, "Popen", return_value = MagicMock()):
            # Should NOT raise FileNotFoundError; cmake path should be attempted
            try:
                save_module.install_llama_cpp_make_non_blocking()
            except FileNotFoundError as exc:
                pytest.fail(f"FileNotFoundError from missing `make` was not caught: {exc}")
            except Exception:
                pass  # Other errors (e.g. RuntimeError from cmake) are acceptable

    assert call_index[0] >= 2, "cmake was never attempted after make was not found"


def test_cmake_not_found_raises_runtime_error():
    """
    When `cmake` is not installed (FileNotFoundError), the code must raise a
    descriptive RuntimeError rather than propagating the raw FileNotFoundError.
    """
    pytest.importorskip("unsloth.save")
    import unsloth.save as save_module

    call_index = [0]

    def mock_run(args, **kwargs):
        call_index[0] += 1
        if call_index[0] == 1:
            # make not found → triggers cmake path
            raise FileNotFoundError("make not found")
        # cmake not found
        raise FileNotFoundError("cmake not found")

    with patch.object(save_module.subprocess, "run", side_effect = mock_run):
        with patch.object(save_module.subprocess, "Popen", return_value = MagicMock()):
            with pytest.raises(RuntimeError):
                save_module.install_llama_cpp_make_non_blocking()

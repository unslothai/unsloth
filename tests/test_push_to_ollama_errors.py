# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A failed `ollama create` / `ollama push` must raise, never reach "Successfully pushed"."""

import ast
import io
import os
import subprocess
import sys
import types

import pytest

_TESTS = os.path.dirname(__file__)
_SAVE = os.path.join(_TESTS, os.pardir, "unsloth", "save.py")

_FUNCTIONS = (
    "_check_ollama_server",
    "_run_ollama_command",
    "create_ollama_model",
    "push_to_ollama_hub",
    "push_to_ollama",
)


def _extract(name):
    with open(_SAVE, encoding = "utf-8") as f:
        source = f.read()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in save.py")


class _FakePopen:
    def __init__(
        self,
        args,
        returncode = 0,
        out = "",
    ):
        self.args = list(args)
        self._returncode = returncode
        self.stdout = io.StringIO(out)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stdout.close()
        return False

    def wait(self):
        return self._returncode


def _load(run = None, popen = None):
    """Exec the real ollama functions beside a fake `subprocess` module."""
    calls = {"run": [], "popen": []}

    def _run(args, **_kwargs):
        calls["run"].append(list(args))
        if run is not None:
            return run(args)
        return types.SimpleNamespace(returncode = 0, stdout = "Ollama is running")

    def _popen(args, **_kwargs):
        calls["popen"].append(list(args))
        if popen is not None:
            return popen(args)
        return _FakePopen(args)

    fake_subprocess = types.SimpleNamespace(
        run = _run,
        Popen = _popen,
        PIPE = subprocess.PIPE,
        STDOUT = subprocess.STDOUT,
        TimeoutExpired = subprocess.TimeoutExpired,
    )
    namespace = {
        "subprocess": fake_subprocess,
        "sys": sys,
        "create_ollama_modelfile": lambda **_: "FROM ./model.gguf\n",
    }
    for name in _FUNCTIONS:
        exec(compile(_extract(name), name, "exec"), namespace)
    return namespace, calls


def _push(namespace):
    namespace["push_to_ollama"](
        tokenizer = "TOKENIZER",
        base_model_name = "unsloth/llama-3-8b-Instruct",
        gguf_location = "./model.gguf",
        username = "user",
        model_name = "my-model",
        tag = "latest",
    )


def test_successful_create_and_push_prints_success(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    namespace, calls = _load()

    _push(namespace)

    out = capsys.readouterr().out
    assert "Successfully pushed to ollama" in out
    assert calls["popen"] == [
        ["ollama", "create", "user/my-model:latest", "-f", "Modelfile_my-model"],
        ["ollama", "push", "user/my-model:latest"],
    ]


def test_failed_create_raises_and_never_pushes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    namespace, calls = _load(popen = lambda args: _FakePopen(args, returncode = 1, out = "boom\n"))

    with pytest.raises(RuntimeError, match = r"ollama create user/my-model:latest"):
        _push(namespace)

    out = capsys.readouterr().out
    assert "Successfully pushed to ollama" not in out
    assert "MODEL PUBLISHED" not in out
    # A failed create must not push whatever stale local tag is lying around.
    assert calls["popen"] == [
        ["ollama", "create", "user/my-model:latest", "-f", "Modelfile_my-model"]
    ]


def test_failed_push_raises_and_prints_no_success(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    returncodes = iter([0, 1])
    namespace, calls = _load(popen = lambda args: _FakePopen(args, returncode = next(returncodes)))

    with pytest.raises(RuntimeError, match = r"ollama push user/my-model:latest"):
        _push(namespace)

    out = capsys.readouterr().out
    assert "MODEL CREATED SUCCESSFULLY" in out
    assert "MODEL PUBLISHED SUCCESSFULLY" not in out
    assert "Successfully pushed to ollama" not in out
    assert len(calls["popen"]) == 2


def test_missing_ollama_executable_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def _popen(args):
        raise FileNotFoundError(2, "No such file or directory", "ollama")

    namespace, calls = _load(popen = _popen)

    with pytest.raises(RuntimeError, match = "not installed or not on PATH"):
        _push(namespace)

    out = capsys.readouterr().out
    assert "Successfully pushed to ollama" not in out
    assert [argv[1] for argv in calls["popen"]] == ["create"]


def test_health_check_timeout_stops_before_create(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def _run(args):
        raise subprocess.TimeoutExpired(cmd = args, timeout = 3)

    namespace, calls = _load(run = _run)

    with pytest.raises(RuntimeError, match = "Timed out"):
        _push(namespace)

    assert "Successfully pushed to ollama" not in capsys.readouterr().out
    assert calls["popen"] == []


def test_missing_curl_still_runs_ollama(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def _run(args):
        raise FileNotFoundError(2, "No such file or directory", "curl")

    namespace, calls = _load(run = _run)

    _push(namespace)

    assert "Successfully pushed to ollama" in capsys.readouterr().out
    assert len(calls["popen"]) == 2

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-fit-params probe: say why a measurement failed, and only cache answers that repeat."""

import sys

import pytest

from core.inference import llama_cpp as lc
from core.inference.llama_cpp import LlamaCppBackend

_TABLE = """0.00.408.825 I common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
0.00.429.748 I common_memory_breakdown_print: |   - MTL0 (Apple M1)    |  5461 = 5460 + (3642 =  3025 +      28 +     589) +       -3642 |
0.00.429.748 I common_memory_breakdown_print: |   - Host               |                 2352 =  2288 +       0 +      64                |
"""

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason = "shell script stands in for the binary"
)


def _install(
    tmp_path,
    script,
    mode = 0o755,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-server").write_text("")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    probe = bin_dir / "llama-fit-params"
    if script is not None:
        probe.write_text(script)
        probe.chmod(mode)
    return str(bin_dir / "llama-server"), str(model), probe


def _counting(tmp_path, body):
    calls = tmp_path / "calls"
    return calls, f'#!/bin/sh\necho x >> "{calls}"\n{body}\n'


def _ncalls(calls):
    return len(calls.read_text(encoding = "utf-8").splitlines()) if calls.exists() else 0


def test_missing_probe_is_named(tmp_path):
    binary, model, _ = _install(tmp_path, None)
    backend = LlamaCppBackend()
    assert backend._metal_measured_model_mib(binary, model) is None
    assert "not found" in backend._metal_probe_failure


@posix_only
def test_non_executable_probe_is_named_and_reprobed_after_repair(tmp_path):
    table = _TABLE.replace("\n", "\\n")
    calls, script = _counting(tmp_path, f'printf "{table}"')
    binary, model, probe = _install(tmp_path, script, mode = 0o644)
    backend = LlamaCppBackend()
    assert backend._metal_measured_model_mib(binary, model) is None
    assert "not executable" in backend._metal_probe_failure
    assert _ncalls(calls) == 0
    probe.chmod(0o755)  # the installer's exec-bit repair
    assert backend._metal_measured_model_mib(binary, model) == (3025, 2288, 2)
    assert backend._metal_probe_failure is None


@posix_only
def test_nonzero_exit_is_named_with_its_output_and_not_cached(tmp_path):
    calls, script = _counting(tmp_path, 'echo "ggml_metal_init: error: boom" >&2\nexit 3')
    binary, model, _ = _install(tmp_path, script)
    backend = LlamaCppBackend()
    assert backend._metal_measured_model_mib(binary, model) is None
    assert "exited with code 3" in backend._metal_probe_failure
    assert "boom" in backend._metal_probe_failure
    assert backend._metal_measured_model_mib(binary, model) is None
    assert _ncalls(calls) == 2


@posix_only
def test_timeout_is_named_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "_METAL_FIT_PROBE_TIMEOUT_S", 0.5)
    calls, script = _counting(tmp_path, "sleep 5")
    binary, model, _ = _install(tmp_path, script)
    backend = LlamaCppBackend()
    assert backend._metal_measured_model_mib(binary, model) is None
    assert "timed out" in backend._metal_probe_failure
    assert backend._metal_measured_model_mib(binary, model) is None
    assert _ncalls(calls) == 2


@posix_only
def test_unparsable_output_is_named_and_cached(tmp_path):
    calls, script = _counting(tmp_path, 'echo "error: invalid argument: -lv"')
    binary, model, _ = _install(tmp_path, script)
    backend = LlamaCppBackend()
    assert backend._metal_measured_model_mib(binary, model) is None
    assert "no Metal memory breakdown" in backend._metal_probe_failure
    assert "invalid argument" in backend._metal_probe_failure
    assert backend._metal_measured_model_mib(binary, model) is None
    assert "no Metal memory breakdown" in backend._metal_probe_failure
    assert _ncalls(calls) == 1


@posix_only
def test_a_valid_table_still_parses_and_caches(tmp_path):
    table = _TABLE.replace("\n", "\\n")
    calls, script = _counting(tmp_path, f'printf "{table}"')
    binary, model, _ = _install(tmp_path, script)
    backend = LlamaCppBackend()
    assert backend._metal_measured_model_mib(binary, model) == (3025, 2288, 2)
    assert backend._metal_measured_model_mib(binary, model) == (3025, 2288, 2)
    assert backend._metal_probe_failure is None
    assert _ncalls(calls) == 1


def test_the_launch_log_carries_the_reason():
    """The "stay charged" line names the cause instead of a fixed guess."""
    import inspect

    src = inspect.getsource(LlamaCppBackend._metal_demand_paged_embedding_bytes)
    assert "_metal_probe_failure" in src
    assert "missing or unreadable" not in src


@posix_only
def test_a_failed_probe_runs_once_per_load(tmp_path):
    calls, script = _counting(tmp_path, "exit 3")
    binary, model, _ = _install(tmp_path, script)
    backend = LlamaCppBackend()
    with lc._load_probe_memo_scope():
        assert backend._metal_measured_model_mib(binary, model) is None
        assert backend._metal_measured_model_mib(binary, model) is None
        assert "exited with code 3" in backend._metal_probe_failure
    assert _ncalls(calls) == 1
    # The next load probes again: a cold shader compile may have finished by then.
    with lc._load_probe_memo_scope():
        assert backend._metal_measured_model_mib(binary, model) is None
    assert _ncalls(calls) == 2

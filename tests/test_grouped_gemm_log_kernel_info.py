# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""log_kernel_info runs after every grouped GEMM launch: with debug logging off it must not
read or format the compiled kernel's metadata (that cost ~12 us per launch)."""

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.fixture(scope = "module")
def interface():
    try:
        from unsloth.kernels.moe.grouped_gemm import interface
    except Exception as exc:
        if torch.cuda.is_available():
            raise  # a real import regression, not an unsupported host
        pytest.skip(f"grouped_gemm needs a GPU build: {exc}")
    return interface


class _Untouchable:
    def __getattr__(self, name):
        raise AssertionError(f"read compiled_kernel.{name} with debug logging off")


def test_no_metadata_read_without_debug_logging(interface):
    level = interface.logger.level
    interface.logger.setLevel(logging.WARNING)
    try:
        interface.log_kernel_info(_Untouchable(), best_config = _Untouchable())
    finally:
        interface.logger.setLevel(level)


def test_debug_logging_still_reports(interface, monkeypatch, caplog):
    class Kernel:
        name, n_regs, n_spills, metadata = "k", 32, 0, "meta"

    with caplog.at_level(logging.DEBUG, logger = interface.logger.name):
        interface.log_kernel_info(Kernel(), best_config = "cfg")
    assert "k: n_regs=32 n_spills=0 metadata=meta" in caplog.text
    assert "k autotuned best_config: cfg" in caplog.text

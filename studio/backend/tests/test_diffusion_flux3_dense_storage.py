# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A dense float16 FLUX.3 transformer widens on the host only when that copy fits host RAM; on a 12 GB host it reads
bf16 as stored and casts per op instead (bit-identical: test_diffusion_flux3_half_storage), so the load is not
killed or stalled converting 11 GB of weights."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_flux3 as f3  # noqa: E402
from core.inference import diffusion_small_host as sh  # noqa: E402

FP16, BF16, FP32 = torch.float16, torch.bfloat16, torch.float32
MIB = 1 << 20


@pytest.fixture
def host(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    # 64 MiB of bf16 weights plus an int64 buffer that is not a weight byte
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    save_file(
        {"w": torch.zeros(32 * MIB, dtype = BF16), "ids": torch.zeros(1024, dtype = torch.int64)},
        str(path),
    )
    state = {"total": 12_000, "available": 9_000, "files": [path]}
    monkeypatch.setattr(sh, "host_ram_mib", lambda: (state["total"], state["available"]))
    monkeypatch.setattr(f3, "_component_files", lambda *a, **k: list(state["files"]))
    for env in (f3.DENSE_STORAGE_ENV, sh.HOST_RAM_CHECK_ENV):
        monkeypatch.delenv(env, raising = False)
    return state


def test_stored_float_bytes_skip_integer_tensors(host):
    assert f3._stored_float_mib(host["files"]) == 64


def _scaled(host, stored_mib):
    # the fixture holds 64 MiB: scale the widen peak so the same file stands for a klein-sized transformer
    return stored_mib / 64


def test_a_widened_copy_that_does_not_fit_reads_as_stored(host, monkeypatch):
    monkeypatch.setattr(f3, "_WIDEN_PEAK", 1.6 * _scaled(host, 11_100))
    why = f3.widened_transformer_exceeds_host("repo", None, True)
    assert why is not None and "9000 MiB available" in why
    assert f3.transformer_load_storage(BF16, FP16, None, "repo", None, True) is BF16


def test_a_widened_copy_that_fits_keeps_the_widened_load(host, monkeypatch):
    monkeypatch.setattr(f3, "_WIDEN_PEAK", 1.6 * _scaled(host, 11_100))
    host["total"], host["available"] = 51_000, 48_000
    assert f3.widened_transformer_exceeds_host("repo", None, True) is None
    assert f3.transformer_load_storage(BF16, FP16, None, "repo", None, True) is None


def test_explicit_choices_and_other_widths_are_unchanged(host, monkeypatch):
    monkeypatch.setattr(f3, "_WIDEN_PEAK", 1e6)
    pick = f3.transformer_load_storage
    # a native quant asked for storage, or the caller refused it: no host reading changes that
    assert pick(BF16, FP16, True, "repo", None, True) is BF16
    assert pick(BF16, FP16, False, "repo", None, True) is None
    # float32 already reads as stored; no storage dtype (bf16 card, kill switch) never does
    assert pick(BF16, FP32, None, "repo", None, True) is BF16
    assert pick(None, FP16, None, "repo", None, True) is None


@pytest.mark.parametrize("env", [f3.DENSE_STORAGE_ENV, sh.HOST_RAM_CHECK_ENV])
def test_kill_switches_keep_the_widened_load(host, monkeypatch, env):
    monkeypatch.setattr(f3, "_WIDEN_PEAK", 1e6)
    monkeypatch.setenv(env, "0")
    assert f3.widened_transformer_exceeds_host("repo", None, True) is None
    assert f3.transformer_load_storage(BF16, FP16, None, "repo", None, True) is None


def test_unreadable_sizes_or_host_keep_the_widened_load(host, monkeypatch):
    monkeypatch.setattr(f3, "_WIDEN_PEAK", 1e6)
    host["files"] = []
    assert f3.widened_transformer_exceeds_host("repo", None, True) is None
    monkeypatch.setattr(sh, "host_ram_mib", lambda: (None, None))
    assert f3.widened_transformer_exceeds_host("repo", None, True) is None

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#9550: forcing MTP after Auto dropped it must re-fit the replayed context against the reserve."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _backend, _launch  # noqa: E402

GB = 1024**3

GPUS = [(0, 45914, 46080), (1, 8032, 8176)]
MODEL_BYTES = int(31.0 * GB)
NATIVE_CTX = 262144
REPORTED_CTX = 112896
KV_BYTES_AT_REPORTED_CTX = int(11.2 * GB)
MTP_BYTES_AT_REPORTED_CTX = int(2.18 * GB)


def _reporter_backend(tmp_path: Path):
    """Reporter-sized backend; KV and MTP terms stubbed linear in context so only the fit is tested."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = GPUS)

    def read_metadata(_path):
        backend._nextn_predict_layers = 1
        backend._n_layers = 48
        backend._n_kv_heads = 8
        backend._n_heads = 40
        backend._embedding_length = 5120
        backend._kv_key_length = 128
        backend._kv_value_length = 128
        backend._context_length = NATIVE_CTX

    def _ctx_of(args, kwargs):
        if "n_ctx" in kwargs:
            return int(kwargs["n_ctx"] or 0)
        return int(args[0]) if args else 0

    backend._read_gguf_metadata = read_metadata
    backend._get_gguf_size_bytes = lambda _path: MODEL_BYTES
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *a, **k: int(
        KV_BYTES_AT_REPORTED_CTX * (_ctx_of(a, k) / REPORTED_CTX)
    )
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._estimate_compute_buffer_bytes = lambda **k: 1
    backend._mtp_draft_kv_bytes = lambda *a, **k: 0
    backend._estimate_mtp_overhead_bytes = lambda *a, **k: int(
        MTP_BYTES_AT_REPORTED_CTX * (_ctx_of(a, k) / REPORTED_CTX)
    )
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    backend._select_gpus = lambda *args, **kwargs: ([0], False)
    backend._select_gpus_split_aware = lambda *args, **kwargs: ([0], False)
    return backend, gguf


def _fit(backend, requested_ctx: int, *, mtp: bool) -> int:
    overhead = backend._estimate_mtp_overhead_bytes if mtp else None
    return backend._fit_context_to_vram(
        requested_ctx,
        GPUS[0][1],
        MODEL_BYTES,
        mtp_engaged = mtp,
        mtp_overhead_fn = (lambda n: overhead(n)) if overhead else None,
        total_mib = GPUS[0][2],
    )


def test_the_mtp_reserve_does_cost_context(tmp_path):
    backend, _gguf = _reporter_backend(tmp_path)
    backend._read_gguf_metadata(None)

    without_mtp = _fit(backend, NATIVE_CTX, mtp = False)
    with_mtp = _fit(backend, NATIVE_CTX, mtp = True)

    assert without_mtp < NATIVE_CTX, "the fit should have reduced the requested context"
    assert (
        with_mtp < without_mtp
    ), "the MTP reserve must cost context, or there is nothing to re-fit"


def _auto_then_forced(
    tmp_path: Path,
    *,
    auto_derived: bool,
    via_extras: bool = False,
):
    """Load 1: Auto drops the drafter. Load 2: forced MTP replays load 1's context."""
    backend, gguf = _reporter_backend(tmp_path)
    first = _launch(backend, gguf, speculative_type = "auto", n_ctx = 0, n_parallel = 4)
    resolved_ctx = int(first["cmd"][first["cmd"].index("-c") + 1])

    backend2, gguf2 = _reporter_backend(tmp_path)
    second = _launch(
        backend2,
        gguf2,
        speculative_type = "auto" if via_extras else "mtp",
        extra_args = ["--spec-type", "draft-mtp"] if via_extras else None,
        n_ctx = resolved_ctx,
        n_ctx_auto_derived = auto_derived,
        n_parallel = 4,
    )
    return backend2, resolved_ctx, second["cmd"]


def test_auto_still_keeps_its_context_and_drops_the_drafter(tmp_path):
    backend, gguf = _reporter_backend(tmp_path)
    result = _launch(backend, gguf, speculative_type = "auto", n_ctx = 0, n_parallel = 4)

    cmd = result["cmd"]
    assert int(cmd[cmd.index("-c") + 1]) < NATIVE_CTX, "the native context was reduced"
    assert "draft-mtp" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"


@pytest.mark.parametrize("via_extras", [False, True])
def test_forcing_mtp_refits_the_context_against_its_reserve(tmp_path, via_extras):
    backend2, resolved_ctx, cmd = _auto_then_forced(
        tmp_path, auto_derived = True, via_extras = via_extras
    )

    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    launched_ctx = int(cmd[cmd.index("-c") + 1])
    assert (
        launched_ctx < resolved_ctx
    ), f"expected a re-fit below {resolved_ctx}, launched at {launched_ctx}"

    backend2._read_gguf_metadata(None)
    assert launched_ctx <= _fit(
        backend2, resolved_ctx, mtp = True
    ), "the launched context must fit the budget that now carries the reserve"


def test_a_user_typed_context_is_still_honored_verbatim(tmp_path):
    _backend2, resolved_ctx, cmd = _auto_then_forced(tmp_path, auto_derived = False)

    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert int(cmd[cmd.index("-c") + 1]) == resolved_ctx

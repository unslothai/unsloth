# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #9550 -- forcing MTP reuses the context Auto picked
while MTP was DROPPED, instead of re-fitting against the MTP reserve.

The reporter's two loads of the same 31.0 GB Qwen3.6-40B MTP GGUF, from the issue:

    # Auto: drafter does not fit, so it is dropped and the context is fit WITHOUT it
    Speculative decoding disabled for this load: ... needs 46.7 GB of a 44.3 GB budget
    Context auto-reduced: 262144 -> 112896 (model: 31.0 GB, est. KV cache: 11.2 GB)
    GGUF size: 31.0 GB, ..., context: 112896, GPUs free: [(0, 45914), (1, 8032)], selected: [0]

    # Then the user forces MTP in the dropdown and reloads
    GGUF size: 31.0 GB, ..., MTP reserve: 2.18 GB (draft KV @ 112896 + verify n_max=2),
    context: 112896, GPUs free: [(0, 45914), (1, 8032)], selected: [0, 1]

The second load runs at the SAME 112896 and pays the 2.18 GB reserve on top, pulling
in the 8 GiB card to cover the overflow, rather than re-deriving the largest context
that fits with the reserve.

Two things combined to produce that:

1. ``llama_cpp.py`` clears ``mtp_overhead_fn`` when the Auto probe drops the drafter
   (deliberately -- the fit must not shrink the context for a drafter that will not
   launch), so the reduced context is fit against a budget with no reserve in it.
2. The frontend then sends that resolved context back as ``max_seq_length`` on the
   next load of the same model (``preset-policy.ts`` returns ``ggufContextLength``
   for a same-model reload), which makes ``explicit_ctx`` true and takes the
   "honor the requested context verbatim" branch -- so the reducer never re-ran,
   even though the reserve was now live.

The fix marks the replayed value's provenance (``max_seq_length_auto_derived`` ->
``GgufLoadIntent.n_ctx_auto_derived``) and re-fits inside the explicit branch when,
and only when, all of: the context was ours, a forced choice re-armed the reserve,
and that reserve costs GPU memory. A context the user typed is still honored
verbatim, which the last test here pins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _backend, _launch  # noqa: E402

GB = 1024**3

# The reporter's box and model, from the #9550 log lines.
GPUS = [(0, 45914, 46080), (1, 8032, 8176)]
MODEL_BYTES = int(31.0 * GB)
NATIVE_CTX = 262144
REPORTED_CTX = 112896
KV_BYTES_AT_REPORTED_CTX = int(11.2 * GB)
MTP_BYTES_AT_REPORTED_CTX = int(2.18 * GB)


def _reporter_backend(tmp_path: Path):
    """A backend sized like the reporter's load, with the real context fitter.

    The KV, compute and MTP terms are stubbed LINEAR in the context so the fit is
    the only thing under test and the arithmetic is checkable by hand; their real
    shapes are covered elsewhere.
    """
    backend, gguf = _backend(tmp_path, vulkan = False, memory = GPUS)

    def read_metadata(_path):
        backend._nextn_predict_layers = 1
        backend._n_layers = 48
        backend._n_kv_heads = 8
        backend._n_heads = 40
        backend._embedding_length = 5120
        backend._kv_key_length = 128
        backend._kv_value_length = 128
        # The model's native context. With no explicit max_seq_length this is
        # what the auto path starts from -- 262144 in the reporter's log.
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
    """The largest context the fitter says card 0 can hold, with and without the
    MTP reserve. This is the comparison the second load never makes."""
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
    """Precondition for the issue: charging the reserve genuinely lowers the
    largest context that fits, so re-fitting would have produced a smaller
    number rather than being a no-op."""
    backend, _gguf = _reporter_backend(tmp_path)
    backend._read_gguf_metadata(None)

    without_mtp = _fit(backend, NATIVE_CTX, mtp = False)
    with_mtp = _fit(backend, NATIVE_CTX, mtp = True)

    assert without_mtp < NATIVE_CTX, "the fit should have reduced the requested context"
    assert (
        with_mtp < without_mtp
    ), "the MTP reserve must cost context, or there is nothing to re-fit"


def _auto_then_forced(tmp_path: Path, *, auto_derived: bool):
    """The two loads the reporter made, in order.

    Load 1 is Auto with no explicit context, so the reducer runs and the drafter
    is dropped. Load 2 forces MTP and carries load 1's resolved context back, the
    way the frontend replays ``ggufContextLength`` for a same-model reload.
    """
    backend, gguf = _reporter_backend(tmp_path)
    first = _launch(backend, gguf, speculative_type = "auto", n_ctx = 0, n_parallel = 4)
    resolved_ctx = int(first["cmd"][first["cmd"].index("-c") + 1])

    backend2, gguf2 = _reporter_backend(tmp_path)
    second = _launch(
        backend2,
        gguf2,
        speculative_type = "mtp",
        n_ctx = resolved_ctx,
        n_ctx_auto_derived = auto_derived,
        n_parallel = 4,
    )
    return backend2, resolved_ctx, second["cmd"]


def test_auto_still_keeps_its_context_and_drops_the_drafter(tmp_path):
    """Unchanged: Auto will not shrink the context to buy a speed option."""
    backend, gguf = _reporter_backend(tmp_path)
    result = _launch(backend, gguf, speculative_type = "auto", n_ctx = 0, n_parallel = 4)

    cmd = result["cmd"]
    assert int(cmd[cmd.index("-c") + 1]) < NATIVE_CTX, "the native context was reduced"
    assert "draft-mtp" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"


def test_forcing_mtp_refits_the_context_against_its_reserve(tmp_path):
    """The fix. Forcing MTP re-derives a context that fits WITH the reserve
    instead of inheriting one that was only valid while the drafter was gone."""
    backend2, resolved_ctx, cmd = _auto_then_forced(tmp_path, auto_derived = True)

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
    """The gate. Same forced drafter, same numbers, but the context came from the
    user, so the shortfall is spent on placement as before and not on their
    context. Everything that pins verbatim honoring elsewhere depends on this."""
    _backend2, resolved_ctx, cmd = _auto_then_forced(tmp_path, auto_derived = False)

    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert int(cmd[cmd.index("-c") + 1]) == resolved_ctx

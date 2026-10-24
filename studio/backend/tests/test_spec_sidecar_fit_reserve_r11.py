# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eleventh-round cells for the DSpark/DFlash sidecar reserve and the Metal probe.

A reranking server or a NONE-pooling embedding server is probed where it really computes
(the native /embedding route), an unmappable MIG mask keeps the broadcast margin in both
planning and spawn, and a multi-device draft pin is not reported as reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_GIB = 1024 * 1024 * 1024


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)


def _spy_fit_sized(backend):
    seen = {}
    build = backend._build_speculative_flags

    def spy(**kw):
        seen["fit_sized"] = kw.get("dspark_fit_sized")
        return build(**kw)

    backend._build_speculative_flags = spy
    return seen


@pytest.mark.parametrize(
    "embedding, extras",
    [
        # --rerank sets embedding mode with RANK pooling: /completion 501s before compute.
        (False, ("--rerank",)),
        (False, ("--reranking",)),
        # /v1/embeddings 400s on NONE pooling, explicit or the model's default.
        (False, ("--embeddings", "--pooling", "none")),
        (True, ()),
    ],
)
def test_embedding_and_rerank_servers_are_probed_where_they_compute(monkeypatch, embedding, extras):
    backend = _lc.LlamaCppBackend()
    backend._port = 1
    backend._extra_args = list(extras)
    monkeypatch.setattr(_lc.LlamaCppBackend, "is_embedding_gguf", property(lambda self: embedding))
    resp = type(
        "Resp",
        (),
        {"status_code": 500, "json": lambda self: {"error": {"message": "Compute error."}}},
    )()
    with patch.object(_lc.httpx, "post", return_value = resp, create = True) as post:
        assert backend._metal_compute_failure_confirmed() is True
    assert post.call_args.args[0].endswith("/embedding")
    assert "content" in post.call_args.kwargs["json"]


def _mig_two_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    B = _lc.LlamaCppBackend
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_smi_uuid_mask_order", staticmethod(lambda: None))
    # A MIG mask (child order: card 1's slice, then card 0's) no resolver can map.
    monkeypatch.setattr(B, "_visibility_mask_is_unmappable", staticmethod(lambda: True))
    backend, gguf = _r2._cuda_backend(tmp_path, memory = ((0, 16_000, 16_000), (1, 4_000, 4_864)))
    backend._arch_gate_survivors = lambda binary = None: []
    return backend, gguf


def test_an_unmappable_mig_mask_gets_no_positional_reserve(tmp_path, monkeypatch):
    backend, gguf = _mig_two_cards(tmp_path, monkeypatch)
    assert len(_r7._margins(_r7._dspark(backend, gguf, tmp_path))) == 1


def test_an_unmappable_mig_mask_is_not_reported_as_reserved(tmp_path, monkeypatch):
    backend, gguf = _mig_two_cards(tmp_path, monkeypatch)
    seen = _spy_fit_sized(backend)
    _r7._dspark(backend, gguf, tmp_path)
    assert seen["fit_sized"] is False


@pytest.mark.parametrize("pin, reserved", [("CUDA0,CUDA1", False), ("CUDA0", True)])
def test_a_multi_device_draft_pin_is_not_reported_as_reserved(tmp_path, monkeypatch, pin, reserved):
    """The spawn reserves only a one-card pin, so a pin across both cards keeps the
    missing-reserve warning."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _r2._cuda_backend(tmp_path, memory = ((0, 24_000, 24_000), (1, 24_000, 24_000)))
    backend._arch_gate_survivors = lambda binary = None: []
    backend._get_gguf_size_bytes = lambda path: (8 if "dspark" in str(path) else 30) * _GIB
    seen = _spy_fit_sized(backend)
    cmd = _r7._dspark(
        backend,
        gguf,
        tmp_path,
        extra_args = ["--device", "CUDA0,CUDA1", "--device-draft", pin],
    )
    assert (len(_r7._margins(cmd)) == 2) is reserved
    assert seen["fit_sized"] is reserved


def _single_card_margin(
    tmp_path,
    monkeypatch,
    extras,
    env = None,
):
    import test_spec_sidecar_fit_reserve_r4 as _r4

    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._mtp_draft_kv_bytes = lambda *a, **k: 768 * 1024 * 1024
    return _r4._margin(_r7._dspark(backend, gguf, tmp_path, extra_args = list(extras)))


@pytest.mark.parametrize(
    "extras, env",
    [
        (["-nkvo"], {}),
        (["--no-kv-offload"], {}),
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": "1"}),
    ],
)
def test_host_resident_draft_kv_is_not_reserved_on_the_card(tmp_path, monkeypatch, extras, env):
    """The draft context copies the target's params (common_base_params_to_speculative), so
    -nkvo puts the draft KV in host buffers too: the card's margin must not carry it."""
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    on = _single_card_margin(tmp_path / "on", monkeypatch, [])
    off = _single_card_margin(tmp_path / "off", monkeypatch, extras, env)
    assert on - off >= 760

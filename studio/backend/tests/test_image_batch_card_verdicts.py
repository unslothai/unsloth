# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-request batching admitted per card by a stored G2c verdict (paired upper bound of batched minus solo LPIPS,
both against a high-precision reference, at most 0.01 over 48 pairs, with the target row bit-identical when only a
neighbour changes). A card without a verdict keeps today's serial rule."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_batch_quality as quality

A100_40 = "NVIDIA A100-SXM4-40GB"
G4 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
OTHER = "NVIDIA T4-LIKE TEST CARD"


@pytest.fixture(autouse = True)
def _no_local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(quality, "_file", lambda: tmp_path / quality.FILE_NAME)


def _on_card(monkeypatch, name):
    monkeypatch.setattr(quality, "pipe_device_class", lambda pipe: name)
    # A word-count tokenizer: "a" and "b" condition to one token, "b c d" to three.
    return types.SimpleNamespace(
        transformer = object(),
        t5_tokenizer = lambda text, **_: {"input_ids": text.split()},
        text_conditioner = types.SimpleNamespace(config = types.SimpleNamespace(min_sequence_length = 0)),
    )


def _anima():
    from core.inference import diffusion_families as families
    from core.inference.diffusion_anima import ANIMA_FAMILY_NAME
    return next(f for f in families._FAMILIES if f.name == ANIMA_FAMILY_NAME)


@pytest.mark.parametrize("card", [A100_40, "NVIDIA L4"])
def test_klein_has_a_passing_verdict_on_the_regated_card(card):
    found = quality.lookup("flux.3-klein", card, "eager")
    assert quality.passes(found)
    assert found["n"] >= 48 and found["neighbour_exact"] is True


def test_klein_keeps_no_verdict_on_an_unmeasured_card():
    assert quality.lookup("flux.3-klein", OTHER, "eager") is None


@pytest.mark.parametrize("card", [A100_40, G4])
def test_anima_fuses_rows_on_a_card_with_a_verdict(monkeypatch, card):
    from core.inference.diffusion import _generation_chunks
    from core.inference.diffusion_batch_adapters import (
        DEFAULT_ADAPTER,
        adapter_for,
        admits_without_pipe,
        batched_forward,
    )

    record = _anima()
    pipe = _on_card(monkeypatch, card)
    assert batched_forward(record, pipe) is True
    assert adapter_for(record, pipe) is DEFAULT_ADAPTER
    # The route has no pipe: it lets the engine decide, as for MOSS.
    assert admits_without_pipe(record) is True
    jobs = [("a", 1), ("a", 2), ("b", 3)]
    assert _generation_chunks(record, jobs, 4, pipe = pipe) == [jobs]


def test_anima_chunks_each_conditioned_length_on_its_own(monkeypatch):
    # One chunk per conditioned-length run.
    from core.inference.diffusion import _generation_chunks

    record = _anima()
    pipe = _on_card(monkeypatch, A100_40)
    jobs = [("a", 1), ("b", 2), ("b c d", 3), ("e f g", 4), ("h", 5)]
    assert _generation_chunks(record, jobs, 4, pipe = pipe) == [jobs[:2], jobs[2:4], jobs[4:]]
    assert _generation_chunks(record, jobs, 2, pipe = pipe) == [jobs[:2], jobs[2:4], jobs[4:]]


def test_anima_chunks_one_image_per_run_when_rows_are_switched_off(monkeypatch):
    from core.inference import diffusion_anima as an
    from core.inference.diffusion import _generation_chunks

    monkeypatch.setenv(an.BATCH_ROWS_ENV, "0")
    pipe = _on_card(monkeypatch, A100_40)
    jobs = [("a", 1), ("a", 2)]
    assert _generation_chunks(_anima(), jobs, 4, pipe = pipe) == [[j] for j in jobs]


def test_anima_stays_one_image_per_forward_on_a_card_without_a_verdict(monkeypatch):
    from core.inference.diffusion import _generation_chunks
    from core.inference.diffusion_batch_adapters import adapter_for, batched_forward

    record = _anima()
    pipe = _on_card(monkeypatch, OTHER)
    assert batched_forward(record, pipe) is False
    assert adapter_for(record, pipe) is None
    jobs = [("a", 1), ("a", 2)]
    assert _generation_chunks(record, jobs, 4, pipe = pipe) == [[j] for j in jobs]


def test_anima_engine_policy_follows_the_card(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    def engine(card, offload):
        eng = DiffusionBackend()
        eng._state = types.SimpleNamespace(
            family = _anima(),
            pipe = _on_card(monkeypatch, card),
            speed_optims = (),
            speed_mode = "off",
            cuda_graphs = (),
            compile_cache_ctx = None,
            offload_policy = offload,
            cache_auto = False,
            transformer_cache = None,
        )
        return eng

    # Streamed: a card with a verdict shares calls; one without renders one at a time.
    monkeypatch.setattr("core.inference.diffusion._per_call_activation_scale", lambda state: None)
    assert engine(A100_40, "sequential").request_batching()["mode"] == "any"
    assert engine(OTHER, "sequential").request_batching()["mode"] == "serial"


def test_a_verdict_lookup_reads_the_pipes_card(monkeypatch):
    calls = []

    def fake(pipe):
        calls.append(pipe)
        return A100_40

    monkeypatch.setattr(quality, "pipe_device_class", fake)
    pipe = types.SimpleNamespace(transformer = object())
    assert quality.family_card_ok("anima", pipe) is True
    assert calls == [pipe]
    assert quality.family_card_ok("anima", None) is False


def _sdxl_engine(monkeypatch, card, verdict):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_image_batch_infra import _engine, _verdicts

    _verdicts(monkeypatch, {2: verdict})
    return _engine(device_name = card, family = types.SimpleNamespace(name = "sdxl"))


NEIGHBOUR_EXACT_WIN = {
    "promoted": True,
    "arm": "batched_eager",
    "path": "eager",
    "neighbour_exact": True,
    "b_exact": False,
}


def test_sdxl_seeded_requests_join_a_neighbour_exact_win_on_the_regated_card(monkeypatch):
    engine = _sdxl_engine(monkeypatch, "NVIDIA L4", NEIGHBOUR_EXACT_WIN)
    assert engine.request_batching()["mode"] == "measured"
    assert engine.coalesce_batch_sizes(1024, 1024, 5.0) == frozenset({2})
    assert engine.coalesce_seeded_ok(1024, 1024, 5.0, 1) is True


@pytest.mark.parametrize(
    "card, verdict",
    [
        (OTHER, NEIGHBOUR_EXACT_WIN),  # no G2c verdict on this card
        (
            "NVIDIA L4",
            dict(NEIGHBOUR_EXACT_WIN, neighbour_exact = False),
        ),  # the race saw a neighbour move a row
        (
            "NVIDIA L4",
            dict(NEIGHBOUR_EXACT_WIN, path = "graph", arm = "batched"),
        ),  # no verdict for graph replay
    ],
)
def test_sdxl_seeded_requests_stay_alone_elsewhere(monkeypatch, card, verdict):
    engine = _sdxl_engine(monkeypatch, card, verdict)
    assert engine.coalesce_seeded_ok(1024, 1024, 5.0, 1) is False


def test_only_the_sdxl_verdict_admits_seeded_rows():
    seeded = {key for key, v in quality.SHIPPED.items() if v.get("seeded")}
    assert seeded == {("sdxl", "NVIDIA L4", "eager")}

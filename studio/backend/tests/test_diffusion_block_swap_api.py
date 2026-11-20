# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""block_swap through the diffusion training API: request validation, forwarding, refusals, family flag, status."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import core.training.diffusion_block_swap as swap
import core.training.diffusion_train_common as common
from core.training.diffusion_train_common import DiffusionLoraConfig
from core.training.diffusion_training_service import DiffusionTrainingService
from models.training import DiffusionTrainingStartRequest, DiffusionTrainingStatusResponse

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_training import _BODY, _FakeCtx, _happy_target, client  # noqa: E402, F401

_FLUX = {**_BODY, "base_model": "black-forest-labs/FLUX.1-dev", "base_precision": "bf16"}


@pytest.fixture
def dit_route(client, monkeypatch):  # noqa: F811
    """A DiT start that reaches the service on a CPU runner: hardware probes pinned to a discrete CUDA card."""
    import routes.training as tr

    freed: list = []
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: freed.append(1))
    monkeypatch.setattr(tr, "_preflight_gated_base", lambda *a, **k: None)
    monkeypatch.setattr(tr, "_refuse_unauthorized_cached_local_paths", lambda *a, **k: None)
    monkeypatch.setattr(common, "training_precision_preflight_error", lambda *a, **k: None)
    monkeypatch.setattr(common, "training_pipeline_import_error", lambda *a, **k: None)
    monkeypatch.setattr(common, "dit_fp16_fallback", lambda *a, **k: False)
    monkeypatch.setattr(swap, "_cuda_device_facts", lambda: ("cuda", False))
    client._freed = freed  # type: ignore[attr-defined]
    return client


def _start(c, **extra):
    return c.post("/api/train/diffusion/start", json = {**_FLUX, **extra})


def test_request_field_accepts_off_auto_and_a_count_and_defaults_unset():
    base = {"base_model": "x", "data_dir": "d", "output_dir": "o"}
    assert DiffusionTrainingStartRequest(**base).block_swap is None
    for value, want in (("off", "off"), ("auto", "auto"), (14, 14), ("14", 14), (0, 0)):
        assert DiffusionTrainingStartRequest(**base, block_swap = value).block_swap == want
    for bad in ("banana", -1, 5000):
        with pytest.raises(ValueError):
            DiffusionTrainingStartRequest(**base, block_swap = bad)


@pytest.mark.parametrize("value, sent", [("auto", "auto"), (14, "14")])
def test_route_forwards_block_swap_to_the_dit_trainer(dit_route, value, sent):
    r = _start(dit_route, block_swap = value)
    assert r.status_code == 200, r.text
    assert dit_route._fake.started_with["block_swap"] == sent
    # The forwarded string is what the trainer parses.
    cfg = common._config_from_dict(dit_route._fake.started_with)
    assert swap.block_swap_request(cfg) == ("auto" if value == "auto" else 14)


def test_route_leaves_block_swap_unset_by_default(dit_route):
    r = _start(dit_route)
    assert r.status_code == 200, r.text
    assert "block_swap" not in dit_route._fake.started_with
    assert common._config_from_dict(dit_route._fake.started_with).block_swap == "off"


@pytest.mark.parametrize(
    "extra, needle",
    [
        ({"base_precision": "nf4"}, "base_precision='bf16'"),
        ({"base_precision": "auto"}, "base_precision='bf16'"),
        ({"gradient_checkpointing": False}, "gradient_checkpointing"),
        ({"base_model": "stabilityai/sdxl-turbo"}, "DiT trainer"),
        ({"base_model": "stabilityai/sdxl-turbo", "base_precision": "nf4"}, "DiT trainer"),
    ],
)
def test_route_refuses_block_swap_the_trainer_cannot_honour_before_freeing_gpu(
    dit_route, extra, needle
):
    r = _start(dit_route, block_swap = "auto", **extra)
    assert r.status_code == 400, r.text
    assert needle in r.json()["detail"]
    assert dit_route._freed == []
    assert dit_route._fake.started_with is None


@pytest.mark.parametrize(
    "facts, needle", [(("cpu", False), "CUDA GPU"), (("cuda", True), "integrated GPU")]
)
def test_route_refuses_block_swap_without_a_discrete_gpu(dit_route, monkeypatch, facts, needle):
    monkeypatch.setattr(swap, "_cuda_device_facts", lambda: facts)
    r = _start(dit_route, block_swap = 8)
    assert r.status_code == 400, r.text
    assert needle in r.json()["detail"]
    assert dit_route._freed == []


def test_route_refuses_block_swap_on_the_fp16_fallback(dit_route, monkeypatch):
    monkeypatch.setattr(common, "dit_fp16_fallback", lambda *a, **k: True)
    r = _start(dit_route, block_swap = "auto")
    assert r.status_code == 400, r.text
    assert "bf16" in r.json()["detail"]


@pytest.mark.parametrize("value", ["off", 0])
def test_route_off_is_accepted_everywhere(dit_route, value):
    # SDXL with an explicit off is today's request plus a no-op field.
    r = dit_route.post("/api/train/diffusion/start", json = {**_BODY, "block_swap": value})
    assert r.status_code == 200, r.text


def test_start_error_reads_only_the_request_not_the_env(monkeypatch):
    # The operator env still reaches the trainer; the route only judges what the request asked for.
    monkeypatch.setenv(swap.BLOCK_SWAP_ENV, "auto")
    cfg = DiffusionLoraConfig(
        base_model = "stabilityai/sdxl-turbo", data_dir = "d", output_dir = "o"
    ).normalized()
    assert swap.block_swap_start_error(cfg, device_facts = ("cpu", False)) is None


def test_family_infos_flag_block_swap_on_dit_families_with_bf16(monkeypatch, dit_train_host):
    monkeypatch.setattr(
        common, "train_precision_modes", lambda: (["nf4", "bf16", "int8", "auto"], "auto")
    )
    monkeypatch.setattr(common, "dit_fp16_fallback", lambda name: False)
    infos = {i["name"]: i for i in common.family_train_infos()}
    assert infos["flux.1"]["supports_block_swap"] is True
    if "sdxl" in infos:
        assert infos["sdxl"]["supports_block_swap"] is False
    for name in ("minimax-h3", "minimax-h3-image"):
        if name in infos:
            assert infos[name]["supports_block_swap"] is False
    # No bf16 on this host: nothing to stream.
    monkeypatch.setattr(common, "train_precision_modes", lambda: (["nf4", "auto"], "nf4"))
    infos = {i["name"]: i for i in common.family_train_infos()}
    assert infos["flux.1"]["supports_block_swap"] is False


def test_status_reports_the_resolved_block_swap():
    svc = DiffusionTrainingService(ctx = _FakeCtx(), target = _happy_target)
    assert svc.status()["block_swap"] is None
    svc._apply_event(
        {
            "type": "block_swap",
            "swapped_blocks": 14,
            "total_blocks": 57,
            "host_gib": 5.5,
            "pinned_gib": 5.5,
        }
    )
    svc._apply_event({"type": "progress", "step": 1, "total_steps": 10, "loss": 0.5})
    st = svc.status()
    assert st["block_swap"] == {
        "swapped_blocks": 14,
        "total_blocks": 57,
        "host_gib": 5.5,
        "pinned_gib": 5.5,
    }
    resp = DiffusionTrainingStatusResponse(
        **{k: v for k, v in st.items() if k in DiffusionTrainingStatusResponse.model_fields}
    )
    assert resp.block_swap["swapped_blocks"] == 14


def test_trainer_emits_the_resolved_block_swap_event():
    # The status fold above is only reachable if the trainer emits the event right after the install.
    import inspect

    from core.training import diffusion_dit_trainer as dit

    src = inspect.getsource(dit._train_dit)
    install = src.index("install_dit_block_swap(")
    event = src.index('"block_swap",', install)
    assert event < src.index("gc_budget = None", install)

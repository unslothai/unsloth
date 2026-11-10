# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How many status GETs one model step costs: the residency checks of a step share one read."""

import asyncio

import pytest

from core.flows.nodes import _common
from studio_mcp import loading

REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
OTHER = "unsloth/Qwen3-14B-GGUF"


class _Ctx:
    def __init__(self, spec = None):
        self.resident, self.spec, self.gpu_shared = {}, spec or {}, False
        self.caller = object()
        self.params = {}

    def set_status(
        self,
        status,
        message = None,
    ):
        pass


def _fakes(
    monkeypatch,
    serving,
    mode = None,
):
    gets = []
    state = {"serving": list(serving)}

    async def fake_read_json(caller, path):
        gets.append(path)
        if path == "/api/inference/status":
            status = {"serving": state["serving"], "loaded": state["serving"], "is_gguf": True}
            status["model_identifier"] = state["serving"][0] if state["serving"] else None
            status["gguf_variant"] = "Q4_K_M"
            if mode:
                status["multi_model_mode"] = mode
            return status
        return {"loaded": False}

    async def fake_load_llm(caller, ctx, **kwargs):
        state["serving"] = [kwargs["model"]]

    monkeypatch.setattr(loading, "_read_json", fake_read_json)
    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_LOADED_OPTIONS", {})
    monkeypatch.setattr(_common, "_ALONGSIDE_OPTIONS", {})
    return gets


@pytest.mark.parametrize(
    "serving, requires, mode, expected",
    [
        # Resident already, with a backend requirement and a variant to compare.
        ([REPO], ["gguf"], None, 3),
        # Another model resident: one read before the load, one after it.
        ([OTHER], ["gguf"], None, 6),
        # Keep multiple models loaded is on but the step replaces nothing it can keep.
        ([REPO], None, "fit", 3),
    ],
)
def test_one_model_check_reads_status_once_per_residency(
    monkeypatch, serving, requires, mode, expected
):
    gets = _fakes(monkeypatch, serving, mode)
    ctx = _Ctx({"requires": requires} if requires else {})
    asyncio.run(_common.ensure_model(ctx, "chat", {"repo": REPO, "variant": "Q4_K_M"}))
    assert len(gets) == expected, gets


def test_a_card_on_the_loaded_model_reads_its_name_and_quant_once(monkeypatch):
    from core.flows.nodes import text

    gets = _fakes(monkeypatch, [REPO])
    ctx = _Ctx()
    ctx.params = {"model": None, "load": {"max_seq_length": 4096}}
    choice, _ = asyncio.run(text._model_and_load(ctx, "chat"))
    assert choice == {"repo": REPO, "variant": "Q4_K_M"}
    assert len(gets) == 3, gets

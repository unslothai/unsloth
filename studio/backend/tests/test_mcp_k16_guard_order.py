# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_model's pre-load guards in their fixed order: a managed download is waited on, never refused, and only a partial variant Unsloth Studio is not downloading is refused. Checkpoint and export loads skip both and carry is_lora."""

import pytest

from studio_mcp import model_downloads

from .mcp_harness import bodies, fast_polls, queries, run_tool, sequence  # noqa: F401
from .test_mcp_tools_loading import PAYLOADS, STATUS_ROUTE
from .test_mcp_train_to_use import _routes as trained_routes

QWEN = "unsloth/Qwen3-0.6B-GGUF"
STATUS = "/api/hub/download-status"
PARTIAL = {
    "default_variant": "Q4_K_M",
    "variants": [{"quant": "Q4_K_M", "filename": "q.gguf", "downloaded": False, "partial": True}],
}
DONE = {
    "default_variant": "Q4_K_M",
    "variants": [{"quant": "Q4_K_M", "filename": "q.gguf", "downloaded": True, "partial": False}],
}


@pytest.fixture(autouse = True)
def quick_waits(monkeypatch, fast_polls):
    monkeypatch.setattr(model_downloads, "WAIT_INTERVAL_S", 0.01)
    # download_model calls elsewhere in the run remember what they staged; start clean.
    monkeypatch.setattr(model_downloads, "_remembered", model_downloads.OrderedDict())


def _load(
    monkeypatch,
    variants,
    states,
    args = None,
):
    routes = {
        **PAYLOADS,
        ("GET", "/api/hub/gguf-variants"): sequence(*variants),
        ("GET", STATUS): sequence(*({"state": s} for s in states)),
    }
    result, studio = run_tool(monkeypatch, routes, "load_model", {"model": QWEN, **(args or {})})
    return result, studio, bodies(studio, "/api/inference/load")


def test_an_attach_to_a_load_in_flight_comes_before_any_download_wait(monkeypatch):
    # Guard 1 (attach) wins: a chat load of the same model already running is followed, and the
    # download it may still be finishing is never read, waited on or refused.
    loading_now = {"loading": [QWEN], "serving": [], "loaded": []}
    loaded = {
        "loading": [],
        "serving": [QWEN],
        "loaded": [QWEN],
        "serving_checkpoints": ["/m/q.gguf"],
    }
    routes = {
        **PAYLOADS,
        STATUS_ROUTE: sequence(loading_now, loading_now, loaded),
        ("GET", "/api/hub/gguf-variants"): PARTIAL,
        ("GET", STATUS): {"state": "running"},
    }
    result, studio = run_tool(monkeypatch, routes, "load_model", {"model": QWEN})
    assert result.get("isError") is not True, result
    assert result["structuredContent"]["model"] == QWEN
    assert queries(studio, STATUS) == [] and bodies(studio, "/api/inference/load") == []


def test_a_managed_download_is_waited_on_then_loaded_not_refused(monkeypatch):
    # The variant reads partial while the manager writes it, then complete.
    result, studio, sent = _load(monkeypatch, [PARTIAL, DONE], ["running", "running", "complete"])
    assert result.get("isError") is not True, result
    assert len(sent) == 1
    reads = queries(studio, STATUS)
    assert len(reads) >= 3 and reads[0] == {"repo_id": QWEN, "gguf_variant": "Q4_K_M"}


def test_an_unmanaged_partial_variant_is_still_refused(monkeypatch):
    result, _studio, sent = _load(monkeypatch, [PARTIAL], ["idle"])
    assert result["isError"] is True
    assert "still downloading or incomplete" in result["content"][0]["text"]
    assert sent == []


def test_a_download_that_outlasts_the_wait_is_reported_with_its_id(monkeypatch):
    result, _studio, sent = _load(monkeypatch, [PARTIAL], ["running"], {"wait_s": 0})
    text = result["content"][0]["text"]
    assert result["isError"] is True and sent == []
    assert "still downloading" in text and f'id="{QWEN}::Q4_K_M"' in text


@pytest.mark.parametrize("name,lora", [("qwen-lora", True), ("qwen-lora-2/checkpoint-30", False)])
def test_a_checkpoint_skips_both_guards_and_carries_is_lora(monkeypatch, name, lora):
    # The fallback records every call, so a Hub lookup for the ref: handle would show up here.
    result, studio = run_tool(
        monkeypatch, trained_routes(), "load_model", {"checkpoint": name}, fallback = {}
    )
    assert result.get("isError") is not True, result
    (body,) = bodies(studio, "/api/inference/load")
    assert body.get("is_lora", False) is lora
    assert queries(studio, STATUS) == [] and queries(studio, "/api/hub/gguf-variants") == []

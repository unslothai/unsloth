# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 sidecar encoder binds its own card (ordinal + UUID); CPU only, torch.cuda faked."""

from __future__ import annotations

import pickle
import types

import pytest
import torch

from core.inference import video_ltx25 as L
from core.inference import video_ltx25_te_worker as W


def _fake_cards(monkeypatch, uuids):
    chosen = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(uuids))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(uuid = uuids[i])
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: chosen.append(i))
    return chosen


def test_worker_binds_the_card_the_server_named(monkeypatch):
    chosen = _fake_cards(monkeypatch, ["GPU-a", "GPU-b"])
    config = {"device": "cuda:1", "bind_device": True, "device_uuid": "GPU-b"}
    W._bind_device(config)
    assert chosen == [1] and config["device"] == "cuda:1"


def test_worker_resolves_the_card_by_uuid_when_its_order_differs(monkeypatch):
    # child enumerates cards in another order: the UUID wins
    chosen = _fake_cards(monkeypatch, ["GPU-b", "GPU-a"])
    config = {"device": "cuda:1", "bind_device": True, "device_uuid": "GPU-b"}
    W._bind_device(config)
    assert chosen == [0] and config["device"] == "cuda:0"
    config = {"device": "cuda:1", "bind_device": True, "device_uuid": "GPU-z"}
    W._bind_device(config)
    assert chosen[-1] == 1 and config["device"] == "cuda:1"


def test_worker_binds_nothing_without_the_lever(monkeypatch):
    chosen = _fake_cards(monkeypatch, ["GPU-a", "GPU-b"])
    for config in (
        {"device": "cuda:1"},
        {"device": "cuda:1", "bind_device": False, "device_uuid": "GPU-b"},
        {"device": "cpu", "bind_device": True},
        {"device": "mps", "bind_device": True},
    ):
        before = dict(config)
        W._bind_device(config)
        assert config == before
    assert chosen == []


@pytest.mark.parametrize("bind", [False, True])
def test_start_text_encoder_hands_the_card_to_the_worker(monkeypatch, bind):
    monkeypatch.setattr(L, "ensure_text_encoder_sidecar", lambda: "/nonexistent/.venv_t5_510")
    monkeypatch.setattr(L.Ltx25TextEncoderClient, "start", lambda self: None)
    monkeypatch.setattr(L, "_cuda_device_uuid", lambda device: f"uuid-of-{device}")
    client = L.start_text_encoder(
        repo = "Lightricks/LTX-2.5-Diffusers",
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cuda:1",
        dtype = "bfloat16",
        resident = True,
        budget_mib = 12000,
        bind_device = bind,
    )
    assert client.config["device"] == "cuda:1" and client.config["budget_mib"] == 12000
    if bind:
        assert client.config["bind_device"] is True
        assert client.config["device_uuid"] == "uuid-of-cuda:1"
    else:
        assert "bind_device" not in client.config and "device_uuid" not in client.config
    assert pickle.loads(pickle.dumps(client.config)) == client.config


def test_cuda_device_uuid_is_none_off_cuda():
    assert L._cuda_device_uuid("cpu") is None
    assert L._cuda_device_uuid("cuda") is None
    assert L._cuda_device_uuid("not a device") is None


def test_a_load_error_read_by_a_placement_preview_reaches_the_load(monkeypatch):
    import queue

    client = L.Ltx25TextEncoderClient(
        repo = "r",
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cuda:1",
        dtype = "bfloat16",
        resident = False,
        venv_dir = "/x",
    )
    client._cmd, client._resp = queue.Queue(), queue.Queue()
    client._proc = types.SimpleNamespace(is_alive = lambda: False, exitcode = 1)
    client._resp.put({"type": "error", "error": "encoder not in the cache", "cache_miss": True})
    for _ in range(2):  # the preview swallows the first, the load's own wait must still see it
        with pytest.raises(L.TextEncoderWorkerError) as err:
            client.wait_ready()
        assert "not in the cache" in str(err.value) and err.value.cache_miss

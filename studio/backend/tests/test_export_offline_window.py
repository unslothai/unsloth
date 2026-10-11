# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the persistent export command loop through a disconnect and recovery."""

import os
import queue
import sys
import threading
import types
from unittest.mock import patch


def _run_worker(monkeypatch, commands, unreachable):
    """Run the command loop with a stub export that makes a real Hub client call. Returns
    (offline flag, request reached the transport) per export, and the export responses."""
    import httpx
    from huggingface_hub import HfApi, constants
    from huggingface_hub.errors import OfflineModeIsEnabled
    from core.export import worker
    from core import import_guards
    from loggers import config as log_config
    from loggers.config import LogConfig
    from utils import transformers_version
    from utils.utils import force_hf_offline

    calls = []

    def handle_request(self, request):
        return httpx.Response(200, json = {"id": "org/model", "siblings": []}, request = request)

    class Backend:
        def cleanup_memory(self):
            pass

        def export_merged_model(self, **kwargs):
            offline = constants.HF_HUB_OFFLINE
            try:
                HfApi(token = False).model_info("org/model")
                calls.append((offline, True))
            except OfflineModeIsEnabled:
                calls.append((offline, False))
            return True, "saved", "/tmp/export"

    fake_export = types.ModuleType("core.export.export")
    fake_export.ExportBackend = Backend
    fake_loader = types.ModuleType("unsloth.models.loader_utils")
    fake_loader._force_hf_offline = force_hf_offline
    monkeypatch.setitem(sys.modules, "core.export.export", fake_export)
    monkeypatch.setitem(sys.modules, "unsloth.models.loader_utils", fake_loader)
    monkeypatch.setattr(import_guards, "ensure_real_packages", lambda *args: None)
    monkeypatch.setattr(LogConfig, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(worker, "_setup_log_capture", lambda _: None)
    monkeypatch.setattr(worker, "_activate_transformers_version", lambda *args: None)
    monkeypatch.setattr(worker, "_handle_load", lambda *args: None)
    # Worker-process globals; keep them from leaking into later tests.
    monkeypatch.setattr(worker, "_log_forward_gate", threading.Event())
    monkeypatch.setattr(log_config, "_BARS_RESTORED", log_config._BARS_RESTORED)
    # Below the client's request hook, which is where huggingface_hub enforces offline mode.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(transformers_version, "hf_endpoint_unreachable", unreachable)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    cmd_queue = queue.Queue()
    for command in [*commands, {"type": "shutdown"}]:
        cmd_queue.put(command)
    resp_queue = queue.Queue()
    # The worker normally owns its process environment; isolate that mutation here.
    with patch.dict(os.environ):
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ["UNSLOTH_OFFLINE_PROBE"] = "1"
        worker.run_export_process(
            cmd_queue = cmd_queue,
            resp_queue = resp_queue,
            config = {"checkpoint_path": "/tmp/ckpt"},
        )
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert constants.HF_HUB_OFFLINE is False
    return calls, [m for m in resp_queue.queue if m.get("type") == "export_merged_done"]


def test_local_export_rechecks_connectivity_and_restores_hub_state(monkeypatch):
    # Activation and load online, first export disconnected, next export recovered.
    verdicts = [False, False, True, False]
    probes = []

    def unreachable():
        probes.append(None)
        return verdicts[len(probes) - 1]

    export = {"type": "export", "export_type": "merged"}
    calls, done = _run_worker(monkeypatch, [export, export], unreachable)
    assert len(probes) == len(verdicts)
    assert calls == [(True, False), (False, True)]
    assert [m["success"] for m in done] == [True, True]


def test_push_export_is_not_pinned_offline(monkeypatch):
    export = {"type": "export", "export_type": "merged", "push_to_hub": True}
    calls, _ = _run_worker(monkeypatch, [export], lambda: True)
    assert calls == [(False, True)]

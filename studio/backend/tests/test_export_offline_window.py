# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the persistent export command loop through a disconnect and recovery."""

import os
import queue
import sys
import types
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("export_type", ["base", "merged", "gguf", "lora"])
@pytest.mark.parametrize("fail_first", [False, True])
def test_export_rechecks_connectivity_and_restores_hub_state(monkeypatch, export_type, fail_first):
    import requests
    import httpx
    from huggingface_hub import HfApi, constants
    from huggingface_hub.errors import OfflineModeIsEnabled
    from core.export import worker
    from core import import_guards
    from loggers.config import LogConfig
    from utils import transformers_version
    from utils.utils import force_hf_offline

    transport_calls = []
    observed = []

    def send(self, request, **kwargs):
        transport_calls.append(request.url)
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"id":"org/model", "siblings":[]}'
        response.url = request.url
        response.request = request
        return response

    def send_httpx(self, request, **kwargs):
        transport_calls.append(str(request.url))
        return httpx.Response(200, json = {"id": "org/model", "siblings": []}, request = request)

    class Backend:
        def cleanup_memory(self):
            pass

        def export(self, **kwargs):
            observed.append(constants.HF_HUB_OFFLINE)
            try:
                HfApi(token = False).model_info("org/model")
            except OfflineModeIsEnabled:
                pass  # Represents a cached export that tolerates unavailable metadata.
            if fail_first and len(observed) == 1:
                raise RuntimeError("export failed")
            return True, "saved", "/tmp/export"

        export_base_model = export
        export_merged_model = export
        export_gguf = export
        export_lora_adapter = export

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
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    monkeypatch.setattr(httpx.Client, "send", send_httpx)
    # Activation and load online, first export disconnected, next export recovered.
    verdicts = iter([False, False, True, False])
    monkeypatch.setattr(transformers_version, "hf_endpoint_unreachable", lambda: next(verdicts))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    commands = queue.Queue()
    for command in ({"type": "export", "export_type": export_type},) * 2:
        commands.put(command)
    commands.put({"type": "shutdown"})
    responses = queue.Queue()
    # The worker normally owns its process environment; isolate that mutation here.
    with patch.dict(os.environ):
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ["UNSLOTH_OFFLINE_PROBE"] = "1"
        worker.run_export_process(
            cmd_queue = commands, resp_queue = responses, config = {"checkpoint_path": "/tmp/checkpoint"}
        )
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ
        assert constants.HF_HUB_OFFLINE is False
    assert observed == [True, False]
    assert transport_calls == ["https://huggingface.co/api/models/org/model"]
    messages = list(responses.queue)
    assert any(m.get("type") == f"export_{export_type}_done" for m in messages)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the full idle lifespan in a fresh interpreter, including its maintenance worker."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent

_COLD_START = r"""
import sys
import main
from fastapi.testclient import TestClient
from utils.torch_warmup import warm_status

main.app.state.suppress_bootstrap_injection = True
with TestClient(main.app) as client:
    # The first response releases startup's import gate if a warm was accidentally started.
    assert client.get('/api/liveness').status_code == 200
    for _ in range(2):
        response = client.get('/api/health')
        assert response.status_code == 200
        assert response.json()['hardware_detection_deferred'] is True
    assert client.get('/api/health?detect_hardware=true').status_code == 401
    assert client.get('/api/settings/ml-warmup').status_code == 401
    assert client.put('/api/settings/ml-warmup', json={'features': ['text']}).status_code == 401
    assert client.get('/api/system?probe_media=true').status_code == 401
    # Authenticated settings reads must not import ML libraries.
    from auth.authentication import get_current_subject
    main.app.dependency_overrides[get_current_subject] = lambda: 'owner'
    try:
        for _ in range(2):
            assert client.get('/api/settings/ml-warmup').json()['features'] == []
    finally:
        main.app.dependency_overrides.clear()
    worker = main._post_warm_thread
    assert worker is not None, 'startup maintenance must still run'
    worker.join(20)
    assert not worker.is_alive(), 'cold startup maintenance did not finish'
    assert warm_status()['started'] is False
    heavy = ('torch', 'transformers', 'datasets', 'diffusers', 'torchao', 'pandas', 'pyarrow')
    loaded = [name for name in heavy if name in sys.modules]
    assert not loaded, loaded
print('IDLE_STARTUP_STAYS_COLD')
"""


_TEXT_START = r"""
import sys
from multiprocessing.process import BaseProcess
import main
from fastapi.testclient import TestClient
from utils.ml_warmup_settings import set_ml_warmup_features
from utils import torch_warmup

set_ml_warmup_features(['text'])
# Model availability must not turn a text-only preference into media or quant warmup.
torch_warmup._a_local_model_would_load_through_diffusers = lambda: True
spawned = []
original_start = BaseProcess.start
def record_start(process, *args, **kwargs):
    spawned.append(process._args[1] if len(process._args) > 1 else None)
    return original_start(process, *args, **kwargs)
BaseProcess.start = record_start

main.app.state.suppress_bootstrap_injection = True
with TestClient(main.app) as client:
    assert client.get('/api/liveness').status_code == 200
    assert client.get('/api/health').status_code == 200
    assert torch_warmup.join_background_warm(60), 'text warmup did not finish'
    main._post_warm_thread.join(20)
    assert not main._post_warm_thread.is_alive()
    state = torch_warmup.warm_status()
    assert list(state['stages']) == ['hardware', 'inference_backend', 'transformers']
    assert all(stage['ok'] for stage in state['stages'].values()), state
    for name in ('torch', 'torch._dynamo', 'transformers.models.auto.modeling_auto'):
        assert name in sys.modules, name
    for name in ('datasets', 'diffusers', 'torchao'):
        assert name not in sys.modules, name
    assert '_child_probe_entry' not in spawned, spawned
print('TEXT_WARMUP_STAYS_SCOPED')
"""


def _run_startup(
    snippet,
    tmp_path,
    timeout = 120,
):
    env = os.environ.copy()
    env.pop("UNSLOTH_STUDIO_DISABLE_TORCH_WARM", None)
    env.pop("UNSLOTH_STUDIO_DISABLE_DIFFUSERS_PREWARM", None)
    env.update(
        {
            "UNSLOTH_HOME": str(tmp_path / "unsloth"),
            "UNSLOTH_STUDIO_HOME": str(tmp_path / "unsloth" / "studio"),
            "UNSLOTH_STUDIO_SANDBOX_HOME": str(tmp_path / "sandbox"),
            "HF_HOME": str(tmp_path / "hf"),
            "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
            "HF_XET_CACHE": str(tmp_path / "hf" / "xet"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "UNSLOTH_DISABLE_UPDATE_CHECK": "1",
            "UNSLOTH_DISABLE_MLX_AUTOREPAIR": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "UNSLOTH_STUDIO_STALL_WATCHDOG": "0",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd = _BACKEND_DIR,
        env = env,
        capture_output = True,
        text = True,
        timeout = timeout,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr[-5000:]}"
    return result.stdout


def test_idle_startup_and_health_leave_ml_libraries_cold(tmp_path):
    output = _run_startup(_COLD_START, tmp_path)
    assert "IDLE_STARTUP_STAYS_COLD" in output


def test_text_only_startup_imports_text_stack_without_media_or_quant_probe(tmp_path):
    for package in ("torch", "transformers"):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"{package} is not installed")
    output = _run_startup(_TEXT_START, tmp_path)
    assert "TEXT_WARMUP_STAYS_SCOPED" in output


_RESTART_SELECTION = r"""
import sys
import main
from fastapi.testclient import TestClient
from auth.authentication import get_current_subject
from utils.ml_warmup_settings import get_ml_warmup_features
from utils import torch_warmup, stall_watchdog

main.app.state.suppress_bootstrap_injection = True
main.app.dependency_overrides[get_current_subject] = lambda: 'owner'
# Record scheduling while avoiding ML imports: this test isolates full-lifespan snapshot wiring.
scheduled = []
watchdog_gates = []
# Check the actual callback main supplies, including preferences saved during this lifespan.
stall_watchdog.start_stall_watchdog = lambda loop, suppress: watchdog_gates.append(suppress)
def record_start(*, gate, features):
    scheduled.append(frozenset(features))
    return True
main.start_background_warm = record_start
main.prewarm_diffusers_if_image_models_exist = lambda **kwargs: None
main.prewarm_quant_probe_if_image_models_exist = lambda: None
for previous, saved, expected in (([], ['text'], []), (['text'], ['datasets'], ['hardware', 'text']), (['datasets'], [], ['datasets']), ([], [], [])):
    with TestClient(main.app) as client:
        assert client.get('/api/liveness').status_code == 200
        assert main._startup_ml_warmup_features == frozenset(expected)
        assert watchdog_gates[-1]() is bool(expected)
        assert client.get('/api/settings/ml-warmup').json()['features'] == previous
        response = client.put('/api/settings/ml-warmup', json={'features': saved})
        assert response.status_code == 200, response.text
        assert response.json()['features'] == saved
        assert get_ml_warmup_features() == frozenset(saved)
        assert main._startup_ml_warmup_features == frozenset(expected)
        assert watchdog_gates[-1]() is bool(expected)
        main._post_warm_thread.join(20)
        assert not main._post_warm_thread.is_alive()
assert scheduled == [frozenset({'hardware', 'text'}), frozenset({'datasets'})], scheduled
print('SAVING_APPLIES_ONLY_TO_NEXT_LIFESPAN')
"""


def test_saving_changes_only_the_next_lifespan_and_turning_off_stops_scheduling(tmp_path):
    output = _run_startup(_RESTART_SELECTION, tmp_path)
    assert "SAVING_APPLIES_ONLY_TO_NEXT_LIFESPAN" in output


_SYSTEM_DEMAND = r"""
import sys
from multiprocessing.process import BaseProcess
import main
from fastapi.testclient import TestClient
from auth.authentication import get_current_subject
from utils.torch_warmup import warm_status

main.app.state.suppress_bootstrap_injection = True
main.app.dependency_overrides[get_current_subject] = lambda: 'owner'
spawned = []
original_start = BaseProcess.start
def record_start(process, *args, **kwargs):
    spawned.append(process._args[1] if len(process._args) > 1 else None)
    return original_start(process, *args, **kwargs)
BaseProcess.start = record_start

with TestClient(main.app) as client:
    assert client.get('/api/liveness').status_code == 200
    main._post_warm_thread.join(20)
    assert not main._post_warm_thread.is_alive()
    assert 'torch' not in sys.modules
    # Opening Resources/monitoring is explicit demand for real hardware information.
    for path in ('/api/system', '/api/system?refresh_memory=true'):
        reply = client.get(path)
        assert reply.status_code == 200, reply.text
        assert 'memory' in reply.json() and 'gpu' in reply.json()
    assert main._hw_module.DETECTION_COMPLETE.is_set()
    assert 'torch' in sys.modules
    unselected = ['datasets', 'diffusers', 'torchao']
    # Apple Silicon detection imports mlx_lm, which pulls in transformers.
    if not main._hw_module.is_apple_silicon():
        unselected += ['torch._dynamo', 'transformers']
    for name in unselected:
        assert name not in sys.modules, name
    assert warm_status()['started'] is False
    assert '_child_probe_entry' not in spawned, spawned
    assert 'hardware_detection_deferred' not in client.get('/api/health').json()
print('SYSTEM_DEMAND_LOADS_HARDWARE_ONLY')
"""


def test_system_information_on_first_use_detects_hardware_without_unselected_ml_stacks(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    output = _run_startup(_SYSTEM_DEMAND, tmp_path)
    assert "SYSTEM_DEMAND_LOADS_HARDWARE_ONLY" in output


_IDLE_CHAT = r"""
import json
import sys
import time
import psutil
import main
from fastapi.testclient import TestClient
from auth.authentication import get_current_subject, authenticated_via_api_key
from utils.torch_warmup import warm_status
from core.inference.orchestrator import peek_inference_backend

main.app.state.suppress_bootstrap_injection = True
main.app.dependency_overrides[get_current_subject] = lambda: 'owner'
main.app.dependency_overrides[authenticated_via_api_key] = lambda: False
heavy = ('torch', 'transformers', 'datasets', 'diffusers', 'torchao', 'pandas', 'pyarrow')
rows = []
# Read-only first-paint requests captured from the actual Chat page in Playwright.
paths = (
    '/api/health',
    '/api/auth/status',
    '/api/providers/registry?include_hidden=true&include_oauth=true',
    '/api/providers/',
    '/api/engines',
    '/api/hub/active-downloads',
    '/api/hub/datasets/active-downloads',
    '/api/export/status',
    '/api/settings/hugging-face-token',
    '/api/settings/pinned-models',
    '/api/settings/personalization',
    '/api/chat/settings',
    '/api/system/disk',
    '/api/inference/monitor',
    '/api/chat/projects?include_archived=false',
    '/api/skills',
    '/api/rag/knowledge-bases',
    '/api/settings/multi-model',
    '/api/inference/images/status',
    '/api/inference/video/status',
    '/api/hub/hidden-models',
    '/api/models/list',
    '/api/inference/status',
    '/api/scheduled-tasks/runs',
    '/api/hub/local',
    '/api/llama/update-status?force_refresh=true',
    '/api/chat/pins',
    '/api/models/loras',
    '/api/studio/update-status',
)
requests = [('GET', path, None) for path in paths]
for archived, unfiled, limit in ((False, True, 20), (False, False, 1), (True, False, 1)):
    requests.append(('POST', '/api/chat/sidebar/rows', {
        'archived': archived, 'projectId': None, 'unfiledOnly': unfiled,
        'includeIds': None, 'excludeIds': [], 'sort': 'updated',
        'manualOrder': None, 'limit': limit, 'cursor': None,
    }))
with TestClient(main.app) as client:
    assert client.get('/api/liveness').status_code == 200
    main._post_warm_thread.join(20)
    assert not main._post_warm_thread.is_alive()
    assert not any(name in sys.modules for name in heavy)
    for pass_index in range(2):
        for method, path, body in requests:
            started = time.monotonic()
            response = client.request(method, path, **({'json': body} if body is not None else {}))
            row = {
                'pass': pass_index + 1,
                'method': method,
                'path': path,
                'status': response.status_code,
                'elapsed_s': round(time.monotonic() - started, 3),
                'rss_mib': round(psutil.Process().memory_info().rss / 1024**2, 1),
                'imports': [name for name in heavy if name in sys.modules],
            }
            rows.append(row)
            print('IDLE_CHAT_ROUTE ' + json.dumps(row), flush=True)
            assert response.status_code == 200, response.text
            if path == '/api/models/list':
                assert response.json()['models'], 'cold catalog must keep model suggestions'
    assert not any(row['imports'] for row in rows), rows
    assert not main._hw_module.DETECTION_COMPLETE.is_set()
    assert peek_inference_backend() is None
    assert warm_status()['started'] is False
print('IDLE_CHAT_REQUESTS_STAY_COLD')
"""


def test_idle_chat_model_status_inventory_and_monitor_requests_leave_ml_cold(tmp_path):
    # 65 sequential requests; TestClient adds seconds per request on some hosts.
    output = _run_startup(_IDLE_CHAT, tmp_path, timeout = 600)
    assert "IDLE_CHAT_REQUESTS_STAY_COLD" in output

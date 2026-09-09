import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from hub.schemas.downloads import CancelDownloadRequest
from hub.services import download_lifecycle, load_downloads
from hub.utils import download_registry
from hub.utils.download_registry import DownloadRegistry


@pytest.fixture
def registry(monkeypatch):
    fresh = DownloadRegistry()
    monkeypatch.setattr(download_registry, "get_models_registry", lambda: fresh)
    return fresh


def _active(registry):
    return [
        (ref.repo_id, ref.owner, ref.state)
        for ref in download_lifecycle.active_download_refs(registry, None, with_variant = False)
    ]


def test_a_load_claims_running_jobs_the_active_list_reports(registry):
    keys = load_downloads.claim_load_downloads(
        ["owner/adapter", "Owner/Base", "owner/adapter", ""],
        xet_disabled = True,
        hub_cache = "/cache/hub",
    )

    assert len(keys) == 2
    assert sorted(_active(registry)) == [
        ("Owner/Base", "load", "running"),
        ("owner/adapter", "load", "running"),
    ]
    metadata = registry.get_job_metadata(keys[1])
    assert metadata.hub_cache == "/cache/hub"
    assert metadata.transport == "http"

    load_downloads.release_load_downloads(keys, "complete")

    assert _active(registry) == []
    assert registry.get_job(keys[0]).state == "complete"


def test_a_load_leaves_a_hub_download_of_the_same_repo_alone(registry):
    assert registry.claim("owner/base::", "http", repo_type = "model", repo_id = "owner/base")[0]

    assert load_downloads.claim_load_downloads(["owner/base"]) == []
    load_downloads.release_load_downloads(["owner/base::"], "complete")

    assert registry.get_job("owner/base::").state == "running"
    assert _active(registry) == [("owner/base", None, "running")]


def test_cancelling_a_load_owned_job_is_refused(registry, monkeypatch):
    from hub.services.models import downloads

    monkeypatch.setattr(downloads, "_registry", registry)
    load_downloads.claim_load_downloads(["owner/base"])

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            downloads.cancel_download_model_response(CancelDownloadRequest(repo_id = "owner/base"))
        )

    assert excinfo.value.status_code == 409
    assert registry.get_job("owner/base::").state == "running"


def test_the_orchestrator_registers_the_downloads_a_load_reports(registry, monkeypatch):
    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    with patch("core.inference.orchestrator.threading.Thread", DummyThread):
        from core.inference.orchestrator import InferenceOrchestrator
        orchestrator = InferenceOrchestrator()

    responses = iter(
        [
            {
                "type": "downloads",
                "repo_ids": ["owner/adapter", "owner/base"],
                "xet_disabled": True,
            },
            {"type": "loaded", "success": True},
        ]
    )
    monkeypatch.setattr(orchestrator, "_read_resp", lambda timeout = 1.0: next(responses))

    assert orchestrator._wait_response("loaded", timeout = 5)["success"]
    assert _active(registry) == [
        ("owner/adapter", "load", "running"),
        ("owner/base", "load", "running"),
    ]

    orchestrator._release_load_downloads("complete")

    assert _active(registry) == []
    orchestrator._release_load_downloads("complete")


def test_the_worker_reports_the_repos_a_lora_load_fetches(monkeypatch):
    import sys
    import types

    from core.inference import worker

    package = types.ModuleType("unsloth")
    models = types.ModuleType("unsloth.models")
    loader_utils = types.ModuleType("unsloth.models.loader_utils")
    loader_utils.get_model_name = (
        lambda name, load_in_4bit = True: "unsloth/base-bnb-4bit" if load_in_4bit else name
    )
    for name, module in (
        ("unsloth", package),
        ("unsloth.models", models),
        ("unsloth.models.loader_utils", loader_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    adapter = SimpleNamespace(identifier = "owner/adapter", base_model = "owner/base")
    cuda = SimpleNamespace(device = "cuda")

    assert worker._load_download_repos(adapter, True, cuda) == [
        "owner/adapter",
        "owner/base",
        "unsloth/base-bnb-4bit",
    ]
    assert worker._load_download_repos(adapter, False, cuda) == ["owner/adapter", "owner/base"]
    assert worker._load_download_repos(adapter, True, SimpleNamespace(device = "mlx")) == [
        "owner/adapter",
        "owner/base",
    ]
    local = SimpleNamespace(identifier = "/models/adapter", base_model = "owner/base")
    assert worker._load_download_repos(local, True, SimpleNamespace(device = "mlx")) == ["owner/base"]
    plain = SimpleNamespace(identifier = "owner/model", base_model = None)
    assert worker._load_download_repos(plain, True, cuda) == ["owner/model"]

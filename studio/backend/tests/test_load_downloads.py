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

    load_downloads.release_load_downloads(keys)

    assert _active(registry) == []
    assert registry.get_job(keys[0]).state == "idle"
    assert registry.claim(keys[0], "http", repo_type = "model", repo_id = "owner/adapter")[0]


def test_a_load_attaches_to_a_hub_download_of_the_same_repo(registry):
    assert registry.claim("owner/base::", "http", repo_type = "model", repo_id = "owner/base")[0]

    keys = load_downloads.claim_load_downloads(["owner/base"])

    assert keys == ["owner/base::"]
    ref = download_lifecycle.active_download_refs(registry, None, with_variant = False)[0]
    assert (ref.owner, ref.load_attached, ref.state) == (None, True, "running")
    assert not load_downloads.is_load_owned(registry, "owner/base::")

    load_downloads.release_load_downloads(keys)

    ref = download_lifecycle.active_download_refs(registry, None, with_variant = False)[0]
    assert (ref.load_attached, ref.state) == (False, "running")


def test_a_load_attaches_to_a_variant_download_of_the_same_repo(registry):
    assert registry.claim(
        "owner/base::q4_k_m", "http", repo_type = "model", repo_id = "owner/base", variant = "Q4_K_M"
    )[0]

    keys = load_downloads.claim_load_downloads(["owner/base"])

    assert keys == ["owner/base::q4_k_m"]
    ref = download_lifecycle.active_download_refs(registry, None, with_variant = True)[0]
    assert (ref.repo_id, ref.variant, ref.load_attached) == ("owner/base", "Q4_K_M", True)

    load_downloads.release_load_downloads(keys)

    ref = download_lifecycle.active_download_refs(registry, None, with_variant = True)[0]
    assert (ref.load_attached, ref.state) == (False, "running")


def test_a_transport_retry_keeps_the_load_attachment(registry):
    assert registry.claim("owner/base::", "xet", repo_type = "model", repo_id = "owner/base")[0]
    load_downloads.claim_load_downloads(["owner/base"])

    assert registry.claim(
        "owner/base::", "http", repo_type = "model", repo_id = "owner/base", replace_active = True
    )[0]

    ref = download_lifecycle.active_download_refs(registry, None, with_variant = False)[0]
    assert (ref.transport, ref.load_attached) == ("http", True)
    registry.set_job("owner/base::", "complete")
    assert registry.claim("owner/base::", "http", repo_type = "model", repo_id = "owner/base")[0]
    assert registry.get_job_metadata("owner/base::").load_attached is False


def test_a_load_placeholder_is_never_adoptable(registry):
    load_downloads.claim_load_downloads(["owner/base"])
    assert registry.adoptable("owner/base::") is False
    assert registry.claim("owner/other::", "http", repo_type = "model", repo_id = "owner/other")[0]
    assert registry.adoptable("owner/other::") is True


def test_an_explicit_download_of_a_load_placeholder_is_refused(registry, monkeypatch):
    from hub.services.models import downloads

    monkeypatch.setattr(downloads, "_registry", registry)
    load_downloads.claim_load_downloads(["owner/base"])

    with pytest.raises(HTTPException) as excinfo:
        downloads._reject_if_load_owned("owner/base::")
    assert excinfo.value.status_code == 409

    load_downloads.release_load_downloads(["owner/base::"])
    downloads._reject_if_load_owned("owner/base::")


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


def test_shutdown_leaves_no_cancel_marker_for_a_load_placeholder(registry, monkeypatch):
    markers = []
    monkeypatch.setattr(
        download_registry, "persist_cancel_marker", lambda *args, **kwargs: markers.append(args[1])
    )
    load_downloads.claim_load_downloads(["owner/adapter", "owner/base"])
    assert registry.claim("owner/other::", "http", repo_type = "model", repo_id = "owner/other")[0]

    registry.terminate_all()

    assert markers == ["owner/other"]
    assert registry.get_job("owner/adapter::").state == "idle"
    assert registry.get_job("owner/base::").state == "idle"
    assert _active(registry) == [("owner/other", None, "cancelling")]


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

    orchestrator._release_load_downloads()

    assert _active(registry) == []
    orchestrator._release_load_downloads()


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
    loader = types.ModuleType("unsloth.models.loader")
    loader.ALLOW_PREQUANTIZED_MODELS = True
    loader.ALLOW_BITSANDBYTES = True
    loader._strip_unsloth_bnb_4bit_suffix = lambda name: name.removesuffix("-bnb-4bit")
    for name, module in (
        ("unsloth", package),
        ("unsloth.models", models),
        ("unsloth.models.loader", loader),
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
    legacy = SimpleNamespace(identifier = "owner/gpt2-lora", base_model = "gpt2")
    assert worker._load_download_repos(legacy, False, cuda) == ["owner/gpt2-lora", "gpt2"]

    loader.ALLOW_PREQUANTIZED_MODELS = False
    assert worker._load_download_repos(adapter, True, cuda) == [
        "owner/adapter",
        "owner/base",
        "unsloth/base",
    ]
    loader.ALLOW_PREQUANTIZED_MODELS = True

    loader.ALLOW_BITSANDBYTES = False
    assert worker._load_download_repos(adapter, True, cuda) == ["owner/adapter", "owner/base"]
    loader.ALLOW_BITSANDBYTES = True

    audio = SimpleNamespace(identifier = "owner/tts-lora", base_model = "owner/base", audio_type = "snac")
    assert worker._load_download_repos(audio, True, cuda, ["owner/tts-lora", "owner/codec"]) == [
        "owner/tts-lora",
        "owner/codec",
        "owner/base",
        "hubertsiuzdak/snac_24khz",
    ]
    spark = SimpleNamespace(
        identifier = "owner/spark-lora", base_model = "unsloth/Spark-TTS-0.5B/LLM", audio_type = "bicodec"
    )
    monkeypatch.setattr(worker, "_load_download_repos", worker._load_download_repos)
    import utils.security.file_security as file_security

    monkeypatch.setattr(
        file_security,
        "load_scan_target",
        lambda name, subdirs: ("unsloth/Spark-TTS-0.5B", ("LLM",))
        if name.endswith("/LLM")
        else (name, subdirs),
    )
    assert worker._load_download_repos(spark, True, SimpleNamespace(device = "mlx")) == [
        "owner/spark-lora",
        "unsloth/Spark-TTS-0.5B",
    ]

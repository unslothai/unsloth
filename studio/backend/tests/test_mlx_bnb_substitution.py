# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.defaults as defaults_mod  # noqa: E402
import utils.hardware.hardware as hw  # noqa: E402
from core.inference.model_ids import mlx_bnb_base_repo, mlx_bnb_substitutions  # noqa: E402


def test_unsloth_bnb_repos_resolve_to_their_base():
    assert (
        mlx_bnb_base_repo("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit") == "unsloth/Qwen2-VL-2B-Instruct"
    )
    assert mlx_bnb_base_repo("unsloth/gemma-3-4b-it-unsloth-bnb-4bit") == "unsloth/gemma-3-4b-it"


def test_repos_mlx_loads_as_given_have_no_base():
    assert mlx_bnb_base_repo("unsloth/Qwen3-4B-Instruct-2507") is None
    assert mlx_bnb_base_repo("unsloth/Llama-3.2-1B-Instruct-GGUF") is None
    assert mlx_bnb_base_repo("someone-else/model-bnb-4bit") is None
    assert mlx_bnb_base_repo(None) is None


def test_a_local_directory_is_never_remapped(monkeypatch, tmp_path):
    (tmp_path / "unsloth" / "model-bnb-4bit").mkdir(parents = True)
    monkeypatch.chdir(tmp_path)

    assert mlx_bnb_base_repo("unsloth/model-bnb-4bit") is None


def test_substitutions_cover_a_loras_base():
    swaps = mlx_bnb_substitutions(["me/my-lora", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"])
    assert swaps == [
        ("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "unsloth/Meta-Llama-3.1-8B-Instruct")
    ]

    already_watched = ["unsloth/gemma-3-4b-it-bnb-4bit", "unsloth/gemma-3-4b-it"]
    assert mlx_bnb_substitutions(already_watched) == [
        ("unsloth/gemma-3-4b-it-bnb-4bit", "unsloth/gemma-3-4b-it")
    ]


def _standard_host(monkeypatch, device):
    monkeypatch.setattr(hw, "CHAT_ONLY", False)
    monkeypatch.setattr(hw, "get_device", lambda: device)


def test_mlx_defaults_recommend_the_repos_mlx_actually_loads(monkeypatch):
    _standard_host(monkeypatch, hw.DeviceType.MLX)

    models = defaults_mod.get_default_models()

    assert [model for model in models if model.endswith("bnb-4bit")] == []
    assert "unsloth/Qwen2-VL-2B-Instruct" in models
    assert len(models) == len(set(models))


def test_cuda_defaults_keep_the_bnb_repos(monkeypatch):
    _standard_host(monkeypatch, hw.DeviceType.CUDA)

    assert defaults_mod.get_default_models() == defaults_mod.DEFAULT_MODELS_STANDARD


def test_chat_only_hosts_are_untouched(monkeypatch):
    monkeypatch.setattr(hw, "CHAT_ONLY", True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)

    assert defaults_mod.get_default_models() == defaults_mod.DEFAULT_MODELS_GGUF


@pytest.mark.parametrize(
    "identifier,base_model,is_lora,expected",
    [
        (
            "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
            None,
            False,
            ["unsloth/Qwen2-VL-2B-Instruct"],
        ),
        (
            "me/my-lora",
            "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
            True,
            ["me/my-lora", "unsloth/Qwen2-VL-2B-Instruct"],
        ),
        (
            "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
            "unsloth/Qwen2-VL-2B-Instruct",
            True,
            ["unsloth/Qwen2-VL-2B-Instruct"],
        ),
    ],
)
def test_worker_only_watches_the_repositories_mlx_downloads(
    monkeypatch, identifier, base_model, is_lora, expected
):
    from types import SimpleNamespace

    import core.inference.worker as worker_mod
    import utils.hf_xet_fallback as fallback_mod

    model_config = SimpleNamespace(
        identifier = identifier,
        base_model = base_model,
        is_lora = is_lora,
    )
    captured = {}

    class StopEvent:
        def set(self):
            pass

    class ResponseQueue:
        def put(self, response):
            pass

    class Backend:
        device = "mlx"

        def load_model(self, **kwargs):
            return False

    def start_watchdog(**kwargs):
        captured.update(kwargs)
        return StopEvent()

    monkeypatch.setattr(worker_mod, "_build_model_config", lambda config: model_config)
    monkeypatch.setattr(worker_mod, "_resolve_lora_4bit", lambda model, requested: False)
    monkeypatch.setattr(worker_mod, "_run_security_gates", lambda *args, **kwargs: True)
    monkeypatch.setattr(fallback_mod, "start_watchdog", start_watchdog)

    worker_mod._handle_load(Backend(), {"model_name": identifier}, ResponseQueue())

    assert captured["repo_ids"] == expected


def test_the_host_rule_only_fires_on_mlx(monkeypatch):
    from core.inference.model_ids import mlx_host_bnb_base_repo

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    assert mlx_host_bnb_base_repo("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit") is None

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    assert (
        mlx_host_bnb_base_repo("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
        == "unsloth/Qwen2-VL-2B-Instruct"
    )


def test_diffusion_bnb_repos_are_loaded_as_named(monkeypatch):
    from core.inference.model_ids import mlx_host_bnb_base_repo

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    assert mlx_host_bnb_base_repo("unsloth/Qwen-Image-2512-unsloth-bnb-4bit") is None
    assert mlx_host_bnb_base_repo("unsloth/Z-Image-Turbo-unsloth-bnb-4bit") is None


def test_validate_reports_the_repo_mlx_will_load(monkeypatch):
    from types import SimpleNamespace

    from routes.inference import _mlx_base_for_config

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    pick = SimpleNamespace(identifier = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit", base_model = None)
    assert _mlx_base_for_config(pick) == "unsloth/Qwen2-VL-2B-Instruct"

    adapter = SimpleNamespace(
        identifier = "me/my-lora",
        base_model = "unsloth/gemma-3-4b-it-bnb-4bit",
    )
    assert _mlx_base_for_config(adapter) == "unsloth/gemma-3-4b-it"

    plain = SimpleNamespace(identifier = "unsloth/Qwen3-4B-Instruct-2507", base_model = None)
    assert _mlx_base_for_config(plain) is None


def test_the_fetched_ranking_is_mapped_on_a_mac_too(monkeypatch):
    import core.inference.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.InferenceOrchestrator, "_fetch_top_models", lambda self: None)
    monkeypatch.setattr(defaults_mod, "get_default_models", lambda: ["unsloth/curated"])
    monkeypatch.setattr(hw, "DETECTION_GENERATION", 1)
    monkeypatch.setattr(hw, "CHAT_ONLY", False)
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.MLX)

    orch = orch_mod.InferenceOrchestrator()
    orch._top_gguf_cache = ["unsloth/Qwen3-4B-GGUF"]
    orch._top_hub_cache = ["unsloth/Qwen3-8B-unsloth-bnb-4bit", "unsloth/Qwen3-4B-Instruct-2507"]

    assert orch.default_models == [
        "unsloth/curated",
        "unsloth/Qwen3-4B-GGUF",
        "unsloth/Qwen3-8B",
        "unsloth/Qwen3-4B-Instruct-2507",
    ]


def test_a_diffusion_bnb_repo_in_the_ranking_keeps_its_name(monkeypatch):
    import core.inference.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.InferenceOrchestrator, "_fetch_top_models", lambda self: None)
    monkeypatch.setattr(defaults_mod, "get_default_models", lambda: ["unsloth/curated"])
    monkeypatch.setattr(hw, "DETECTION_GENERATION", 1)
    monkeypatch.setattr(hw, "CHAT_ONLY", False)
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.MLX)

    orch = orch_mod.InferenceOrchestrator()
    orch._top_hub_cache = [
        "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
        "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    ]

    assert orch.default_models == [
        "unsloth/curated",
        # diffusion: read as named by diffusers/MPS
        "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
        "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
        # text: mlx-lm cannot read it, so name the repo it really loads
        "unsloth/Qwen3-8B",
    ]


def test_the_fetched_ranking_is_untouched_off_mlx(monkeypatch):
    import core.inference.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.InferenceOrchestrator, "_fetch_top_models", lambda self: None)
    monkeypatch.setattr(defaults_mod, "get_default_models", lambda: ["unsloth/curated"])
    monkeypatch.setattr(hw, "DETECTION_GENERATION", 1)
    monkeypatch.setattr(hw, "CHAT_ONLY", False)
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CUDA)

    orch = orch_mod.InferenceOrchestrator()
    orch._top_hub_cache = ["unsloth/Qwen3-8B-unsloth-bnb-4bit"]

    assert orch.default_models == ["unsloth/curated", "unsloth/Qwen3-8B-unsloth-bnb-4bit"]


def test_the_mirror_still_matches_the_loader_it_mirrors():
    loader = pytest.importorskip("unsloth_zoo.mlx.loader")

    for name in (
        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Instruct-2507",
        "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "someone-else/model-bnb-4bit",
    ):
        remapped, _revision, swapped_from = loader._remap_unsloth_bnb_hub_id_for_mlx(
            name, "some-revision"
        )
        expected = remapped if swapped_from is not None else None
        assert mlx_bnb_base_repo(name) == expected, name


@pytest.mark.parametrize(
    "weight_names",
    [
        ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
        ("adapter_model.safetensors",),
        ("adapters.safetensors",),
    ],
)
@pytest.mark.parametrize("offline", [False, True])
def test_mlx_progress_verifies_loader_files_without_a_manifest(
    monkeypatch, tmp_path, weight_names, offline
):
    import asyncio
    from collections import OrderedDict
    from types import SimpleNamespace
    from huggingface_hub import HfApi
    from hub.services import snapshot_progress
    from hub.services.models import cache_inventory, downloads

    repo = "unsloth/progress-model"
    entry = tmp_path / "models--unsloth--progress-model"
    snap = entry / "snapshots" / ("a" * 40)
    snap.mkdir(parents = True)
    blobs = entry / "blobs"
    blobs.mkdir()
    names = ("config.json", *weight_names, "README.md")
    siblings = [
        SimpleNamespace(rfilename = name, size = 4, blob_id = f"blob{i}", lfs = None)
        for i, name in enumerate(names)
    ]
    monkeypatch.setattr(HfApi, "model_info", lambda *_a, **_k: SimpleNamespace(siblings = siblings))
    monkeypatch.setattr(cache_inventory, "_mlx_plan_cache", OrderedDict())
    if offline:
        from huggingface_hub._tree_cache import TreeCacheEntry, write_tree_cache
        import hub.utils.hf_cache_state as cache_state

        (entry / "refs").mkdir()
        (entry / "refs" / "main").write_text(snap.name)
        write_tree_cache(
            str(entry),
            snap.name,
            {
                item.rfilename: TreeCacheEntry(
                    size = item.size, blob_id = "pointer", lfs_sha256 = item.blob_id
                )
                for item in siblings
            },
        )

        def offline_info(*_a, **_k):
            raise OSError("offline")

        monkeypatch.setattr(HfApi, "model_info", offline_info)
        monkeypatch.setattr(cache_state, "preferred_repo_cache_dirs", lambda *_a, **_k: [entry])
    monkeypatch.setattr(snapshot_progress, "preferred_repo_cache_dirs", lambda *_a, **_k: [entry])
    monkeypatch.setattr(
        downloads, "_registry", SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "idle"))
    )

    def progress():
        return asyncio.run(downloads.get_download_progress_response(repo, mlx_load = True))

    for i, name in enumerate(names[:-1]):
        (blobs / f"blob{i}").write_bytes(b"data")
        (snap / name).symlink_to(blobs / f"blob{i}")
        reading = progress()
        assert reading["expected_bytes"] == 4 * (len(names) - 1)
        assert reading["complete_on_disk"] is (i == len(names) - 2)
    assert reading["progress"] == 1
    generic = asyncio.run(downloads.get_download_progress_response(repo))
    assert generic["complete_on_disk"] is False
    assert generic["progress"] < 1
    for name in weight_names:
        (snap / name).unlink()
        assert progress()["complete_on_disk"] is False
        i = names.index(name)
        (snap / name).symlink_to(blobs / f"blob{i}")
        assert progress()["progress"] == 1


@pytest.mark.parametrize("warm", [False, True])
def test_mlx_metadata_outage_reuses_plan_and_bounds_retries(monkeypatch, warm):
    from collections import OrderedDict
    from huggingface_hub import HfApi
    from hub.services.models import cache_inventory

    key = ("unsloth/cached-base", cache_inventory.hf_cache_scan.token_fingerprint(None))
    plan = (4, frozenset({"weight"}), ())
    expired = cache_inventory.time.monotonic() - cache_inventory._REPO_SIZE_POS_TTL - 1
    monkeypatch.setattr(
        cache_inventory, "_mlx_plan_cache", OrderedDict({key: (plan, expired)} if warm else {})
    )
    monkeypatch.setattr(cache_inventory, "_cached_mlx_siblings", lambda _repo: [])
    calls = []

    def unavailable(*_a, **_k):
        calls.append(True)
        raise OSError("unreachable")

    monkeypatch.setattr(HfApi, "model_info", unavailable)
    expected = plan if warm else (0, frozenset(), ())
    for _ in range(2):
        assert cache_inventory.get_mlx_load_plan_cached(key[0]) == expected
    assert len(calls) == 1

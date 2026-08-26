# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MLX loads the full-precision base of an unsloth bnb-4bit repo, never the bnb weights.

The swap happens inside ``unsloth_zoo.mlx.loader``, which only prints it, so a curated
list naming a bnb repo costs an Apple Silicon user a download that is then discarded for
a much larger one, and the load-time stall watchdog measures a repo nothing is writing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.defaults as defaults_mod  # noqa: E402
import utils.hardware.hardware as hw  # noqa: E402
from core.inference.mlx_bnb import mlx_bnb_base_repo, mlx_bnb_substitutions  # noqa: E402


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
    """Mirrors the loader: an on-disk path that looks like a repo id is loaded as given."""
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
    from core.inference.mlx_bnb import mlx_host_bnb_base_repo

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    assert mlx_host_bnb_base_repo("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit") is None

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    assert (
        mlx_host_bnb_base_repo("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
        == "unsloth/Qwen2-VL-2B-Instruct"
    )


def test_diffusion_bnb_repos_are_loaded_as_named(monkeypatch):
    """Diffusion runs on diffusers/MPS, which reads bnb weights; only the MLX loader cannot."""
    from core.inference.mlx_bnb import mlx_host_bnb_base_repo

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
    """/api/models/list serves curated + the remote ranking, and that ranking is mostly
    bnb repos, so leaving it alone keeps offering the download MLX discards."""
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
    """The suggestion list must make the same diffusion exception the host rule makes.

    unsloth publishes diffusion bnb repos and the ranking is ordered by downloads, so one
    can surface there at any time. They load on the diffusers/MPS path exactly as named,
    so mapping one would advertise the BIGGER full-precision repo in place of the one the
    loader uses -- the opposite of what this change is for.
    """
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
    """The rule lives in unsloth_zoo; this module only restates it, so pin the two together.

    importorskip for the same reason test_mlx_inference_backend.py uses it on this exact
    module: bare backend CI does not install Zoo, so skip rather than error there. Every
    behaviour asserted above stands on its own; this only catches Zoo moving underneath it.
    """
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

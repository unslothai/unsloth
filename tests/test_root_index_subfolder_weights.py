# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A root safetensors index mapping weights into subfolders (nvidia/Cosmos3-*) must keep those weights in the prefetch."""

import json
import sys
import types
from pathlib import Path

import pytest
from huggingface_hub.utils import filter_repo_objects

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unsloth.models._utils as U  # noqa: E402

SHARD = "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
VISION = "vision_encoder/model.safetensors"
COSMOS3 = {"weight_map": {"a": SHARD, "b": VISION}}
ROOT_SHARDS = {
    "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
}
LISTING = ["config.json", "model.safetensors.index.json", SHARD, VISION, "transformer/config.json"]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Serve a controlled root index (None = 404), a fixed listing, and a fake downloader; returns (request, Hub calls)."""
    import huggingface_hub

    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(flag, raising = False)
    state = {"index": None, "seen": {}, "calls": []}

    def fake_hf_hub_download(repo_id, filename, *a, **k):
        state["calls"].append({"filename": filename, **k})
        index = state["index"]
        if filename == "model.safetensors.index.json" and index is not None:
            p = tmp_path / filename
            p.write_text(index if isinstance(index, str) else json.dumps(index), encoding = "utf-8")
            return str(p)
        raise OSError(f"404 {filename}")

    class _Api:
        def model_info(self, *a, **k):
            return types.SimpleNamespace(
                siblings = [types.SimpleNamespace(rfilename = f) for f in LISTING]
            )

    def fake_download(repo_id, **kwargs):
        state["seen"] = kwargs
        return str(tmp_path / "snapshot")

    zoo = types.ModuleType("unsloth_zoo.hf_xet_fallback")
    zoo.snapshot_download_with_xet_fallback = fake_download
    zoo.DownloadStallError = type("DownloadStallError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "unsloth_zoo.hf_xet_fallback", zoo)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

    def run(
        index,
        model = "nvidia/Cosmos3-Edge",
        **kwargs,
    ):
        state.update(index = index, seen = {}, calls = [])
        U.maybe_prefetch_hf_snapshot(model, **{"weights_at_root": True, **kwargs})
        return state["seen"], state["calls"]

    run.state = state
    return run


@pytest.mark.parametrize(
    "index, expected",
    [
        (COSMOS3, ("transformer", "vision_encoder")),
        (ROOT_SHARDS, ()),
        (None, ()),
        (
            {"weight_map": {"a": "/transformer\\shard.safetensors", "b": "root.safetensors"}},
            ("transformer",),
        ),
        ("{ not json", ()),
        (["a", "list"], ()),
        ({"weight_map": ["not", "a", "dict"]}, ()),
        ({"weight_map": {"a": None, "b": 3, "c": "  /  "}}, ()),
        ({"no": "weight_map"}, ()),
    ],
    ids = [
        "subfolders",
        "root_only",
        "no_index",
        "normalised",
        "bad_json",
        "list",
        "map_list",
        "bad_values",
        "no_map",
    ],
)
def test_root_index_weight_subfolders(hub, index, expected):
    hub.state["index"] = index
    assert U._root_index_weight_subfolders("org/repo") == expected


@pytest.mark.parametrize(
    "index, keeps_subfolder_weights", [(COSMOS3, True), (ROOT_SHARDS, False), (None, False)]
)
def test_prefetch_keeps_subfolder_weights_only_when_the_root_index_maps_there(
    hub, index, keeps_subfolder_weights
):
    seen, _ = hub(index)
    assert seen, "the downloader was never reached"
    selected = set(
        filter_repo_objects(
            LISTING,
            allow_patterns = seen.get("allow_patterns"),
            ignore_patterns = seen.get("ignore_patterns"),
        )
    )
    assert ({SHARD, VISION} <= selected) is keeps_subfolder_weights, seen
    assert "transformer/config.json" in selected
    pruned, _ = hub(ROOT_SHARDS)
    diff = set(pruned["ignore_patterns"]) - set(seen["ignore_patterns"])
    assert diff == (set(U._SUBDIR_WEIGHT_IGNORE_PATTERNS) if keeps_subfolder_weights else set())
    assert set(seen["ignore_patterns"]) <= set(pruned["ignore_patterns"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokenizer_only": True},
        {"adapter_only": True},
        {"gguf_file": "m.gguf"},
        {"subfolder": "transformer"},
        {"weights_at_root": False},
        {"local_files_only": True},
    ],
)
def test_index_is_not_probed_without_a_root_weight_load(hub, kwargs):
    _, calls = hub(COSMOS3, **kwargs)
    assert all(c["filename"] != "model.safetensors.index.json" for c in calls), calls


def test_probe_forwards_revision_token_and_cache_dir(hub, tmp_path):
    _, calls = hub(COSMOS3, revision = "344d602b", token = "hf_test", cache_dir = str(tmp_path / "c"))
    probe = [c for c in calls if c["filename"] == "model.safetensors.index.json"]
    assert probe and probe[0]["revision"] == "344d602b" and probe[0]["token"] == "hf_test"
    assert probe[0]["cache_dir"] == str(tmp_path / "c")

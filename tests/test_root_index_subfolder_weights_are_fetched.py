# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A root safetensors index can map every weight into subfolders; the subfolder prune must skip them."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unsloth.models._utils as U  # noqa: E402


# nvidia/Cosmos3-Nano @ e59a53c2: config.json and the index sit at the root, every shard under a subfolder.
COSMOS3_INDEX = {
    "metadata": {"total_size": 1},
    "weight_map": {
        "model.language_model.embed_tokens.weight": "transformer/diffusion_pytorch_model-00001-of-00004.safetensors",
        "model.language_model.norm.weight": "transformer/diffusion_pytorch_model-00004-of-00004.safetensors",
        "model.visual.patch_embed.proj.weight": "vision_encoder/model.safetensors",
    },
}

ROOT_SHARDS_INDEX = {
    "metadata": {"total_size": 1},
    "weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    },
}


@pytest.fixture
def hub_files(tmp_path, monkeypatch):
    """Stub hf_hub_download so it serves the root index we control and 404s everything else."""

    def _install(index):
        import huggingface_hub
        def fake(repo_id, filename, *a, **k):
            if filename == "model.safetensors.index.json" and index is not None:
                p = tmp_path / filename
                p.write_text(
                    index if isinstance(index, str) else json.dumps(index), encoding = "utf-8"
                )
                return str(p)
            raise OSError(f"404 {filename}")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)

    return _install


def test_the_cosmos3_layout_is_detected(hub_files):
    hub_files(COSMOS3_INDEX)
    assert U._root_index_weight_subfolders("nvidia/Cosmos3-Nano") == (
        "transformer",
        "vision_encoder",
    )


def test_an_index_of_root_shards_is_not(hub_files):
    """The common sharded checkpoint: nothing lives in a subfolder, so the prune stays in force."""
    hub_files(ROOT_SHARDS_INDEX)
    assert U._root_index_weight_subfolders("unsloth/Qwen3-8B") == ()


def test_a_repo_without_a_root_index_is_not(hub_files):
    """A single-file model ships no index; it must keep the old behaviour exactly."""
    hub_files(None)
    assert U._root_index_weight_subfolders("unsloth/Qwen3-0.6B") == ()


@pytest.mark.parametrize(
    "payload",
    [
        "{ not json",
        json.dumps(["a", "list"]),
        json.dumps({"weight_map": ["not", "a", "dict"]}),
        json.dumps({"weight_map": {"a": None, "b": 3}}),
        json.dumps({"weight_map": {"a": "  /  "}}),
        json.dumps({"no": "weight_map"}),
    ],
)
def test_a_malformed_index_falls_back_to_the_old_behaviour(hub_files, payload):
    """Best-effort by design: an unreadable index must not start failing loads that work today."""
    hub_files(payload)
    assert U._root_index_weight_subfolders("org/whatever") == ()


def test_backslashes_and_stray_slashes_are_normalised(hub_files):
    hub_files({"weight_map": {"a": "/transformer\\shard.safetensors", "b": "root.safetensors"}})
    assert U._root_index_weight_subfolders("org/x") == ("transformer",)


def _ignores(model_name, monkeypatch):
    """Drives maybe_prefetch_hf_snapshot itself and returns the ignore_patterns it hands the downloader."""
    seen = {}

    def fake_download(name, **kwargs):
        seen.update(kwargs)
        return "/nonexistent/snapshot"

    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(flag, raising = False)

    import unsloth_zoo.hf_xet_fallback as XF

    monkeypatch.setattr(XF, "snapshot_download_with_xet_fallback", fake_download)
    import huggingface_hub

    class _Api:
        def model_info(self, *a, **k):
            raise RuntimeError("no network in test")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    U.maybe_prefetch_hf_snapshot(model_name, weights_at_root = True)
    assert seen, "the downloader was never reached; the call bailed out early"
    return list(seen.get("ignore_patterns") or [])


def test_the_subdir_weight_patterns_are_dropped_for_such_a_repo(hub_files, monkeypatch):
    hub_files(COSMOS3_INDEX)
    got = _ignores("nvidia/Cosmos3-Nano", monkeypatch)
    assert "*/*.safetensors" not in got, got


def test_the_subdir_weight_patterns_are_kept_for_root_shards(hub_files, monkeypatch):
    """The other half of the claim. Without this, the test above would pass
    just as well if the patterns had been deleted outright."""
    hub_files(ROOT_SHARDS_INDEX)
    got = _ignores("unsloth/Qwen3-8B", monkeypatch)
    assert "*/*.safetensors" in got, got


def test_only_the_subdir_weight_patterns_differ(hub_files, monkeypatch):
    """The fix must not quietly change anything else about the request."""
    hub_files(ROOT_SHARDS_INDEX)
    plain = set(_ignores("unsloth/Qwen3-8B", monkeypatch))
    hub_files(COSMOS3_INDEX)
    cosmos = set(_ignores("nvidia/Cosmos3-Nano", monkeypatch))
    assert plain - cosmos == set(U._SUBDIR_WEIGHT_IGNORE_PATTERNS)
    assert cosmos - plain == set()


def test_a_hub_failure_keeps_the_patterns(monkeypatch):
    """Network trouble must not silently enlarge every download."""
    import huggingface_hub

    def boom(*a, **k):
        raise RuntimeError("hub down")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    got = _ignores("org/anything", monkeypatch)
    assert "*/*.safetensors" in got


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

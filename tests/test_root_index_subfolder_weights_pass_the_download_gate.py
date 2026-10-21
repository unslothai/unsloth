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

"""The request the prefetch builds for a root index that maps weights into subfolders must pass unsloth_zoo's own
post-download gate. That contract is what broke for nvidia/Cosmos3-Edge: the subdir prune removed every weight, the
gate judged the snapshot through the same ignore list, and a complete download was reported as "incomplete snapshot
... check your network connection" on both transports."""

import json
import struct
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unsloth.models._utils as U  # noqa: E402

gate = pytest.importorskip("unsloth_zoo_downloads.hf_xet_fallback")
from huggingface_hub.utils import filter_repo_objects  # noqa: E402


TRANSFORMER_SHARDS = (
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
)
VISION_ENCODER = "vision_encoder/model.safetensors"
VAE = "vae/diffusion_pytorch_model.safetensors"

# nvidia/Cosmos3-Edge @ 344d602b, shape only: the reasoner's root index names the omni transformer's shards and the vision
# encoder; the diffusers pipeline reads transformer/, vae/, scheduler/ and text_tokenizer/.
ROOT_INDEX = {
    "metadata": {"total_size": 12},
    "weight_map": {
        "model.language_model.embed_tokens.weight": TRANSFORMER_SHARDS[0],
        "model.language_model.norm.weight": TRANSFORMER_SHARDS[1],
        "model.visual.patch_embed.proj.weight": VISION_ENCODER,
    },
}
MODEL_INDEX = {
    "_class_name": "Cosmos3OmniPipeline",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_tokenizer": ["transformers", "PreTrainedTokenizerFast"],
    "transformer": ["diffusers", "Cosmos3OmniTransformer"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "safety_checker": [None, None],
    "sound_tokenizer": [None, None],
}


def _safetensors(path, name = "w"):
    """A valid one-tensor safetensors file (4 bytes of float32), so a reader that parses the header accepts it."""
    header = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)


def _write_json(path, payload):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload), encoding = "utf-8")


def _cosmos3_tree(root, *, with_model_index = True):
    """A complete local snapshot shaped like Cosmos3-Edge, every weight a tiny real safetensors file."""
    _write_json(
        root / "config.json",
        {"model_type": "cosmos3_edge", "architectures": ["Cosmos3EdgeForConditionalGeneration"]},
    )
    _write_json(root / "model.safetensors.index.json", ROOT_INDEX)
    _write_json(root / "tokenizer_config.json", {})
    _write_json(root / "tokenizer.json", {})
    for shard in TRANSFORMER_SHARDS:
        _safetensors(root / shard)
    _safetensors(root / VISION_ENCODER)
    if with_model_index:
        _write_json(root / "model_index.json", MODEL_INDEX)
        _write_json(root / "transformer" / "config.json", {"_class_name": "Cosmos3OmniTransformer"})
        _write_json(
            root / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
            {
                "metadata": {"total_size": 8},
                "weight_map": {
                    "a": TRANSFORMER_SHARDS[0].split("/", 1)[1],
                    "b": TRANSFORMER_SHARDS[1].split("/", 1)[1],
                },
            },
        )
        _write_json(root / "vae" / "config.json", {"_class_name": "AutoencoderKLWan"})
        _safetensors(root / VAE)
        _write_json(
            root / "scheduler" / "scheduler_config.json", {"_class_name": "UniPCMultistepScheduler"}
        )
        _write_json(root / "text_tokenizer" / "tokenizer_config.json", {})
        _write_json(root / "text_tokenizer" / "tokenizer.json", {})
    return root


def _listing(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


@pytest.fixture
def prefetch(tmp_path, monkeypatch):
    """Drive maybe_prefetch_hf_snapshot against a fake downloader (injected, independent of the installed
    unsloth_zoo) and a stubbed Hub that serves the root index and the Edge-shaped file listing, so the format prune
    runs as it does in production; returns (kwargs the downloader got, Hub calls)."""
    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(flag, raising = False)
    seen, hub_calls = {}, []

    def fake_download(repo_id, **kwargs):
        seen.clear()
        seen.update(kwargs, repo_id = repo_id)
        return str(tmp_path / "snapshot")

    fake_module = types.ModuleType("unsloth_zoo.hf_xet_fallback")
    fake_module.snapshot_download_with_xet_fallback = fake_download
    fake_module.DownloadStallError = type("DownloadStallError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "unsloth_zoo.hf_xet_fallback", fake_module)

    import huggingface_hub

    siblings = _listing(_cosmos3_tree(tmp_path / "hub_listing"))

    class _Api:
        def model_info(self, *a, **k):
            return types.SimpleNamespace(
                siblings = [types.SimpleNamespace(rfilename = f) for f in siblings]
            )

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    index_file = tmp_path / "served" / "model.safetensors.index.json"
    _write_json(index_file, ROOT_INDEX)

    def fake_hf_hub_download(repo_id, filename, *a, **k):
        hub_calls.append({"repo_id": repo_id, "filename": filename, **k})
        if filename == "model.safetensors.index.json":
            return str(index_file)
        raise OSError(f"404 {filename}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    def run(model_name = "nvidia/Cosmos3-Edge", **kwargs):
        seen.clear()
        hub_calls.clear()
        U.maybe_prefetch_hf_snapshot(model_name, **kwargs)
        return dict(seen), list(hub_calls)

    return run


def _selected(names, request):
    """What the request selects from a listing, with the matcher snapshot_download and the zoo gate use."""
    return set(
        filter_repo_objects(
            names,
            allow_patterns = request.get("allow_patterns"),
            ignore_patterns = request.get("ignore_patterns"),
        )
    )


def _usable(snapshot, request):
    return gate._download_result_usable(
        Path(snapshot),
        repo_type = "model",
        allow_patterns = request.get("allow_patterns"),
        ignore_patterns = request.get("ignore_patterns"),
    )


def _fetched(full, dest, request):
    """What a fetch of this request leaves in an EMPTY cache: the files of the complete tree the request selects."""
    import shutil

    for name in _selected(_listing(full), request):
        target = dest / name
        target.parent.mkdir(parents = True, exist_ok = True)
        shutil.copyfile(full / name, target)
    return dest


def _pruned(request):
    """The request as it was before the fix: the same, plus the subdir weight prune."""
    return {
        **request,
        "ignore_patterns": [
            *(request.get("ignore_patterns") or []),
            *U._SUBDIR_WEIGHT_IGNORE_PATTERNS,
        ],
    }


def test_the_request_passes_the_gate_on_a_complete_snapshot(prefetch, tmp_path):
    seen, _ = prefetch(weights_at_root = True)
    assert seen, "the downloader was never reached"
    snapshot = _cosmos3_tree(tmp_path / "complete")
    assert _usable(snapshot, seen), seen


def test_the_old_request_fails_the_same_gate(prefetch, tmp_path):
    """The other half of the contract: the request before the fix is what the gate rejected, even over a complete
    cache, so this file would have caught the Cosmos3-Edge report."""
    seen, _ = prefetch(weights_at_root = True)
    snapshot = _cosmos3_tree(tmp_path / "complete")
    assert not _usable(snapshot, _pruned(seen)), _pruned(seen)


def test_the_request_fetches_every_weight_the_index_names(prefetch, tmp_path):
    seen, _ = prefetch(weights_at_root = True)
    selected = _selected(_listing(_cosmos3_tree(tmp_path / "full")), seen)
    for name in (*TRANSFORMER_SHARDS, VISION_ENCODER, VAE):
        assert name in selected, (name, seen)


@pytest.mark.parametrize("with_model_index", [True, False], ids = ["pipeline_repo", "reasoner_only"])
def test_a_fresh_download_of_the_request_passes_the_gate(prefetch, tmp_path, with_model_index):
    """An empty cache gets exactly what the request selects. Without model_index.json the gate judges the root index's
    shard set, so the pruned request (no subfolder shards fetched) is incomplete there too; with the fix it is whole."""
    seen, _ = prefetch(weights_at_root = True)
    full = _cosmos3_tree(tmp_path / "full", with_model_index = with_model_index)
    assert _usable(_fetched(full, tmp_path / "fixed", seen), seen), seen
    old = _pruned(seen)
    assert not _usable(_fetched(full, tmp_path / "pruned", old), old), old


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_files_only": True},
        {"tokenizer_only": True},
        {"adapter_only": True},
        {"gguf_file": "model-Q4_K_M.gguf"},
        {"subfolder": "transformer"},
        {"fast_inference": True},
        {"weights_at_root": False},
    ],
    ids = [
        "local_files_only",
        "tokenizer_only",
        "adapter_only",
        "gguf",
        "subfolder",
        "fast_inference",
        "whole_repo",
    ],
)
def test_the_index_is_not_probed_when_no_root_weights_are_warmed(prefetch, kwargs):
    _, hub_calls = prefetch(**{"weights_at_root": True, **kwargs})
    assert all(c["filename"] != "model.safetensors.index.json" for c in hub_calls), hub_calls


def test_the_index_is_not_probed_offline(prefetch, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _, hub_calls = prefetch(weights_at_root = True)
    assert all(c["filename"] != "model.safetensors.index.json" for c in hub_calls), hub_calls


def test_the_index_is_not_probed_for_a_local_folder(prefetch, tmp_path):
    local = _cosmos3_tree(tmp_path / "local")
    seen, hub_calls = prefetch(str(local), weights_at_root = True)
    assert not seen and not hub_calls


def test_the_probe_reads_the_requested_revision(prefetch, tmp_path):
    revision = "344d602b128d1bbdacb43b08d0a3626f46343e29"
    seen, hub_calls = prefetch(
        weights_at_root = True, revision = revision, token = "hf_test", cache_dir = str(tmp_path / "cache")
    )
    probe = [c for c in hub_calls if c["filename"] == "model.safetensors.index.json"]
    assert probe and probe[0].get("revision") == revision
    assert probe[0].get("token") == "hf_test" and probe[0].get("cache_dir") == str(
        tmp_path / "cache"
    )
    assert seen.get("revision") == revision


def test_the_vllm_snapshot_path_gets_the_same_request(prefetch, monkeypatch):
    """resolve_vllm_model_path forwards weights_at_root, so a tokened vLLM load builds the same request as a direct
    load, which differs from a whole-repo (weights_at_root=False) request."""
    direct, _ = prefetch(weights_at_root = True, token = "hf_test")
    whole, _ = prefetch(weights_at_root = False, token = "hf_test")
    captured = {}
    fake = sys.modules["unsloth_zoo.hf_xet_fallback"]
    inner = fake.snapshot_download_with_xet_fallback

    def spy(repo_id, **kwargs):
        captured.update(kwargs)
        return inner(repo_id, **kwargs)

    monkeypatch.setattr(fake, "snapshot_download_with_xet_fallback", spy)
    U.resolve_vllm_model_path("nvidia/Cosmos3-Edge", "hf_test", weights_at_root = True)
    for key in ("allow_patterns", "ignore_patterns"):
        assert captured.get(key) == direct.get(key), key
    assert (direct.get("allow_patterns"), direct.get("ignore_patterns")) != (
        whole.get("allow_patterns"),
        whole.get("ignore_patterns"),
    )

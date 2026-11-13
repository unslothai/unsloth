# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import time

import pytest

import utils.hf_xet_fallback as xet
from core.inference import diffusion as diffusion_mod
from core.inference import hub_reachability
from core.inference import video as video_mod
from core.inference.video import VideoBackend
from core.inference.video_families import detect_video_family

WAN_GGUF = "unsloth/Wan2.2-TI2V-5B-GGUF"
WAN_BASE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_FILE = "wan2.2-ti2v-5b-Q4_K_M.gguf"


def _cache(root, repo, *names):
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents = True, exist_ok = True)
    (folder / "refs" / "main").write_text("abc123")
    snap = folder / "snapshots" / "abc123"
    snap.mkdir(parents = True, exist_ok = True)
    for name in names:
        (snap / name).write_text("{}")


@pytest.fixture
def hub_down(monkeypatch, tmp_path):
    import huggingface_hub

    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(diffusion_mod, "hub_cache_dir", lambda: str(cache))
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(cache))
    monkeypatch.setitem(
        hub_reachability._UNREACHABLE, hub_reachability.endpoint(), time.monotonic()
    )

    def _model_info(self, repo_id, **_kwargs):
        raise ConnectionError(f"model_info({repo_id!r}): the Hub did not connect")

    def _download(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        path = tmp_path / "dl" / filename
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"")
        return str(path)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _model_info, raising = False)
    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)
    return cache


def _backend(monkeypatch, load_pipeline):
    backend = VideoBackend()
    backend._load_token = 1
    backend._loading = video_mod._VideoLoadingState(repo_id = WAN_GGUF, base_repo = WAN_BASE)
    fam = detect_video_family(WAN_BASE)
    monkeypatch.setattr(video_mod, "_detect_load_family", lambda *_a, **_k: fam)
    monkeypatch.setattr(backend, "load_pipeline", load_pipeline)
    return backend


def _cache_everything(cache):
    from core.inference.diffusion_families import mirror_repo
    for repo in {WAN_BASE, mirror_repo(WAN_BASE)}:
        _cache(cache, repo, "model_index.json")
    _cache(cache, WAN_GGUF, WAN_FILE)


def test_a_cached_base_loads_offline_while_the_hub_is_down(monkeypatch, hub_down):
    _cache_everything(hub_down)
    calls: list[dict] = []
    backend = _backend(monkeypatch, lambda **kw: calls.append(dict(kw)))

    backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert [c.get("local_files_only") for c in calls] == [True]


def test_a_cache_miss_offline_retries_online(monkeypatch, hub_down):
    _cache_everything(hub_down)
    calls: list[dict] = []

    def _load(**kw):
        calls.append(dict(kw))
        if kw.get("local_files_only"):
            raise OSError("transformer/config.json is not in the cache")

    backend = _backend(monkeypatch, _load)
    backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert [c.get("local_files_only") for c in calls] == [True, False]


def test_an_uncached_base_still_loads_online(monkeypatch, hub_down):
    _cache(hub_down, WAN_GGUF, WAN_FILE)
    calls: list[dict] = []
    backend = _backend(monkeypatch, lambda **kw: calls.append(dict(kw)))

    backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert [c.get("local_files_only") for c in calls] in ([False], [None])


def test_a_pipeline_cached_only_in_its_mirror_loads_offline(monkeypatch, hub_down):
    from core.inference.diffusion_families import mirror_repo

    mirror = mirror_repo(WAN_BASE)
    _cache(hub_down, mirror, "model_index.json")
    calls = []
    backend = _backend(monkeypatch, lambda **kw: calls.append(dict(kw)))
    backend._run_load(
        repo_id = WAN_BASE,
        model_kind = "pipeline",
        _load_token = 1,
        transformer_quant = "none",
        text_encoder_quant = "none",
    )

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert [c.get("local_files_only") for c in calls] == [True]
    assert calls[0]["_base_fetch_repo"] == mirror


def test_a_modular_pipeline_manifest_is_a_cached_base(hub_down):
    repo = "MiniMaxAI/MiniMax-H3"
    _cache(hub_down, repo, "modular_model_index.json")

    assert video_mod._video_cached_while_hub_unreachable(None, repo, None, None)


def test_a_local_base_keeps_its_path_when_a_matching_hub_cache_exists(
    monkeypatch, hub_down, tmp_path
):
    _cache(hub_down, "org/local-base", "model_index.json")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org/local-base").mkdir(parents = True)

    assert not video_mod._video_cached_while_hub_unreachable(None, "org/local-base", None, None)


@pytest.fixture
def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    reference = Tokenizer(
        models.Unigram(
            [("<pad>", 0), ("</s>", 0), ("<unk>", 0), ("▁", -1), ("hello", -1), ("world", -1)],
            unk_id = 2,
        )
    )
    reference.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Metaspace(replacement = "▁", prepend_scheme = "always"),
        ]
    )
    reference.post_processor = processors.TemplateProcessing(
        single = "$A </s>",
        special_tokens = [("</s>", 1)],
    )
    return reference


@pytest.mark.parametrize("complete", [False, True])
def test_automatic_offline_loading_preserves_the_tokenizer_vocabulary(
    monkeypatch, hub_down, complete, tiny_tokenizer
):
    import json
    from transformers import AutoTokenizer
    from core.inference.diffusion_families import mirror_repo

    reference = tiny_tokenizer
    _cache_everything(hub_down)
    for repo in {WAN_BASE, mirror_repo(WAN_BASE)}:
        snapshot = hub_down / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"
        (snapshot / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "WanPipeline",
                    "tokenizer": ["transformers", "T5TokenizerFast"],
                }
            )
        )
        folder = snapshot / "tokenizer"
        folder.mkdir()
        (folder / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "T5TokenizerFast",
                    "extra_ids": 0,
                    "pad_token": "<pad>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                }
            )
        )
        if complete:
            reference.save(str(folder / "tokenizer.json"))
    calls, encoded = [], []

    def _load(**kw):
        calls.append(kw.get("local_files_only"))
        repo = kw.get("_base_fetch_repo") or WAN_BASE
        folder = (
            hub_down / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123" / "tokenizer"
        )
        if not kw.get("local_files_only"):
            reference.save(str(folder / "tokenizer.json"))
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only = True)
        encoded.append(tokenizer.encode("hello world"))

    backend = _backend(monkeypatch, _load)
    backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert encoded == [reference.encode("hello world").ids]
    assert calls == ([True] if complete else [None])


@pytest.mark.parametrize("cls", ["T5TokenizerFast", "Qwen3VLProcessor"])
@pytest.mark.parametrize("subfolder", [None, "nested"])
@pytest.mark.parametrize("complete", [False, True])
def test_modular_tokenizer_sources_use_their_own_snapshot_folder(
    hub_down, tiny_tokenizer, subfolder, complete, cls
):
    import json

    base, external = "org/modular-base", "org/external-tokenizer"
    _cache(hub_down, base, "modular_model_index.json")
    _cache(hub_down, external, "placeholder")
    source = {"pretrained_model_name_or_path": external}
    if subfolder is not None:
        source["subfolder"] = subfolder
    snapshot = hub_down / "models--org--modular-base/snapshots/abc123"
    (snapshot / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "TestModularPipeline",
                "tokenizer": ["transformers", cls, source],
            }
        )
    )
    folder = hub_down / "models--org--external-tokenizer/snapshots/abc123" / (subfolder or "")
    folder.mkdir(exist_ok = True)
    (
        folder / ("processor_config.json" if "Processor" in cls else "tokenizer_config.json")
    ).write_text("{}")
    if complete:
        tiny_tokenizer.save(str(folder / "tokenizer.json"))

    assert (
        video_mod._video_cached_while_hub_unreachable(
            None, base, None, None, family = detect_video_family("MiniMaxAI/MiniMax-H3")
        )
        is complete
    )


def test_a_modular_load_validates_its_preferred_manifest(hub_down):
    import json

    repo = "org/both-manifests"
    _cache(hub_down, repo, "model_index.json", "modular_model_index.json")
    snapshot = hub_down / "models--org--both-manifests/snapshots/abc123"
    (snapshot / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "UnusedPipeline",
                "tokenizer": ["transformers", "T5TokenizerFast"],
            }
        )
    )

    assert video_mod._video_cached_while_hub_unreachable(
        None, repo, None, None, family = detect_video_family("MiniMaxAI/MiniMax-H3")
    )
    assert not video_mod._video_cached_while_hub_unreachable(None, repo, None, None)


def test_a_mirrored_single_file_pick_uses_its_resolved_cached_checkpoint(monkeypatch, hub_down):
    import json
    import struct
    import huggingface_hub

    from core.inference.diffusion_families import mirror_repo

    repo, base = "Lightricks/LTX-2.3", "Lightricks/LTX-2"
    filename = "ltx-2.3-22b-distilled.safetensors"
    checkpoint_repo = mirror_repo(repo)
    for source in {base, mirror_repo(base)}:
        _cache(hub_down, source, "model_index.json")
    _cache(hub_down, checkpoint_repo, filename)
    checkpoint = (
        hub_down
        / ("models--" + checkpoint_repo.replace("/", "--"))
        / "snapshots"
        / "abc123"
        / filename
    )
    header = json.dumps(
        {
            "model.diffusion_model.transformer_blocks.0.scale_shift_table": {
                "dtype": "F32",
                "shape": [9, 8],
                "data_offsets": [0, 288],
            },
        }
    ).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + bytes(288))
    downloaded = []

    def cached_download(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        path = huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            cache_dir = str(hub_down),
            local_files_only = True,
        )
        downloaded.append((repo_id, filename, path))
        return path

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", cached_download)
    backend = VideoBackend()
    backend._load_token = 1
    backend._loading = video_mod._VideoLoadingState(repo_id = repo, base_repo = base)
    family = detect_video_family(base)
    monkeypatch.setattr(video_mod, "_detect_load_family", lambda *_a, **_k: family)
    calls = []
    monkeypatch.setattr(backend, "load_pipeline", lambda **kw: calls.append(dict(kw)))
    backend._run_load(
        repo_id = repo,
        gguf_filename = filename,
        _load_token = 1,
        transformer_quant = "none",
        text_encoder_quant = "none",
    )

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert downloaded == [(checkpoint_repo, filename, str(checkpoint))]
    assert [call.get("local_files_only") for call in calls] == [True]


@pytest.mark.parametrize("deferred", [False, True])
def test_the_failed_offline_attempt_is_released_before_the_online_retry(
    monkeypatch, hub_down, deferred
):
    import contextlib
    import weakref

    from core.inference.diffusion_prequant_cache import PendingWrite, _WriteJob

    _cache_everything(hub_down)

    class _Weights:
        pass

    built: list = []
    alive_at_retry: list[bool] = []

    def _load(**kw):
        if kw.get("local_files_only"):
            transformer = _Weights()  # held by this frame, so by the failure's traceback
            built.append(weakref.ref(transformer))
            if deferred:
                job = _WriteJob(
                    "failed-load", hub_down / "pending", {"weight": transformer}, {}, {}, 1, 0, None
                )
                backend._resident_quant_write = (1, PendingWrite(job, contextlib.ExitStack()))
            raise OSError("vae/config.json is not in the cache")
        alive_at_retry.append(built[0]() is not None)

    backend = _backend(monkeypatch, _load)
    try:
        backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)
        assert backend._loading is None, getattr(backend._loading, "error", None)
        assert alive_at_retry == [False]
        assert getattr(backend, "_resident_quant_write", None) is None
    finally:
        video_mod._discard_resident_quant_write(backend, 1)


@pytest.mark.parametrize("cache_miss, retried", [(True, True), (False, False)])
def test_an_encoder_worker_cache_miss_retries_online(monkeypatch, hub_down, cache_miss, retried):
    from core.inference.video_ltx25 import TextEncoderWorkerError

    _cache_everything(hub_down)
    calls: list[dict] = []

    def _load(**kw):
        calls.append(dict(kw))
        if kw.get("local_files_only"):
            raise TextEncoderWorkerError("encoder failed to load", cache_miss = cache_miss)

    backend = _backend(monkeypatch, _load)
    backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)

    flags = [c.get("local_files_only") for c in calls]
    if retried:
        assert backend._loading is None, getattr(backend._loading, "error", None)
        assert flags == [True, False]
    else:
        assert flags == [True] and "encoder failed to load" in backend._loading.error


def test_the_encoder_worker_flags_a_cache_miss_across_the_process_boundary(monkeypatch):
    import queue

    from core.inference import video_ltx25_te_worker as worker
    from core.inference.video_ltx25 import Ltx25TextEncoderClient, TextEncoderWorkerError

    def _replies(exc):
        def _fail(_config):
            raise exc

        monkeypatch.setattr(worker, "_activate_sidecar", lambda _venv: None)
        monkeypatch.setattr(worker, "_load", _fail)
        resp: queue.Queue = queue.Queue()
        worker.run_text_encoder_worker(cmd_queue = queue.Queue(), resp_queue = resp, config = {})
        client = Ltx25TextEncoderClient.__new__(Ltx25TextEncoderClient)
        client._resp, client._proc, client._ready = resp, None, False
        with pytest.raises(TextEncoderWorkerError) as info:
            client.wait_ready()
        return info.value.cache_miss

    from huggingface_hub.errors import LocalEntryNotFoundError

    assert _replies(LocalEntryNotFoundError("tokenizer.json is not in the cache")) is True
    assert _replies(OSError("We couldn't connect and the file is not cached")) is True
    assert _replies(RuntimeError("CUDA out of memory")) is False


@pytest.mark.parametrize("missing, expected", [(True, True), (False, False)])
def test_hosted_encoder_cache_miss_survives_dense_memory_refusal(monkeypatch, missing, expected):
    import queue
    import sys
    import types
    import transformers

    from huggingface_hub.errors import LocalEntryNotFoundError
    from core.inference import video_ltx25_te_worker as worker

    def _hosted(*_args):
        if missing:
            raise LocalEntryNotFoundError("hosted int8 encoder is not cached")
        raise RuntimeError("invalid encoder checkpoint")

    monkeypatch.setattr(transformers, "__version__", "5.10.1")
    monkeypatch.setattr(
        transformers, "Gemma4UnifiedForConditionalGeneration", object, raising = False
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: object())
    monkeypatch.setattr(worker, "_activate_sidecar", lambda _venv: None)
    monkeypatch.setattr(worker, "_load_hosted", _hosted)
    monkeypatch.setattr(worker, "_available_host_bytes", lambda: 16 * 1000**3)
    replies = queue.Queue()
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16 = "bfloat16"))
        worker.run_text_encoder_worker(
            cmd_queue = queue.Queue(),
            resp_queue = replies,
            config = {
                "repo": "org/cached-base",
                "local_files_only": True,
                "te_quant": "int8",
                "te_prequant": {"repo": "org/encoder"},
                "dense_te_gb": 23.9,
            },
        )
    reply = replies.get_nowait()
    assert reply["type"] == "error"
    assert "24 GB of free memory (16 GB free)" in reply["error"]
    assert reply["cache_miss"] is expected

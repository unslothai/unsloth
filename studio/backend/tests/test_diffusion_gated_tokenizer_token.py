# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A gated repo (krea/Krea-2-Turbo) must send the token on every Hub read: transformers 5.x
``AutoTokenizer`` with no ``tokenizer/config.json`` falls back to a config load that drops ``token=``."""

from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

REPO = "krea/Krea-2-Turbo"
TOKEN = "hf_request_token_for_test"  # secret-scrub: allow


def _write_repo(root: pathlib.Path) -> pathlib.Path:
    """A Krea-2-shaped repo: model_index.json and a tokenizer/ folder with no config.json."""
    tokenizers = pytest.importorskip("tokenizers")
    repo = root / "fake_repo"
    (repo / "tokenizer").mkdir(parents = True)
    (repo / "model_index.json").write_text(
        json.dumps({"is_distilled": True, "patch_size": 2}), encoding = "utf-8"
    )
    for sub in ("tokenizer", "t5_tokenizer"):
        folder = repo / sub
        folder.mkdir(exist_ok = True)
        vocab = {"[UNK]": 0, "hello": 1, "world": 2}
        tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token = "[UNK]"))
        tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        tok.save(str(folder / "tokenizer.json"))
        (folder / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}),
            encoding = "utf-8",
        )
    return repo


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Records the token of every Hub read; a file the fake repo lacks is a Hub 404."""
    pytest.importorskip("transformers")
    import httpx
    import huggingface_hub
    import huggingface_hub.constants as constants
    import transformers.utils.hub as tf_hub
    from huggingface_hub.errors import EntryNotFoundError

    import utils.hf_cache_settings as cache_settings

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    for key in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising = False)
    cache = tmp_path / "hub_cache"
    cache.mkdir()
    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: str(cache))
    repo = _write_repo(tmp_path)
    calls: list[tuple[str, str, object]] = []

    def hf_hub_download(
        repo_id,
        filename,
        *args,
        subfolder = None,
        token = None,
        **kwargs,
    ):
        rel = "/".join(p for p in (subfolder, filename) if p)
        calls.append(("hf_hub_download", rel, token))
        path = repo / rel
        if not path.is_file():
            raise EntryNotFoundError(f"404 {repo_id}/{rel}")
        return str(path)

    def snapshot_download(
        repo_id,
        *args,
        token = None,
        **kwargs,
    ):
        calls.append(("snapshot_download", ",".join(kwargs.get("allow_patterns") or ()), token))
        return str(repo)

    def no_network(self, request, *args, **kwargs):
        calls.append(("http", str(request.url), request.headers.get("authorization")))
        return httpx.Response(404, request = request)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(tf_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(httpx.Client, "send", no_network)
    return calls


def _assert_every_call_had_the_token(calls):
    assert calls, "no Hub read was recorded"
    anonymous = [c for c in calls if c[0] == "http" or c[2] != TOKEN]
    assert not anonymous, f"Hub reads without the request token: {anonymous}"


def test_krea2_image_load_sends_the_token_on_every_hub_read(fake_hub, monkeypatch):
    from core.inference import diffusion_krea2

    components: dict = {}

    class _Component:
        def __init__(self, tag):
            self.tag = tag

        def from_pretrained(self, repo_id, **kwargs):
            fake_hub.append((f"diffusers.{self.tag}", repo_id, kwargs.get("token")))
            return SimpleNamespace(tag = self.tag)

    def _pipeline(**kwargs):
        components.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            Krea2Pipeline = _pipeline,
            FlowMatchEulerDiscreteScheduler = _Component("scheduler"),
            AutoencoderKLQwenImage = _Component("vae"),
            Krea2Transformer2DModel = _Component("transformer"),
        ),
    )
    try:
        diffusion_krea2.load_krea2_pipeline(
            REPO,
            "bf16",
            hf_token = TOKEN,
            text_encoder = SimpleNamespace(tag = "te"),
            with_transformer = False,
        )
    finally:
        # Checked even when the load raised: the anonymous read is the root cause, not what follows it.
        _assert_every_call_had_the_token(fake_hub)
    assert components["tokenizer"]("hello world")["input_ids"] == [1, 2]


def test_krea2_tokenizer_alone_sends_the_token(fake_hub):
    from core.inference import diffusion_krea2
    try:
        tokenizer = diffusion_krea2.load_krea2_tokenizer(REPO, hf_token = TOKEN)
    finally:
        _assert_every_call_had_the_token(fake_hub)
    assert tokenizer("hello world")["input_ids"] == [1, 2]


def test_krea2_text_encoder_sends_the_token(fake_hub, monkeypatch):
    from core.inference import diffusion_krea2

    class _Config:
        @staticmethod
        def from_pretrained(repo_id, **kwargs):
            fake_hub.append(("AutoConfig", repo_id, kwargs.get("token")))
            return SimpleNamespace(text_config = SimpleNamespace(rope_scaling = {"x": 1}))

    class _Model:
        @staticmethod
        def from_pretrained(repo_id, **kwargs):
            fake_hub.append(("Qwen3VLModel", repo_id, kwargs.get("token")))
            return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoConfig = _Config, Qwen3VLModel = _Model)
    )
    diffusion_krea2.load_krea2_text_encoder(REPO, "bf16", hf_token = TOKEN)
    _assert_every_call_had_the_token(fake_hub)


def test_anima_tokenizers_send_the_token(fake_hub):
    """Anima's per-subfolder tokenizer loader shares the helper."""
    from transformers import AutoTokenizer

    from core.inference import diffusion_anima

    common = {"local_files_only": False, "token": TOKEN}
    for sub in ("tokenizer", "t5_tokenizer"):
        try:
            tok = diffusion_anima._load_tokenizer(AutoTokenizer, REPO, sub, dict(common))
        finally:
            _assert_every_call_had_the_token(fake_hub)
        assert tok("hello world")["input_ids"] == [1, 2]


def test_an_offline_load_still_skips_the_hub(fake_hub, monkeypatch, tmp_path):
    """local_files_only never reaches snapshot_download: offline keeps the cached-folder lookup."""
    from core.inference import diffusion_offline_source

    got = diffusion_offline_source.tokenizer_snapshot_source(
        REPO, "tokenizer", local_files_only = True, token = TOKEN, cache_dir = str(tmp_path)
    )
    assert got == REPO
    assert not [c for c in fake_hub if c[0] == "snapshot_download"]


def test_a_local_dir_is_returned_unchanged(fake_hub, tmp_path):
    from core.inference import diffusion_offline_source

    got = diffusion_offline_source.tokenizer_snapshot_source(
        str(tmp_path), "tokenizer", local_files_only = False, token = TOKEN
    )
    assert got == str(tmp_path)
    assert fake_hub == []

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A fully downloaded model must load its tokenizer / processor offline (hand-built hub cache)."""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from core.inference import diffusion_offline_source as offline_source

REPO = "acme/tiny-diffusion"
SHA = "0123456789abcdef0123456789abcdef01234567"


def _fake_cache(
    root: pathlib.Path,
    *,
    subfolders = ("tokenizer",),
    no_exist_marker = True,
) -> pathlib.Path:
    """A completed download; the ``.no_exist`` marker (an online 404) does not rescue a repo-id load."""
    repo_dir = root / ("models--" + REPO.replace("/", "--"))
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text(SHA, encoding = "utf-8")
    snapshot = repo_dir / "snapshots" / SHA
    for sub in subfolders:
        (snapshot / sub).mkdir(parents = True)
        (snapshot / sub / "tokenizer_config.json").write_text("{}", encoding = "utf-8")
        if no_exist_marker:
            marker = repo_dir / ".no_exist" / SHA / sub / "config.json"
            marker.parent.mkdir(parents = True, exist_ok = True)
            marker.touch()
    return snapshot


@pytest.fixture
def online(monkeypatch):
    import huggingface_hub.constants as constants
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)


def test_an_offline_load_opens_the_cached_snapshot_folder(tmp_path, online):
    snapshot = _fake_cache(tmp_path)
    got = offline_source.offline_snapshot_source(
        REPO, "tokenizer", local_files_only = True, cache_dir = str(tmp_path)
    )
    assert pathlib.Path(got) == snapshot


def test_hf_hub_offline_counts_as_offline(tmp_path, monkeypatch):
    import huggingface_hub.constants as constants

    snapshot = _fake_cache(tmp_path)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    got = offline_source.offline_snapshot_source(
        REPO, "tokenizer", local_files_only = False, cache_dir = str(tmp_path)
    )
    assert pathlib.Path(got) == snapshot


def test_an_online_load_keeps_the_repo_id(tmp_path, online):
    _fake_cache(tmp_path)
    assert (
        offline_source.offline_snapshot_source(
            REPO, "tokenizer", local_files_only = False, cache_dir = str(tmp_path)
        )
        == REPO
    )


def test_nothing_cached_keeps_the_repo_id(tmp_path, online):
    _fake_cache(tmp_path, subfolders = ("vae",))
    assert (
        offline_source.offline_snapshot_source(
            REPO, "tokenizer", local_files_only = True, cache_dir = str(tmp_path)
        )
        == REPO
    )
    assert (
        offline_source.offline_snapshot_source(
            "acme/never-downloaded", "tokenizer", local_files_only = True, cache_dir = str(tmp_path)
        )
        == "acme/never-downloaded"
    )


def test_a_local_folder_is_left_alone(tmp_path, online):
    local = tmp_path / "local_model"
    (local / "tokenizer").mkdir(parents = True)
    assert offline_source.offline_snapshot_source(
        str(local), "tokenizer", local_files_only = True, cache_dir = str(tmp_path)
    ) == str(local)


def _spec(
    type_hint,
    subfolder,
    repo = REPO,
):
    return SimpleNamespace(
        type_hint = type_hint,
        subfolder = subfolder,
        pretrained_model_name_or_path = repo,
        revision = None,
        default_creation_method = "from_pretrained",
    )


def test_modular_tokenizer_and_processor_specs_are_pointed_at_the_snapshot(tmp_path, online):
    transformers = pytest.importorskip("transformers")

    class _Processor(transformers.ProcessorMixin):
        pass

    class _Tokenizer(transformers.PreTrainedTokenizerBase):
        pass

    class _Encoder(transformers.PreTrainedModel):
        pass

    snapshot = _fake_cache(tmp_path, subfolders = ("processor", "tokenizer", "text_encoder"))
    pipe = SimpleNamespace(
        _component_specs = {
            "processor": _spec(_Processor, "processor"),
            "tokenizer": _spec(_Tokenizer, "tokenizer"),
            # Models carry their own config.json and load offline by repo id already.
            "text_encoder": _spec(_Encoder, "text_encoder"),
            "guider": SimpleNamespace(type_hint = _Tokenizer, default_creation_method = "from_config"),
        }
    )
    offline = offline_source.offline_component_sources(
        pipe, local_files_only = True, cache_dir = str(tmp_path)
    )
    assert {k: pathlib.Path(v) for k, v in offline.items()} == {
        "processor": snapshot,
        "tokenizer": snapshot,
    }
    assert (
        offline_source.offline_component_sources(
            pipe, local_files_only = False, cache_dir = str(tmp_path)
        )
        == {}
    )


def test_the_h3_modular_build_hands_the_offline_sources_to_load_components():
    """The H3 build needs a real Modular Diffusers pipeline to reach, so read the call site: the
    per-component source override has to be spelled on load_components."""
    backend_root = pathlib.Path(offline_source.__file__).resolve().parents[2]
    tree = ast.parse((backend_root / "core/inference/video.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_h3_modular_pipeline":
            calls = [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and getattr(call.func, "attr", None) == "load_components"
            ]
            assert calls, "no load_components call"
            sources = ast.unparse(node)
            assert "offline_component_sources(" in sources
            for call in calls:
                spelled = ast.unparse(call)
                assert "pretrained_model_name_or_path" in spelled, spelled
            return
    raise AssertionError("_load_h3_modular_pipeline not found")


def _live_cache(monkeypatch, root: pathlib.Path) -> None:
    import utils.hf_cache_settings as cache_settings
    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: str(root))


def test_the_krea_tokenizer_and_its_fallback_open_the_snapshot_offline(
    monkeypatch, tmp_path, online
):
    from core.inference import diffusion_krea2

    snapshot = _fake_cache(tmp_path)
    _live_cache(monkeypatch, tmp_path)
    sources: list = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            sources.append(source)
            if "extra_special_tokens" not in kwargs:
                raise OSError("4.x compat failure")
            return SimpleNamespace(source = source)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer = _AutoTokenizer))
    diffusion_krea2.load_krea2_tokenizer(REPO, local_files_only = True)
    assert [pathlib.Path(s) for s in sources] == [snapshot, snapshot]


def test_an_online_krea_tokenizer_load_still_uses_the_repo_id(monkeypatch, tmp_path, online):
    from core.inference import diffusion_krea2

    _fake_cache(tmp_path)
    _live_cache(monkeypatch, tmp_path)
    sources: list = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            sources.append(source)
            return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer = _AutoTokenizer))
    diffusion_krea2.load_krea2_tokenizer(REPO, local_files_only = False)
    assert sources == [REPO]


def _write_real_tokenizer(folder: pathlib.Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token = "[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}),
        encoding = "utf-8",
    )


def test_a_downloaded_krea_tokenizer_really_loads_offline(monkeypatch, tmp_path, online):
    """Real transformers, real files, no network: on transformers 5.x the repo-id load raises even
    with the .no_exist marker present, which is the Krea-2-Turbo failure the benchmark hit."""
    pytest.importorskip("transformers")
    from core.inference import diffusion_krea2

    snapshot = _fake_cache(tmp_path)
    _write_real_tokenizer(snapshot / "tokenizer")
    _live_cache(monkeypatch, tmp_path)
    tokenizer = diffusion_krea2.load_krea2_tokenizer(REPO, local_files_only = True)
    assert tokenizer("hello world")["input_ids"] == [1, 2]

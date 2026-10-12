# SPDX-License-Identifier: AGPL-3.0-only
"""Probe the real BiCodec handoff without importing the model or audio stack."""

import ast
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def _repo_root():
    configured = os.environ.get("UNSLOTH_REPO_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "studio/backend/core/training/trainer.py").is_file():
            return candidate
    pytest.skip("Could not locate studio/backend/training source.")


def _method(namespace):
    source = (_repo_root() / "studio/backend/core/training/trainer.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnslothTrainer"
    )
    method = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_preprocess_bicodec_dataset"
    )
    # The constructor sentinel stops before any encoding or nested tensor functions.
    exec(
        compile(ast.Module(body = [method], type_ignores = []), "<bicodec-handoff>", "exec"), namespace
    )
    return namespace[method.name]


class _CodecReached(Exception):
    pass


class _UnresolvedCodecRoot(AssertionError):
    pass


def _probe(repo_dir, *, missing_column = None):
    calls = []

    def codec(root, device):
        calls.append((root, device))
        raise _CodecReached

    def import_module(name, source):
        assert source == "<mock-source>"
        if name == "sparktts.models.audio_tokenizer":
            return SimpleNamespace(BiCodecTokenizer = codec)
        if name == "sparktts.utils.audio":
            return SimpleNamespace(audio_volume_normalize = lambda value: value)
        raise AssertionError(f"Unexpected Spark-TTS import: {name}")

    namespace = {
        "ensure_spark_tts_source": lambda root: "<mock-source>",
        "import_sparktts_module": import_module,
        "logger": SimpleNamespace(info = lambda *args: None),
    }
    method = _method(namespace)
    owner = SimpleNamespace(
        _spark_tts_repo_dir = repo_dir,
        _update_progress = lambda **kwargs: None,
        _resolve_audio_columns = lambda *args: {
            "audio_col": None if missing_column == "audio" else "audio",
            "text_col": None if missing_column == "text" else "text",
            "speaker_col": None,
        },
    )
    dataset = SimpleNamespace(column_names = ["audio", "text"])
    dataset.cast_column = lambda *args: dataset
    audio = ModuleType("torchaudio")
    audio.transforms = ModuleType("torchaudio.transforms")
    modules = {
        "torch": SimpleNamespace(cuda = SimpleNamespace(is_available = lambda: False)),
        "numpy": ModuleType("numpy"),
        "torchaudio": audio,
        "torchaudio.transforms": audio.transforms,
        "datasets": SimpleNamespace(Audio = lambda: object()),
    }
    with patch.dict("sys.modules", modules):
        if missing_column:
            with pytest.raises(ValueError, match = "needs 'audio' and 'text'"):
                method(owner, dataset)
        else:
            with pytest.raises(_CodecReached):
                method(owner, dataset)
    return calls


@pytest.mark.parametrize("repo_dir", [r"C:\models\Spark-TTS-0.5B", "/srv/Spark-TTS-0.5B"])
def test_resolved_root_reaches_codec_unchanged(repo_dir):
    assert _probe(repo_dir) == [(repo_dir, "cpu")]


@pytest.mark.parametrize("column", ["audio", "text"])
def test_missing_required_column_stops_before_codec(column):
    assert _probe(None, missing_column = column) == []


@pytest.mark.xfail(
    strict = True,
    raises = _UnresolvedCodecRoot,
    reason = "Issue #13458: preprocessing can run before load_model sets the codec repo root",
)
def test_dataset_before_model_does_not_pass_an_unresolved_root():
    calls = _probe(None)
    assert len(calls) == 1
    assert calls[0][1] == "cpu"
    if calls[0][0] is None:
        raise _UnresolvedCodecRoot("BiCodecTokenizer received an unresolved repository root")

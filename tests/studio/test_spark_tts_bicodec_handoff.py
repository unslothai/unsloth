# SPDX-License-Identifier: AGPL-3.0-only
"""Spark-TTS: the BiCodec codec must get the repo root when the dataset is formatted before load_model (#13458).

The worker runs detect -> dataset -> model, so `_preprocess_bicodec_dataset` runs before `load_model` has set
`_spark_tts_repo_dir`; the codec then read `<cwd>/None/config.yaml`. Source-level: the real methods are exec'd
with the model, audio and Hub layers stubbed.
"""

import ast
import os
import typing
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

_TRAINER = Path(__file__).resolve().parents[2] / "studio/backend/core/training/trainer.py"
_WORKER = _TRAINER.with_name("worker.py")
_NAMES = (
    "_spark_tts_hub_repo",
    "_spark_tts_repo_dir",
    "pre_detect_and_load_tokenizer",
    "resolve_spark_tts_repo",
    "_preprocess_bicodec_dataset",
)


class _CodecReached(Exception):
    pass


def _run(
    model_name,
    model_load_name = None,
    local_files_only = False,
    revision = None,
    repo_dir = None,
):
    """pre_detect, then dataset preprocessing up to the BiCodec constructor, in the worker's order."""
    tree = ast.parse(_TRAINER.read_text(encoding = "utf-8"))
    trainer = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnslothTrainer"
    )
    nodes = [
        n
        for n in (*tree.body, *trainer.body)
        if isinstance(n, ast.FunctionDef) and n.name in _NAMES
    ]
    downloads, codec_roots = [], []

    def snapshot_download(repo, revision = None):
        downloads.append((repo, revision))
        return f"/hf/hub/{repo}/snapshots/abc"

    def codec(root, device):
        codec_roots.append(root)
        raise _CodecReached

    def import_module(name, source):
        if name == "sparktts.models.audio_tokenizer":
            return SimpleNamespace(BiCodecTokenizer = codec)
        return SimpleNamespace(audio_volume_normalize = lambda value: value)

    ns = {
        "os": os,
        "Optional": typing.Optional,
        "Callable": typing.Callable,
        "logger": SimpleNamespace(info = lambda *args: None),
        "_metadata_lookup_name": lambda name, lookup, *args: lookup,
        "detect_audio_type_checked": lambda *args, **kwargs: ("bicodec", True),
        "is_vision_model": lambda *args, **kwargs: False,
        "_load_adapter_aware_tokenizer": lambda *args, **kwargs: object(),
        "_spark_tts_tokenizer_kwargs": lambda *args: {},
        "_tokenizer_is_empty": lambda tokenizer: False,
        "ensure_spark_tts_source": lambda root: "<source>",
        "import_sparktts_module": import_module,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(_TRAINER), "exec"), ns)
    owner = SimpleNamespace(
        _spark_tts_repo_dir = repo_dir,
        _spark_tts_repo_source = None,
        _update_progress = lambda **kwargs: None,
        # pre_detect also classifies text-diffusion LMs; a Spark-TTS repo is never one.
        _detect_diffusion_lm = lambda *args: None,
        diffusion_lm_profile = None,
        _resolve_audio_columns = lambda *args: {
            "audio_col": "audio",
            "text_col": "text",
            "speaker_col": None,
        },
    )
    owner.resolve_spark_tts_repo = MethodType(ns["resolve_spark_tts_repo"], owner)
    dataset = SimpleNamespace(column_names = ["audio", "text"])
    dataset.cast_column = lambda *args: dataset
    audio = ModuleType("torchaudio")
    audio.transforms = ModuleType("torchaudio.transforms")
    modules = {
        "huggingface_hub": SimpleNamespace(snapshot_download = snapshot_download),
        "torch": SimpleNamespace(cuda = SimpleNamespace(is_available = lambda: False)),
        "numpy": ModuleType("numpy"),
        "torchaudio": audio,
        "torchaudio.transforms": audio.transforms,
        "datasets": SimpleNamespace(Audio = lambda: object()),
    }
    with patch.dict("sys.modules", modules):
        ns["pre_detect_and_load_tokenizer"](
            owner,
            model_name = model_name,
            model_load_name = model_load_name,
            local_files_only = local_files_only,
            model_revision = revision,
        )
        # pre_detect stays config/tokenizer only: the codec weights are fetched when the dataset needs them.
        assert downloads == []
        with pytest.raises(_CodecReached):
            ns["_preprocess_bicodec_dataset"](owner, dataset)
    return downloads, codec_roots


@pytest.mark.parametrize(
    "model_name, revision",
    [
        ("unsloth/Spark-TTS-0.5B", None),
        ("Spark-TTS-0.5B/LLM", None),
        ("unsloth/Spark-TTS-0.5B", "c80b03b6"),
    ],
)
def test_hub_model_codec_gets_the_downloaded_root_before_load_model(model_name, revision):
    downloads, codec_roots = _run(model_name, revision = revision)
    assert downloads == [("unsloth/Spark-TTS-0.5B", revision)]
    assert codec_roots == [os.path.abspath("/hf/hub/unsloth/Spark-TTS-0.5B/snapshots/abc")]


def test_cached_snapshot_codec_gets_the_snapshot_root_offline(tmp_path):
    snapshot = tmp_path / "models--unsloth--Spark-TTS-0.5B" / "snapshots" / "c80b03b6"
    downloads, codec_roots = _run("unsloth/Spark-TTS-0.5B", str(snapshot), local_files_only = True)
    assert downloads == []
    assert codec_roots == [str(snapshot)]


@pytest.mark.parametrize("pick_llm", [False, True])
def test_a_local_spark_folder_is_used_in_place(tmp_path, pick_llm):
    root = tmp_path / "Spark-TTS-0.5B"
    (root / "LLM").mkdir(parents = True)
    (root / "BiCodec").mkdir()
    picked = str(root / "LLM") if pick_llm else str(root)
    downloads, codec_roots = _run(picked, picked)
    assert downloads == []
    assert codec_roots == [str(root)]


@pytest.mark.parametrize("repo_dir", [r"C:\models\Spark-TTS-0.5B", "/srv/Spark-TTS-0.5B"])
def test_a_root_load_model_already_set_is_kept(repo_dir):
    downloads, codec_roots = _run("unsloth/Spark-TTS-0.5B", repo_dir = repo_dir)
    assert downloads == []
    assert codec_roots == [repo_dir]


def _fetch(resolve, model_name = "unsloth/Spark-TTS-0.5B"):
    tree = ast.parse(_WORKER.read_text(encoding = "utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_fetch_spark_tts_repo"
    )
    events, watched = [], []

    class _Stop:
        def set(self):
            events.append("watchdog_stopped")

    def start_watchdog(**kwargs):
        watched.append(kwargs["repo_ids"])
        return _Stop()

    ns = {
        "os": os,
        "time": SimpleNamespace(time = lambda: 0.0),
        "Any": typing.Any,
        "_send_status": lambda queue, message: None,
    }
    exec(compile(ast.Module(body = [node], type_ignores = []), str(_WORKER), "exec"), ns)
    queue = SimpleNamespace(put = lambda event: events.append(event["type"]))
    trainer = SimpleNamespace(
        resolve_spark_tts_repo = lambda on_model_resolved: resolve(on_model_resolved),
        spark_tts_hub_repo = lambda: "unsloth/Spark-TTS-0.5B",
    )
    hf_xet = SimpleNamespace(start_watchdog = start_watchdog)
    with patch.dict("sys.modules", {"utils": SimpleNamespace(), "utils.hf_xet_fallback": hf_xet}):
        try:
            ns["_fetch_spark_tts_repo"](trainer, queue, model_name)
        except OSError:
            events.append("raised")
    return events, watched


def test_the_early_spark_download_runs_under_the_model_load_watchdog():
    events, watched = _fetch(lambda report: report("unsloth/Spark-TTS-0.5B"))
    assert watched == [["unsloth/Spark-TTS-0.5B"]]
    assert events == [
        "model_load_started",
        "model_load_resolved",
        "watchdog_stopped",
        "model_load_completed",
    ]


def test_a_failed_spark_download_still_closes_the_model_load_lifecycle():
    def fail(report):
        raise OSError("connection reset")

    events, _ = _fetch(fail)
    assert events == ["model_load_started", "watchdog_stopped", "model_load_completed", "raised"]


def test_the_watchdog_watches_the_repo_behind_the_llm_alias():
    _, watched = _fetch(lambda report: None, model_name = "Spark-TTS-0.5B/LLM")
    assert watched == [["Spark-TTS-0.5B/LLM", "unsloth/Spark-TTS-0.5B"]]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Spark-TTS LoRA checkpoints must stay exportable (#13411).

Spark-TTS trains from ``<snapshot>/LLM``. Rewriting that pinned path to the bare repo id dropped the
subfolder, so adapter_config.json named a repo root with no config.json and every export failed.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import huggingface_hub  # noqa: E402

from utils.models.spark_tts_adapter import spark_tts_adapter_load_target  # noqa: E402
from utils.models.model_identity import restore_hf_cache_repo_identity  # noqa: E402


_SNAPSHOT = (
    "/home/user/.cache/huggingface/hub/models--unsloth--Spark-TTS-0.5B/snapshots/0123456789abcdef"
)


def test_pinned_spark_tts_llm_subfolder_keeps_its_path():
    llm = f"{_SNAPSHOT}/LLM"
    config = SimpleNamespace(_name_or_path = llm)
    adapter_config = SimpleNamespace(base_model_name_or_path = llm)
    model = SimpleNamespace(
        config = config, name_or_path = llm, peft_config = {"default": adapter_config}
    )

    restored = restore_hf_cache_repo_identity(
        model, _SNAPSHOT, expected_repo_id = "unsloth/Spark-TTS-0.5B"
    )

    assert restored is None
    assert config._name_or_path == llm
    assert model.name_or_path == llm
    assert adapter_config.base_model_name_or_path == llm


def test_snapshot_root_with_trailing_separator_is_still_restored():
    config = SimpleNamespace(_name_or_path = _SNAPSHOT + "/")
    restored = restore_hf_cache_repo_identity(
        SimpleNamespace(config = config), _SNAPSHOT, expected_repo_id = "unsloth/Spark-TTS-0.5B"
    )
    assert restored == "unsloth/Spark-TTS-0.5B"
    assert config._name_or_path == "unsloth/Spark-TTS-0.5B"


def _spark_repo(root: Path) -> Path:
    (root / "LLM").mkdir(parents = True)
    (root / "LLM" / "config.json").write_text('{"model_type": "qwen2"}', encoding = "utf-8")
    (root / "config.yaml").write_text("sample_rate: 16000\n", encoding = "utf-8")
    return root


def _checkpoint(path: Path, base: str) -> Path:
    path.mkdir(parents = True)
    adapter = {"base_model_name_or_path": base, "r": 8, "peft_type": "LORA"}
    (path / "adapter_config.json").write_text(json.dumps(adapter), encoding = "utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "tokenizer_config.json").write_text("{}", encoding = "utf-8")
    (path / "optimizer.pt").write_bytes(b"optimizer state")
    return path


def test_legacy_adapter_naming_repo_root_loads_through_llm_subfolder(tmp_path):
    repo = _spark_repo(tmp_path / "Spark-TTS-0.5B")
    ckpt = _checkpoint(tmp_path / "checkpoint-30", str(repo))
    original = (ckpt / "adapter_config.json").read_bytes()

    with spark_tts_adapter_load_target(str(ckpt)) as target:
        staged = Path(target)
        assert staged != ckpt
        config = json.loads((staged / "adapter_config.json").read_text(encoding = "utf-8"))
        assert config["base_model_name_or_path"] == str(repo / "LLM")
        assert config["r"] == 8
        assert (staged / "adapter_model.safetensors").read_bytes() == b"weights"
        assert (staged / "tokenizer_config.json").is_file()
        assert not (staged / "optimizer.pt").exists()

    assert not staged.exists()
    assert (ckpt / "adapter_config.json").read_bytes() == original


def test_legacy_adapter_naming_hub_repo_id_uses_cached_snapshot(tmp_path, monkeypatch):
    snapshot = _spark_repo(tmp_path / "snapshot")
    ckpt = _checkpoint(tmp_path / "checkpoint-30", "unsloth/Spark-TTS-0.5B")
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    with spark_tts_adapter_load_target(str(ckpt), hf_token = False, local_files_only = True) as target:
        config = json.loads((Path(target) / "adapter_config.json").read_text(encoding = "utf-8"))
        assert config["base_model_name_or_path"] == str(snapshot / "LLM")

    assert calls[0][0] == "unsloth/Spark-TTS-0.5B"
    assert calls[0][1]["token"] is False
    assert calls[0][1]["local_files_only"] is True


def test_loadable_bases_and_non_adapters_are_unchanged(tmp_path, monkeypatch):
    repo = _spark_repo(tmp_path / "Spark-TTS-0.5B")
    llm_ckpt = _checkpoint(tmp_path / "llm_base", str(repo / "LLM"))
    plain = tmp_path / "plain_base"
    plain.mkdir()
    (plain / "config.json").write_text('{"model_type": "qwen2"}', encoding = "utf-8")
    (plain / "LLM").mkdir()
    (plain / "LLM" / "config.json").write_text("{}", encoding = "utf-8")
    plain_ckpt = _checkpoint(tmp_path / "plain_ckpt", str(plain))
    merged = tmp_path / "merged"
    merged.mkdir()

    def offline_failure(repo_id, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline_failure)
    unresolvable = _checkpoint(tmp_path / "unresolvable", "unsloth/Spark-TTS-0.5B")

    for path in (llm_ckpt, plain_ckpt, merged, unresolvable):
        with spark_tts_adapter_load_target(str(path)) as target:
            assert target == str(path)

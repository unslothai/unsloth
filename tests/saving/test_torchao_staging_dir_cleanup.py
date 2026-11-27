# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import transformers

import unsloth.save as save_mod


class _TorchAoConfig:
    def __init__(self, **kwargs):
        pass


@pytest.mark.parametrize("failure", ["merge", "reload", "save", None])
@pytest.mark.parametrize("push", [False, True])
def test_torchao_staging_is_cleaned_without_touching_the_user_directory(
    monkeypatch, tmp_path, failure, push
):
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "model"
    destination.mkdir()
    keep = destination / "model.safetensors"
    keep.write_bytes(b"user checkpoint")
    staging = []
    outputs = []
    restores = []

    def write_merge(path):
        path = Path(path)
        staging.append(path)
        path.mkdir(parents = True, exist_ok = True)
        (path / "model.safetensors").write_bytes(b"intermediate checkpoint")
        if failure == "merge":
            raise RuntimeError("failed merge")

    def write_quantized(path, **kwargs):
        outputs.append(path)
        if failure == "save":
            raise RuntimeError("failed save")

    class Adapter:
        pass

    def merge_adapter(**kwargs):
        assert not kwargs["push_to_hub"]
        write_merge(kwargs["save_directory"])

    def reload(path, **kwargs):
        assert Path(path) == staging[0]
        if failure == "reload":
            raise RuntimeError("failed reload")
        return SimpleNamespace(save_pretrained = write_quantized, push_to_hub = write_quantized)

    torchao = ModuleType("torchao")
    torchao.quantize_ = lambda *args: None
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setattr(transformers, "TorchAoConfig", _TorchAoConfig)
    monkeypatch.setattr(save_mod, "PeftModel", Adapter)
    monkeypatch.setattr(save_mod, "unsloth_generic_save", merge_adapter)
    monkeypatch.setattr(
        transformers, "AutoModelForCausalLM", SimpleNamespace(from_pretrained = reload)
    )

    def save_tokenizer(*args, **kwargs):
        pass

    tokenizer = SimpleNamespace(save_pretrained = save_tokenizer, push_to_hub = save_tokenizer)
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained = lambda path: tokenizer),
    )
    monkeypatch.setattr(save_mod, "sync_load_when_quantizing", lambda *args: nullcontext())
    monkeypatch.setattr(save_mod, "importlib_version", lambda *args: "0.0.0")
    monkeypatch.setattr(save_mod, "_offload_model_for_quantize_subprocess", lambda model: "state")
    monkeypatch.setattr(
        save_mod,
        "_restore_model_after_quantize_subprocess",
        lambda model, state: restores.append(state),
    )

    result = (
        nullcontext() if failure is None else pytest.raises(RuntimeError, match = f"failed {failure}")
    )
    model = Adapter() if push else SimpleNamespace(save_pretrained = write_merge)
    target = "owner/model" if push else str(destination)
    with result:
        save_mod._unsloth_save_torchao_with_given_config(
            model,
            target,
            tokenizer,
            torchao_config = object(),
            push_to_hub = push,
            token = "test" if push else None,
        )

    assert keep.read_bytes() == b"user checkpoint"
    assert len(staging) == 1
    assert staging[0].parent.resolve() == tmp_path.resolve()
    assert not staging[0].exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model"]
    assert outputs == ([target + "-torchao"] if failure in ("save", None) else [])
    assert restores == ([] if failure == "merge" else ["state"])

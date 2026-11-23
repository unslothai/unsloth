# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Rank handling for the PEFT merged and GGUF saves (unsloth/save.py), with a faked process group:
is_main_process gating, rank-0-only plain DDP Hub uploads, and no veto inside an FSDP2 gather."""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import unsloth_zoo.saving_utils as zoo_saving_utils  # noqa: E402
from unsloth import save as unsloth_save  # noqa: E402
from unsloth.models import fsdp2_save  # noqa: E402


def _tiny_peft_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model

    config = LlamaConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 32,
    )
    model = LlamaForCausalLM(config)
    return get_peft_model(
        model,
        LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"]),
    )


def _counted_merge(monkeypatch):
    """merge_and_overwrite_lora -> call recorder; the Hub prewarm never runs."""
    calls = []

    def fake_merge(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(zoo_saving_utils, "merge_and_overwrite_lora", fake_merge)
    monkeypatch.setattr(unsloth_save, "_prewarm_base_model_hub_cache", lambda *a, **k: None)
    return calls


def _fake_ddp(
    monkeypatch,
    rank,
    world = 2,
):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: world)


def _fake_gguf_push(monkeypatch, tmp_path):
    """unsloth_save_pretrained_gguf and HfApi -> call recorders."""
    calls = {"save": 0, "hub": []}
    gguf_file = tmp_path / "model.Q8_0.gguf"
    gguf_file.write_bytes(b"gguf")

    def fake_save_gguf(self, save_directory, **kwargs):
        calls["save"] += 1
        return {
            "save_directory": save_directory,
            "gguf_directory": save_directory,
            "gguf_files": [str(gguf_file)],
            "modelfile_location": None,
            "want_full_precision": False,
            "is_vlm": False,
            "fix_bos_token": False,
        }

    class FakeHub:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            def record(*args, **kwargs):
                calls["hub"].append(name)
                return {"name": "user"} if name == "whoami" else None

            return record

    monkeypatch.setattr(unsloth_save, "unsloth_save_pretrained_gguf", fake_save_gguf)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHub)
    return calls


def _generic_save(model, directory, **kwargs):
    return unsloth_save.unsloth_generic_save(
        model = model,
        tokenizer = None,
        save_directory = directory,
        save_method = "merged_16bit",
        **kwargs,
    )


def test_non_main_process_skips_the_merge(monkeypatch, tmp_path):
    calls = _counted_merge(monkeypatch)
    model = _tiny_peft_model()
    _generic_save(model, str(tmp_path / "main"), is_main_process = True)
    assert len(calls) == 1
    _generic_save(model, str(tmp_path / "other"), is_main_process = False)
    assert len(calls) == 1
    assert not (tmp_path / "other").exists()


def test_fsdp2_writer_inside_gather_ignores_the_flag(monkeypatch, tmp_path):
    """The gather already chose its writer, which need not be global rank 0: an
    `is_main_process = False` there must not veto the save."""
    calls = _counted_merge(monkeypatch)
    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: True)
    model = _tiny_peft_model()
    _generic_save(model, str(tmp_path / "writer"), is_main_process = False)
    assert len(calls) == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_push_to_hub_merged_uploads_from_rank0_only(monkeypatch, tmp_path, rank):
    calls = _counted_merge(monkeypatch)
    _fake_ddp(monkeypatch, rank)
    out = unsloth_save.unsloth_generic_push_to_hub_merged(
        _tiny_peft_model(), "user/repo", None, token = "hf_x"
    )
    if rank == 0:
        assert len(calls) == 1 and calls[0]["push_to_hub"] is True
    else:
        assert out is None and calls == []


@pytest.mark.parametrize("rank", [0, 1])
def test_save_pretrained_merged_under_ddp_pushes_from_rank0_only(monkeypatch, tmp_path, rank):
    calls = _counted_merge(monkeypatch)
    _fake_ddp(monkeypatch, rank)
    model = _tiny_peft_model()
    unsloth_save.unsloth_generic_save_pretrained_merged(
        model, str(tmp_path / "local"), None, save_method = "merged_16bit"
    )
    assert len(calls) == 1 and calls[0]["push_to_hub"] is False
    out = unsloth_save.unsloth_generic_save_pretrained_merged(
        model,
        "user/repo",
        None,
        save_method = "merged_16bit",
        push_to_hub = True,
        token = "hf_x",
    )
    if rank == 0:
        assert len(calls) == 2 and calls[-1]["push_to_hub"] is True
    else:
        assert out is None and len(calls) == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_push_to_hub_gguf_uploads_from_rank0_only(monkeypatch, tmp_path, rank):
    calls = _fake_gguf_push(monkeypatch, tmp_path)
    _fake_ddp(monkeypatch, rank)
    out = unsloth_save.unsloth_push_to_hub_gguf(
        _tiny_peft_model(), "user/repo", object(), token = "hf_x"
    )
    if rank == 0:
        assert calls["save"] == 1 and calls["hub"]
    else:
        assert out is None and calls["save"] == 0 and calls["hub"] == []


def test_save_pretrained_gguf_skips_everything_off_main(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(unsloth_save, "unsloth_generic_save", lambda **kwargs: ran.append(kwargs))
    out = unsloth_save.unsloth_save_pretrained_gguf(
        _tiny_peft_model(), str(tmp_path / "gguf"), object(), is_main_process = False
    )
    assert out is None and ran == []


def test_save_pretrained_gguf_inside_fsdp2_gather_ignores_the_flag(monkeypatch, tmp_path):
    """The same flag gate must yield to an FSDP2 gather already in progress."""
    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: True)
    monkeypatch.setattr(unsloth_save, "_gguf_remote_code_approved", lambda *a, **k: True)
    monkeypatch.setattr(
        unsloth_save, "_preflight_gguf_disk", lambda **kwargs: (kwargs["save_directory"], True)
    )
    monkeypatch.setattr(unsloth_save, "_resolve_imatrix_file", lambda *a, **k: None)
    monkeypatch.setattr(unsloth_save, "fix_tokenizer_bos_token", lambda *a, **k: (False, None))
    monkeypatch.setattr(unsloth_save, "create_ollama_modelfile", lambda *a, **k: None)
    from unsloth import tokenizer_utils

    monkeypatch.setattr(tokenizer_utils, "_tokenizer_objects", lambda *a: [])
    merged, converted = [], []

    import inspect

    real_save = inspect.signature(unsloth_save.unsloth_generic_save)
    # bind(): the forwarded kwargs must be ones the real merge accepts.
    monkeypatch.setattr(
        unsloth_save,
        "unsloth_generic_save",
        lambda **kwargs: merged.append(real_save.bind(**kwargs)),
    )
    monkeypatch.setattr(
        unsloth_save,
        "save_to_gguf",
        lambda **kwargs: (converted.append(1) or ([str(tmp_path / "m.gguf")], False, False)),
    )
    out = unsloth_save.unsloth_save_pretrained_gguf(
        _tiny_peft_model(),
        str(tmp_path / "gguf"),
        object(),
        is_main_process = False,
        quantization_method = "q8_0",
    )
    assert len(merged) == 1 and converted == [1]
    assert out["gguf_files"] == [str(tmp_path / "m.gguf")]


def test_push_to_hub_gguf_inside_fsdp2_gather_ignores_the_flag(monkeypatch, tmp_path):
    """An FSDP2 sub-mesh writer with is_main_process=False must still run push_to_hub_gguf."""
    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: True)
    calls = _fake_gguf_push(monkeypatch, tmp_path)
    unsloth_save.unsloth_push_to_hub_gguf(
        _tiny_peft_model(), "user/repo", object(), token = "hf_x", is_main_process = False
    )
    assert calls["save"] == 1 and calls["hub"]


def test_full_model_inside_fsdp2_gather_ignores_the_flag(monkeypatch, tmp_path):
    """The non-PEFT branch (also the staged save of a merged Hub push) honours the gather too."""
    from transformers import LlamaConfig, LlamaForCausalLM

    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: True)
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            vocab_size = 32,
        )
    )
    _generic_save(model, str(tmp_path / "full"), is_main_process = False)
    assert (tmp_path / "full" / "config.json").exists()

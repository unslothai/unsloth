# SPDX-License-Identifier: Apache-2.0
"""#4037 integration: the guard fires INSIDE FastBaseModel.from_pretrained, before any
model construction — so a Jamba 4-bit request dies with the clear error and never
reaches auto_model.from_pretrained (no shard download, no opaque quantizer crash).
"""
from types import SimpleNamespace

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.models import vision


def _fake_jamba_config():
    return SimpleNamespace(
        model_type = "jamba",
        architectures = ["JambaForCausalLM"],
        hidden_size = 2048,
        num_attention_heads = 16,
        max_position_embeddings = 4096,
        torch_dtype = "float16",
    )


def _patch_cuda(monkeypatch):
    """Stub the CUDA calls that run between entry and the guard on a CPU-only build."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (80 * 1024**3, 80 * 1024**3), raising = False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "FakeGPU", raising = False)
    # The device-map planner probes per-GPU capacity; CPU torch has no such API.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: SimpleNamespace(total_memory = 80 * 1024**3, name = "FakeGPU", major = 8, minor = 0),
        raising = False,
    )


def test_preflight_guard_fires_inside_from_pretrained(monkeypatch):
    """A Jamba 4-bit load must raise the clear error before auto_model.from_pretrained."""
    fake_config = _fake_jamba_config()
    _patch_cuda(monkeypatch)

    def _boom(*args, **kwargs):
        pytest.fail("auto_model.from_pretrained was reached — preflight guard did not fire first")

    monkeypatch.setattr(vision.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: fake_config))
    monkeypatch.setattr(vision.AutoModelForCausalLM, "from_pretrained", classmethod(_boom), raising = False)
    monkeypatch.setattr(vision.AutoModelForVision2Seq, "from_pretrained", classmethod(_boom), raising = False)

    with pytest.raises(ValueError, match = "not supported yet"):
        vision.FastBaseModel.from_pretrained(
            model_name = "ai21labs/Jamba2-Mini",
            max_seq_length = 512,
            load_in_4bit = True,
            local_files_only = True,
            model_types = ["jamba"],
        )


def test_llama_load_not_blocked_by_preflight(monkeypatch):
    """Sanity: a non-hybrid model_type passes the guard — the guard's error must NOT fire.

    On a CPU-only runner the load itself fails later (device-map planning / no weights),
    so we assert the *negative*: whatever downstream error happens, it is not the
    hybrid-Mamba preflight message. That is the property under test — Llama sails past
    the guard that blocks Jamba.
    """
    fake_config = SimpleNamespace(
        model_type = "llama",
        architectures = ["LlamaForCausalLM"],
        hidden_size = 1024,
        num_attention_heads = 8,
        max_position_embeddings = 2048,
        torch_dtype = "bfloat16",
    )
    _patch_cuda(monkeypatch)

    monkeypatch.setattr(vision.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: fake_config))

    with pytest.raises(Exception) as exc_info:
        vision.FastBaseModel.from_pretrained(
            model_name = "unsloth/Llama-3.2-1B-Instruct",
            max_seq_length = 512,
            load_in_4bit = True,
            local_files_only = True,
            model_types = ["llama"],
        )
    # The guard must NOT have fired for a non-hybrid model.
    assert "not supported yet" not in str(exc_info.value), (
        f"preflight guard wrongly blocked llama: {exc_info.value}"
    )


def test_16bit_attribute_error_passes_through_untouched(monkeypatch):
    """A 16-bit load (no bnb) raising AttributeError is a real bug — it must NOT be
    re-dressed as a quantizer/layout-mismatch message. Original error propagates."""
    fake_config = SimpleNamespace(
        model_type = "llama",
        architectures = ["LlamaForCausalLM"],
        hidden_size = 1024,
        num_attention_heads = 8,
        max_position_embeddings = 2048,
        torch_dtype = "bfloat16",
    )
    _patch_cuda(monkeypatch)

    def _real_bug(*args, **kwargs):
        raise AttributeError("LlamaRMSNorm has no attribute 'some_new_field'")

    monkeypatch.setattr(vision.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: fake_config))
    # The dispatch may pick any of the three auto classes depending on config shape;
    # patch all so whichever is reached hits the except clause under test.
    for _cls in (vision.AutoModelForCausalLM, vision.AutoModelForImageTextToText, vision.AutoModelForVision2Seq):
        monkeypatch.setattr(_cls, "from_pretrained", classmethod(_real_bug), raising = False)

    with pytest.raises(AttributeError) as exc_info:
        vision.FastBaseModel.from_pretrained(
            model_name = "unsloth/Llama-3.2-1B-Instruct",
            max_seq_length = 512,
            load_in_4bit = False,  # no bnb -> translation must stay silent
            local_files_only = True,
            model_types = ["llama"],
        )
    assert "bitsandbytes" not in str(exc_info.value), (
        f"16-bit AttributeError was mislabeled as a quantizer error: {exc_info.value}"
    )


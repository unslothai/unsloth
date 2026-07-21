"""CPU-only behavioral routing tests for the export API.

With the heavy save helpers monkeypatched, confirm each `save_method` / `quantization_method`
reaches the correct export path with the correct arguments. A bare object stands in for the
model, so these run on CPU-only CI with no GPU and no real weights, yet they catch routing
regressions that pure AST checks cannot (e.g. wrong scheme/suffix/outtype passed through).
"""

from __future__ import annotations

import inspect

import pytest

import unsloth.save as save_mod


class _FakeModel:
    """Minimal model stand-in; routing reads nothing meaningful off it before dispatch."""

    config = type(
        "cfg",
        (),
        {"_name_or_path": "fake/model", "architectures": ["LlamaForCausalLM"]},
    )()


# -- merged_*  ->  compressed-tensors dispatch ---------------------------------------------


def test_merged_fp8_routes_to_compressed(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod, "_unsloth_save_compressed_tensors", lambda **kw: seen.update(kw)
    )
    monkeypatch.setattr(
        save_mod, "unsloth_generic_save", lambda **kw: seen.update(generic = True)
    )
    save_mod.unsloth_generic_save_pretrained_merged(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "fp8",
    )
    assert seen.get("scheme") == "FP8_DYNAMIC"
    assert seen.get("suffix") == "fp8"
    assert seen.get("needs_calibration") is False
    assert (
        "generic" not in seen
    ), "compressed save_method must not fall through to the plain merge"


def test_merged_nvfp4_marks_calibration(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod, "_unsloth_save_compressed_tensors", lambda **kw: seen.update(kw)
    )
    monkeypatch.setattr(save_mod, "unsloth_generic_save", lambda **kw: None)
    save_mod.unsloth_generic_save_pretrained_merged(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "nvfp4",
    )
    assert seen.get("scheme") == "NVFP4"
    assert seen.get("needs_calibration") is True


def test_merged_16bit_does_not_route_compressed(monkeypatch, tmp_path):
    calls = {"compressed": 0, "generic": 0}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_compressed_tensors",
        lambda **kw: calls.__setitem__("compressed", calls["compressed"] + 1),
    )
    monkeypatch.setattr(
        save_mod,
        "unsloth_generic_save",
        lambda **kw: calls.__setitem__("generic", calls["generic"] + 1),
    )
    save_mod.unsloth_generic_save_pretrained_merged(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "merged_16bit",
    )
    assert calls["compressed"] == 0, "merged_16bit must not hit the compressed export"
    assert calls["generic"] == 1, "merged_16bit must go through the normal merge path"


# -- save_method='lora'  ->  LoRA GGUF dispatch --------------------------------------------


def test_gguf_lora_passes_valid_outtype(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda model, tok, sd, outtype = None: seen.update(outtype = outtype),
    )
    save_mod.unsloth_save_pretrained_gguf(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "lora",
        quantization_method = "q8_0",
    )
    assert seen.get("outtype") == "q8_0"


def test_gguf_lora_invalid_outtype_falls_back_to_f16(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda model, tok, sd, outtype = None: seen.update(outtype = outtype),
    )
    save_mod.unsloth_save_pretrained_gguf(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "lora",
        quantization_method = "q4_k_m",
    )
    assert (
        seen.get("outtype") == "f16"
    ), "a GGUF model quant (q4_k_m) is not a valid LoRA outtype -> f16"


def test_gguf_lora_push_to_hub_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_mod.unsloth_save_pretrained_gguf(
            _FakeModel(),
            "repo/id",
            tokenizer = object(),
            save_method = "lora",
            push_to_hub = True,
        )


# The above rejection points users at push_to_hub_gguf(save_method='lora'), so that path
# has to work; it is only ever exercised here.


def test_push_to_hub_gguf_lora_dispatches(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda model, tok, sd, **kw: seen.update(kw),
    )
    save_mod.unsloth_push_to_hub_gguf(
        _FakeModel(),
        "repo/id",
        tokenizer = object(),
        save_method = "lora",
        quantization_method = "q8_0",
    )
    assert seen.get("outtype") == "q8_0"
    assert seen.get("push_to_hub") is True


def test_push_to_hub_gguf_lora_skips_non_main_process(monkeypatch):
    calls = []
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda *a, **kw: calls.append(kw),
    )
    result = save_mod.unsloth_push_to_hub_gguf(
        _FakeModel(),
        "repo/id",
        tokenizer = object(),
        save_method = "lora",
        is_main_process = False,
    )
    assert result is None
    assert calls == []


def test_push_to_hub_gguf_skips_non_main_process_before_merged_conversion(monkeypatch):
    calls = []
    monkeypatch.setattr(
        save_mod,
        "unsloth_save_pretrained_gguf",
        lambda **kw: calls.append(kw),
    )
    result = save_mod.unsloth_push_to_hub_gguf(
        _FakeModel(),
        "repo/id",
        tokenizer = object(),
        is_main_process = False,
    )
    assert result is None
    assert calls == []


def test_push_to_hub_gguf_preserves_positional_max_shard_size():
    bound = inspect.signature(save_mod.unsloth_push_to_hub_gguf).bind(
        _FakeModel(),
        "repo/id",
        object(),
        "q4_k_m",
        None,
        None,
        None,
        None,
        "token",
        "50GB",
    )
    assert bound.arguments["max_shard_size"] == "50GB"
    assert "is_main_process" not in bound.arguments


# -- torchao PTQ / QAT dispatch ------------------------------------------------------------


def test_torchao_ptq_routes_to_given_config(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_torchao_with_given_config",
        lambda **kw: seen.update(given = True),
    )
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_torchao_with_attached_config",
        lambda **kw: seen.update(attached = True),
    )
    save_mod.unsloth_save_pretrained_torchao(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        torchao_config = object(),
    )
    assert seen.get("given") and not seen.get("attached")


def test_torchao_qat_routes_to_attached_config(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_torchao_with_given_config",
        lambda **kw: seen.update(given = True),
    )
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_torchao_with_attached_config",
        lambda **kw: seen.update(attached = True),
    )
    model = _FakeModel()
    model._torchao_config = object()  # simulates a model trained with qat_scheme
    save_mod.unsloth_save_pretrained_torchao(
        model,
        str(tmp_path),
        tokenizer = object(),
        torchao_config = None,
    )
    assert seen.get("attached") and not seen.get("given")


def test_torchao_requires_config_or_qat(tmp_path):
    # No torchao_config and no attached QAT config is a user error, surfaced eagerly.
    with pytest.raises(AssertionError):
        save_mod.unsloth_save_pretrained_torchao(
            _FakeModel(),
            str(tmp_path),
            tokenizer = object(),
            torchao_config = None,
        )

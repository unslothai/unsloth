# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""load_in_8bit dispatch: 'int8' (rotated int8 LoRA base), 'bnb' (bitsandbytes), True (bitsandbytes unless the
GPU is one the rotated int8 base is measured faster on), and which LoRA bases the int8 base converts.
CPU only: the GPU class and the checkpoint config are stubbed."""

import json

import pytest
import torch

from unsloth.models import float8_training as f8t
from unsloth.models import onthefly_quant as oq


def _from_pretrained(
    model_name = "m",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    load_in_8bit = False,
    load_in_16bit = False,
    full_finetuning = False,
    token = None,
    fast_inference = False,
    revision = None,
    trust_remote_code = False,
    qat_scheme = None,
    load_in_fp8 = False,
    *args,
    **kwargs,
):
    return (load_in_4bit, load_in_8bit, load_in_fp8, kwargs.get("float8_training"))


@pytest.fixture
def wrapped():
    return oq.onthefly_lowbit_load(_from_pretrained)


@pytest.fixture
def int8_gpu(monkeypatch):
    """An RTX 50 / RTX PRO 6000 class GPU and a 16-bit checkpoint wide enough for the int8 base."""
    monkeypatch.setattr(oq, "_int8_auto_device_reason", lambda: None)
    monkeypatch.setattr(oq, "_int8_auto_config_reason", lambda arguments, extra: None)
    monkeypatch.delenv(oq.AUTO_8BIT_ENV, raising = False)
    monkeypatch.setattr(oq, "_AUTO_NOTICE", [])


@pytest.mark.parametrize("name", ["int8", "INT8", "int8-rotated", "int8_lora"])
def test_int8_string_is_the_rotated_int8_base(wrapped, name):
    assert wrapped("m", load_in_8bit = name) == (False, False, False, "int8_lora")
    assert wrapped("m", load_in_8bit = name, load_in_4bit = False) == (False, False, False, "int8_lora")
    assert wrapped("m", load_in_8bit = name, float8_training = "int8_lora") == (
        False,
        False,
        False,
        "int8_lora",
    )


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"full_finetuning": True}, ValueError),
        ({"fast_inference": True}, NotImplementedError),
        ({"load_in_4bit": True}, ValueError),
        ({"load_in_fp8": True}, ValueError),
        ({"float8_training": "fp8_lora"}, ValueError),
    ],
)
def test_int8_string_conflicts_raise(wrapped, kwargs, exc):
    with pytest.raises(exc):
        wrapped("m", load_in_8bit = "int8", **kwargs)


@pytest.mark.parametrize("name", ["bnb", "bitsandbytes", "llm.int8", "BNB-8bit"])
def test_bnb_string_is_bitsandbytes_even_where_auto_picks_int8(wrapped, int8_gpu, name):
    # Like the fp8 strings, a bitsandbytes string replaces the load_in_4bit = True default.
    assert wrapped("m", load_in_8bit = name) == (False, True, False, None)
    assert wrapped("m", load_in_8bit = name, load_in_4bit = False) == (False, True, False, None)
    assert oq.normalize_lowbit_flags(False, name) == (False, True, False, None)


def test_explicit_bnb_survives_a_nested_load(int8_gpu):
    # FastLanguageModel delegates load_in_8bit loads to FastModel, which is wrapped too: the inner call must not
    # re-run the True -> int8 pick on the boolean the outer call normalized 'bnb' to.
    inner = oq.onthefly_lowbit_load(_from_pretrained)

    def outer_from_pretrained(
        model_name = "m",
        load_in_4bit = True,
        load_in_8bit = False,
        **kwargs,
    ):
        return inner(model_name, load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit, **kwargs)

    outer = oq.onthefly_lowbit_load(outer_from_pretrained)
    assert outer("m", load_in_4bit = False, load_in_8bit = "bnb") == (False, True, False, None)
    assert outer("m", load_in_4bit = False, load_in_8bit = True) == (False, False, False, "int8_lora")
    assert inner("m", load_in_4bit = False, load_in_8bit = True) == (False, False, False, "int8_lora")
    assert outer("m", load_in_8bit = "bnb") == (False, True, False, None)


def test_true_stays_bitsandbytes_off_the_measured_gpus(wrapped, monkeypatch):
    monkeypatch.setattr(oq, "_int8_auto_device_reason", lambda: "compute capability 9.0 is not one")
    assert wrapped("m", load_in_8bit = True, load_in_4bit = False) == (False, True, False, None)


@pytest.mark.parametrize("l4", [True, "nf4", "nvfp4", "int4"])
def test_explicit_4bit_strings_are_not_dropped(wrapped, int8_gpu, l4):
    with pytest.raises(ValueError, match = "conflicts with load_in_4bit"):
        wrapped("m", load_in_8bit = "int8", load_in_4bit = l4)
    # True leaves an explicit 4-bit request (and its validation) alone.
    arguments = {"load_in_8bit": True, "load_in_4bit": l4}
    assert oq.resolve_int8_8bit(arguments) == "bnb" and arguments["load_in_4bit"] == l4
    if l4 == "int4":
        with pytest.raises(ValueError, match = "not supported"):
            wrapped("m", load_in_8bit = True, load_in_4bit = l4)


def test_true_picks_int8_on_the_measured_gpus_and_says_so_once(wrapped, int8_gpu, capsys):
    for _ in range(2):
        assert wrapped("m", load_in_8bit = True, load_in_4bit = False) == (
            False,
            False,
            False,
            "int8_lora",
        )
    assert capsys.readouterr().out.count("load_in_8bit = True runs the rotated int8") == 1
    # load_in_4bit left at its True default: unchanged (the loader refuses 4-bit + 8-bit as before).
    assert wrapped("m", load_in_8bit = True) == (True, True, False, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"full_finetuning": True},
        {"fast_inference": True},
        {"qat_scheme": "int4"},
        {"load_in_4bit": True},
        {"dtype": torch.float32},
        {"float8_training": "fp8_lora"},
    ],
)
def test_true_keeps_bitsandbytes_where_int8_cannot_replace_it(wrapped, int8_gpu, kwargs):
    kwargs = {"load_in_4bit": False, **kwargs}
    l4, l8, fp8, f8 = wrapped("m", load_in_8bit = True, **kwargs)
    assert l8 is True and f8 == kwargs.get("float8_training")


def test_auto_kill_switch(wrapped, int8_gpu, monkeypatch):
    monkeypatch.setenv(oq.AUTO_8BIT_ENV, "0")
    assert wrapped("m", load_in_8bit = True, load_in_4bit = False) == (False, True, False, None)


def test_false_and_fp8_untouched(wrapped, int8_gpu):
    assert wrapped("m") == (True, False, False, None)
    assert wrapped("m", load_in_8bit = "fp8") == (False, False, True, None)


@pytest.mark.parametrize(
    "capability, hip, ok",
    [
        ((12, 0), None, True),
        ((12, 1), None, False),
        ((10, 0), None, False),
        ((9, 0), None, False),
        ((8, 9), None, False),
        ((8, 0), None, False),
        ((7, 5), None, False),
        ((12, 0), "6.4", False),
    ],
)
def test_device_rule(monkeypatch, capability, hip, ok):
    import unsloth.kernels.int8_lora as IL

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: capability)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(IL, "int8_lora_supported", lambda device: None)
    assert (oq._int8_auto_device_reason() is None) is ok


def _config_dir(tmp_path, **extra):
    cfg = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "head_dim": 128,
    }
    cfg.update(extra)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def test_config_rule(tmp_path):
    assert oq._int8_auto_config_reason({"model_name": _config_dir(tmp_path)}, {}) is None
    (tmp_path / "n").mkdir()
    narrow = _config_dir(tmp_path / "n", hidden_size = 1024)
    assert "hidden size 1024" in oq._int8_auto_config_reason({"model_name": narrow}, {})
    (tmp_path / "q").mkdir()
    quant = _config_dir(
        tmp_path / "q", quantization_config = {"quant_method": "bitsandbytes", "load_in_8bit": True}
    )
    assert "pre-quantized" in oq._int8_auto_config_reason({"model_name": quant}, {})
    assert oq._int8_auto_config_reason({"model_name": str(tmp_path / "missing")}, {}) is not None
    assert (
        oq._int8_auto_config_reason({"model_name": _config_dir(tmp_path)}, {"gguf_file": "x.gguf"})
        is not None
    )


def test_config_rule_keeps_bitsandbytes_for_text_diffusion(tmp_path, monkeypatch):
    from unsloth.models import diffusion

    path = _config_dir(tmp_path)
    monkeypatch.setattr(
        diffusion, "is_diffusion_model_type", lambda model_types: "qwen3" in model_types
    )
    assert "text-diffusion" in oq._int8_auto_config_reason({"model_name": path}, {})


def test_config_probe_forwards_offline_and_cache_options(monkeypatch):
    import transformers

    import os

    calls = []

    def fake(name, **kwargs):
        calls.append((dict(kwargs), os.environ.get("HF_HUB_OFFLINE")))
        raise OSError("offline")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    extra = {
        "local_files_only": True,
        "cache_dir": "/c",
        "subfolder": None,
        "attn_implementation": "sdpa",
    }
    assert oq._int8_auto_config_reason({"model_name": "org/m", "revision": "r"}, extra) is not None
    ((seen, env),) = calls  # offline asked: one offline read, no network attempt
    assert (
        seen["local_files_only"] is True and seen["cache_dir"] == "/c" and seen["revision"] == "r"
    )
    assert env == "1" and "subfolder" not in seen and "attn_implementation" not in seen
    calls.clear()
    assert oq._int8_auto_config_reason({"model_name": "org/m"}, {"cache_dir": "/c"}) is not None
    assert [c[0].get("local_files_only") for c in calls] == [
        None,
        True,
    ]  # online, then the offline retry


class _Block(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.q_proj = torch.nn.Linear(i, o, bias = False)
        self.in_proj_a = torch.nn.Linear(i, 16, bias = False)


class _VLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import PretrainedConfig

        self.config = PretrainedConfig(hidden_size = 256)  # peft reads config.get(...)
        self.visual = torch.nn.ModuleDict({"blocks": torch.nn.ModuleList([_Block(256, 256)])})
        self.language_model = torch.nn.ModuleDict(
            {"layers": torch.nn.ModuleList([_Block(256, 256)])}
        )

    def forward(self, x):
        return self.language_model["layers"][0].q_proj(x)


def test_int8_base_skips_towers_and_tiny_outputs(monkeypatch):
    peft = pytest.importorskip("peft")
    monkeypatch.setenv("UNSLOTH_INT8_LORA_MIN_HIDDEN", "0")
    import unsloth.kernels.int8_lora as IL

    monkeypatch.setattr(IL, "MIN_HIDDEN", 0)
    model = _VLM().to(torch.bfloat16)
    model = peft.get_peft_model(model, peft.LoraConfig(r = 4, target_modules = ["q_proj", "in_proj_a"]))
    converted = f8t.enable_fp8_lora(model, f8t.INT8_LORA_RECIPE, allow_cpu = True)
    assert [n.endswith("language_model.layers.0.q_proj") for n in converted] == [True]
    monkeypatch.setenv("UNSLOTH_INT8_LORA_TOWERS", "1")
    assert f8t._int8_tower_skips(model) == ()


_ROCM = torch.cuda.is_available() and torch.version.hip is not None


@pytest.mark.skipif(not _ROCM, reason = "needs a ROCm GPU")
def test_rocm_true_stays_bitsandbytes_and_int8_refuses(wrapped):
    assert oq._int8_auto_device_reason() == "not an NVIDIA GPU"
    assert wrapped("m", load_in_8bit = True, load_in_4bit = False) == (False, True, False, None)
    from unsloth import precision as P

    with pytest.raises(P.PrecisionRefused):
        f8t.resolve_fp8_lora("int8_lora", dtype = torch.bfloat16)


@pytest.mark.skipif(not _ROCM, reason = "needs a ROCm GPU")
def test_rocm_int_mm_matches_reference():
    # Not used by Unsloth on ROCm; records whether the int8 GEMM a future ROCm int8 base would need is exact.
    a = torch.randint(-127, 128, (256, 512), dtype = torch.int8, device = "cuda")
    b = torch.randint(-127, 128, (512, 256), dtype = torch.int8, device = "cuda")
    out = torch._int_mm(a, b)
    assert out.dtype == torch.int32
    assert torch.equal(out.cpu(), (a.cpu().long() @ b.cpu().long()).int())

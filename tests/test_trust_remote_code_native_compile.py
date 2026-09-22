"""`trust_remote_code = True` on a native architecture must not switch the compiler off.

The compiler pass (fast LoRA forward, fused linear cross entropy, compiled norms and
attention) was skipped whenever the flag was set, on the grounds that remote code
cannot be traced. That is only true when the checkpoint actually ships its own
modeling files. Gemma-4 loaded with the flag lost all of it: PEFT's own Linear4bit
forward ran, casting every activation to the float32 LoRA dtype and running both
LoRA matmuls as fp32 SIMT GEMMs, and the 262k-vocab logits were materialised in
full instead of going through the fused loss.
"""
import os
from types import SimpleNamespace

import pytest


def _helper():
    from unsloth.models.loader import _config_uses_remote_code
    return _config_uses_remote_code


def test_native_config_is_not_remote_code():
    f = _helper()
    assert f(SimpleNamespace(auto_map=None)) is False
    assert f(SimpleNamespace()) is False


def test_auto_map_means_remote_code():
    f = _helper()
    assert f(SimpleNamespace(auto_map={"AutoModelForCausalLM": "modeling_x.XForCausalLM"})) is True


def test_sub_config_auto_map_counts():
    f = _helper()
    cfg = SimpleNamespace(auto_map=None, text_config=SimpleNamespace(auto_map={"AutoConfig": "configuration_x.XConfig"}))
    assert f(cfg) is True


def test_transformers_modules_config_class_counts():
    f = _helper()

    class RemoteConfig:  # what a dynamically loaded config looks like
        auto_map = None
    RemoteConfig.__module__ = "transformers_modules.some_repo.configuration_x"
    assert f(RemoteConfig()) is True


def test_no_config_keeps_the_conservative_answer():
    assert _helper()(None) is True


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a GPU to load a 4-bit model")
def test_native_model_with_trust_remote_code_keeps_fast_lora(tmp_path, monkeypatch):
    """The arm that fails without the fix: PEFT's Linear4bit forward is left in place."""
    monkeypatch.chdir(tmp_path)  # fresh unsloth_compiled_cache
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastModel
    model, _ = FastModel.from_pretrained(
        "tiny-random/gemma-4-moe", max_seq_length=256, dtype=torch.bfloat16,
        load_in_4bit=True, trust_remote_code=True,
    )
    model = FastModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0, bias="none")
    from peft.tuners.lora.bnb import Linear4bit
    assert Linear4bit.forward.__name__ == "unsloth_forward", Linear4bit.forward.__module__
    assert any(f.startswith("unsloth_compiled_module_gemma4") for f in os.listdir("unsloth_compiled_cache"))

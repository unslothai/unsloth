# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

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
    assert f(SimpleNamespace(auto_map = None)) is False
    assert f(SimpleNamespace()) is False


def test_auto_map_means_remote_code():
    f = _helper()
    assert f(SimpleNamespace(auto_map = {"AutoModelForCausalLM": "modeling_x.XForCausalLM"})) is True


def test_sub_config_auto_map_counts():
    f = _helper()
    cfg = SimpleNamespace(
        auto_map = None,
        text_config = SimpleNamespace(auto_map = {"AutoConfig": "configuration_x.XConfig"}),
    )
    assert f(cfg) is True


def test_transformers_modules_config_class_counts():
    f = _helper()

    class RemoteConfig:  # what a dynamically loaded config looks like
        auto_map = None

    RemoteConfig.__module__ = "transformers_modules.some_repo.configuration_x"
    assert f(RemoteConfig()) is True


def test_no_config_keeps_the_conservative_answer():
    assert _helper()(None) is True


def test_tokenizer_only_auto_map_is_still_native():
    """A custom tokenizer or processor is not model code the compiler must trace."""
    f = _helper()
    assert (
        f(SimpleNamespace(auto_map = {"AutoTokenizer": ["tokenization_x.XTokenizer", None]})) is False
    )
    assert (
        f(
            SimpleNamespace(
                auto_map = {
                    "AutoProcessor": "processing_x.XProcessor",
                    "AutoImageProcessor": "image_processing_x.XImageProcessor",
                }
            )
        )
        is False
    )
    assert (
        f(
            SimpleNamespace(
                auto_map = {
                    "AutoTokenizer": "tokenization_x.XTokenizer",
                    "AutoModelForCausalLM": "modeling_x.XForCausalLM",
                }
            )
        )
        is True
    )


def test_sub_config_from_transformers_modules_counts():
    """The remote-class check applies to sub-configs the same way as to the root."""
    f = _helper()

    class RemoteTextConfig:
        auto_map = None

    RemoteTextConfig.__module__ = "transformers_modules.some_repo.configuration_x"
    assert f(SimpleNamespace(auto_map = None, text_config = RemoteTextConfig())) is True


def test_dict_shaped_configs_are_handled():
    f = _helper()
    assert f({"auto_map": None, "model_type": "gemma4"}) is False
    assert f({"auto_map": {"AutoConfig": "configuration_x.XConfig"}}) is True
    assert f({"text_config": {"auto_map": {"AutoModel": "modeling_x.XModel"}}}) is True


def _cuda_is_available():
    # Importing torch in the decorator itself turns this skip into a collection error on
    # a runner that does not ship torch, taking the whole module with it.
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_is_available(), reason = "needs a GPU to load a 4-bit model")
def test_native_model_with_trust_remote_code_keeps_fast_lora(tmp_path, monkeypatch):
    """The arm that fails without the fix: PEFT's Linear4bit forward is left in place.

    The checkpoint is fetched from the Hub and is a Gemma-4, so three things this
    test cannot control can stop it before it has measured anything: a
    transformers that predates the architecture (`ValueError: ... model type
    gemma4 ... Transformers does not recognize this architecture` on 4.57.6), a
    host without torchvision (`ImportError: Unsloth: Could not load the vision
    processor`), and no network. All three were live on the declared support
    range, and each turned into a hard failure that says nothing about the fix.
    Skip on them; still fail on anything else, which is the arm that matters.
    """
    monkeypatch.chdir(tmp_path)  # fresh unsloth_compiled_cache
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastModel

    try:
        model, _ = FastModel.from_pretrained(
            "tiny-random/gemma-4-moe",
            max_seq_length = 256,
            dtype = torch.bfloat16,
            load_in_4bit = True,
            trust_remote_code = True,
        )
    except Exception as exception:
        text = str(exception)
        if any(
            marker in text
            for marker in (
                "does not recognize this architecture",
                "is not supported yet in",
                "torchvision",
                "Could not load the vision processor",
                "We couldn't connect to",
                "offline mode",
                "Connection error",
            )
        ):
            pytest.skip(
                f"the checkpoint cannot be built on this host ({type(exception).__name__}: {text[:160]})"
            )
        raise
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none")
    from peft.tuners.lora.bnb import Linear4bit

    assert Linear4bit.forward.__name__ == "unsloth_forward", Linear4bit.forward.__module__
    assert any(
        f.startswith("unsloth_compiled_module_gemma4") for f in os.listdir("unsloth_compiled_cache")
    )


def test_compiler_call_site_gates_the_flag_on_the_config():
    """Every other test here calls the predicate directly, so all of them stay green if
    the one line that uses it is reverted. This is the only test that fails on main."""
    import ast
    import inspect

    from unsloth.models import loader

    gated = []
    for node in ast.walk(ast.parse(inspect.getsource(loader))):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "unsloth_compile_transformers":
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        assert "trust_remote_code" in keywords, "the compiler call lost its trust_remote_code"
        gated.append(
            any(
                isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "_config_uses_remote_code"
                for n in ast.walk(keywords["trust_remote_code"])
            )
        )

    assert gated, "no unsloth_compile_transformers call site found"
    assert all(gated), (
        f"{gated.count(False)} of {len(gated)} compiler call sites pass trust_remote_code "
        "straight through instead of gating it on _config_uses_remote_code(model_config)"
    )

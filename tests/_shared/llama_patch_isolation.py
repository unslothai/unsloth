# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastLlamaModel.pre_patch rewrites the stock Llama classes process-wide; restore them after an in-process load."""

import pytest


@pytest.fixture
def restore_llama_patches():
    import peft
    import transformers.models.llama.modeling_llama as modeling_llama

    module_globals = dict(vars(modeling_llama))
    classes = {
        cls: dict(vars(cls))
        for cls in module_globals.values()
        if isinstance(cls, type) and cls.__module__ == modeling_llama.__name__
    }
    peft_forward = peft.PeftModelForCausalLM.__dict__["forward"]
    try:
        yield
    finally:
        for name in set(vars(modeling_llama)) - set(module_globals):
            delattr(modeling_llama, name)
        for name, value in module_globals.items():
            if vars(modeling_llama).get(name) is not value:
                setattr(modeling_llama, name, value)
        for cls, attrs in classes.items():
            for name in set(vars(cls)) - set(attrs):
                delattr(cls, name)
            for name, value in attrs.items():
                if vars(cls).get(name) is not value:
                    setattr(cls, name, value)
        peft.PeftModelForCausalLM.forward = peft_forward

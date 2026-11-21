"""Regression for unsloth_base_fast_generate leaving for_inference()'s eval /
gradient_checkpointing-off state behind (C9): generate() must hand a training
model back to training, on success and on exception, verbatim GC mode included.
AST-extracted so no unsloth/CUDA import is needed, like test_fa2_fast_generate_bypass."""

import ast
import inspect
import os
import types
from contextlib import nullcontext
from importlib.metadata import version as installed_version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version


VISION_PATH = Path(__file__).parents[1] / "unsloth" / "models" / "vision.py"
TRANSFORMERS_VERSION = installed_version("transformers")


def _load_function(name, namespace):
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(ast.Module(body = [function], type_ignores = []), str(VISION_PATH), "exec"),
        namespace,
    )
    return namespace[name]


class _CM:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TinyVLM(torch.nn.Module):
    def __init__(self, gc):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(
            architectures = ["TinyForConditionalGeneration"],
            eos_token_id = 2,
        )
        self.generation_config = SimpleNamespace()
        self._supports_static_cache = False
        if gc != "absent":
            self.gradient_checkpointing = gc
            self.lin.gradient_checkpointing = gc

    def forward(self, input_ids = None):
        return self.lin(input_ids.float())


def _namespace(events, flash):
    class FakeFastBaseModel:
        @staticmethod
        def for_inference(model):
            events.append("for_inference")
            for m in model.modules():
                if hasattr(m, "gradient_checkpointing"):
                    m.gradient_checkpointing = False
            model.eval()
            model._flag_for_generation = True

        @staticmethod
        def for_training(model, use_gradient_checkpointing = None):
            events.append(("for_training", use_gradient_checkpointing))
            for m in model.modules():
                if hasattr(m, "gradient_checkpointing"):
                    m.gradient_checkpointing = use_gradient_checkpointing
            model.train()
            if hasattr(model, "_flag_for_generation"):
                del model._flag_for_generation

    return {
        "torch": torch,
        "os": os,
        "inspect": inspect,
        "contextlib": __import__("contextlib"),
        "FastBaseModel": FakeFastBaseModel,
        "dtype_from_config": lambda config: "bfloat16",
        "_get_dtype": lambda dtype: torch.bfloat16,
        "_get_inference_mode_context_manager": lambda model: torch.inference_mode(),
        "_lora_fold_generate": lambda model, args, kwargs: nullcontext(),
        "_unsloth_generate_accepts_kwarg": lambda model, name: False,
        "NUM_LOGITS_TO_KEEP": {},
        "DEVICE_TYPE_TORCH": "cpu",
        "Version": Version,
        "transformers_version": TRANSFORMERS_VERSION,
        "_uses_flash_attention_for_generation": lambda config: flash,
        "_clear_generation_caches": lambda model: None,
        "_is_text_seq2seq_config": lambda config: False,
        "_generic_decode_graphs": SimpleNamespace(
            ineligible = lambda model: "eager",
            plan = lambda *args, **kwargs: "eager",
            kept_cache = lambda model: None,
        ),
        "_decode_graphs": SimpleNamespace(
            generate_scope = lambda: nullcontext(),
            plan_scope = lambda plan: nullcontext(),
            eager_scope = lambda: nullcontext(),
        ),
        "_needs_bidirectional_multimodal_mask": lambda model, kwargs: False,
        "_static_cache_does_not_fit": lambda model, input_ids, kwargs: False,
        "_dynamic_cache_choice": lambda kwargs: "dynamic",
        "_compiles_decode": lambda model: False,
        "_is_decode_compile_model": lambda model: False,
        "_match_compiled_call": lambda model, compile_decode: None,
        "_decode_compile_config": object(),
        "_compile_config": object(),
        "_CompileDecodeOnRepeat": _CM,
        "_eager_decodes": lambda model: False,
        "_EagerDecodeSteps": _CM,
        "_NonFlexGenerate": _CM,
        "_wants_attentions": lambda model, kwargs: False,
        "_GraphDecodeSteps": _CM,
    }


def _run(flash, start_training, gc, fail):
    events, seen = [], {}
    model = TinyVLM(gc)
    model.eval() if not start_training else model.train()

    def _old_generate(self, *args, **kwargs):
        seen["training"] = self.training
        seen["gc"] = {id(m): getattr(m, "gradient_checkpointing", "absent") for m in self.modules()}
        if fail:
            raise fail("boom")
        return "generated"

    model._old_generate = types.MethodType(_old_generate, model)
    fast_generate = _load_function("unsloth_base_fast_generate", _namespace(events, flash))
    model.generate = types.MethodType(fast_generate, model)

    ids = torch.ones((1, 3), dtype = torch.long)
    if fail:
        with pytest.raises(fail):
            model.generate(input_ids = ids)
        result = None
    else:
        result = model.generate(input_ids = ids)
    return model, events, seen, result


@pytest.mark.parametrize("flash", [True, False])
@pytest.mark.parametrize("start_training", [True, False])
@pytest.mark.parametrize("gc", [True, "unsloth", False, "absent"])
@pytest.mark.parametrize("fail", [None, RuntimeError, KeyboardInterrupt])
def test_generate_restores_training_state(flash, start_training, gc, fail):
    model, events, seen, result = _run(flash, start_training, gc, fail)

    if fail is None:
        assert result == "generated"
    assert events[0] == "for_inference"
    assert seen["training"] is False
    assert all(v in (False, "absent") for v in seen["gc"].values())

    expected_gc = False if gc == "absent" else gc
    if start_training:
        assert events == ["for_inference", ("for_training", expected_gc)]
        assert model.training is True
        for m in model.modules():
            assert getattr(m, "gradient_checkpointing", "absent") == (
                "absent" if gc == "absent" else gc
            )
        assert not hasattr(model, "_flag_for_generation")
    else:
        assert events == ["for_inference"]
        assert model.training is False
        for m in model.modules():
            assert getattr(m, "gradient_checkpointing", "absent") in (False, "absent")
        assert hasattr(model, "_flag_for_generation")


def test_training_step_works_after_generate():
    model, _, _, result = _run(flash = False, start_training = True, gc = True, fail = None)
    assert result == "generated"
    out = model.lin(torch.ones(2, 4)).sum()
    out.backward()
    assert model.lin.weight.grad is not None


if __name__ == "__main__":
    for flash in (True, False):
        for start_training in (True, False):
            for gc in (True, "unsloth", False, "absent"):
                for fail in (None, RuntimeError, KeyboardInterrupt):
                    test_generate_restores_training_state(flash, start_training, gc, fail)
    test_training_step_works_after_generate()
    print("OK: generate() restores training state")

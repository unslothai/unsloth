"""Regression for unsloth_fast_generate leaving for_inference()'s eval /
gradient_checkpointing-off state behind (C9b): llama generate() must hand a
training model back to training, on success and on exception, verbatim GC mode
included, and free the decode graphs it held. AST-extracted so no unsloth/CUDA
import is needed, like test_generate_training_state_restore."""

import ast
import inspect
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


LLAMA_PATH = Path(__file__).parents[1] / "unsloth" / "models" / "llama.py"
HELD = "_unsloth_decode_graphs_held"


def _load_functions(names, namespace):
    tree = ast.parse(LLAMA_PATH.read_text(encoding = "utf-8"))
    wanted = set(names)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {f.name for f in functions} == wanted
    exec(
        compile(ast.Module(body = functions, type_ignores = []), str(LLAMA_PATH), "exec"),
        namespace,
    )
    return namespace


class TinyLlama(torch.nn.Module):
    def __init__(self, gc, max_position_embeddings = 128):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(
            max_position_embeddings = max_position_embeddings,
            eos_token_id = 2,
            model_type = "llama",
        )
        self.generation_config = SimpleNamespace()
        self._saved_temp_tokenizer = SimpleNamespace(padding_side = "right")
        if gc != "absent":
            self.gradient_checkpointing = gc
            self.lin.gradient_checkpointing = gc

    def forward(self, input_ids = None):
        return input_ids


def _record_cm(events, event):
    events.append(event)
    return nullcontext()


def _namespace(events, graphs_enabled, fail):
    class FakeFastLlamaModel:
        @staticmethod
        def for_inference(model):
            events.append("for_inference")
            for m in model.modules():
                if hasattr(m, "gradient_checkpointing"):
                    m.gradient_checkpointing = False
                if hasattr(m, "training"):
                    m.training = False
                if hasattr(m, "_saved_temp_tokenizer"):
                    m._saved_temp_tokenizer.padding_side = "left"
                m._flag_for_generation = True

        @staticmethod
        def for_training(model, use_gradient_checkpointing = None):
            events.append(("for_training", use_gradient_checkpointing))
            for m in model.modules():
                if hasattr(m, "gradient_checkpointing"):
                    m.gradient_checkpointing = use_gradient_checkpointing
                if hasattr(m, "training"):
                    m.training = True
                if hasattr(m, "_saved_temp_tokenizer"):
                    m._saved_temp_tokenizer.padding_side = "right"
                if hasattr(m, "_flag_for_generation"):
                    del m._flag_for_generation

    def generate_with_fallback(model, old_generate, graphs, args, kwargs):
        if fail is not None:
            raise fail("boom")
        return old_generate(*args, **kwargs)

    return {
        "torch": torch,
        "inspect": inspect,
        "contextlib": __import__("contextlib"),
        "FastLlamaModel": FakeFastLlamaModel,
        "dtype_from_config": lambda config: "bfloat16",
        "_get_dtype": lambda dtype: torch.bfloat16,
        "_ensure_cache_is_dynamic": lambda cache: cache,
        "_get_inference_mode_context_manager": lambda model: torch.inference_mode(),
        "_lora_fold_generate": lambda model, args, kwargs: nullcontext(),
        "DEVICE_TYPE_TORCH": "cpu",
        "KV_CACHE_INCREMENT": 512,
        "_decode_graphs": SimpleNamespace(
            generate_scope = lambda: nullcontext(),
            enabled = lambda: graphs_enabled,
            decode_batch = lambda bsz, kwargs, generation_config: bsz,
            short_call = lambda model, plan_len, prompt_len, bsz: False,
            plan_call = lambda model, ids, dtype, increment, plan_len, bsz = None, eager = False: (
                "graph",
                "token",
            ),
            start_generation = lambda plan_len: events.append("start_generation"),
            end_generation = lambda: events.append("end_generation"),
            EAGER_REDOS = 0,
            eager_scope = lambda: nullcontext(),
            plan_scope = lambda plan: _record_cm(events, ("plan_scope", plan)),
            generate_with_fallback = generate_with_fallback,
            record_need = lambda model, token: events.append("record_need"),
            release_for_training = lambda model: events.append("release_for_training"),
            release = lambda model: events.append("release"),
            keep_only = lambda model, expected: events.append(("keep_only", expected)),
            HELD = HELD,
        ),
    }


def _run(graphs, start_training, gc, fail = None, fail_at = "generate", output_past = False):
    events, seen = [], {}
    model = TinyLlama(
        gc, max_position_embeddings = 4 if fail_at == "maxlen" else 128
    )
    model.train() if start_training else model.eval()

    def _old_generate(*args, **kwargs):
        seen["training"] = model.training
        seen["gc"] = [
            getattr(m, "gradient_checkpointing", "absent") for m in model.modules()
        ]
        seen["padding_side"] = model._saved_temp_tokenizer.padding_side
        if output_past:
            return SimpleNamespace(past_key_values = object())
        return "generated"

    model._old_generate = _old_generate
    namespace = _namespace(events, graphs, fail if fail_at == "generate" else None)
    _load_functions(
        [
            "_generation_max_len",
            "_generation_plan_len",
            "_generation_input_ids",
            "unsloth_fast_generate",
        ],
        namespace,
    )
    model.generate = types.MethodType(namespace["unsloth_fast_generate"], model)

    ids = torch.ones((1, 3), dtype = torch.long)
    raised = fail if fail_at == "generate" else (ValueError if fail else None)
    if raised is not None:
        with pytest.raises(raised):
            model.generate(input_ids = ids, max_new_tokens = 5)
        result = None
    else:
        result = model.generate(input_ids = ids, max_new_tokens = 5)
    return model, events, seen, result


def _assert_inference_state_seen(seen):
    assert seen["training"] is False
    assert all(v in (False, "absent") for v in seen["gc"])
    assert seen["padding_side"] == "left"


def _assert_restored(model, gc):
    expected_gc = False if gc == "absent" else gc
    assert model.training is True
    for m in model.modules():
        assert getattr(m, "gradient_checkpointing", "absent") == (
            "absent" if gc == "absent" else expected_gc
        )
    assert not hasattr(model, "_flag_for_generation")
    assert model._saved_temp_tokenizer.padding_side == "right"


def _assert_still_inference(model):
    assert model.training is False
    for m in model.modules():
        assert getattr(m, "gradient_checkpointing", "absent") in (False, "absent")
    assert hasattr(model, "_flag_for_generation")
    assert model._saved_temp_tokenizer.padding_side == "left"


@pytest.mark.parametrize("graphs", [True, False])
@pytest.mark.parametrize("start_training", [True, False])
@pytest.mark.parametrize("gc", [True, "unsloth", False, "absent"])
@pytest.mark.parametrize("fail", [None, RuntimeError, KeyboardInterrupt])
def test_generate_restores_training_state(graphs, start_training, gc, fail):
    model, events, seen, result = _run(graphs, start_training, gc, fail = fail)

    assert events[0] == "for_inference"
    if fail is None:
        assert result == "generated"
        _assert_inference_state_seen(seen)

    expected_gc = False if gc == "absent" else gc
    if start_training:
        assert events[-1] == ("for_training", expected_gc)
        if graphs:
            assert "start_generation" in events
            assert "end_generation" in events
            assert "release_for_training" in events
            assert events.index("release_for_training") < events.index(
                ("for_training", expected_gc)
            )
        else:
            assert "release_for_training" not in events
        _assert_restored(model, gc)
    else:
        assert not any(
            e == ("for_training", expected_gc) or e == "release_for_training"
            for e in events
        )
        _assert_still_inference(model)


@pytest.mark.parametrize("start_training", [True, False])
@pytest.mark.parametrize("gc", [True, "unsloth", False, "absent"])
def test_max_position_embeddings_error_restores(start_training, gc):
    model, events, seen, result = _run(
        True, start_training, gc, fail = ValueError, fail_at = "maxlen"
    )

    expected_gc = False if gc == "absent" else gc
    if start_training:
        assert events == ["for_inference", ("for_training", expected_gc)]
        _assert_restored(model, gc)
    else:
        assert events == ["for_inference"]
        _assert_still_inference(model)


def test_generate_without_restore_keeps_graphs_on_success():
    model, events, seen, result = _run(True, False, True)
    assert result == "generated"
    assert "release_for_training" not in events
    assert "release" not in events
    assert ("keep_only", 1) in events
    assert model.__dict__.get(HELD) is True
    _assert_still_inference(model)


def test_generate_without_restore_hands_over_returned_cache():
    model, events, seen, result = _run(True, False, True, output_past = True)
    assert result.past_key_values is not None
    assert "release" in events
    assert "release_for_training" not in events
    assert HELD not in model.__dict__
    _assert_still_inference(model)


def test_training_step_works_after_generate():
    model, _, _, result = _run(True, True, "unsloth")
    assert result == "generated"
    out = model.lin(torch.ones(2, 4)).sum()
    out.backward()
    assert model.lin.weight.grad is not None


if __name__ == "__main__":
    for graphs in (True, False):
        for start_training in (True, False):
            for gc in (True, "unsloth", False, "absent"):
                for fail in (None, RuntimeError, KeyboardInterrupt):
                    test_generate_restores_training_state(graphs, start_training, gc, fail)
                test_max_position_embeddings_error_restores(start_training, gc)
    test_generate_without_restore_keeps_graphs_on_success()
    test_generate_without_restore_hands_over_returned_cache()
    test_training_step_works_after_generate()
    print("OK: llama generate() restores training state")

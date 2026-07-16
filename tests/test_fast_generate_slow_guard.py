"""GPU-free test for the fast_generate slow-mode guard in _utils.py.

When fast_inference=False, model.fast_generate falls back to HuggingFace generate, so vLLM-only
inputs must be rejected with a clear message instead of leaking into transformers.generate. Covers
a string prompt, a vLLM {"prompt":..., "multi_modal_data":...} dict, SamplingParams passed both
positionally and as a kwarg, and a normal tokenized call passing through.
"""

import ast, functools, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")


def _load_factory():
    src = open(UTILS).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "make_fast_generate_wrapper":
            ns = {"functools": functools}
            exec(ast.get_source_segment(src, node), ns)
            return ns["make_fast_generate_wrapper"]
    raise AssertionError("make_fast_generate_wrapper not found in _utils.py")


make_fast_generate_wrapper = _load_factory()


class _SamplingParams:
    pass


_SamplingParams.__name__ = "SamplingParams"  # match by class name, no vllm import needed


def _wrapper():
    state = {}

    def original_generate(*a, **k):
        state["hit"] = True
        return "ok"

    return make_fast_generate_wrapper(original_generate), state


def _rejects(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), str(e)
        return True
    raise AssertionError("expected ValueError")


def test_fast_generate_slow_guard():
    w, _ = _wrapper()
    # reject every vLLM-only shape
    assert _rejects(lambda: w("hello"), "fast_inference=True")
    assert _rejects(
        lambda: w({"prompt": "hi", "multi_modal_data": {"image": None}}), "fast_inference=True"
    )
    assert _rejects(lambda: w(["a", "b"]), "fast_inference=True")
    assert _rejects(lambda: w([{"prompt": "hi"}]), "fast_inference=True")  # list of prompt dicts
    assert _rejects(
        lambda: w({"prompt_token_ids": [1, 2, 3]}), "fast_inference=True"
    )  # vLLM TokensPrompt
    assert _rejects(lambda: w(prompts = "hello"), "fast_inference=True")  # vLLM `prompts` kwarg
    assert _rejects(
        lambda: w(prompts = [{"prompt": "hi"}]), "fast_inference=True"
    )  # vLLM `prompts` kwarg list
    assert _rejects(
        lambda: w(prompt_token_ids = [1, 2, 3]), "fast_inference=True"
    )  # vLLM legacy tokenized kwarg
    assert _rejects(
        lambda: w(prompts = [1, 2, 3]), "fast_inference=True"
    )  # token-id list via vLLM-only `prompts` kwarg
    assert _rejects(
        lambda: w(prompts = None), "fast_inference=True"
    )  # vLLM-only kwarg present even if None
    assert _rejects(lambda: w({"prompt": "hi"}, _SamplingParams()), "sampling_params")
    assert _rejects(
        lambda: w({"prompt": "hi"}, [_SamplingParams()]), "sampling_params"
    )  # list of SamplingParams
    assert _rejects(lambda: w(sampling_params = object()), "sampling_params")

    # pass normal tokenized calls with no false positives
    w, state = _wrapper()
    assert w(input_ids = "TOKENS", max_new_tokens = 8) == "ok" and state.get("hit")
    assert w([1, 2, 3], max_new_tokens = 8) == "ok"  # positional token ids
    assert w([], max_new_tokens = 8) == "ok"  # empty positional
    print("13 reject + 3 pass fast_generate slow-mode guard cases passed")


if __name__ == "__main__":
    test_fast_generate_slow_guard()
    print("OK: fast_generate rejects vLLM-style inputs when fast_inference=False")

"""GPU-free regression test for unslothai/unsloth#5162.

`unsloth/gpt-oss-*` ship `generation_config.json` with `eos_token_id = [200002, 199999]`,
where upstream `openai/gpt-oss-*` ship `[200002, 199999, 200012]`. 200012 is `<|call|>`, the
harmony token that ends a TOOL CALL. Without it in the stop set, greedy generation does not
halt when the model finishes a tool call, and the continuation is out of distribution: the
model writes the next harmony role or channel as plain BPE text ("commentary", "assistant")
where a special token was required, which `openai_harmony.StreamableParser` raises as
HarmonyError 200006.

Measured on one B200 with `unsloth/gpt-oss-20b` at bf16, greedy, one tool schema:

    eos = [200002, 199999]          -> 160 new tokens, did not stop, 125 tokens past <|call|>
    eos = [200002, 199999, 200012]  ->  35 new tokens, stopped exactly on <|call|>

`patch_harmony_tool_call_eos` closes the gap at load time for any harmony checkpoint,
including a local fine-tune that inherited the short list through `save_pretrained`.

AST-extracted the way `test_generate_kwarg_gate.py` does it, so nothing imports unsloth,
transformers or CUDA.
"""

import ast
import os

import pytest


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")

# The real ids, so a renumbering of the harmony vocabulary would be caught here rather than
# silently passing with made-up numbers.
CALL_ID = 200012
RETURN_ID = 200002
CHANNEL_ID = 200005
ENDOFTEXT_ID = 199999

_HARMONY_IDS = {
    "<|call|>": CALL_ID,
    "<|channel|>": CHANNEL_ID,
    "<|return|>": RETURN_ID,
    "<|endoftext|>": ENDOFTEXT_ID,
}


class _StubLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


def _namespace():
    """Exec just the harmony helpers out of `_utils.py` into a bare namespace."""
    src = open(UTILS, encoding = "utf-8").read()
    mod = ast.parse(src)
    wanted_functions = {
        "_harmony_tool_call_token_id",
        "patch_harmony_tool_call_eos",
        "_vllm_generation_config_fields",
        "patch_harmony_tool_call_eos_vllm",
        "patch_tokenizer",
    }
    wanted_constants = {
        "_HARMONY_TOOL_CALL_TOKEN",
        "_HARMONY_FINGERPRINT_TOKENS",
        "_VLLM_GENERATION_CONFIG_FIELD_PATHS",
        "_VLLM_MAX_STOP_TOKEN_IDS",
    }
    ns = {"logger": _StubLogger()}
    found = set()
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            exec(ast.get_source_segment(src, node), ns)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_constants:
                    exec(ast.get_source_segment(src, node), ns)
                    found.add(target.id)
    missing = (wanted_functions | wanted_constants) - found
    assert not missing, f"not found in _utils.py: {sorted(missing)}"
    return ns


class _Tokenizer:
    """Only the surface `_harmony_tool_call_token_id` is allowed to touch."""

    def __init__(
        self,
        ids,
        unk_token_id = None,
    ):
        self._ids = dict(ids)
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, self.unk_token_id if self.unk_token_id is not None else -1)


class _GenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _Model:
    def __init__(self, eos_token_id):
        self.generation_config = _GenerationConfig(eos_token_id)


@pytest.fixture(scope = "module")
def ns():
    return _namespace()


@pytest.fixture
def harmony_tokenizer():
    return _Tokenizer(_HARMONY_IDS)


def test_the_shipped_unsloth_stop_set_gains_the_tool_call_token(ns, harmony_tokenizer):
    """The exact list in `unsloth/gpt-oss-20b/generation_config.json` today."""
    model = _Model([RETURN_ID, ENDOFTEXT_ID])
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == [RETURN_ID, ENDOFTEXT_ID, CALL_ID]


def test_the_upstream_stop_set_is_left_exactly_alone(ns, harmony_tokenizer):
    """`openai/gpt-oss-*` already stop on `<|call|>`; order and contents must not move."""
    upstream = [RETURN_ID, ENDOFTEXT_ID, CALL_ID]
    model = _Model(list(upstream))
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == upstream


def test_it_is_idempotent(ns, harmony_tokenizer):
    model = _Model([RETURN_ID, ENDOFTEXT_ID])
    for _ in range(3):
        ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == [RETURN_ID, ENDOFTEXT_ID, CALL_ID]


def test_a_scalar_stop_set_is_widened_not_replaced(ns, harmony_tokenizer):
    model = _Model(RETURN_ID)
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == [RETURN_ID, CALL_ID]


def test_a_missing_stop_set_is_left_alone(ns, harmony_tokenizer):
    """No stop set means there is nothing to widen, so do not invent one.

    Creating `[CALL_ID]` here would make `<|call|>` the only terminator and drop
    `<|return|>`, so an ordinary reply would run past its own end: the reported
    defect pointed the other way. A real repo hits this, reaperdoesntknow/Mini-oss-0.6b.
    """
    model = _Model(None)
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id is None


@pytest.mark.parametrize("empty", [[], ()])
def test_an_empty_stop_set_is_left_alone(ns, harmony_tokenizer, empty):
    """The same reasoning as the `None` case, in a shape that passes the list check.

    `isinstance([], list)` is true, so an empty list reaches the list branch, the loop over
    its entries adds nothing, and an unguarded append would leave `[CALL_ID]`: `<|call|>` as
    the only terminator, with `<|return|>` gone. That is the reported defect pointed the
    other way, which is exactly what the missing-stop-set branch refuses to do.
    """
    model = _Model(empty)
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == empty


@pytest.mark.parametrize(
    "shape",
    [
        ["<|return|>"],  # str entries, int() would raise ValueError
        [200002, None],  # None entry, int() would raise TypeError
        [[200002], 199999],  # nested list, int() would raise TypeError
        True,  # bool is an int subclass
        [True, 199999],
    ],
)
def test_an_unparseable_stop_set_declines_instead_of_raising(ns, harmony_tokenizer, shape):
    """This runs inside patch_tokenizer during from_pretrained.

    llama.py wraps that call in no `except`, and vision.py responds to a raise by
    re-fetching the tokenizer over the network. Declining to widen costs a user the
    tool-call stop token; raising costs them the model load.
    """
    model = _Model(shape)
    before = model.generation_config.eos_token_id
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == before


def test_a_tuple_keeps_its_entries(ns, harmony_tokenizer):
    model = _Model((RETURN_ID, ENDOFTEXT_ID))
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == [RETURN_ID, ENDOFTEXT_ID, CALL_ID]


# ── Everything that is NOT harmony must be untouched ───────────────────────────


def test_a_non_harmony_tokenizer_is_untouched(ns):
    """Qwen, Llama, Gemma, Mistral: no `<|call|>`, so nothing may change."""
    qwen_like = _Tokenizer({"<|im_start|>": 151644, "<|im_end|>": 151645})
    model = _Model([151645, 151643])
    ns["patch_harmony_tool_call_eos"](model, qwen_like)
    assert model.generation_config.eos_token_id == [151645, 151643]


def test_a_tokenizer_that_folds_unknown_text_onto_unk_is_untouched(ns):
    """`convert_tokens_to_ids` returning the unk id for every harmony token is not harmony."""
    folding = _Tokenizer({}, unk_token_id = 0)
    model = _Model([2])
    ns["patch_harmony_tool_call_eos"](model, folding)
    assert model.generation_config.eos_token_id == [2]


def test_a_tokenizer_that_answers_one_id_for_every_token_is_untouched(ns):
    """Distinct-id guard: three harmony tokens that all map to 7 are not a harmony vocab."""
    degenerate = _Tokenizer({"<|call|>": 7, "<|channel|>": 7, "<|return|>": 7})
    model = _Model([2])
    ns["patch_harmony_tool_call_eos"](model, degenerate)
    assert model.generation_config.eos_token_id == [2]


def test_a_partial_harmony_vocabulary_is_untouched(ns):
    """`<|call|>` on its own is not a fingerprint; a model with only that token is skipped."""
    partial = _Tokenizer({"<|call|>": CALL_ID})
    model = _Model([2])
    ns["patch_harmony_tool_call_eos"](model, partial)
    assert model.generation_config.eos_token_id == [2]


def test_no_tokenizer_and_no_model_are_both_safe(ns, harmony_tokenizer):
    assert ns["patch_harmony_tool_call_eos"](None, harmony_tokenizer) is None
    model = _Model([RETURN_ID])
    ns["patch_harmony_tool_call_eos"](model, None)
    assert model.generation_config.eos_token_id == [RETURN_ID]


def test_a_model_with_no_generation_config_is_safe(ns, harmony_tokenizer):
    class _Bare:
        pass

    bare = _Bare()
    assert ns["patch_harmony_tool_call_eos"](bare, harmony_tokenizer) is bare


def test_an_unrecognised_stop_set_shape_is_left_alone(ns, harmony_tokenizer):
    sentinel = object()
    model = _Model(sentinel)
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id is sentinel


def test_a_read_only_generation_config_does_not_raise(ns, harmony_tokenizer):
    """The shape checks decline rather than raise; the ASSIGNMENT must do the same.

    A `generation_config` whose `eos_token_id` is a read-only property raises on
    assignment, and that raise escapes `patch_tokenizer` mid-`from_pretrained`, which is
    the one failure mode this whole function is written to avoid.
    """

    class _ReadOnly:
        @property
        def eos_token_id(self):
            return [RETURN_ID, ENDOFTEXT_ID]

    model = type("_M", (), {})()
    model.generation_config = _ReadOnly()
    assert ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer) is model
    assert model.generation_config.eos_token_id == [RETURN_ID, ENDOFTEXT_ID]


def test_a_validating_generation_config_setter_does_not_raise(ns, harmony_tokenizer):
    """Same again for a setter that rejects the widened list outright."""

    class _Validating:
        def __init__(self):
            self._value = [RETURN_ID, ENDOFTEXT_ID]

        @property
        def eos_token_id(self):
            return self._value

        @eos_token_id.setter
        def eos_token_id(self, value):
            if len(value) > 2:
                raise ValueError("this config accepts at most two eos ids")
            self._value = value

    model = type("_M", (), {})()
    model.generation_config = _Validating()
    assert ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer) is model
    assert model.generation_config.eos_token_id == [RETURN_ID, ENDOFTEXT_ID]


def test_a_tokenizer_whose_conversion_raises_is_safe(ns):
    class _Raises:
        unk_token_id = None

        def convert_tokens_to_ids(self, token):
            raise RuntimeError("third-party tokenizer adapter")

    model = _Model([2])
    ns["patch_harmony_tool_call_eos"](model, _Raises())
    assert model.generation_config.eos_token_id == [2]


# ── Wiring: the load path must actually call it ────────────────────────────────


def test_patch_tokenizer_applies_the_harmony_fix():
    """The wiring must be a real call, not the name appearing in the source text.

    A substring check over the function body is satisfied by a comment such as
    `# TODO: wire up patch_harmony_tool_call_eos(...)`, so it cannot fail in the one
    way that matters. Walk for an `ast.Call` whose callee is actually that name.
    """
    src = open(UTILS, encoding = "utf-8").read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "patch_tokenizer":
            called = {
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            assert "patch_harmony_tool_call_eos" in called, sorted(called)
            assert "patch_harmony_tool_call_eos_vllm" in called, sorted(called)
            return
    raise AssertionError("patch_tokenizer not found in _utils.py")


# ── The vLLM stop set, which `model.generate` never touches ────────────────────
#
# `fast_inference = True` binds `model.fast_generate` straight to `vllm_engine.generate`
# before the tokenizer exists, so the HF `generation_config` edit above cannot reach it.
# These cover the second patch, which widens the dict vLLM applies per request.


class _InputProcessor:
    def __init__(self, fields):
        self.generation_config_fields = fields


class _Engine:
    """Shaped like `vllm.LLM`: the dict hangs off `llm_engine.input_processor`."""

    def __init__(self, fields):
        self.llm_engine = type("_LLMEngine", (), {})()
        self.llm_engine.input_processor = _InputProcessor(fields)


class _VLLMModel(_Model):
    def __init__(self, eos_token_id, fields):
        super().__init__(eos_token_id)
        self.vllm_engine = _Engine(fields)


def test_the_vllm_stop_set_gains_the_tool_call_token(ns, harmony_tokenizer):
    model = _VLLMModel([RETURN_ID, ENDOFTEXT_ID], {"eos_token_id": [RETURN_ID, ENDOFTEXT_ID]})
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    fields = model.vllm_engine.llm_engine.input_processor.generation_config_fields
    assert fields["eos_token_id"] == [RETURN_ID, ENDOFTEXT_ID, CALL_ID]


def test_the_vllm_patch_is_idempotent(ns, harmony_tokenizer):
    model = _VLLMModel([RETURN_ID], {"eos_token_id": [RETURN_ID, ENDOFTEXT_ID, CALL_ID]})
    for _ in range(3):
        ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    fields = model.vllm_engine.llm_engine.input_processor.generation_config_fields
    assert fields["eos_token_id"] == [RETURN_ID, ENDOFTEXT_ID, CALL_ID]


@pytest.mark.parametrize("fields", [{}, {"eos_token_id": None}, {"eos_token_id": []}])
def test_an_absent_vllm_eos_key_is_seeded(ns, harmony_tokenizer, fields):
    """`to_diff_dict()` omits defaults, and vLLM ignores an absent or empty list entirely.

    Seeding `[CALL_ID]` is safe on THIS side, unlike the HF side, because vLLM tracks the
    primary eos separately off the tokenizer and this dict only ever adds to it. Leaving the
    key absent means the widening never fires at all.
    """
    model = _VLLMModel([RETURN_ID], dict(fields))
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    got = model.vllm_engine.llm_engine.input_processor.generation_config_fields
    assert got["eos_token_id"] == [CALL_ID]


def test_a_non_harmony_model_leaves_the_vllm_stop_set_alone(ns):
    qwen_like = _Tokenizer({"<|im_start|>": 151644, "<|im_end|>": 151645})
    model = _VLLMModel([151645], {"eos_token_id": [151645, 151643]})
    ns["patch_harmony_tool_call_eos_vllm"](model, qwen_like)
    fields = model.vllm_engine.llm_engine.input_processor.generation_config_fields
    assert fields["eos_token_id"] == [151645, 151643]


def test_no_engine_means_no_work(ns, harmony_tokenizer):
    """The overwhelmingly common case: `fast_inference` was never asked for."""
    model = _Model([RETURN_ID, ENDOFTEXT_ID])
    assert ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer) is model
    assert not hasattr(model, "vllm_engine")


def test_an_unreachable_engine_dict_does_not_raise(ns, harmony_tokenizer):
    """A vLLM whose internals moved must cost the stop token, never the model load."""

    class _Opaque:
        pass

    model = _Model([RETURN_ID])
    model.vllm_engine = _Opaque()
    assert ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer) is model


def test_an_engine_that_raises_on_attribute_access_does_not_raise(ns, harmony_tokenizer):
    class _Hostile:
        def __getattr__(self, name):
            raise RuntimeError("engine is shutting down")

    model = _Model([RETURN_ID])
    model.vllm_engine = _Hostile()
    assert ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer) is model


@pytest.mark.parametrize("shape", [True, ["<|call|>"], [None], object()])
def test_an_unparseable_vllm_stop_set_declines(ns, harmony_tokenizer, shape):
    model = _VLLMModel([RETURN_ID], {"eos_token_id": shape})
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    fields = model.vllm_engine.llm_engine.input_processor.generation_config_fields
    assert fields["eos_token_id"] is shape


def test_the_vllm_stop_token_cap_is_respected(ns, harmony_tokenizer):
    """vLLM rejects a request whose stop set exceeds its cap when `min_tokens > 0`."""
    cap = ns["_VLLM_MAX_STOP_TOKEN_IDS"]
    full = list(range(cap))
    model = _VLLMModel([RETURN_ID], {"eos_token_id": list(full)})
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    fields = model.vllm_engine.llm_engine.input_processor.generation_config_fields
    assert fields["eos_token_id"] == full


@pytest.mark.parametrize(
    "path", [("llm_engine", "input_processor"), ("input_processor",), ("processor",)]
)
def test_every_supported_attribute_path_is_found(ns, harmony_tokenizer, path):
    """vLLM has moved this attribute between releases; each spelling must resolve."""
    engine = type("_Engine", (), {})()
    holder = engine
    for attribute in path[:-1]:
        setattr(holder, attribute, type("_Node", (), {})())
        holder = getattr(holder, attribute)
    setattr(holder, path[-1], _InputProcessor({"eos_token_id": [RETURN_ID]}))

    model = _Model([RETURN_ID])
    model.vllm_engine = engine
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    assert ns["_vllm_generation_config_fields"](engine)["eos_token_id"] == [RETURN_ID, CALL_ID]


# ── Against the real vLLM, when it is installed ───────────────────────────────


def _real_sampling_params(**kwargs):
    vllm = pytest.importorskip("vllm", reason = "vLLM is not installed")
    return vllm.SamplingParams(**kwargs)


def test_the_widened_dict_really_reaches_vllms_stop_token_ids(ns, harmony_tokenizer):
    """The claim under test is vLLM's, so assert it with vLLM's own code.

    `SamplingParams.update_from_generation_config` is what the engine applies to every
    request. A hand-rolled stand-in would only prove the stand-in.
    """
    fields = {"eos_token_id": [RETURN_ID, ENDOFTEXT_ID]}
    model = _VLLMModel([RETURN_ID, ENDOFTEXT_ID], fields)

    before = _real_sampling_params()
    before.update_from_generation_config(dict(fields), RETURN_ID)
    assert CALL_ID not in (before.stop_token_ids or [])

    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    after = _real_sampling_params()
    after.update_from_generation_config(dict(fields), RETURN_ID)
    assert CALL_ID in after.stop_token_ids


def test_an_explicit_sampling_params_still_gets_the_tool_call_token(ns, harmony_tokenizer):
    """GRPO always passes its own `SamplingParams`, so defaults-only would not cover it.

    The caller's own stop ids must survive alongside.
    """
    fields = {"eos_token_id": [RETURN_ID, ENDOFTEXT_ID]}
    model = _VLLMModel([RETURN_ID], fields)
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)

    explicit = _real_sampling_params(stop_token_ids = [42])
    explicit.update_from_generation_config(dict(fields), RETURN_ID)
    assert CALL_ID in explicit.stop_token_ids
    assert 42 in explicit.stop_token_ids


def test_ignore_eos_still_opts_out(ns, harmony_tokenizer):
    """A caller who asked not to stop on eos must not be given a new way to stop."""
    fields = {"eos_token_id": [RETURN_ID, ENDOFTEXT_ID]}
    model = _VLLMModel([RETURN_ID], fields)
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)

    opted_out = _real_sampling_params(ignore_eos = True)
    opted_out.update_from_generation_config(dict(fields), RETURN_ID)
    assert CALL_ID not in (opted_out.stop_token_ids or [])


def test_the_ordinary_terminator_survives_a_seeded_stop_set(ns, harmony_tokenizer):
    """Seeding `[CALL_ID]` into an empty dict must not cost `<|return|>`.

    That is the one thing the HF side refuses to do, and the reason it is safe here is
    vLLM-specific, so prove it with vLLM rather than by reading the source.
    """
    model = _VLLMModel([RETURN_ID], {})
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    fields = model.vllm_engine.llm_engine.input_processor.generation_config_fields

    params = _real_sampling_params()
    params.update_from_generation_config(dict(fields), RETURN_ID)
    assert CALL_ID in params.all_stop_token_ids
    assert RETURN_ID in params.all_stop_token_ids


# ── The real artefact, when it happens to be on disk ───────────────────────────

_LOCAL_GPT_OSS = os.environ.get("UNSLOTH_TEST_GPT_OSS_DIR", "")


@pytest.mark.skipif(
    not (_LOCAL_GPT_OSS and os.path.isfile(os.path.join(_LOCAL_GPT_OSS, "generation_config.json"))),
    reason = "set UNSLOTH_TEST_GPT_OSS_DIR to a local unsloth/gpt-oss-* snapshot",
)
def test_the_real_shipped_generation_config_is_the_short_list(ns, harmony_tokenizer):
    """Guards the premise: if the Hub repo is corrected, this fails and the patch becomes a
    no-op rather than a silent change of meaning."""
    import json

    shipped = json.load(
        open(os.path.join(_LOCAL_GPT_OSS, "generation_config.json"), encoding = "utf-8")
    )
    eos = shipped["eos_token_id"]
    assert isinstance(eos, list)
    model = _Model(list(eos))
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert CALL_ID in model.generation_config.eos_token_id
    assert all(token_id in model.generation_config.eos_token_id for token_id in eos)

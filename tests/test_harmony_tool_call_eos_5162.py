# SPDX-License-Identifier: AGPL-3.0-only
"""unslothai/unsloth#5162: gpt-oss must stop on <|call|> (200012). AST-extracted, no unsloth/CUDA import."""

import ast
import os

import pytest


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")

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
    # [CALL_ID] alone would drop <|return|>; real case: reaperdoesntknow/Mini-oss-0.6b.
    model = _Model(None)
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id is None


@pytest.mark.parametrize("empty", [[], ()])
def test_an_empty_stop_set_is_left_alone(ns, harmony_tokenizer, empty):
    model = _Model(empty)
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == empty


@pytest.mark.parametrize(
    "shape",
    [
        ["<|return|>"],
        [200002, None],
        [[200002], 199999],
        True,
        [True, 199999],
    ],
)
def test_an_unparseable_stop_set_declines_instead_of_raising(ns, harmony_tokenizer, shape):
    # Runs inside from_pretrained: a raise would cost the model load.
    model = _Model(shape)
    before = model.generation_config.eos_token_id
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == before


def test_a_tuple_keeps_its_entries(ns, harmony_tokenizer):
    model = _Model((RETURN_ID, ENDOFTEXT_ID))
    ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer)
    assert model.generation_config.eos_token_id == [RETURN_ID, ENDOFTEXT_ID, CALL_ID]


def test_a_non_harmony_tokenizer_is_untouched(ns):
    qwen_like = _Tokenizer({"<|im_start|>": 151644, "<|im_end|>": 151645})
    model = _Model([151645, 151643])
    ns["patch_harmony_tool_call_eos"](model, qwen_like)
    assert model.generation_config.eos_token_id == [151645, 151643]


def test_a_tokenizer_that_folds_unknown_text_onto_unk_is_untouched(ns):
    folding = _Tokenizer({}, unk_token_id = 0)
    model = _Model([2])
    ns["patch_harmony_tool_call_eos"](model, folding)
    assert model.generation_config.eos_token_id == [2]


def test_a_tokenizer_that_answers_one_id_for_every_token_is_untouched(ns):
    degenerate = _Tokenizer({"<|call|>": 7, "<|channel|>": 7, "<|return|>": 7})
    model = _Model([2])
    ns["patch_harmony_tool_call_eos"](model, degenerate)
    assert model.generation_config.eos_token_id == [2]


def test_a_partial_harmony_vocabulary_is_untouched(ns):
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
    class _ReadOnly:
        @property
        def eos_token_id(self):
            return [RETURN_ID, ENDOFTEXT_ID]

    model = type("_M", (), {})()
    model.generation_config = _ReadOnly()
    assert ns["patch_harmony_tool_call_eos"](model, harmony_tokenizer) is model
    assert model.generation_config.eos_token_id == [RETURN_ID, ENDOFTEXT_ID]


def test_a_validating_generation_config_setter_does_not_raise(ns, harmony_tokenizer):
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


def test_patch_tokenizer_applies_the_harmony_fix():
    # A real ast.Call, not a substring (a comment naming it would pass that).
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


class _InputProcessor:
    def __init__(self, fields):
        self.generation_config_fields = fields


class _Engine:
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
    # to_diff_dict() can omit the key; vLLM keeps the primary eos separately, so seeding is safe.
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
    model = _Model([RETURN_ID, ENDOFTEXT_ID])
    assert ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer) is model
    assert not hasattr(model, "vllm_engine")


def test_an_unreachable_engine_dict_does_not_raise(ns, harmony_tokenizer):
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


def _real_sampling_params(**kwargs):
    vllm = pytest.importorskip("vllm", reason = "vLLM is not installed")
    return vllm.SamplingParams(**kwargs)


def test_the_widened_dict_really_reaches_vllms_stop_token_ids(ns, harmony_tokenizer):
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
    fields = {"eos_token_id": [RETURN_ID, ENDOFTEXT_ID]}
    model = _VLLMModel([RETURN_ID], fields)
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)

    explicit = _real_sampling_params(stop_token_ids = [42])
    explicit.update_from_generation_config(dict(fields), RETURN_ID)
    assert CALL_ID in explicit.stop_token_ids
    assert 42 in explicit.stop_token_ids


def test_ignore_eos_still_opts_out(ns, harmony_tokenizer):
    fields = {"eos_token_id": [RETURN_ID, ENDOFTEXT_ID]}
    model = _VLLMModel([RETURN_ID], fields)
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)

    opted_out = _real_sampling_params(ignore_eos = True)
    opted_out.update_from_generation_config(dict(fields), RETURN_ID)
    assert CALL_ID not in (opted_out.stop_token_ids or [])


def test_the_ordinary_terminator_survives_a_seeded_stop_set(ns, harmony_tokenizer):
    model = _VLLMModel([RETURN_ID], {})
    ns["patch_harmony_tool_call_eos_vllm"](model, harmony_tokenizer)
    fields = model.vllm_engine.llm_engine.input_processor.generation_config_fields

    params = _real_sampling_params()
    params.update_from_generation_config(dict(fields), RETURN_ID)
    assert CALL_ID in params.all_stop_token_ids
    assert RETURN_ID in params.all_stop_token_ids


_LOCAL_GPT_OSS = os.environ.get("UNSLOTH_TEST_GPT_OSS_DIR", "")


@pytest.mark.skipif(
    not (_LOCAL_GPT_OSS and os.path.isfile(os.path.join(_LOCAL_GPT_OSS, "generation_config.json"))),
    reason = "set UNSLOTH_TEST_GPT_OSS_DIR to a local unsloth/gpt-oss-* snapshot",
)
def test_the_real_shipped_generation_config_is_the_short_list(ns, harmony_tokenizer):
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

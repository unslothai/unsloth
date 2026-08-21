# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the opt-in UEmbed (Qwen3.5) instruction / chat-template input formatting.

Upstream `Qwen35Embedder.format_model_input` wraps every input in a two-message
conversation - a system message carrying the instruction ("Represent the user's input.")
and a user message carrying the content in the order video, image, text - and renders it
with `processor.apply_chat_template(..., add_generation_prompt = True, tokenize = False)`
before the processor is called. Unsloth's #2 encode path does NOT wrap inputs, so dense
parity needs this path; it must stay opt-in so existing embedders are unchanged.

Everything here is CPU-only and deterministic: the processor is a synthetic stub that
renders a minimal chat template in-process, so there is no download, no GPU and no
network.

Layers:
- Baseline characterization: the un-wrapped ("plain #2") formatting has no instruction,
  and a module that was never opted in keeps its own preprocess.
- Behavioural tests: the conversation structure and the rendered prompt are asserted
  against the reference ordering, and the processor call is asserted on its arguments.
- Structural wiring test: proves the attachment sits behind the `num_eos_tokens > 0`
  guard in `from_pretrained`, without importing unsloth (the package import needs an
  accelerator + unsloth_zoo, which CPU boxes lack).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "sentence_transformer.py"
_POOLING_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_pooling.py"

_INSTRUCTION = "Represent the user's input."


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _load_uembed_pooling():
    """Load `unsloth.models.uembed_pooling`, falling back to a direct file load.

    `import unsloth` runs the accelerator / unsloth_zoo gate, which legitimately refuses
    to import on a CPU-only machine. The module itself only needs torch, so the fallback
    executes the exact same source file rather than a stub.
    """
    try:
        from unsloth.models import uembed_pooling  # noqa: PLC0415
        return uembed_pooling
    except Exception:  # accelerator gate / missing unsloth_zoo / heavy optional deps
        pass

    name = "unsloth_uembed_pooling_direct"
    if name in sys.modules:
        return sys.modules[name]
    assert _POOLING_SOURCE_PATH.exists(), f"missing module file: {_POOLING_SOURCE_PATH}"
    spec = importlib.util.spec_from_file_location(name, _POOLING_SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def uembed():
    return _load_uembed_pooling()


class _StubImage:
    """Stands in for a PIL image: identity matters, contents do not."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"_StubImage({self.name!r})"


class _StubProcessor:
    """A minimal chat-template renderer + processor, recording every call it receives.

    `apply_chat_template` renders one line per content chunk so the ORDER of the chunks is
    observable in the output string; `__call__` records the keyword arguments instead of
    producing tensors, so nothing here needs a model or an accelerator.
    """

    def __init__(self) -> None:
        self.chat_template_calls: list[dict] = []
        self.processor_calls: list[dict] = []

    def apply_chat_template(
        self,
        conversations,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        self.chat_template_calls.append(
            {
                "conversations": conversations,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "kwargs": kwargs,
            }
        )
        rendered = []
        for conversation in conversations:
            lines = []
            for message in conversation:
                lines.append(f"<|im_start|>{message['role']}")
                for chunk in message["content"]:
                    kind = chunk["type"]
                    lines.append(chunk["text"] if kind == "text" else f"<|{kind}_pad|>")
                lines.append("<|im_end|>")
            if add_generation_prompt:
                lines.append("<|im_start|>assistant")
            rendered.append("\n".join(lines))
        return rendered

    def __call__(self, **kwargs):
        self.processor_calls.append(kwargs)
        return {"input_ids": [[0]] * len(kwargs.get("text", []))}


class _StubTransformerModule:
    """Stand-in for sentence-transformers' Transformer module (preprocess + processor)."""

    def __init__(
        self,
        processor,
        max_seq_length: int | None = None,
    ) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.plain_calls: list[list] = []

    def preprocess(self, inputs, **kwargs):
        self.plain_calls.append(inputs)
        return {"plain": inputs}


def _plain_message(model_input: dict) -> list[dict]:
    """Independent oracle for the un-wrapped ("plain #2") sentence-transformers format.

    Mirrors `sentence_transformers.base.modality.InputFormatter.to_message`: one user
    message whose content follows the caller's own key order, and no system message.
    """
    return [
        {
            "role": "user",
            "content": [{"type": key, key: value} for key, value in model_input.items()],
        }
    ]


def _content_types(conversation: list[dict]) -> list[str]:
    user = [message for message in conversation if message["role"] == "user"]
    assert len(user) == 1, f"expected exactly one user message, got {conversation}"
    return [chunk["type"] for chunk in user[0]["content"]]


def _st_source_tree() -> ast.Module:
    return ast.parse(_ST_SOURCE_PATH.read_text(encoding = "utf-8"))


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_ST_SOURCE_PATH}")


def _calls_to(node: ast.AST, func_name: str) -> list[ast.Call]:
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == func_name:
                calls.append(child)
    return calls


# --------------------------------------------------------------------------------------
# baseline characterization -- the un-wrapped formatting these tests must not change
# --------------------------------------------------------------------------------------
def test_baseline_plain_format_carries_no_instruction():
    """The plain #2 conversation is a bare user message: no system role, no instruction."""
    processor = _StubProcessor()
    conversation = _plain_message({"text": "hello"})

    (rendered,) = processor.apply_chat_template([conversation], tokenize = False)

    assert "system" not in rendered
    assert _INSTRUCTION not in rendered
    assert [message["role"] for message in conversation] == ["user"]


def test_baseline_module_without_optin_keeps_its_own_preprocess():
    """A module nobody opted in stays exactly as sentence-transformers built it."""
    module = _StubTransformerModule(_StubProcessor())
    original = module.preprocess

    assert module.preprocess(["hello"]) == {"plain": ["hello"]}
    # Bound methods are rebuilt per access, so compare by equality (same func + instance).
    assert module.preprocess == original
    assert module.processor.chat_template_calls == []


# --------------------------------------------------------------------------------------
# behaviour -- the reference conversation structure
# --------------------------------------------------------------------------------------
def test_text_input_is_wrapped_in_the_reference_conversation(uembed):
    conversation = uembed.build_uembed_conversation({"text": "hello"})

    assert conversation == [
        {"role": "system", "content": [{"type": "text", "text": _INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]


def test_bare_string_input_is_treated_as_text(uembed):
    assert uembed.build_uembed_conversation("hello") == uembed.build_uembed_conversation(
        {"text": "hello"}
    )


def test_image_and_text_follow_the_reference_order(uembed):
    """Upstream emits image before text regardless of the caller's dict order."""
    image = _StubImage("a")

    text_first = uembed.build_uembed_conversation({"text": "hello", "image": image})
    image_first = uembed.build_uembed_conversation({"image": image, "text": "hello"})

    assert _content_types(text_first) == ["image", "text"]
    assert text_first == image_first
    user_content = text_first[1]["content"]
    assert user_content[0] == {"type": "image", "image": image}
    assert user_content[1] == {"type": "text", "text": "hello"}


def test_video_comes_before_image_and_text(uembed):
    conversation = uembed.build_uembed_conversation(
        {"text": "hello", "image": _StubImage("a"), "video": _StubImage("v")}
    )

    assert _content_types(conversation) == ["video", "image", "text"]


def test_instruction_can_be_overridden(uembed):
    conversation = uembed.build_uembed_conversation(
        {"text": "hello"}, instruction = "Encode the query."
    )

    assert conversation[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "Encode the query."}],
    }


# --------------------------------------------------------------------------------------
# behaviour -- malformed / empty input falls back to NULL
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_input",
    [None, {}, "", {"text": ""}, {"text": None}, {"image": None, "text": None}],
    ids = ["none", "empty-dict", "empty-str", "empty-text", "none-text", "all-none"],
)
def test_empty_input_falls_back_to_a_single_null_text_chunk(uembed, model_input):
    conversation = uembed.build_uembed_conversation(model_input)

    assert conversation[1]["content"] == [{"type": "text", "text": "NULL"}]


def test_a_pil_shaped_object_is_treated_as_an_image(uembed):
    class _PILLike:
        mode = "RGB"
        size = (2, 2)

    image = _PILLike()

    conversation = uembed.build_uembed_conversation(image)

    assert conversation[1]["content"] == [{"type": "image", "image": image}]


def test_an_unsupported_input_type_is_refused(uembed):
    with pytest.raises(ValueError, match = "UEmbed input formatting"):
        uembed.build_uembed_conversation(object())


def test_image_only_input_keeps_the_image_and_adds_no_null(uembed):
    image = _StubImage("a")

    conversation = uembed.build_uembed_conversation({"image": image})

    assert conversation[1]["content"] == [{"type": "image", "image": image}]


# --------------------------------------------------------------------------------------
# behaviour -- rendering through the processor's chat template
# --------------------------------------------------------------------------------------
def test_rendered_prompt_contains_the_instruction_then_the_content(uembed):
    processor = _StubProcessor()

    (rendered,) = uembed.render_uembed_prompts(processor, [{"text": "hello"}])

    assert _INSTRUCTION in rendered
    assert rendered.index(_INSTRUCTION) < rendered.index("hello")
    assert "system" in rendered


def test_rendered_prompt_orders_image_before_text(uembed):
    processor = _StubProcessor()

    (rendered,) = uembed.render_uembed_prompts(
        processor, [{"text": "hello", "image": _StubImage("a")}]
    )

    assert _INSTRUCTION in rendered
    assert rendered.index("<|image_pad|>") < rendered.index("hello")
    assert rendered.index(_INSTRUCTION) < rendered.index("<|image_pad|>")


def test_chat_template_is_asked_for_a_generation_prompt_without_tokenizing(uembed):
    processor = _StubProcessor()

    uembed.render_uembed_prompts(processor, [{"text": "hello"}, {"text": "world"}])

    assert len(processor.chat_template_calls) == 1, "one batched apply_chat_template call"
    call = processor.chat_template_calls[0]
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False
    assert len(call["conversations"]) == 2


def test_rendering_is_deterministic(uembed):
    inputs = [{"text": "hello"}, {"image": _StubImage("a"), "text": "hi"}, {}]

    first = uembed.render_uembed_prompts(_StubProcessor(), inputs)
    second = uembed.render_uembed_prompts(_StubProcessor(), inputs)

    assert first == second


def test_a_string_returning_chat_template_is_normalized_to_a_list(uembed):
    class _SingleStringProcessor(_StubProcessor):
        def apply_chat_template(self, conversations, **kwargs):
            return super().apply_chat_template(conversations, **kwargs)[0]

    rendered = uembed.render_uembed_prompts(_SingleStringProcessor(), [{"text": "hello"}])

    assert isinstance(rendered, list) and len(rendered) == 1
    assert _INSTRUCTION in rendered[0]


# --------------------------------------------------------------------------------------
# behaviour -- the processor call built from the wrapped inputs
# --------------------------------------------------------------------------------------
def test_preprocess_sends_rendered_text_and_images_to_the_processor(uembed):
    processor = _StubProcessor()
    image = _StubImage("a")

    uembed.uembed_preprocess_inputs(processor, [{"image": image, "text": "hello"}])

    assert len(processor.processor_calls) == 1
    call = processor.processor_calls[0]
    assert len(call["text"]) == 1 and _INSTRUCTION in call["text"][0]
    assert call["images"] == [image]
    assert "videos" not in call, "no video input must not send an empty videos list"
    assert call["padding"] is True
    assert call["return_tensors"] == "pt"


def test_preprocess_omits_images_for_text_only_inputs(uembed):
    processor = _StubProcessor()

    uembed.uembed_preprocess_inputs(processor, [{"text": "hello"}])

    call = processor.processor_calls[0]
    assert "images" not in call and "videos" not in call


def test_preprocess_forwards_extra_processor_kwargs(uembed):
    processor = _StubProcessor()

    uembed.uembed_preprocess_inputs(processor, [{"text": "hello"}], truncation = True, max_length = 32)

    call = processor.processor_calls[0]
    assert call["truncation"] is True and call["max_length"] == 32


# --------------------------------------------------------------------------------------
# behaviour -- attaching the path to a module (opt-in) and leaving it off (default)
# --------------------------------------------------------------------------------------
def test_attaching_replaces_preprocess_with_the_instruction_path(uembed):
    module = _StubTransformerModule(_StubProcessor())

    uembed.attach_uembed_input_format(module)
    module.preprocess([{"text": "hello"}])

    assert module.plain_calls == [], "the un-wrapped path must not run once opted in"
    call = module.processor.processor_calls[0]
    assert _INSTRUCTION in call["text"][0]


def test_attaching_threads_the_modules_max_seq_length(uembed):
    module = _StubTransformerModule(_StubProcessor(), max_seq_length = 24)

    uembed.attach_uembed_input_format(module)
    module.preprocess([{"text": "hello"}])

    call = module.processor.processor_calls[0]
    assert call["truncation"] is True and call["max_length"] == 24


def test_attaching_is_idempotent(uembed):
    module = _StubTransformerModule(_StubProcessor())

    assert uembed.attach_uembed_input_format(module) is True
    patched = module.preprocess
    assert uembed.attach_uembed_input_format(module) is False
    assert module.preprocess is patched


def test_attaching_keeps_the_original_preprocess_reachable(uembed):
    module = _StubTransformerModule(_StubProcessor())
    original = module.preprocess

    uembed.attach_uembed_input_format(module)

    assert module._unsloth_uembed_original_preprocess == original
    assert original([{"text": "hello"}]) == {"plain": [{"text": "hello"}]}


def test_attaching_refuses_a_processor_without_a_chat_template(uembed):
    class _NoTemplate:
        def __call__(self, **kwargs):
            return {}

    module = _StubTransformerModule(_NoTemplate())

    with pytest.raises(ValueError, match = "chat template"):
        uembed.attach_uembed_input_format(module)


def test_a_single_input_is_accepted_as_well_as_a_list(uembed):
    processor = _StubProcessor()

    uembed.uembed_preprocess_inputs(processor, {"text": "hello"})

    assert len(processor.processor_calls[0]["text"]) == 1


# --------------------------------------------------------------------------------------
# structural wiring -- the path is opt-in inside from_pretrained
# --------------------------------------------------------------------------------------
def test_input_formatting_is_attached_only_when_num_eos_tokens_is_positive():
    from_pretrained = _function_def(_st_source_tree(), "from_pretrained")

    calls = _calls_to(from_pretrained, "attach_uembed_input_format")
    assert len(calls) == 1, "expected exactly one input-format attachment"

    guarded = [
        node
        for node in ast.walk(from_pretrained)
        if isinstance(node, ast.If)
        and _calls_to(node, "attach_uembed_input_format")
        and any(
            isinstance(compare, ast.Compare)
            and isinstance(compare.left, ast.Name)
            and compare.left.id == "num_eos_tokens"
            and isinstance(compare.ops[0], ast.Gt)
            and isinstance(compare.comparators[0], ast.Constant)
            and compare.comparators[0].value == 0
            for compare in ast.walk(node.test)
        )
    ]
    assert len(guarded) == 1, "attachment must sit behind `num_eos_tokens > 0`"


def test_eos_post_processor_wiring_is_still_in_place():
    """T2's attachment must survive: the EOS block is what the offset pooling counts back from."""
    from_pretrained = _function_def(_st_source_tree(), "from_pretrained")

    assert len(_calls_to(from_pretrained, "build_eos_post_processor")) == 1
    assert _calls_to(from_pretrained, "read_num_eos_tokens")

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

"""`num_eos_tokens` handling for UEmbed (Qwen3.5) style embedding checkpoints.

UEmbed appends `num_eos_tokens` `<|endoftext|>` tokens after the content and reads the
dense vector from the hidden state that *precedes* that block, so plain last-token
pooling would pool an EOS filler position instead of the sentence representation:

    last_index = (attention_mask.cumsum(dim = 1) * attention_mask).argmax(dim = 1)
    target     = last_index - num_eos_tokens
    dense      = hidden_state[batch_arange, target]

`num_eos_tokens` is read from the checkpoint's `sparse_info.json`; when that file is
absent it defaults to 0, which makes this module behave exactly like sentence-
transformers' `lasttoken` pooling. The module is opt-in: `_load_modules` only selects it
for the pooling modes in `OFFSET_POOLING_MODES`, every other mode keeps stock `Pooling`.

The same number drives the tokenizer: `build_eos_post_processor` attaches a `tokenizers`
post-processor that emits the trailing `<|endoftext|>` block which the pooling above then
skips. Both sides are opt-in and are no-ops at `num_eos_tokens = 0`.

The same checkpoints also wrap every input in an instruction conversation before
tokenization -- system instruction "Represent the user's input." plus a user message
carrying `video, image, text` -- rendered through the processor's chat template
(`build_uembed_conversation` / `attach_uembed_input_format`). That path is opt-in behind
the same `num_eos_tokens > 0` signal, so existing embedders keep their formatting.

Depends on torch only (`tokenizers` is imported lazily), so it stays importable without an
accelerator.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch


# Sidecar file UEmbed ships next to the weights, and the key holding the EOS block size.
SPARSE_INFO_FILENAME = "sparse_info.json"
NUM_EOS_TOKENS_KEY = "num_eos_tokens"

# The token UEmbed repeats `num_eos_tokens` times after the content.
EOS_TOKEN = "<|endoftext|>"

# Pooling modes that select this module instead of sentence-transformers' `Pooling`.
# Deliberately disjoint from the stock modes so existing embedders are untouched.
OFFSET_POOLING_MODES = frozenset({"offset_lasttoken"})


def is_offset_pooling_mode(pooling_mode: Any) -> bool:
    """True when the caller explicitly asked for UEmbed-style offset pooling."""
    return isinstance(pooling_mode, str) and pooling_mode in OFFSET_POOLING_MODES


def _local_sparse_info_path(model_name_or_path: str) -> str | None:
    if not isinstance(model_name_or_path, str) or not os.path.isdir(model_name_or_path):
        return None
    candidate = os.path.join(model_name_or_path, SPARSE_INFO_FILENAME)
    return candidate if os.path.isfile(candidate) else None


def _hub_sparse_info_path(
    model_name_or_path: str,
    token: str | bool | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> str | None:
    """Download `sparse_info.json` from the Hub, or None when the repo has no such file.

    A missing file is the documented "not a UEmbed checkpoint" case and stays silent.
    Anything else (network / auth failure) is reported so a misconfigured run does not
    quietly fall back to `num_eos_tokens = 0` and pool the wrong position.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
        try:
            from huggingface_hub.errors import RemoteEntryNotFoundError
        except ImportError:
            # huggingface_hub < 1 uses EntryNotFoundError for remote 404s. Its
            # LocalEntryNotFoundError subclass is excluded explicitly below.
            RemoteEntryNotFoundError = EntryNotFoundError
    except ImportError as exception:
        raise RuntimeError(
            f"Unsloth: cannot resolve optional `{SPARSE_INFO_FILENAME}` for "
            f"`{model_name_or_path}` because `huggingface_hub` is unavailable. Install "
            f"`huggingface_hub` or load from a complete local checkpoint."
        ) from exception

    try:
        return hf_hub_download(
            model_name_or_path,
            SPARSE_INFO_FILENAME,
            token = token,
            cache_dir = cache_dir,
            revision = revision,
        )
    except Exception as exception:
        if isinstance(exception, RemoteEntryNotFoundError) and not isinstance(
            exception, LocalEntryNotFoundError
        ):
            return None
        raise RuntimeError(
            f"Unsloth: failed to resolve `{SPARSE_INFO_FILENAME}` for "
            f"`{model_name_or_path}`. Only a confirmed missing file is treated as an "
            f"optional dense-only checkpoint; this failure may be caused by the network, "
            f"an invalid repository/revision, authentication, or gated-repository access. "
            f"Check connectivity and pass a token with access. Original error: {exception}"
        ) from exception


def read_num_eos_tokens(
    model_name_or_path: str,
    token: str | bool | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> int:
    """Read `num_eos_tokens` from the checkpoint's `sparse_info.json`, else 0.

    0 is the "no EOS block" default and makes offset pooling identical to `lasttoken`.
    A present-but-malformed value is an error: silently pooling the wrong position would
    corrupt every embedding the model produces.
    """
    path = _local_sparse_info_path(model_name_or_path)
    if path is None:
        # A real local directory without the optional sidecar is a confirmed absence, not
        # a malformed Hub repo id. Absolute nonexistent paths retain the same local
        # checkpoint semantics for callers probing a not-yet-created directory.
        is_local = isinstance(model_name_or_path, str) and (
            os.path.isdir(model_name_or_path) or os.path.isabs(model_name_or_path)
        )
        if is_local:
            return 0
        path = _hub_sparse_info_path(
            model_name_or_path, token = token, cache_dir = cache_dir, revision = revision
        )
    if path is None:
        return 0

    with open(path, encoding = "utf-8") as file:
        sparse_info = json.load(file)

    if not isinstance(sparse_info, dict) or NUM_EOS_TOKENS_KEY not in sparse_info:
        return 0

    value = sparse_info[NUM_EOS_TOKENS_KEY]
    _validate_num_eos_tokens(value, source = path)
    return int(value)


def _validate_num_eos_tokens(value: Any, source: str | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        location = f" in {source}" if source else ""
        raise ValueError(
            f"Unsloth: `{NUM_EOS_TOKENS_KEY}`{location} must be a non-negative integer, "
            f"got {value!r}."
        )


def _resolve_fast_tokenizer(tokenizer: Any) -> Any:
    """Unwrap a processor to the tokenizer that actually encodes text."""
    inner = getattr(tokenizer, "tokenizer", None)
    return tokenizer if inner is None else inner


def _backend_tokenizer(tokenizer: Any) -> Any:
    """The `tokenizers.Tokenizer` behind a fast tokenizer; slow tokenizers have none."""
    for attribute in ("_tokenizer", "backend_tokenizer"):
        backend = getattr(tokenizer, attribute, None)
        if backend is not None and hasattr(backend, "post_processor"):
            return backend
    raise ValueError(
        f"Unsloth: {type(tokenizer).__name__} has no fast (`tokenizers`) backend, so the "
        f"UEmbed `{EOS_TOKEN}` block cannot be appended. Load the checkpoint with a fast "
        f"tokenizer / processor."
    )


def _eos_token_id(tokenizer: Any, eos_token: str) -> int:
    """The vocabulary id of `eos_token`, refusing an unknown-token fallback."""
    token_id = tokenizer.convert_tokens_to_ids(eos_token)
    maps_to_unknown = token_id == getattr(tokenizer, "unk_token_id", None) and eos_token != getattr(
        tokenizer, "unk_token", None
    )
    if token_id is None or maps_to_unknown:
        raise ValueError(
            f"Unsloth: `{eos_token}` is not in this tokenizer's vocabulary (id {token_id!r}), so "
            f"the UEmbed EOS block would be appended as unknown tokens. This checkpoint does not "
            f"look like a UEmbed model."
        )
    return int(token_id)


def build_eos_post_processor(
    tokenizer: Any,
    num_eos_tokens: int,
    eos_token: str = EOS_TOKEN,
) -> Any:
    """Make `tokenizer` append `num_eos_tokens` x `eos_token` after every encoded input.

    Mirrors upstream `Qwen35Embedder.update_processor()`: a `TemplateProcessing`
    post-processor appends the block to a single sequence and to both halves of a pair,
    and padding moves to the right so the block stays at the end of the real tokens --
    that is the position `offset_last_token_pool` counts back from.

    Opt-in: `num_eos_tokens = 0` (i.e. every checkpoint without a `sparse_info.json`)
    attaches nothing and leaves the tokenizer exactly as it was.

    Args:
        tokenizer: a fast tokenizer, or a processor owning one as `.tokenizer`.
        num_eos_tokens: size of the EOS block; 0 disables the post-processor entirely.
        eos_token: the token to repeat, `<|endoftext|>` for UEmbed.

    Returns:
        The attached `TemplateProcessing`, or None when nothing was attached.

    Raises:
        ValueError: if `num_eos_tokens` is not a non-negative integer, if the tokenizer
            has no fast backend, or if `eos_token` is missing from the vocabulary.
    """
    _validate_num_eos_tokens(num_eos_tokens)
    if num_eos_tokens == 0:
        return None

    from tokenizers.processors import TemplateProcessing

    fast_tokenizer = _resolve_fast_tokenizer(tokenizer)
    backend = _backend_tokenizer(fast_tokenizer)
    eos_token_id = _eos_token_id(fast_tokenizer, eos_token)

    block = " ".join([eos_token] * num_eos_tokens)
    template = TemplateProcessing(
        single = f"$A {block}",
        pair = f"$A {block} $B {block}",
        special_tokens = [(eos_token, eos_token_id)],
    )
    backend.post_processor = template
    fast_tokenizer.padding_side = "right"
    return template


# The instruction the reference wraps every input in, the placeholder it emits when an
# input carries no content at all, and the order it emits the content chunks in.
DEFAULT_INSTRUCTION = "Represent the user's input."
NULL_CONTENT_TEXT = "NULL"
CONTENT_ORDER = ("video", "image", "text")


def _content_value(field: str, value: Any) -> Any | None:
    """The chunk value for `field`, or None when the caller supplied nothing usable."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _input_fields(model_input: Any) -> dict[str, Any]:
    """Normalise one `encode()` item into `{modality: value}`.

    A bare string is text and a PIL-shaped object is an image, matching what the existing
    embedding path already accepts. Anything else is refused rather than guessed at:
    silently filing a video tensor under `image` would corrupt every embedding.
    """
    if model_input is None:
        return {}
    if isinstance(model_input, str):
        return {"text": model_input}
    if isinstance(model_input, dict):
        return dict(model_input)
    if hasattr(model_input, "mode") and hasattr(model_input, "size"):  # PIL.Image.Image
        return {"image": model_input}
    raise ValueError(
        f"Unsloth: UEmbed input formatting expects a string, a PIL image, or a dict of "
        f"{list(CONTENT_ORDER)} entries, got {type(model_input).__name__}. Pass e.g. "
        f"`{{'image': image, 'text': 'a caption'}}`."
    )


def _as_input_list(model_inputs: Any) -> list[Any]:
    """A batch of one when the caller passed a single item (a dict is one item, not many)."""
    if isinstance(model_inputs, (list, tuple)):
        return list(model_inputs)
    return [model_inputs]


def build_uembed_conversation(
    model_input: Any, instruction: str | None = None
) -> list[dict[str, Any]]:
    """Wrap one input in the reference instruction conversation.

    Mirrors upstream `Qwen35Embedder.format_model_input`: a system message carrying the
    instruction, then a user message carrying the content as `video, image, text` -- the
    reference order, not the caller's dict order, because the rendered prompt is what the
    model was trained on. An input with no usable content becomes a single `NULL` text
    chunk so the conversation is never empty.

    Args:
        model_input: a dict of `{"text": ..., "image": ..., "video": ...}` entries, a bare
            string (text), a PIL image, or None.
        instruction: overrides `DEFAULT_INSTRUCTION` for this input.

    Returns:
        `[{"role": "system", ...}, {"role": "user", ...}]` in the reference content order.
    """
    fields = _input_fields(model_input)
    content = []
    for field in CONTENT_ORDER:
        value = _content_value(field, fields.get(field))
        if value is not None:
            content.append({"type": field, field: value})
    if not content:
        content = [{"type": "text", "text": NULL_CONTENT_TEXT}]

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": DEFAULT_INSTRUCTION if instruction is None else instruction,
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def _render_conversations(processor: Any, conversations: list[list[dict[str, Any]]]) -> list[str]:
    """Render whole conversations to prompt strings via the processor's chat template."""
    rendered = processor.apply_chat_template(
        conversations,
        add_generation_prompt = True,
        tokenize = False,
    )
    if isinstance(rendered, str):
        # Some processors collapse a single conversation to one string.
        return [rendered]
    return list(rendered)


def render_uembed_prompts(
    processor: Any,
    model_inputs: Any,
    instruction: str | None = None,
) -> list[str]:
    """The instruction-wrapped prompt string for every input, ready for tokenization."""
    conversations = [
        build_uembed_conversation(model_input, instruction)
        for model_input in _as_input_list(model_inputs)
    ]
    return _render_conversations(processor, conversations)


def _collect_vision_inputs(
    conversations: list[list[dict[str, Any]]],
) -> tuple[list[Any], list[Any]]:
    """The images and videos referenced by `conversations`, in prompt order."""
    images, videos = [], []
    for conversation in conversations:
        for message in conversation:
            for chunk in message["content"]:
                if chunk["type"] == "image":
                    images.append(chunk["image"])
                elif chunk["type"] == "video":
                    videos.append(chunk["video"])
    return images, videos


def uembed_preprocess_inputs(
    processor: Any,
    model_inputs: Any,
    instruction: str | None = None,
    **processor_kwargs: Any,
) -> Any:
    """Tokenize `model_inputs` the way the reference does: wrap, render, then process.

    Mirrors upstream `Qwen35Embedder._preprocess_inputs`: the rendered prompts go to the
    processor as `text`, with the images / videos they reference alongside. Modality
    arguments are omitted entirely when the batch has none, since processors reject an
    empty list for a modality they were not given.

    Args:
        processor: the multimodal processor (must own a chat template).
        model_inputs: one input or a list of them (see `build_uembed_conversation`).
        instruction: overrides `DEFAULT_INSTRUCTION` for the whole batch.
        **processor_kwargs: merged over `padding` / `return_tensors` on the processor call.

    Returns:
        Whatever the processor returns (a `BatchFeature` of model-ready tensors).
    """
    conversations = [
        build_uembed_conversation(model_input, instruction)
        for model_input in _as_input_list(model_inputs)
    ]
    texts = _render_conversations(processor, conversations)
    images, videos = _collect_vision_inputs(conversations)

    call_kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
    if images:
        call_kwargs["images"] = images
    if videos:
        call_kwargs["videos"] = videos
    call_kwargs.update(processor_kwargs)
    return processor(**call_kwargs)


def _module_processor(transformer_module: Any) -> Any:
    """The processor a sentence-transformers Transformer module encodes with."""
    processor = getattr(transformer_module, "processor", None)
    if processor is None:
        processor = getattr(transformer_module, "tokenizer", None)
    if processor is None:
        raise ValueError(
            f"Unsloth: {type(transformer_module).__name__} exposes neither `processor` nor "
            f"`tokenizer`, so UEmbed input formatting has nothing to tokenize with."
        )
    return processor


def attach_uembed_input_format(transformer_module: Any, instruction: str | None = None) -> bool:
    """Make `transformer_module` wrap every input in the reference instruction conversation.

    Replaces the module's input-preparation method (`preprocess`, or `tokenize` on older
    sentence-transformers) with the reference path, because the stock one applies the
    processor's chat template itself -- letting it also see a pre-rendered prompt would
    wrap the input twice. The original method stays reachable as
    `_unsloth_uembed_original_preprocess`.

    Opt-in: only `from_pretrained` calls this, and only for a checkpoint that asks for the
    trailing EOS block (`num_eos_tokens > 0`). Every other embedder keeps the stock path.

    Args:
        transformer_module: the first module of the SentenceTransformer.
        instruction: overrides `DEFAULT_INSTRUCTION` for this model.

    Returns:
        True when the path was attached, False when it already was (idempotent).

    Raises:
        ValueError: if the module has no input-preparation method, or its processor has no
            chat template to render the conversation with.
    """
    if getattr(transformer_module, "_unsloth_uembed_input_format", False):
        return False

    method_name = "preprocess" if hasattr(transformer_module, "preprocess") else "tokenize"
    original = getattr(transformer_module, method_name, None)
    if original is None:
        raise ValueError(
            f"Unsloth: {type(transformer_module).__name__} has neither `preprocess` nor "
            f"`tokenize`, so UEmbed input formatting cannot be attached to it."
        )

    processor = _module_processor(transformer_module)
    if not hasattr(processor, "apply_chat_template"):
        raise ValueError(
            f"Unsloth: {type(processor).__name__} has no chat template "
            f"(`apply_chat_template`), so the UEmbed instruction conversation cannot be "
            f"rendered. This checkpoint does not look like a UEmbed model."
        )

    def preprocess_with_instruction(
        inputs: Any,
        prompt: Any = None,
        **kwargs: Any,
    ) -> Any:
        if prompt is not None:
            # The instruction IS this path's system prompt; accepting a second one would
            # silently drop one of the two.
            raise ValueError(
                "Unsloth: this UEmbed embedder already wraps inputs in its own system "
                "instruction, so `prompt` cannot be used. Pass `instruction` to "
                "`attach_uembed_input_format` instead."
            )
        # Remaining kwargs are sentence-transformers plumbing (e.g. `processing_kwargs`),
        # not processor arguments; the reference call below owns the processor kwargs.
        extra_kwargs: dict[str, Any] = {}
        max_length = getattr(transformer_module, "max_seq_length", None)
        if isinstance(max_length, int) and not isinstance(max_length, bool) and max_length > 0:
            extra_kwargs["truncation"] = True
            extra_kwargs["max_length"] = max_length
        return uembed_preprocess_inputs(
            _module_processor(transformer_module),
            inputs,
            instruction = instruction,
            **extra_kwargs,
        )

    setattr(transformer_module, method_name, preprocess_with_instruction)
    transformer_module._unsloth_uembed_original_preprocess = original
    transformer_module._unsloth_uembed_input_format = True
    return True


def offset_last_token_pool(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor, num_eos_tokens: int
) -> torch.Tensor:
    """Pool `hidden_state` at `last_non_pad_index - num_eos_tokens` for every row.

    Args:
        hidden_state: `(batch, sequence, hidden)` transformer output.
        attention_mask: `(batch, sequence)` mask, 1 for real tokens and 0 for padding.
            Left, right and interior padding all resolve to the last unmasked position.
        num_eos_tokens: size of the trailing EOS block to skip; 0 = plain last token.

    Raises:
        ValueError: if a row is entirely padding, or if `num_eos_tokens` reaches past the
            start of the sequence. Both would otherwise index backwards from the end and
            return a silently wrong vector.
    """
    _validate_num_eos_tokens(num_eos_tokens)
    if hidden_state.dim() != 3:
        raise ValueError(
            f"Unsloth: offset pooling expects a (batch, sequence, hidden) hidden state, "
            f"got shape {tuple(hidden_state.shape)}."
        )

    mask = attention_mask.to(device = hidden_state.device, dtype = torch.long)
    empty_rows = (mask.sum(dim = 1) == 0).nonzero(as_tuple = False).flatten()
    if empty_rows.numel():
        raise ValueError(
            f"Unsloth: attention_mask has no unmasked position for batch row(s) "
            f"{empty_rows.tolist()}; offset pooling has nothing to pool there."
        )

    # Last unmasked position: cumsum peaks at the final real token, and multiplying by
    # the mask keeps padding from tying that peak.
    last_indices = (mask.cumsum(dim = 1) * mask).argmax(dim = 1)
    target_indices = last_indices - num_eos_tokens

    short_rows = (target_indices < 0).nonzero(as_tuple = False).flatten()
    if short_rows.numel():
        raise ValueError(
            f"Unsloth: num_eos_tokens = {num_eos_tokens} exceeds the content length of "
            f"batch row(s) {short_rows.tolist()} (last unmasked index "
            f"{last_indices[short_rows].tolist()}). Every sequence must be longer than "
            f"the trailing EOS block."
        )

    batch_indices = torch.arange(hidden_state.shape[0], device = hidden_state.device)
    return hidden_state[batch_indices, target_indices]


class OffsetLastTokenPooling(torch.nn.Module):
    """sentence-transformers pooling module for UEmbed-style dense vectors.

    Drop-in replacement for `Pooling(pooling_mode = "lasttoken")` that skips the trailing
    `num_eos_tokens` positions. With `num_eos_tokens = 0` it is exactly `lasttoken`.
    """

    config_keys = ["word_embedding_dimension", "num_eos_tokens"]

    def __init__(
        self,
        word_embedding_dimension: int,
        num_eos_tokens: int = 0,
    ) -> None:
        super().__init__()
        _validate_num_eos_tokens(num_eos_tokens)
        self.word_embedding_dimension = int(word_embedding_dimension)
        self.num_eos_tokens = int(num_eos_tokens)

    def forward(self, features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        token_embeddings = features["token_embeddings"]
        attention_mask = features.get("attention_mask")
        if attention_mask is None or attention_mask.size(-1) != token_embeddings.size(1):
            # Same fallback as sentence-transformers' Pooling: no mask means no padding.
            attention_mask = torch.ones(
                token_embeddings.shape[:-1], device = token_embeddings.device, dtype = torch.long
            )

        features["sentence_embedding"] = offset_last_token_pool(
            token_embeddings, attention_mask, self.num_eos_tokens
        )
        return features

    def get_embedding_dimension(self) -> int:
        return self.word_embedding_dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.word_embedding_dimension

    def get_config_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.config_keys}

    def save(self, output_path: str, *args: Any, **kwargs: Any) -> None:
        os.makedirs(output_path, exist_ok = True)
        with open(os.path.join(output_path, "config.json"), "w", encoding = "utf-8") as file:
            json.dump(self.get_config_dict(), file, indent = 2)

    @classmethod
    def load(cls, input_path: str, *args: Any, **kwargs: Any) -> OffsetLastTokenPooling:
        subfolder = kwargs.get("subfolder", "")
        if subfolder:
            if os.path.isdir(input_path):
                input_path = os.path.join(input_path, subfolder)
            else:
                try:
                    from sentence_transformers.util import load_dir_path
                except ImportError as exception:
                    raise RuntimeError(
                        "Unsloth: sentence-transformers is required to resolve a remote "
                        "serialized OffsetLastTokenPooling module."
                    ) from exception
                input_path = load_dir_path(
                    model_name_or_path = input_path,
                    subfolder = subfolder,
                    token = kwargs.get("token"),
                    cache_folder = kwargs.get("cache_folder"),
                    revision = kwargs.get("revision"),
                    local_files_only = kwargs.get("local_files_only", False),
                )
        with open(os.path.join(input_path, "config.json"), encoding = "utf-8") as file:
            config = json.load(file)
        return cls(**{key: config[key] for key in cls.config_keys if key in config})

    def __repr__(self) -> str:
        return f"OffsetLastTokenPooling({self.get_config_dict()})"

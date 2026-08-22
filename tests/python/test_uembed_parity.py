# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Opt-in, GPU-only parity tests for UEmbed (Qwen3.5) embedding support.

The dense test asserts that ``FastSentenceTransformer.encode()`` reproduces the upstream
UEmbed reference (``Qwen35Embedder(pooling = "last.normal")``) at cosine >= 0.99, plus a
collapse guard: two deliberately dissimilar inputs must not land on the same vector. The
three things under test - the instruction conversation, the trailing ``<|endoftext|>``
block, and pooling at ``last_index - num_eos_tokens`` - all fail *silently* (a plausible
but wrong vector), so only a numeric comparison against the reference catches them. The
sparse tests do the same for ``encode(output_mode = "sparse")`` against
``Qwen35Embedder(pooling = "splade.last")`` and ``"splade.max"``: a head read off the
wrong EOS slot, or a max taken over the padding, is just as silently plausible.

Gating, in this order, so a CPU-only machine never pays for a heavy import:
1. ``UNSLOTH_UEMBED_PARITY_MODEL`` unset  -> skip. Nothing is imported, nothing downloads.
2. CUDA (or bf16) unavailable            -> skip. qwen3_5 is fp16-blocklisted and needs a
                                            bf16 GPU, so there is no CPU fallback to run.
Every import (torch, transformers, unsloth, PIL, numpy) lives INSIDE the test bodies, and
no weights are fetched at collection time.

Reference selection defaults to the tracked pinned-upstream adapter at
``tests/python/fixtures/uembed_reference/reference_module.py``. Set
``UNSLOTH_UEMBED_REFERENCE_MODULE`` only to override that location; set it to an empty
string to build the reference from stock ``transformers`` primitives here in the test file,
re-implementing the upstream recipe independently of the unsloth code path under test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


PARITY_MODEL_ENV = "UNSLOTH_UEMBED_PARITY_MODEL"
REFERENCE_MODULE_ENV = "UNSLOTH_UEMBED_REFERENCE_MODULE"
DEFAULT_REFERENCE_MODULE = (
    Path(__file__).with_name("fixtures") / "uembed_reference" / "reference_module.py"
)

# The token UEmbed repeats num_eos_tokens times after the content.
EOS_TOKEN = "<|endoftext|>"
# The system instruction the reference wraps every input in.
DEFAULT_INSTRUCTION = "Represent the user's input."
# Reference content order inside the user message.
CONTENT_ORDER = ("video", "image", "text")

# SPLADE pooling modes and the sidecar UEmbed ships its sparse heads in. Spelled out as
# literals instead of imported from unsloth: `parametrize` runs at COLLECTION time, which
# is before (and on a CPU box, instead of) the gate.
SPLADE_LAST = "splade.last"
SPLADE_MAX = "splade.max"
SPARSE_WEIGHTS_FILENAME = "sparse_weights.pt"
SPARSE_LM_HEADS_KEY = "sparse_lm_heads"
SPARSE_BIAS_KEY = "sparse_bias"


def _gate():
    """Resolve the parity checkpoint, or skip. Decides BEFORE importing anything heavy."""
    model_id = os.environ.get(PARITY_MODEL_ENV)
    if not model_id:
        pytest.skip(
            f"{PARITY_MODEL_ENV} not set; export it to a UEmbed checkpoint "
            f"(e.g. Alibaba-NLP/UEmbed-2B) to run the GPU parity test"
        )

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; UEmbed parity needs a bf16 GPU (qwen3_5 is bf16-only)")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA unavailable in bf16; qwen3_5 is fp16-blocklisted so there is no fallback")
    return model_id, torch


def _read_num_eos_tokens(model_id):
    """num_eos_tokens straight from the checkpoint's sparse_info.json (never hardcoded)."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, "sparse_info.json")
    with open(path, encoding = "utf-8") as file:
        sparse_info = json.load(file)
    num_eos_tokens = sparse_info.get("num_eos_tokens")
    assert isinstance(num_eos_tokens, int) and num_eos_tokens > 0, (
        f"{model_id} sparse_info.json has num_eos_tokens = {num_eos_tokens!r}; "
        f"{PARITY_MODEL_ENV} must point at a UEmbed-style checkpoint."
    )
    return num_eos_tokens


def _synthetic_images():
    """Two deterministic, network-free RGB images that are as unalike as two 112x112
    rectangles can be, so the collapse guard below is not measuring image noise."""
    from PIL import Image

    dark = Image.new("RGB", (112, 112), color = (8, 8, 24))
    dark.paste((250, 240, 30), (0, 0, 56, 56))

    light = Image.new("RGB", (112, 112), color = (245, 245, 235))
    light.paste((10, 40, 160), (56, 56, 112, 112))

    return dark, light


def _conversation(model_input):
    """Wrap one input the way upstream ``format_model_input`` does."""
    fields = {"text": model_input} if isinstance(model_input, str) else dict(model_input)
    content = [
        {"type": field, field: fields[field]}
        for field in CONTENT_ORDER
        if fields.get(field) is not None
    ]
    if not content:
        content = [{"type": "text", "text": "NULL"}]
    return [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
        {"role": "user", "content": content},
    ]


def _comparison_array(values):
    """Detach reference tensors only at the NumPy comparison boundary."""
    import numpy as np

    detach = getattr(values, "detach", None)
    if detach is not None:
        values = detach()
    cpu = getattr(values, "cpu", None)
    if cpu is not None:
        values = cpu()
    if str(getattr(values, "dtype", "")) in ("torch.bfloat16", "torch.float16"):
        values = values.float()
    return np.asarray(values, dtype = np.float32)


def _reference_module_path():
    """Configured adapter path, defaulting to the tracked pinned fixture without reading it."""
    return os.environ.get(REFERENCE_MODULE_ENV, os.fspath(DEFAULT_REFERENCE_MODULE))


def _reference_embeddings(
    model_id,
    model_inputs,
    num_eos_tokens,
    torch,
    device = "cuda",
):
    """Upstream dense ("last.normal") embeddings for ``model_inputs``.

    Uses the tracked pinned upstream ``Qwen35Embedder`` by default; an empty
    ``UNSLOTH_UEMBED_REFERENCE_MODULE`` re-implements the recipe on stock transformers.
    """
    reference_path = _reference_module_path()
    if reference_path:
        import importlib.util

        spec = importlib.util.spec_from_file_location("uembed_reference", reference_path)
        assert (
            spec is not None and spec.loader is not None
        ), f"{REFERENCE_MODULE_ENV} = {reference_path!r} is not an importable python file"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        embedder = module.Qwen35Embedder(model_id, pooling = "last.normal")
        embeddings = embedder.encode(model_inputs)
        return _comparison_array(embeddings)

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)
    # Upstream pins right padding so the EOS block stays at the end of the real tokens,
    # which is the position the offset pooling below counts back from.
    processor.tokenizer.padding_side = "right"
    model = (
        AutoModel.from_pretrained(model_id, dtype = torch.bfloat16, trust_remote_code = True)
        .to(device)
        .eval()
    )

    conversations = [_conversation(model_input) for model_input in model_inputs]
    prompts = processor.apply_chat_template(
        conversations, add_generation_prompt = True, tokenize = False
    )
    if isinstance(prompts, str):
        prompts = [prompts]
    # The trailing EOS block, appended as text instead of via a tokenizer post-processor:
    # a different mechanism than the code under test, on purpose.
    prompts = [prompt + EOS_TOKEN * num_eos_tokens for prompt in prompts]

    images = [
        chunk["image"]
        for conversation in conversations
        for message in conversation
        for chunk in message["content"]
        if chunk["type"] == "image"
    ]
    call_kwargs = {"text": prompts, "padding": True, "return_tensors": "pt"}
    if images:
        call_kwargs["images"] = images
    batch = processor(**call_kwargs)
    batch = {
        key: (value.to(device, torch.bfloat16) if value.is_floating_point() else value.to(device))
        for key, value in batch.items()
        if hasattr(value, "to")
    }

    with torch.inference_mode():
        hidden_state = model(**batch).last_hidden_state

    mask = batch["attention_mask"].to(dtype = torch.long)
    last_indices = (mask.cumsum(dim = 1) * mask).argmax(dim = 1)
    pooled = hidden_state[
        torch.arange(hidden_state.shape[0], device = hidden_state.device),
        last_indices - num_eos_tokens,
    ]
    pooled = torch.nn.functional.normalize(pooled.float(), p = 2, dim = -1)
    return _comparison_array(pooled)


def _cosines(reference, candidate):
    import numpy as np
    return (reference * candidate).sum(1) / (
        np.linalg.norm(reference, axis = 1) * np.linalg.norm(candidate, axis = 1)
    )


def test_uembed_dense_encode_parity_matches_reference():
    """Dense parity: FastSentenceTransformer must match the upstream "last.normal" vectors
    at cosine >= 0.99 on text and on image+text, and must not collapse distinct images."""
    model_id, torch = _gate()
    np = pytest.importorskip("numpy")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("transformers")
    pytest.importorskip("PIL")

    num_eos_tokens = _read_num_eos_tokens(model_id)
    image_a, image_b = _synthetic_images()
    # Each encode() call stays modality-homogeneous: a mixed batch takes a different
    # sentence-transformers route and would not test the path this feature adds.
    text_inputs = ["a photo of a cat", "the capital of France is Paris"]
    image_inputs = [
        {"image": image_a, "text": "a bright block in the corner"},
        {"image": image_b, "text": "a bright block in the corner"},
    ]

    # Reference FIRST, before unsloth is imported, so its global patches never touch it.
    reference = {
        "text": _reference_embeddings(model_id, text_inputs, num_eos_tokens, torch),
        "image+text": _reference_embeddings(model_id, image_inputs, num_eos_tokens, torch),
    }

    from unsloth import FastSentenceTransformer

    fast = FastSentenceTransformer.from_pretrained(
        model_id,
        load_in_16bit = True,
        dtype = torch.bfloat16,
        pooling_mode = "offset_lasttoken",
        processor_kwargs = {"min_pixels": 28 * 28, "max_pixels": 600 * 600},
        trust_remote_code = True,
    )

    # The offset pooling must actually be the module doing the pooling, reading the
    # checkpoint's own num_eos_tokens: plain lasttoken pooling would pool an EOS filler.
    pooling_modules = [
        module for module in fast if type(module).__name__ == "OffsetLastTokenPooling"
    ]
    assert pooling_modules, (
        f"pooling_mode='offset_lasttoken' did not install OffsetLastTokenPooling; "
        f"modules are {[type(module).__name__ for module in fast]}"
    )
    assert pooling_modules[0].num_eos_tokens == num_eos_tokens, (
        f"offset pooling uses num_eos_tokens = {pooling_modules[0].num_eos_tokens}, "
        f"checkpoint declares {num_eos_tokens}"
    )

    for label, inputs in (("text", text_inputs), ("image+text", image_inputs)):
        embeddings = np.asarray(
            fast.encode(inputs, normalize_embeddings = True, batch_size = 2), dtype = np.float32
        )
        cos = _cosines(reference[label], embeddings)
        assert (
            float(cos.min()) >= 0.99
        ), f"UEmbed dense {label} parity regressed: min cosine {float(cos.min()):.5f} < 0.99"

    # Collapse guard: same text, two very different images. Near-identical vectors mean the
    # image never reached the vision tower (or pooling landed on a constant EOS position).
    image_embeddings = np.asarray(
        fast.encode(image_inputs, normalize_embeddings = True, batch_size = 2), dtype = np.float32
    )
    pair_cos = float(
        (image_embeddings[0] * image_embeddings[1]).sum()
        / (np.linalg.norm(image_embeddings[0]) * np.linalg.norm(image_embeddings[1]))
    )
    assert pair_cos < 0.999, (
        f"Distinct images produced near-identical dense embeddings (cos {pair_cos:.5f}); "
        f"the image input is not reaching the model."
    )


def _reference_sparse_heads(model_id, torch, device):
    """The checkpoint's own SPLADE heads, read straight from ``sparse_weights.pt``.

    Loaded here with plain ``torch.load`` rather than through the unsloth loader under
    test, so a regression in that loader cannot move the reference with it.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, SPARSE_WEIGHTS_FILENAME)
    state = torch.load(path, map_location = "cpu", weights_only = True)
    heads, biases = state[SPARSE_LM_HEADS_KEY], state[SPARSE_BIAS_KEY]
    assert len(heads) == len(biases) and len(heads) > 0, (
        f"{model_id} {SPARSE_WEIGHTS_FILENAME} holds {len(heads)} head(s) and "
        f"{len(biases)} bias(es); {PARITY_MODEL_ENV} must point at a SPLADE checkpoint."
    )
    return (
        [head.to(device = device, dtype = torch.float32) for head in heads],
        [bias.to(device = device, dtype = torch.float32) for bias in biases],
    )


def _splade_reference(hidden_state, attention_mask, heads, biases, num_eos_tokens, mode, torch):
    """The upstream SPLADE recipe (paper Eq. 3-4) in float32, written out here."""
    functional = torch.nn.functional
    mask = attention_mask.to(dtype = torch.long)
    hidden_state = hidden_state.float()
    # Last unmasked position; padding is on the right, so the EOS block ends here.
    last_indices = (mask.cumsum(dim = 1) * mask).argmax(dim = 1)

    if mode == SPLADE_LAST:
        assert num_eos_tokens <= len(
            heads
        ), f"{SPLADE_LAST} needs {num_eos_tokens} heads, checkpoint ships {len(heads)}"
        rows = torch.arange(hidden_state.shape[0], device = hidden_state.device)
        logits = [
            functional.linear(
                hidden_state[rows, last_indices - ((num_eos_tokens - 1) - index)],
                heads[index],
                biases[index],
            )
            for index in range(num_eos_tokens)
        ]
        return torch.log1p(torch.relu(torch.cat(logits, dim = -1)))

    weights = torch.log1p(torch.relu(functional.linear(hidden_state, heads[0], biases[0])))
    # log1p(relu(.)) is >= 0 everywhere, so zeroing the padding and taking the max is the
    # same selection as excluding the padding from the max.
    weights = weights * mask.unsqueeze(-1).to(weights.dtype)
    return weights.max(dim = 1).values


def _reference_sparse_embeddings(
    model_id,
    texts,
    num_eos_tokens,
    torch,
    mode,
    device = "cuda",
):
    """Upstream sparse ("splade.last" / "splade.max") vectors for ``texts``.

    Same reference resolution as the dense helper: the tracked pinned upstream
    ``Qwen35Embedder`` by default, or the recipe rebuilt on stock ``transformers`` plus the
    checkpoint's own heads when the override is empty, independently of the unsloth
    code path under test. Text-only: the dense test already covers the image plumbing, so
    this stays a comparison of the sparse pooling and nothing else. The tracked adapter is
    the default; an empty ``UNSLOTH_UEMBED_REFERENCE_MODULE`` selects the stock recipe.
    """
    reference_path = _reference_module_path()
    if reference_path:
        import importlib.util

        spec = importlib.util.spec_from_file_location("uembed_reference", reference_path)
        assert (
            spec is not None and spec.loader is not None
        ), f"{REFERENCE_MODULE_ENV} = {reference_path!r} is not an importable python file"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        embedder = module.Qwen35Embedder(model_id, pooling = mode)
        embeddings = embedder.encode(texts)
        if getattr(embeddings, "is_sparse", False):
            embeddings = embeddings.to_dense()
        return _comparison_array(embeddings)

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)
    # Right padding keeps the EOS block at the end of the real tokens, which is where the
    # splade.last heads count back from.
    processor.tokenizer.padding_side = "right"
    model = (
        AutoModel.from_pretrained(model_id, dtype = torch.bfloat16, trust_remote_code = True)
        .to(device)
        .eval()
    )

    prompts = processor.apply_chat_template(
        [_conversation(text) for text in texts], add_generation_prompt = True, tokenize = False
    )
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [prompt + EOS_TOKEN * num_eos_tokens for prompt in prompts]

    batch = processor(text = prompts, padding = True, return_tensors = "pt")
    batch = {
        key: (value.to(device, torch.bfloat16) if value.is_floating_point() else value.to(device))
        for key, value in batch.items()
        if hasattr(value, "to")
    }
    with torch.inference_mode():
        hidden_state = model(**batch).last_hidden_state

    heads, biases = _reference_sparse_heads(model_id, torch, hidden_state.device)
    sparse = _splade_reference(
        hidden_state, batch["attention_mask"], heads, biases, num_eos_tokens, mode, torch
    )
    return _comparison_array(sparse)


@pytest.mark.parametrize("sparse_mode", [SPLADE_LAST, SPLADE_MAX])
def test_uembed_sparse_encode_parity_matches_reference(sparse_mode):
    """Sparse parity: ``encode(output_mode = "sparse")`` must reproduce the upstream
    SPLADE vectors at cosine >= 0.99 for both pooling modes, and must not hand two
    unrelated sentences the same sparse vector."""
    model_id, torch = _gate()
    np = pytest.importorskip("numpy")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("transformers")

    num_eos_tokens = _read_num_eos_tokens(model_id)
    # Deliberately unrelated sentences: shared vocabulary would make the collapse guard
    # below measure the topic overlap instead of the pooling.
    text_inputs = [
        "a photo of a cat sitting on a windowsill",
        "the capital of France is Paris",
    ]

    # Reference FIRST, before unsloth is imported, so its global patches never touch it.
    reference = _reference_sparse_embeddings(
        model_id, text_inputs, num_eos_tokens, torch, sparse_mode
    )

    from unsloth import FastSentenceTransformer
    from unsloth.models.uembed_wiring import require_uembed_sparse_output

    fast = FastSentenceTransformer.from_pretrained(
        model_id,
        load_in_16bit = True,
        dtype = torch.bfloat16,
        pooling_mode = "offset_lasttoken",
        processor_kwargs = {"min_pixels": 28 * 28, "max_pixels": 600 * 600},
        trust_remote_code = True,
    )

    # The sparse head has to be attached by the load itself (from the checkpoint's own
    # sparse_weights.pt); without it `output_mode = "sparse"` has nothing to pool.
    sparse_output = require_uembed_sparse_output(fast)
    assert sparse_output.num_eos_tokens == num_eos_tokens, (
        f"sparse head uses num_eos_tokens = {sparse_output.num_eos_tokens}, "
        f"checkpoint declares {num_eos_tokens}"
    )
    sparse_output.set_mode(sparse_mode)
    assert sparse_output.mode == sparse_mode

    embeddings = np.asarray(
        fast.encode(text_inputs, output_mode = "sparse", batch_size = 2), dtype = np.float32
    )
    assert embeddings.shape == reference.shape, (
        f"UEmbed {sparse_mode} produced {embeddings.shape} sparse vectors, "
        f"reference is {reference.shape}"
    )
    assert (
        float(embeddings.min()) >= 0.0
    ), f"UEmbed {sparse_mode} produced negative weights; SPLADE is log1p(relu(.))"

    cos = _cosines(reference, embeddings)
    assert (
        float(cos.min()) >= 0.99
    ), f"UEmbed {sparse_mode} parity regressed: min cosine {float(cos.min()):.5f} < 0.99"

    # Collapse guard: two unrelated sentences. Near-identical sparse vectors mean the head
    # is pooling a constant position (an EOS filler, or the padding) instead of the input.
    pair_cos = float(
        (embeddings[0] * embeddings[1]).sum()
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )
    assert pair_cos < 0.999, (
        f"Unrelated sentences produced near-identical {sparse_mode} vectors "
        f"(cos {pair_cos:.5f}); the sparse head is not reading the content."
    )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Tests for multimodal (vision) decoder embedding support in FastSentenceTransformer
(e.g. Qwen/Qwen3-VL-Embedding-2B).

Three layers:
- Static source checks that never import unsloth (so they never hit the accelerator
  gate). They compile unsloth/models/sentence_transformer.py and assert, via ast, that
  the symbols the rest of this file depends on still exist. These ALWAYS run: a syntax
  error or a renamed/removed symbol turns this file red even on a CPU-only runner.
- Fast, no-GPU unit tests for the VLM-detection helper and the get_peft_model vision
  flags. They import unsloth but never touch weights. They skip only on the one
  explicitly detected condition where unsloth legitimately refuses to import
  (NotImplementedError with no CUDA device); every other import failure fails.
- An opt-in, GPU-only end-to-end parity test (UNSLOTH_VLM_EMBEDDING_PARITY_MODEL) that
  loads a real VLM embedding checkpoint and compares against a stock SentenceTransformer.

Design note: for a VLM embedding model, FastSentenceTransformer must keep
auto_model = AutoModel (the base model, e.g. Qwen3VLModel, returns last_hidden_state
which the Pooling layer needs); AutoModelForImageTextToText maps to
*ForConditionalGeneration, which returns logits and corrupts pooling. The image path is
enabled by swapping the text tokenizer for an AutoProcessor, not by changing the model
class.
"""

from __future__ import annotations

import ast
import inspect
import os
import types
from pathlib import Path

import pytest

# unsloth/models/sentence_transformer.py, located relative to this test file (not the
# CWD) so the static checks below work from any invocation directory.
_SOURCE_PATH = (
    Path(__file__).resolve().parents[2] / "unsloth" / "models" / "sentence_transformer.py"
)


def _import_fast_sentence_transformer():
    """Import FastSentenceTransformer, skipping ONLY on the documented CPU-only refusal.

    unsloth raises NotImplementedError when it finds no supported accelerator; that is
    an environment fact and may skip. Anything else -- SyntaxError, ImportError, a
    missing symbol -- is a real regression and is re-raised so the test fails."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("unsloth_zoo")
    try:
        from unsloth import FastSentenceTransformer
    except NotImplementedError as exc:
        if torch.cuda.is_available():
            raise
        pytest.skip(f"unsloth requires an accelerator; none available: {exc}")
    return FastSentenceTransformer


def _cfg(**attrs):
    return types.SimpleNamespace(**attrs)


def _parse_source_module():
    """Compile and parse the module under test without importing it, so these checks
    run on every machine including CPU-only runners with no torch installed."""
    source = _SOURCE_PATH.read_text(encoding = "utf-8")
    # compile() raises SyntaxError on a broken module -> the test fails, never skips.
    compile(source, str(_SOURCE_PATH), "exec")
    return ast.parse(source, filename = str(_SOURCE_PATH))


def _class_node(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {_SOURCE_PATH}")


def _method_node(cls_node, name):
    for node in cls_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{cls_node.name}.{name} not found in {_SOURCE_PATH}")


def _arg_names(func_node):
    args = func_node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return set(names)


def test_source_module_compiles_and_defines_expected_symbols():
    """Always-runs regression guard: never skips, so a syntax error or a renamed symbol
    in sentence_transformer.py turns this file red even where unsloth cannot import."""
    tree = _parse_source_module()

    module_level_assignments = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            module_level_assignments.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_level_assignments.add(node.target.id)
    assert "_DEFAULT_TARGET_MODULES" in module_level_assignments

    cls = _class_node(tree, "FastSentenceTransformer")
    for method in ("_is_vlm_embedding_config", "from_pretrained", "get_peft_model"):
        _method_node(cls, method)


def test_source_get_peft_model_declares_vision_selectors():
    """Signature-level mirror of test_get_peft_model_exposes_vision_lora_flags that does
    not require importing unsloth, so renaming a selector cannot pass as a skip."""
    cls = _class_node(_parse_source_module(), "FastSentenceTransformer")
    params = _arg_names(_method_node(cls, "get_peft_model"))
    for name in (
        "finetune_vision_layers",
        "finetune_language_layers",
        "finetune_attention_modules",
        "finetune_mlp_modules",
        "finetune_last_n_layers",
    ):
        assert name in params, f"get_peft_model lost the {name} selector"

    from_pretrained_params = _arg_names(_method_node(cls, "from_pretrained"))
    assert "processor_kwargs" in from_pretrained_params


def test_is_vlm_embedding_config_detects_vision_config():
    FastSentenceTransformer = _import_fast_sentence_transformer()
    detect = FastSentenceTransformer._is_vlm_embedding_config

    # A Qwen3-VL-Embedding-style config carries a nested vision_config.
    assert detect(_cfg(model_type = "qwen3_vl", vision_config = _cfg(depth = 24))) is True
    # Architecture-name fallback (no vision_config attribute surfaced).
    assert detect(_cfg(architectures = ["Qwen3VLForConditionalGeneration"])) is True


def test_is_vlm_embedding_config_rejects_text_models():
    FastSentenceTransformer = _import_fast_sentence_transformer()
    detect = FastSentenceTransformer._is_vlm_embedding_config

    assert detect(None) is False
    # A plain decoder text embedder (e.g. Qwen3-Embedding) must NOT be treated as VLM.
    assert detect(_cfg(model_type = "qwen3", architectures = ["Qwen3Model"])) is False
    assert detect(_cfg(model_type = "bert", architectures = ["BertModel"])) is False


def test_get_peft_model_exposes_vision_lora_flags():
    FastSentenceTransformer = _import_fast_sentence_transformer()
    params = inspect.signature(FastSentenceTransformer.get_peft_model).parameters

    # Vision selectors must exist so VLM embedders can restrict LoRA to the language
    # tower (the cheap, common contrastive-tuning recipe) by default.
    assert params["finetune_vision_layers"].default is False
    assert params["finetune_language_layers"].default is True
    assert params["finetune_attention_modules"].default is True
    assert params["finetune_mlp_modules"].default is True
    assert params["finetune_last_n_layers"].default is None


def test_from_pretrained_accepts_processor_kwargs():
    FastSentenceTransformer = _import_fast_sentence_transformer()
    params = inspect.signature(FastSentenceTransformer.from_pretrained).parameters
    # min_pixels / max_pixels style image controls must be threadable to the processor.
    assert "processor_kwargs" in params
    assert params["processor_kwargs"].default is None


def _synthetic_images():
    """Two deterministic, network-free RGB images that are as visually unalike as a pair of
    112x112 rectangles can be: a dark frame with a bright top-left block, and its inverse
    with a bright frame and a dark bottom-right block. Near-identical inputs would make the
    collapse check below flaky, which is the whole reason they are this far apart."""
    from PIL import Image

    dark = Image.new("RGB", (112, 112), color = (8, 8, 24))
    dark.paste((250, 240, 30), (0, 0, 56, 56))

    light = Image.new("RGB", (112, 112), color = (245, 245, 235))
    light.paste((10, 40, 160), (56, 56, 112, 112))

    return dark, light


def test_vlm_embedding_matches_stock_st():
    """End-to-end parity: FastSentenceTransformer must match a stock SentenceTransformer
    load of the same VLM embedding checkpoint on BOTH text and image inputs, and must NOT
    swap in a logits-returning *ForConditionalGeneration model. Opt-in + GPU-only.

    The image half is the point of the feature: stock sentence-transformers encodes this
    checkpoint's images today (see its model card), so a text-only parity check would pass
    while the vision path stays unreachable -- exactly the bug this file exists to prevent.
    """
    model_id = os.environ.get("UNSLOTH_VLM_EMBEDDING_PARITY_MODEL")
    if not model_id:
        pytest.skip(
            "set UNSLOTH_VLM_EMBEDDING_PARITY_MODEL to a vision embedding model "
            "(e.g. Qwen/Qwen3-VL-Embedding-2B) to run the multimodal parity test"
        )

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("FastSentenceTransformer requires CUDA; skipping on CPU-only runner")
    np = pytest.importorskip("numpy")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("PIL")
    from sentence_transformers import SentenceTransformer

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    texts = ["a photo of a cat", "the capital of France is Paris"]
    img_a, img_b = _synthetic_images()
    # Each encode() call must be modality-homogeneous: sentence-transformers collapses a
    # mixed-modality batch to the "message" (chat-template) route, which would silently test
    # a different code path than the one this feature adds.
    image_text_inputs = [
        {"image": img_a, "text": "a bright block in the corner"},
        {"image": img_b, "text": "a bright block in the corner"},
    ]
    image_only_inputs = [{"image": img_a}, {"image": img_b}]

    # Control FIRST, before importing unsloth, so its global patches never touch the ref.
    ctrl = SentenceTransformer(model_id, device = device, model_kwargs = {"torch_dtype": dtype})
    ctrl_emb = np.asarray(
        ctrl.encode(texts, normalize_embeddings = True, batch_size = 2), dtype = np.float32
    )
    ctrl_img_text_emb = np.asarray(
        ctrl.encode(image_text_inputs, normalize_embeddings = True, batch_size = 2), dtype = np.float32
    )
    ctrl_img_emb = np.asarray(
        ctrl.encode(image_only_inputs, normalize_embeddings = True, batch_size = 2), dtype = np.float32
    )

    import unsloth  # noqa: F401
    from unsloth import FastSentenceTransformer

    fast = FastSentenceTransformer.from_pretrained(
        model_id,
        dtype = dtype,
        load_in_4bit = False,
        load_in_16bit = True,
    )

    # Must keep the base multimodal model (returns last_hidden_state), not the LM head.
    inner_name = fast[0].auto_model.__class__.__name__
    assert not inner_name.endswith("ForConditionalGeneration"), (
        f"FastSentenceTransformer loaded {inner_name}; a *ForConditionalGeneration model "
        f"returns logits and corrupts the Pooling layer. Expected the base model."
    )

    # The processor, not the tokenizer: on sentence-transformers >= 5.4 `.tokenizer`
    # unwraps to the inner text tokenizer either way, so it cannot detect the failure.
    processor = fast[0].processor
    assert hasattr(processor, "image_processor"), (
        f"FastSentenceTransformer kept {type(processor).__name__} as the processor; a text "
        f"tokenizer cannot turn images into pixel_values, so the vision tower is unreachable."
    )

    fast_emb = np.asarray(
        fast.encode(texts, normalize_embeddings = True, batch_size = 2), dtype = np.float32
    )

    cos = (ctrl_emb * fast_emb).sum(1) / (
        np.linalg.norm(ctrl_emb, axis = 1) * np.linalg.norm(fast_emb, axis = 1)
    )
    assert (
        float(cos.min()) > 0.99
    ), f"VLM embedding text parity regressed: min cosine {float(cos.min()):.5f} <= 0.99"

    def _cosines(reference, candidate):
        return (reference * candidate).sum(1) / (
            np.linalg.norm(reference, axis = 1) * np.linalg.norm(candidate, axis = 1)
        )

    for label, inputs, reference in (
        ("image+text", image_text_inputs, ctrl_img_text_emb),
        ("image", image_only_inputs, ctrl_img_emb),
    ):
        fast_img_emb = np.asarray(
            fast.encode(inputs, normalize_embeddings = True, batch_size = 2), dtype = np.float32
        )
        img_cos = _cosines(reference, fast_img_emb)
        assert (
            float(img_cos.min()) > 0.99
        ), f"VLM {label} parity regressed: min cosine {float(img_cos.min()):.5f} <= 0.99"

        # Two deliberately dissimilar images must not collapse to the same vector. With the
        # same text on both rows of the image+text batch, a collapse means only the text is
        # reaching the model, i.e. the image was dropped somewhere in preprocessing.
        pair_cos = float(
            (fast_img_emb[0] * fast_img_emb[1]).sum()
            / (np.linalg.norm(fast_img_emb[0]) * np.linalg.norm(fast_img_emb[1]))
        )
        assert pair_cos < 0.999, (
            f"Distinct images produced near-identical {label} embeddings (cos {pair_cos:.5f}); "
            f"the image input is not reaching the vision tower."
        )

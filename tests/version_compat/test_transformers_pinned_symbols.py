# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol + source-pattern transformers compat checks via GitHub raw-fetch + grep.

Catches breakage classes from unsloth#3998/5036/5155/5259 and
unsloth-zoo#572/571/549/543/541/495/491/488/472/393/388/583/584/159.
CPU-only, no install. Anchors: transformers 4.57.6 (floor), 5.17.0 (ceiling) and 5.5.0
(the old ceiling, still the Apple Silicon cap).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.version_compat._fetch import (
    fetch_text,
    first_match,
    function_params,
    has_def,
    is_bound,
)


# Read from pyproject: a hardcoded floor went stale.
_FLOOR_RE = re.compile(r"^\s*\"transformers[^\"]*?>=\s*([0-9]+(?:\.[0-9]+)*)", re.M)
_FLOOR_FALLBACK = (4, 52, 4)


def _declared_floor() -> tuple[int, ...]:
    """The oldest transformers pyproject admits; the fallback keeps a bad read from
    widening the matrix to every release ever published."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        found = _FLOOR_RE.findall(pyproject.read_text(encoding = "utf-8"))
    except OSError:
        return _FLOOR_FALLBACK
    if not found:
        return _FLOOR_FALLBACK
    return min(tuple(int(part) for part in v.split(".")) for v in found)


_FLOOR = _declared_floor()

# 4.57.6 old floor, 5.5.0 the Apple Silicon cap, 5.16.0 the tokenizers>=0.23.1 break.
_ALWAYS = ("v4.57.6", "v5.5.0", "v5.16.0", "v5.10.1", "v5.15.1")

# Exact pins notebooks use; one-tag-per-minor would evict them.

_CAP = re.compile(r"^\s*\"transformers[^\"]*?<=\s*([0-9]+(?:\.[0-9]+)*)", re.M)


def _declared_ceiling_tag() -> tuple[str, ...]:
    """The tag for the newest transformers pyproject admits, or empty if unreadable."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        found = _CAP.findall(pyproject.read_text(encoding = "utf-8"))
    except OSError:
        return ()
    if not found:
        return ()
    ceiling = max(found, key = lambda v: tuple(int(p) for p in v.split(".")))
    return (_TAG_OVERRIDES.get(ceiling, "v" + ceiling),)


# PyPI version -> tag where upstream disagrees.
_TAG_OVERRIDES = {"5.10.4": "v5.10.3", "4.54.1": "v4.54-release"}

# Outage fallback; must start at the declared floor.
_TAGS_FALLBACK = (
    "v4.52.4",
    "v4.53.3",
    "v4.54-release",
    "v4.55.4",
    "v4.56.2",
    "v4.57.6",
    "v5.0.0",
    "v5.1.0",
    "v5.2.0",
    "v5.3.0",
    "v5.4.0",
    "v5.5.4",
    "v5.6.2",
    "v5.7.0",
    "v5.8.1",
    "v5.9.0",
    "v5.10.2",
    "v5.10.3",
    "v5.11.0",
    "v5.12.1",
    "v5.13.1",
    "v5.14.1",
    "v5.15.1",
    "v5.16.1",
    "v5.17.0",
)


# Shared across xdist workers so all collect identical parameters.
_MATRIX_CACHE_ENV = "PYTEST_TRANSFORMERS_MATRIX_FILE"


def _cached_matrix() -> list[str] | None:
    """The shared matrix, or None when there is no cache or it is unreadable."""
    path = os.environ.get(_MATRIX_CACHE_ENV)
    if not path:
        return None
    try:
        tags = json.loads(Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return [str(tag) for tag in tags] if isinstance(tags, list) and tags else None


def _write_cached_matrix(tags: list[str]) -> None:
    """Publish `tags` for the other workers. Atomic, so no worker reads a partial file."""
    path = os.environ.get(_MATRIX_CACHE_ENV)
    if not path:
        return
    target = Path(path)
    try:
        target.parent.mkdir(parents = True, exist_ok = True)
        handle, temporary = tempfile.mkstemp(dir = str(target.parent), suffix = ".json")
        with os.fdopen(handle, "w", encoding = "utf-8") as stream:
            json.dump(tags, stream)
        os.replace(temporary, target)
    except OSError:
        pass


def _resolved_tags() -> list[str]:
    """`_release_tags()`, resolved once per run and shared across xdist workers.

    Every return goes through `_with_always`, the published one included: the cache agrees on the
    PyPI half of the answer, it is not a second source of truth for the anchors. A file written by
    another revision and returned verbatim dropped the floor, the old ceiling and the notebook pins
    while reporting green.
    """
    cached = _cached_matrix()
    if cached is not None:
        return _with_always(cached)
    tags = _release_tags()
    cached = _cached_matrix()
    if cached is not None:
        return _with_always(cached)
    _write_cached_matrix(tags)
    return tags


def _release_tags() -> list[str]:
    """Every transformers minor at or above the floor, latest patch of each, oldest first.

    Read from PyPI, not pinned: this suite says which versions the cap may be lifted to, and a
    hand-maintained list answers for the day it was edited. One tag per minor bounds the matrix; a
    patch that broke something earns a place in `_ALWAYS`. Yanked and rc/dev/post builds are
    skipped, since pip will not install them.
    """
    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/transformers/json",
            timeout = 20,
        ) as response:
            releases = json.loads(response.read().decode("utf-8"))["releases"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        return _with_always(_TAGS_FALLBACK)

    latest_by_minor: dict[tuple[int, int], tuple[int, ...]] = {}
    for version, files in releases.items():
        if not files or all(f.get("yanked") for f in files):
            continue
        if re.fullmatch(r"\d+(?:\.\d+){2,}", version) is None:
            continue
        parts = tuple(int(g) for g in version.split("."))
        if parts < _FLOOR:
            continue
        minor = parts[:2]
        if parts > latest_by_minor.get(minor, ()):
            latest_by_minor[minor] = parts
    if not latest_by_minor:
        return _with_always(_TAGS_FALLBACK)

    tags = []
    for parts in sorted(latest_by_minor.values()):
        name = ".".join(str(p) for p in parts)
        tags.append(_TAG_OVERRIDES.get(name, "v" + name))
    return _with_always(tags)


def _with_always(tags) -> list[str]:
    """`tags` with every `_ALWAYS` anchor present, sorted, deduplicated.

    Every return path goes through here, the fallback ones included: `_TAGS_FALLBACK` carries one
    tag per minor so it holds neither v5.5.0 nor v5.16.0, and returning it unmerged let an outage
    drop the Apple Silicon ceiling and the tokenizers breakpoint and still report green.
    """
    return sorted(set(tuple(tags) + _ALWAYS + _declared_ceiling_tag()), key = _sort_key)


_TAG_TO_RELEASE = {tag: release for release, tag in _TAG_OVERRIDES.items()}


def _sort_key(tag: str) -> tuple[int, ...]:
    name = _TAG_TO_RELEASE.get(tag, tag).lstrip("v")
    return tuple(int(g) for g in name.split("."))


TRANSFORMERS_TAGS = _resolved_tags() + ["main"]

pytestmark = pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)


def test_trainer_class_importable_path(tag: str):
    """transformers.Trainer must remain at trainer.py or trainer/__init__.py."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None, f"{tag}: src/transformers/trainer[.py|/__init__.py] both missing"
    _, src = hit
    assert has_def(src, "Trainer", "class"), f"{tag}: class Trainer missing"


def test_trainer_compute_loss_num_items_in_batch_param(tag: str):
    """unsloth-zoo#159 + unsloth#4998 + #4616: Trainer.compute_loss must accept num_items_in_batch kwarg."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    params = function_params(src, "compute_loss", cls = "Trainer")
    if params is None:
        pytest.fail(f"{tag}: Trainer.compute_loss not found in source")
    assert "num_items_in_batch" in params, (
        f"{tag}: Trainer.compute_loss signature missing num_items_in_batch param; "
        f"unsloth grad-accum patches assume this kwarg present"
    )


def test_trainer_training_step_grad_accum_pattern(tag: str):
    """unsloth#3598 patches Trainer.training_step source; drift = silent no-op = double-scale loss bug."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    needed = (
        "loss *= self.args.gradient_accumulation_steps",
        "if self.model_accepts_loss_kwargs:",
        "self.accelerator.backward(loss",
    )
    missing = [s for s in needed if s not in src]
    if len(missing) == len(needed):
        pytest.fail(
            f"{tag}: Trainer.training_step has none of the grad-accum "
            f"fingerprints {needed}; unsloth/models/_utils.py:1689-1791 "
            f"patch silently no-ops -> double-scale loss"
        )


def test_trainer_get_batch_samples_returns_num_items(tag: str):
    """unsloth-zoo loss_utils.py:241 replaces Trainer.get_batch_samples; must keep the num_items_in_batch return."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    if not has_def(src, "get_batch_samples", "func"):
        pytest.skip(f"{tag}: get_batch_samples not yet on Trainer")
    assert (
        "num_items_in_batch" in src
    ), f"{tag}: Trainer.get_batch_samples / num_items_in_batch contract missing"


def test_trainer_inner_training_loop_inplace_loss_v5(tag: str):
    """unsloth-zoo#543: transformers 5.0+ switched out-of-place tr_loss add to in-place `self._tr_loss +=`."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    has_inplace = "self._tr_loss +=" in src
    has_outplace = "tr_loss = tr_loss + tr_loss_step" in src
    assert has_inplace or has_outplace, (
        f"{tag}: Trainer._inner_training_loop has neither "
        f"`tr_loss = tr_loss + tr_loss_step` nor `self._tr_loss +=`; "
        f"unsloth-zoo#543 patch breaks"
    )


def test_modeling_utils_exposes_checkpoint(tag: str):
    """unsloth-zoo#549: transformers 5.2+ uses modeling_utils.checkpoint; patch must replace it, not just torch's."""
    src = fetch_text("huggingface/transformers", tag, "src/transformers/modeling_utils.py")
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    has_import = bool(
        re.search(
            r"^from\s+torch\.utils\.checkpoint\s+import\s+checkpoint",
            src,
            re.MULTILINE,
        )
        or re.search(r"^import\s+torch\.utils\.checkpoint", src, re.MULTILINE)
        or "checkpoint = torch.utils.checkpoint.checkpoint" in src
    )
    assert has_import, (
        f"{tag}: transformers.modeling_utils does not import / re-bind "
        f"torch.utils.checkpoint.checkpoint; unsloth-zoo#549 patch breaks"
    )


def test_pushtohubmixin_create_repo_status(tag: str):
    """unsloth-zoo#393: transformers 5.x removed PushToHubMixin._create_repo; snapshot which side."""
    src = fetch_text("huggingface/transformers", tag, "src/transformers/modeling_utils.py")
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    has_create = bool(re.search(r"def _create_repo\b", src) or "_create_repo" in src)
    _ = has_create


def test_integrations_bitsandbytes_module_present(tag: str):
    src = fetch_text(
        "huggingface/transformers", tag, "src/transformers/integrations/bitsandbytes.py"
    )
    if src is None:
        pytest.skip(f"{tag}: integrations/bitsandbytes.py missing (legacy layout)")
    assert (
        "Linear4bit" in src or "linear" in src.lower()
    ), f"{tag}: integrations/bitsandbytes.py has no Linear4bit reference"


def test_quantizers_should_convert_module_signature(tag: str):
    """unsloth-zoo#491/#488: 5.x moved is_replaceable to quantizers_utils.should_convert_module; snapshot its form."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/quantizers/quantizers_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: quantizers/quantizers_utils.py missing")
    if not has_def(src, "should_convert_module", "func"):
        pytest.skip(f"{tag}: should_convert_module not yet present (4.x)")
    has_dot_form = ".{key}." in src or "f'.{key}.'" in src or 'f".{key}."' in src
    _ = has_dot_form


def test_fp8linear_init_param_names(tag: str):
    """unsloth-zoo#572: transformers 5.x renamed FP8Linear.__init__ `bias` -> `has_bias`."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/integrations/finegrained_fp8.py",
    )
    if src is None:
        pytest.skip(f"{tag}: integrations/finegrained_fp8.py missing")
    if not has_def(src, "FP8Linear", "class"):
        pytest.skip(f"{tag}: FP8Linear not yet defined")
    params = function_params(src, "__init__", cls = "FP8Linear") or ()
    assert (
        "bias" in params or "has_bias" in params
    ), f"{tag}: FP8Linear.__init__ has neither `bias` nor `has_bias` param"


def test_processing_utils_unpack_importable(tag: str):
    """unsloth-zoo#583/584: transformers.processing_utils.Unpack must keep importing."""
    src = fetch_text("huggingface/transformers", tag, "src/transformers/processing_utils.py")
    if src is None:
        pytest.skip(f"{tag}: processing_utils.py missing")
    assert is_bound(src, "Unpack"), (
        f"{tag}: transformers.processing_utils.Unpack missing; "
        f"unsloth-zoo#583/584 import guard breaks"
    )


# NamedTuple, not dataclass: test_transformers_cap_sites.py exec_module()s this file outside sys.modules.
class ModelingContract(NamedTuple):
    name: str
    path: str
    classes: tuple[str, ...]
    rope_name: str | None = None
    extra_bindings: tuple[str, ...] = ()
    patch_site: str = ""


MODELING_CONTRACTS: tuple[ModelingContract, ...] = (
    ModelingContract(
        name = "llama",
        path = "src/transformers/models/llama/modeling_llama.py",
        classes = ("LlamaAttention", "LlamaDecoderLayer", "LlamaModel", "LlamaForCausalLM"),
        rope_name = "LlamaRotaryEmbedding",
        patch_site = "unsloth/models/llama.py:FastLlamaModel.pre_patch",
    ),
    ModelingContract(
        name = "qwen2",
        path = "src/transformers/models/qwen2/modeling_qwen2.py",
        classes = ("Qwen2Attention", "Qwen2DecoderLayer", "Qwen2Model", "Qwen2ForCausalLM"),
        rope_name = "Qwen2RotaryEmbedding",
        patch_site = "unsloth/models/qwen2.py:FastQwen2Model.pre_patch",
    ),
    ModelingContract(
        name = "qwen3",
        path = "src/transformers/models/qwen3/modeling_qwen3.py",
        classes = ("Qwen3Attention", "Qwen3DecoderLayer", "Qwen3Model", "Qwen3ForCausalLM"),
        rope_name = "Qwen3RotaryEmbedding",
        patch_site = "unsloth/models/qwen3.py:FastQwen3Model.pre_patch",
    ),
    ModelingContract(
        name = "qwen3_moe",
        path = "src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
        classes = (
            "Qwen3MoeAttention",
            "Qwen3MoeSparseMoeBlock",
            "Qwen3MoeMLP",
            "Qwen3MoeDecoderLayer",
            "Qwen3MoeModel",
            "Qwen3MoeForCausalLM",
        ),
        rope_name = "Qwen3MoeRotaryEmbedding",
        patch_site = "unsloth/models/qwen3_moe.py:FastQwen3MoeModel.pre_patch",
    ),
    ModelingContract(
        name = "gemma",
        path = "src/transformers/models/gemma/modeling_gemma.py",
        classes = ("GemmaAttention", "GemmaDecoderLayer", "GemmaModel", "GemmaForCausalLM"),
        rope_name = "GemmaRotaryEmbedding",
        patch_site = "unsloth/models/gemma.py:FastGemmaModel.pre_patch",
    ),
    ModelingContract(
        name = "gemma2",
        path = "src/transformers/models/gemma2/modeling_gemma2.py",
        classes = ("Gemma2Attention", "Gemma2DecoderLayer", "Gemma2Model", "Gemma2ForCausalLM"),
        rope_name = "Gemma2RotaryEmbedding",
        patch_site = "unsloth/models/gemma2.py:FastGemma2Model.pre_patch",
    ),
    ModelingContract(
        name = "gemma3",
        path = "src/transformers/models/gemma3/modeling_gemma3.py",
        classes = ("Gemma3Attention",),
        patch_site = "early-warning: gemma3 modeling still present upstream",
    ),
    ModelingContract(
        name = "gpt_oss",
        path = "src/transformers/models/gpt_oss/modeling_gpt_oss.py",
        classes = ("GptOssModel",),
        patch_site = "early-warning: gpt_oss modeling still present upstream",
    ),
    ModelingContract(
        name = "mistral",
        path = "src/transformers/models/mistral/modeling_mistral.py",
        classes = (
            "MistralAttention",
            "MistralDecoderLayer",
            "MistralModel",
            "MistralForCausalLM",
        ),
        rope_name = "MistralRotaryEmbedding",
        patch_site = "unsloth/models/mistral.py:FastMistralModel.pre_patch",
    ),
    ModelingContract(
        name = "cohere",
        path = "src/transformers/models/cohere/modeling_cohere.py",
        classes = ("CohereAttention", "CohereDecoderLayer", "CohereModel", "CohereForCausalLM"),
        rope_name = "CohereRotaryEmbedding",
        patch_site = "unsloth/models/cohere.py:FastCohereModel.pre_patch",
    ),
    ModelingContract(
        name = "granite",
        path = "src/transformers/models/granite/modeling_granite.py",
        classes = (
            "GraniteAttention",
            "GraniteDecoderLayer",
            "GraniteModel",
            "GraniteForCausalLM",
        ),
        rope_name = "GraniteRotaryEmbedding",
        patch_site = "unsloth/models/granite.py:FastGraniteModel.pre_patch",
    ),
    ModelingContract(
        name = "falcon_h1",
        path = "src/transformers/models/falcon_h1/modeling_falcon_h1.py",
        classes = (
            "FalconH1Attention",
            "FalconH1DecoderLayer",
            "FalconH1Model",
            "FalconH1ForCausalLM",
        ),
        rope_name = "FalconH1RotaryEmbedding",
        extra_bindings = ("FalconH1RMSNorm",),
        patch_site = "unsloth/models/falcon_h1.py:FastFalconH1Model.pre_patch",
    ),
    # glm4_moe omitted: upstream renamed Glm4MoeLiteNaiveMoe -> Glm4MoeLiteExperts.
)


@pytest.mark.parametrize("contract", MODELING_CONTRACTS, ids = lambda c: c.name)
def test_modeling_contract(tag: str, contract: ModelingContract):
    src = fetch_text("huggingface/transformers", tag, contract.path)
    if src is None:
        pytest.skip(f"{tag}/{contract.name}: {contract.path} missing")
    for cls in contract.classes:
        assert has_def(
            src, cls, "class"
        ), f"{tag}/{contract.name}: class {cls} missing: {contract.patch_site}"
    if contract.rope_name is not None:
        assert is_bound(src, contract.rope_name), (
            f"{tag}/{contract.name}: {contract.rope_name} missing; "
            f"{contract.patch_site} RoPE rebind silently no-ops"
        )
    for n in contract.extra_bindings:
        assert is_bound(src, n), f"{tag}/{contract.name}: {n} missing: {contract.patch_site}"


def test_auto_factory_lazy_mapping_private_api(tag: str):
    """unsloth#5155: resolve_model_class needs all four _LazyAutoMapping private attrs to remain."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/auto/auto_factory.py",
    )
    if src is None:
        pytest.skip(f"{tag}: auto/auto_factory.py missing")
    needed = (
        "_model_mapping",
        "_config_mapping",
        "_extra_content",
        "_load_attr_from_module",
    )
    missing = [n for n in needed if n not in src]
    assert not missing, (
        f"{tag}: _LazyAutoMapping private API missing {missing}; "
        f"unsloth/models/_utils.py:resolve_model_class breaks (unsloth#5155)"
    )


def test_configuration_utils_alias(tag: str):
    """transformers 5.x renamed PretrainedConfig -> PreTrainedConfig; unsloth-zoo imports both defensively."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/configuration_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: configuration_utils.py missing")
    has_old = has_def(src, "PretrainedConfig", "class")
    has_new = has_def(src, "PreTrainedConfig", "class")
    assert has_old or has_new, (
        f"{tag}: neither PretrainedConfig (4.x) nor PreTrainedConfig (5.x) "
        f"defined in configuration_utils.py"
    )


def test_apply_chat_template_signature_present(tag: str):
    """unsloth-zoo#572: apply_chat_template `return_dict` default flipped False -> True in transformers 5.x."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/tokenization_utils_base.py",
    )
    if src is None:
        pytest.skip(f"{tag}: tokenization_utils_base.py missing")
    assert has_def(
        src, "apply_chat_template", "func"
    ), f"{tag}: apply_chat_template missing in tokenization_utils_base.py"


def test_modeling_attn_mask_utils_symbols(tag: str):
    """_prepare_4d_attention_mask_for_sdpa is imported by unsloth/models/llama.py + sentence_transformer.py."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/modeling_attn_mask_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_attn_mask_utils.py missing")
    assert has_def(src, "AttentionMaskConverter", "class"), f"{tag}: AttentionMaskConverter missing"
    assert is_bound(
        src, "_prepare_4d_attention_mask_for_sdpa"
    ), f"{tag}: _prepare_4d_attention_mask_for_sdpa missing"


def test_cache_utils_classes(tag: str):
    src = fetch_text("huggingface/transformers", tag, "src/transformers/cache_utils.py")
    if src is None:
        pytest.skip(f"{tag}: cache_utils.py missing")
    needed = ("Cache", "DynamicCache")
    for cls in needed:
        assert has_def(src, cls, "class"), f"{tag}: transformers.cache_utils.{cls} missing"


def test_training_args_parallel_mode_importable(tag: str):
    src = fetch_text("huggingface/transformers", tag, "src/transformers/training_args.py")
    if src is None:
        pytest.skip(f"{tag}: training_args.py missing")
    assert is_bound(src, "ParallelMode"), (
        f"{tag}: transformers.training_args.ParallelMode missing; "
        f"unsloth-zoo loss_utils.py:232 ImportError"
    )


def test_the_matrix_starts_at_the_declared_floor(tag: str) -> None:
    """The matrix is only a compatibility claim if it begins where the claim does.

    `_FLOOR` was a literal 4.57.6 while pyproject declared 4.52.4, so every 4.52-4.56 release was
    discarded and a change landing after the floor could break supported users, matrix still green.
    """
    declared = _declared_floor()
    assert _FLOOR == declared, (
        f"_FLOOR is {_FLOOR} but pyproject declares {declared}; the matrix would skip "
        f"every release between them"
    )
    if tag == "main":
        return
    assert _sort_key(tag) >= declared, (
        f"{tag} sits below the declared floor {declared}, so the matrix is checking a "
        f"release no supported install can resolve"
    )


def test_trainer_training_step_model_train_call_is_standalone(tag: str):
    """unsloth#11238 rewrites the FIRST `model.train()` in Trainer.training_step.

    A release writing `self.model.train()` earlier in that method turns the same replace into
    `self._unsloth_train_if_needed(model)` and every training step dies with AttributeError.
    """
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    m = re.search(r"\n    def training_step\(.*?\n    def ", src, re.DOTALL)
    if m is None:
        pytest.skip(f"{tag}: Trainer.training_step not found")
    body = m.group(0)
    idx = body.find("model.train()")
    if idx < 0:
        pytest.skip(f"{tag}: Trainer.training_step no longer calls model.train()")
    assert body[max(0, idx - 11) : idx] != "self.model.", (
        f"{tag}: the first `model.train()` in Trainer.training_step is `self.model.train()`; "
        f"unsloth/models/_utils.py would rewrite it into `self._unsloth_train_if_needed(model)`"
    )

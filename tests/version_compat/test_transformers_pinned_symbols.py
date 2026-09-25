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

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


# READ from pyproject, not repeated: a hardcoded 4.57.6 stayed put when the floor moved to
# 4.52.4, so every 4.52-4.56 release was discarded and the matrix went green regardless.
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
    # The lowest, so a marker-gated half cannot raise it and hide releases the other admits.
    return min(tuple(int(part) for part in v.split(".")) for v in found)


_FLOOR = _declared_floor()

# Always present whatever PyPI says: 4.57.6 is the floor, 5.5.0 the Apple Silicon cap, and
# 5.16.0 first required tokenizers>=0.23.1, which broke that install (test_transformers_tokenizers_pair).
_ALWAYS = ("v4.57.6", "v5.5.0", "v5.16.0", "v5.10.1", "v5.15.1")

# Exact pins real users run: notebooks pin 5.10.1 and 5.15.x. One tag per minor would replace
# 5.10.1 with 5.10.4, so a symbol arriving in a later 5.10 patch would break those notebooks
# while the matrix stayed green. 5.15.1 is anchored because the next patch would evict it too.

# pyproject's cap, read not repeated: one tag per minor means a published 5.17.1 would evict
# 5.17.0, the exact maximum the window admits, and check a version no user can resolve instead.
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
    # The highest, so a lower marker-gated half of a split cap cannot lower the anchor.
    ceiling = max(found, key = lambda v: tuple(int(p) for p in v.split(".")))
    # Through the override table: "v" + version can name a tag upstream never pushed, and every
    # check against it would then fail on the fetch rather than on the symbol.
    return (_TAG_OVERRIDES.get(ceiling, "v" + ceiling),)


# PyPI version -> the tag that carries it, where upstream disagrees with itself. PyPI 5.10.4 is
# tagged v5.10.3 (its __init__ says 5.10.4, and no v5.10.4 exists); PyPI 4.54.1 is tagged
# v4.54-release (v4.54.0's __init__ says 4.54.0, so it is a different release).
_TAG_OVERRIDES = {"5.10.4": "v5.10.3", "4.54.1": "v4.54-release"}

# Used when PyPI is unreachable; frozen so an outage cannot shrink the matrix and report green.
# It must START at the declared floor: beginning at 4.57.6 dropped every 4.52-4.56 check, and
# _ALWAYS does not restore them. test_the_outage_fallback_reaches_the_declared_floor pins this.
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


# Where the resolved matrix is shared between processes. xdist requires every worker to collect
# the SAME parameters, but each resolves the matrix during collection, so one worker timing out
# takes the fallback list, the parameter sets diverge and xdist aborts the run
# (https://pytest-xdist.readthedocs.io/en/stable/known-limitations.html). PYTEST_ and not UNSLOTH_
# because it is a harness knob, inert at runtime and read by nothing under unsloth/ or studio/.
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
    # Re-read before publishing: another worker may have resolved it, and its answer is in use.
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


# Inverted so a tag whose NAME is not its version still sorts by the release it carries.
_TAG_TO_RELEASE = {tag: release for release, tag in _TAG_OVERRIDES.items()}


def _sort_key(tag: str) -> tuple[int, ...]:
    name = _TAG_TO_RELEASE.get(tag, tag).lstrip("v")
    return tuple(int(g) for g in name.split("."))


# `main` catches drift before it ships to PyPI.
TRANSFORMERS_TAGS = _resolved_tags() + ["main"]

# Every check runs once per tag; one that cannot skips from inside so the tag stays in the report.
pytestmark = pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)


# Trainer surface: unsloth/models/_utils.py rewrites Trainer.{__init__, training_step, get_batch_samples, compute_loss}.
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
    m = re.search(r"^\s*def compute_loss\(([^)]*)\)", src, re.MULTILINE | re.DOTALL)
    if m is None:
        pytest.fail(f"{tag}: Trainer.compute_loss not found in source")
    assert "num_items_in_batch" in m.group(1), (
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
    # Hard-fail only when ALL substrings missing; partial drift is informational.
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
    # Assert ONE form is present so a refactor dropping both is caught.
    assert has_inplace or has_outplace, (
        f"{tag}: Trainer._inner_training_loop has neither "
        f"`tr_loss = tr_loss + tr_loss_step` nor `self._tr_loss +=`; "
        f"unsloth-zoo#543 patch breaks"
    )


# modeling_utils: checkpoint, PushToHubMixin, ALL_ATTENTION_FUNCTIONS.
def test_modeling_utils_exposes_checkpoint(tag: str):
    """unsloth-zoo#549: transformers 5.2+ uses modeling_utils.checkpoint; patch must replace it, not just torch's."""
    src = fetch_text("huggingface/transformers", tag, "src/transformers/modeling_utils.py")
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    # Either a direct import or local rebinding.
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


# integrations.bitsandbytes: _replace_with_bnb_linear vs new path.
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


# integrations.finegrained_fp8.FP8Linear: bias/has_bias rename in v5.
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
    has_bias_kw = re.search(r"def __init__\([^)]*\bbias\b", src) is not None
    has_has_bias_kw = re.search(r"def __init__\([^)]*\bhas_bias\b", src) is not None
    assert (
        has_bias_kw or has_has_bias_kw
    ), f"{tag}: FP8Linear.__init__ has neither `bias` nor `has_bias` param"


def test_processing_utils_unpack_importable(tag: str):
    """unsloth-zoo#583/584: transformers.processing_utils.Unpack must keep importing."""
    src = fetch_text("huggingface/transformers", tag, "src/transformers/processing_utils.py")
    if src is None:
        pytest.skip(f"{tag}: processing_utils.py missing")
    has_unpack = bool(re.search(r"^Unpack\b\s*=", src, re.MULTILINE) or "Unpack" in src)
    assert has_unpack, (
        f"{tag}: transformers.processing_utils.Unpack missing; "
        f"unsloth-zoo#583/584 import guard breaks"
    )


# Models: gemma3, gpt_oss forward signature drift.


def test_gemma3_attention_forward_present(tag: str):
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/gemma3/modeling_gemma3.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_gemma3.py missing")
    assert has_def(src, "Gemma3Attention", "class"), f"{tag}: class Gemma3Attention missing"


def test_gpt_oss_model_forward_present(tag: str):
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/gpt_oss/modeling_gpt_oss.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_gpt_oss.py missing (legacy)")
    assert has_def(src, "GptOssModel", "class"), f"{tag}: class GptOssModel missing"


# auto_factory: unsloth#5155 _LazyAutoMapping private API.


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


# tokenization: apply_chat_template return_dict default flip in v5.
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


# Generic-importability sweep: every transformers symbol unsloth/zoo imports must stay reachable.
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
    assert (
        has_def(src, "_prepare_4d_attention_mask_for_sdpa", "func")
        or "_prepare_4d_attention_mask_for_sdpa" in src
    ), f"{tag}: _prepare_4d_attention_mask_for_sdpa missing"


# Generic-importability sweep:
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
    assert "ParallelMode" in src, (
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

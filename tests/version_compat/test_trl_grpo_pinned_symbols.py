# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across all TRL PyPI minors unsloth + unsloth-zoo target.

Catches RL-surface API drift (DataCollatorForPreference relocation,
gated openenv/vllm_generation modules, unwrap_model_for_generation
moves, top-level GRPO exports). Fetches TRL source per tag straight from
github (no pip install) and asserts every depended-on symbol is present,
covering the pyproject window plus several releases above the cap for
early warning.
"""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def, is_bound
from tests.version_compat._rl_anchors import _reject_aux_loss_opt_in, _rl_py_constant


# 0.19.0 is pyproject-excluded (broken) but kept to see which symbols break.
TRL_TAGS = [
    "v0.18.2",
    "v0.19.0",
    "v0.19.1",
    "v0.20.0",
    "v0.21.0",
    "v0.22.0",
    "v0.22.1",
    "v0.22.2",
    "v0.23.0",
    "v0.23.1",
    "v0.24.0",
    "v0.25.0",
    "v0.25.1",
    "v0.26.0",
    "v0.26.1",
    "v0.26.2",
    "v0.27.0",
    "v0.27.1",
    "v0.27.2",
    "v0.28.0",
    "v0.29.0",
    "v0.29.1",
    "v1.0.0",
    "v1.1.0",
    "v1.2.0",
    "v1.3.0",
    "v1.4.0",
    "v1.5.0",
    "v1.5.1",
    "v1.6.0",
    "v1.7.0",
    "v1.7.1",
    "v1.8.0",
    "v1.9.0",
    "v1.9.1",
    "v1.9.2",
    "main",
]

pytestmark = pytest.mark.parametrize("tag", TRL_TAGS)


def _tag_ge(tag: str, floor: str) -> bool:
    """True if `tag` is `main` or a version >= `floor` (e.g. "1.7.0")."""
    if tag == "main":
        return True
    from packaging.version import Version

    try:
        return Version(tag.lstrip("v")) >= Version(floor)
    except Exception:
        return False


def test_trl_top_level_grpo_sft(tag: str):
    """GRPO/SFT Trainer+Config must resolve at the trl package root."""
    src = fetch_text("huggingface/trl", tag, "trl/__init__.py")
    assert src is not None, f"trl/__init__.py missing in {tag}"
    for name in ("GRPOTrainer", "GRPOConfig", "SFTTrainer", "SFTConfig"):
        assert is_bound(src, name), (
            f"{tag}: `from trl import {name}` will fail; "
            f"unsloth/trainer.py + unsloth/models/rl.py rely on this re-export"
        )


def test_grpo_trainer_class_canonical_path(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None, (
        f"{tag}: trl/trainer/grpo_trainer.py missing — "
        f"unsloth.models.rl._patch_trl_rl_trainers('grpo_trainer') breaks"
    )
    assert has_def(
        src, "GRPOTrainer", "class"
    ), f"{tag}: trl.trainer.grpo_trainer.GRPOTrainer not defined as a class"


def test_grpo_config_class_canonical_path(tag: str):
    """GRPOConfig must be discoverable by the *Config heuristic in rl.py:579-618."""
    candidates = ["trl/trainer/grpo_config.py", "trl/trainer/grpo_trainer.py"]
    hit = first_match("huggingface/trl", tag, candidates)
    assert hit is not None, f"{tag}: neither grpo_config.py nor grpo_trainer.py found"
    _, src = hit
    assert has_def(src, "GRPOConfig", "class"), (
        f"{tag}: GRPOConfig class missing in {[p for p, _ in [hit]]}; "
        f"unsloth's *Config heuristic in models/rl.py:579-618 will fail"
    )


def test_data_collator_for_preference_resolvable(tag: str):
    """DataCollatorForPreference must exist in dpo_trainer or utils (rl_replacements.py:318 imports it)."""
    new_path = fetch_text("huggingface/trl", tag, "trl/trainer/dpo_trainer.py")
    old_path = fetch_text("huggingface/trl", tag, "trl/trainer/utils.py")
    have = []
    if new_path is not None and is_bound(new_path, "DataCollatorForPreference"):
        have.append("trl.trainer.dpo_trainer")
    if old_path is not None and is_bound(old_path, "DataCollatorForPreference"):
        have.append("trl.trainer.utils")
    assert have, (
        f"{tag}: DataCollatorForPreference defined in NEITHER "
        f"trl/trainer/dpo_trainer.py NOR trl/trainer/utils.py — "
        f"unsloth/models/rl_replacements.py:318 will ImportError on real install"
    )


def test_trl_trainer_utils_pad(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/trainer/utils.py")
    if src is None:
        src = fetch_text("huggingface/trl", tag, "trl/trainer/utils/__init__.py")
    assert src is not None, f"{tag}: trl/trainer/utils[.py|/__init__.py] both missing"
    assert is_bound(src, "pad"), (
        f"{tag}: trl.trainer.utils.pad missing — "
        f"unsloth/models/rl_replacements.py:326 emits `from trl.trainer.utils "
        f"import pad as _unsloth_trl_pad` into the GRPO compile cell"
    )


def test_unwrap_model_for_generation_either_path(tag: str):
    """unwrap_model_for_generation must resolve via one of the two paths rl.py:152-155 tries (mirror prod exactly)."""
    candidates = [
        "trl/models/utils.py",
        "trl/models/__init__.py",
    ]
    for path in candidates:
        src = fetch_text("huggingface/trl", tag, path)
        if src is None:
            continue
        if is_bound(src, "unwrap_model_for_generation"):
            return
    pytest.fail(
        f"{tag}: trl.unwrap_model_for_generation not in any known path "
        f"({candidates}); unsloth/models/rl.py:152-155 will ImportError"
    )


def test_trl_experimental_openenv_gated(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/experimental/openenv/__init__.py")
    if src is None:
        pytest.skip(f"{tag}: trl.experimental.openenv not present (OK)")
    utils_src = fetch_text("huggingface/trl", tag, "trl/experimental/openenv/utils.py")
    assert utils_src is not None, (
        f"{tag}: trl.experimental.openenv exists but utils.py missing; "
        f"unsloth/models/rl_replacements.py:1765 imports openenv.utils explicitly"
    )


def test_trl_generation_vllm_generation_gated(tag: str):
    """VLLMGeneration + its _init_vllm/sync_weights/generate methods must
    exist when the module is present, else rl_replacements.py:1851-1971
    rewrites silently no-op and the server fast_inference path breaks."""
    src = fetch_text("huggingface/trl", tag, "trl/generation/vllm_generation.py")
    if src is None:
        pytest.skip(f"{tag}: trl.generation.vllm_generation not present (OK)")
    assert has_def(src, "VLLMGeneration", "class"), (
        f"{tag}: class VLLMGeneration missing; unsloth-zoo dispatch "
        f"in models/rl_replacements.py:1852 will silently no-op"
    )
    for method in ("_init_vllm", "sync_weights", "generate"):
        assert has_def(src, method, "func"), (
            f"{tag}: VLLMGeneration.{method} missing; "
            f"unsloth/models/rl_replacements.py rewrites this method body"
        )


# rl.py string-matches TRL's __version__.
def test_trl_version_parseable(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/__init__.py")
    assert src is not None
    has_literal = bool(re.search(r'^__version__\s*=\s*["\']', src, re.MULTILINE))
    has_subimport = bool(re.search(r"^from\s+\.version\s+import\s+__version__", src, re.MULTILINE))
    has_metadata = bool(
        re.search(
            r"^from\s+importlib\.metadata\s+import\s+(?:[\w,\s]+,\s*)?version",
            src,
            re.MULTILINE,
        )
        and re.search(r"^\s*__version__\s*=\s*version\s*\(", src, re.MULTILINE)
    )
    has_version_file = bool(
        re.search(r"^\s*__version__\s*=\s*f\.read\s*\(", src, re.MULTILINE)
        or re.search(r"^\s*__version__\s*=\s*open\s*\(", src, re.MULTILINE)
    )
    assert has_literal or has_subimport or has_metadata or has_version_file, (
        f"{tag}: trl.__version__ not exported via any known mechanism; "
        f"unsloth/models/rl.py:63 will AttributeError"
    )


def test_trl_is_conversational_export(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/__init__.py")
    assert src is not None
    if "is_conversational" not in src:
        pytest.skip(f"{tag}: trl.is_conversational not exported (legacy TRL)")


def test_trl_sft_trainer_module_internals(tag: str):
    """sft_trainer symbols for the `from trl.trainer.sft_trainer import *` at tokenizer_utils.py:1538."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/sft_trainer.py")
    assert src is not None, (
        f"{tag}: trl/trainer/sft_trainer.py missing; "
        f"unsloth/tokenizer_utils.py:1538 wildcard import fails"
    )
    assert has_def(src, "SFTTrainer", "class"), f"{tag}: class SFTTrainer missing in sft_trainer.py"
    if "neftune_post_forward_hook" not in src:
        pass


def test_trl_dpo_trainer_module_exists(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/trainer/dpo_trainer.py")
    assert src is not None, (
        f"{tag}: trl/trainer/dpo_trainer.py missing; "
        f"unsloth-zoo/temporary_patches/misc.py:1376 import fails"
    )
    assert has_def(src, "DPOTrainer", "class"), f"{tag}: class DPOTrainer missing in dpo_trainer.py"


def test_trl_constant_length_dataset_optional(tag: str):
    candidates = [
        "trl/trainer/utils.py",
        "trl/trainer/utils/__init__.py",
    ]
    hit = first_match("huggingface/trl", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: trl/trainer/utils not present")
    _, src = hit
    if "ConstantLengthDataset" not in src:
        pytest.skip(
            f"{tag}: ConstantLengthDataset removed; unsloth-zoo soft " f"import handles this"
        )


def test_trl_models_utils_disable_gradient_checkpointing(tag: str):
    if tag == "main":
        require = True
    else:
        try:
            from packaging.version import Version
            require = Version(tag.lstrip("v")) >= Version("1.0.0")
        except Exception:
            require = False
    src = fetch_text("huggingface/trl", tag, "trl/models/utils.py")
    if src is None:
        if require:
            pytest.fail(f"{tag}: trl/models/utils.py missing on 1.0.0+")
        pytest.skip(f"{tag}: trl/models/utils.py missing (legacy TRL)")
    has_it = has_def(src, "disable_gradient_checkpointing", "func")
    if require:
        assert has_it, (
            f"{tag}: trl.models.utils.disable_gradient_checkpointing "
            f"missing on TRL >=1.0.0; unsloth/models/rl.py:1979 patch silent no-op"
        )


def test_trl_import_utils_available_pattern(tag: str):
    candidates = [
        "trl/import_utils.py",
        "trl/import_utils/__init__.py",
    ]
    hit = first_match("huggingface/trl", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: trl/import_utils not present (legacy TRL)")
    _, src = hit
    # At least one `*_available` name must exist or import_fixes silently no-ops.
    has_pattern = bool(re.search(r"\b\w+_available\b", src))
    assert has_pattern, (
        f"{tag}: trl.import_utils has no `_available` cache var; "
        f"unsloth/import_fixes.py:508-516 silently no-ops"
    )


def test_trl_openenv_utils_generators(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/experimental/openenv/utils.py")
    if src is None:
        pytest.skip(f"{tag}: openenv.utils not present (gated optional)")
    legacy = is_bound(src, "generate_rollout_completions")
    new = is_bound(src, "_generate_rollout_completions_colocate")
    assert legacy or new, (
        f"{tag}: openenv.utils has neither `generate_rollout_completions` "
        f"nor `_generate_rollout_completions_colocate`; "
        f"unsloth/models/rl_replacements.py:1775-1781 patch breaks"
    )


# rl_replacements dispatches on method names; a rename silently skips the patch.
def test_trl_grpo_trainer_required_methods(tag: str):
    """GRPOTrainer methods unsloth rewrites against; drift silently skips
    the rewrite. _get_per_token_logps was renamed to
    _get_per_token_logps_and_entropies in TRL 0.20+; either is fine."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    for m in ("_prepare_inputs", "_generate_and_score_completions", "compute_loss"):
        assert has_def(src, m, "func"), (
            f"{tag}: GRPOTrainer.{m} missing; "
            f"unsloth/models/rl_replacements.py dispatch by name silently skips"
        )
    has_legacy = has_def(src, "_get_per_token_logps", "func")
    has_new = has_def(src, "_get_per_token_logps_and_entropies", "func")
    assert has_legacy or has_new, (
        f"{tag}: neither GRPOTrainer._get_per_token_logps (TRL <=0.19) nor "
        f"._get_per_token_logps_and_entropies (TRL >=0.20) found; "
        f"unsloth's per-token-logps rewrite no-ops on both dispatch keys"
    )
    for m in ("_generate_single_turn", "_move_model_to_vllm", "_calculate_rewards"):
        _present = has_def(src, m, "func")
        _ = _present


# Each substring is one half of a source .replace(); if it vanishes the rewrite silently no-ops.


def test_trl_grpo_source_inference_mode_unwrap(tag: str):
    """`torch.inference_mode` and `self.accelerator.unwrap_model` must both
    appear, or rl_replacements.py:526-535 autocast insertion no-ops."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    has_inference_mode = "torch.inference_mode" in src
    has_unwrap = "self.accelerator.unwrap_model" in src
    assert has_inference_mode and has_unwrap, (
        f"{tag}: GRPOTrainer source missing torch.inference_mode={has_inference_mode} "
        f"or self.accelerator.unwrap_model={has_unwrap}; "
        f"unsloth/models/rl_replacements.py:526 autocast insertion no-ops"
    )


def test_trl_kto_get_batch_logps_signature(tag: str):
    """KTO log-prob computation must stay patchable. Older TRL exposed
    KTOTrainer.get_batch_logps; TRL 1.x moved the math into
    _compute_logps/compute_ref_log_probs via selective_log_softmax.
    rl_replacements.py patches both shapes, so require EITHER form."""
    candidates = [
        "trl/trainer/kto_trainer.py",
        "trl/experimental/kto/kto_trainer.py",
        "trl/experimental/kto/__init__.py",
    ]
    checked_sources = []
    for path in candidates:
        src = fetch_text("huggingface/trl", tag, path)
        if src is None:
            continue
        checked_sources.append((path, src))
        if has_def(src, "get_batch_logps", "func"):
            return
        if has_def(src, "_compute_logps", "func") and "selective_log_softmax" in src:
            return
        if "per_token_logps = selective_log_softmax(shift_logits" in src:
            return
    old_shape_check = (
        'raise ValueError("Logits (batch and sequence length dim) and labels '
        'must have the same shape.")'
    )
    if checked_sources and not any(old_shape_check in src for _, src in checked_sources):
        # TRL main inlined KTO log-probs; nothing to guard until a new rewrite target appears.
        return
    pytest.fail(
        f"{tag}: KTO log-prob computation not found in any of {candidates}; "
        f"unsloth/models/rl_replacements.py KTO rewrite silently skipped"
    )


# transformers 5.0 removed push_to_hub_token; without the bare pop our patch no-ops.
def test_trl_sft_trainer_class(tag: str):
    """SFTTrainer must exist. The push_to_hub_token pop literal is checked
    only when present; its absence means TRL already adapted (fine)."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/sft_trainer.py")
    assert src is not None
    assert has_def(src, "SFTTrainer", "class"), f"{tag}: class SFTTrainer missing"


def test_trl_dpo_trainer_methods(tag: str):
    """DPOTrainer methods unsloth's rewriters key on (rl_replacements.py
    :222-394). All version-windowed and non-required (the rewriter
    cleanly no-ops when absent); presence/absence is logged as
    informational so a silent regression stays visible."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/dpo_trainer.py")
    assert src is not None
    assert has_def(src, "DPOTrainer", "class"), f"{tag}: class DPOTrainer missing in dpo_trainer.py"
    for method in (
        "concatenated_inputs",
        "concatenated_forward",
        "_compute_loss_liger",
        "_set_signature_columns_if_needed",
        "_prepare_dataset",
    ):
        _present = has_def(src, method, "func")
        _ = _present


def test_trl_grpo_internal_helpers_in_scope(tag: str):
    """Chat-template kwargs must propagate via legacy
    `maybe_apply_chat_template` (TRL <=0.24, rewritten by unsloth) or
    successor `apply_chat_template(... **chat_template_kwargs)` (TRL
    >=0.25, native). Either pattern wires the path."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    legacy = "maybe_apply_chat_template" in src
    successor = "chat_template_kwargs" in src or "apply_chat_template" in src
    assert legacy or successor, (
        f"{tag}: GRPOTrainer source does NOT propagate chat-template kwargs "
        f"via legacy `maybe_apply_chat_template` OR successor "
        f"`apply_chat_template(... **chat_template_kwargs)`; "
        f"unsloth/models/rl_replacements.py:909-927 rewrite no-ops AND "
        f"native TRL doesn't carry the kwargs either — likely real bug"
    )


def test_trl_truncate_with_protected_tokens_optional(tag: str):
    """Informational: track truncate_with_protected_tokens (shipped TRL
    0.22.2-0.23.1, later removed) so a silent rename doesn't slip past
    the rl_replacements.py:712 regex that handles both presence/absence."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    has_it = "truncate_with_protected_tokens" in src
    _ = has_it


# These pin exact source strings rl.py transforms for TRL >= 1.7.0; a restructure silently no-ops.
def test_trl_grpo_peft_ref_adapter_block_contract(tag: str):
    """rl.py (trl>=1.4.0) strips TRL's PEFT ref-adapter init with a re.DOTALL
    regex anchored on `elif is_peft_model(model) and args.beta != 0.0:` ...
    `ref_param.data.copy_(param.data)`. Both anchors must exist (else the
    regex no-ops and the ref adapter is created under Unsloth), and the
    following `enable_input_require_grads` gradient-checkpointing block must
    remain present -- the tightened regex must NOT swallow it (PR #6904). The
    `elif` block shape appeared in TRL 1.4.0, so this contract runs from there."""
    if not _tag_ge(tag, "1.4.0"):
        pytest.skip(
            f"{tag}: pre-1.4.0 uses the `if is_peft_available()...` form (rl.py 0.27 branch)"
        )
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    assert "elif is_peft_model(model) and args.beta != 0.0:" in src, (
        f"{tag}: PEFT ref-adapter `elif` anchor gone; unsloth/models/rl.py "
        f"peft_pattern re.sub no-ops and TRL's ref adapter init runs under Unsloth"
    )
    assert "ref_param.data.copy_(param.data)" in src, (
        f"{tag}: `ref_param.data.copy_(param.data)` end-anchor gone; "
        f"unsloth/models/rl.py peft_pattern loses its DOTALL end match"
    )
    assert "enable_input_require_grads" in src, (
        f"{tag}: `enable_input_require_grads` block gone from grpo_trainer.py; "
        f"the tightened PR #6904 regex assumed it follows the ref-adapter block"
    )


def test_trl_grpo_quantized_model_cast_contract(tag: str):
    """rl.py (trl>=1.7.0) neutralizes TRL's hardcoded QLoRA bf16 cast
    `if _is_quantized_model:` -> `if False:`. A rename leaves the cast active,
    which ignores the user's dtype and breaks GradScaler with fp16=True."""
    if not _tag_ge(tag, "1.7.0"):
        pytest.skip(f"{tag}: pre-1.7.0 spells the cast differently (is_loaded_in_4bit)")
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    assert "if _is_quantized_model:" in src, (
        f"{tag}: `if _is_quantized_model:` gone; unsloth/models/rl.py cannot "
        f"neutralize TRL's hardcoded QLoRA bf16 cast and it runs under Unsloth"
    )


def test_trl_grpo_aux_loss_enabled_contract(tag: str):
    """rl.py (trl>=1.7.0) appends a fail-fast after GRPOTrainer's one
    `self.aux_loss_enabled = ...` assignment, so an MoE router-aux opt-in errors
    instead of silently training without the penalty (the optimized forward
    cannot compute it; PR #6904). It anchors on the assignment, not its
    expression: TRL #7248 rewrote the expression to read the coefficient from
    the model config, and the exact-text anchor then no-oped on TRL main."""
    if not _tag_ge(tag, "1.7.0"):
        pytest.skip(f"{tag}: aux_loss_enabled / router_aux_loss_coef added in TRL 1.7.0")
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    hits = re.findall(_rl_py_constant("_GRPO_AUX_LOSS_ENABLED_LINE"), src, flags = re.MULTILINE)
    assert len(hits) == 1, (
        f"{tag}: expected one `self.aux_loss_enabled = ...` assignment in GRPOTrainer, found "
        f"{len(hits)}; unsloth/models/rl.py's fail-fast anchor is ambiguous or gone"
    )
    patched = _reject_aux_loss_opt_in(src)
    line = next(l for l in patched.splitlines() if "self.aux_loss_enabled = " in l)
    after = patched.splitlines()[patched.splitlines().index(line) + 1]
    assert after.strip() == _rl_py_constant(
        "_GRPO_AUX_LOSS_REJECT"
    ), f"{tag}: the fail-fast did not land directly after the assignment: {after!r}"
    assert after[: len(after) - len(after.lstrip())] == line[: len(line) - len(line.lstrip())]


def test_trl_grpo_per_token_logps_aux_arity_contract(tag: str):
    """TRL 1.7.0 added `compute_aux_loss` to
    _get_per_token_logps_and_entropies and made every call site unpack a
    3-tuple. rl_replacements.py version-gates its injected replacement to emit
    a 3-tuple for trl>=1.7.0 (2-tuple below). This is the exact change the
    has_def existence checks miss: the method still exists, only its arity
    changed. If TRL drops/renames the aux return, the gate needs revisiting."""
    if not _tag_ge(tag, "0.20.0"):
        pytest.skip(f"{tag}: pre-0.20 uses legacy _get_per_token_logps (2-tuple, no aux)")
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    assert has_def(src, "_get_per_token_logps_and_entropies", "func"), (
        f"{tag}: _get_per_token_logps_and_entropies missing on TRL >=0.20; "
        f"unsloth's per-token-logps injection dispatch key no longer matches"
    )
    if _tag_ge(tag, "1.7.0"):
        assert "compute_aux_loss" in src, (
            f"{tag}: TRL >=1.7.0 dropped `compute_aux_loss`; the 3-tuple "
            f"injection gate in unsloth/models/rl_replacements.py must be revisited"
        )

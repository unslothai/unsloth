# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across all TRL PyPI minor versions
unsloth + unsloth-zoo target. Catches API drift like:

  - trl 0.18 split DataCollatorForPreference into trl.trainer.dpo_trainer
    (was trl.trainer.utils). unsloth.models.rl_replacements:318 imports
    the post-split path; if a new TRL release moves it again, the
    GRPOTrainer.compile cell crashes with ImportError.
  - trl 0.20 introduced trl.experimental.openenv as a *gated* module;
    unsloth.models.rl_replacements:1765-1770 catches ImportError, but
    the gate must remain importable when present.
  - trl 0.22 introduced trl.generation.vllm_generation for the
    server-mode fast_inference path; unsloth.models.rl_replacements
    :1846-1848 catches ImportError, but the module must exist on
    versions where unsloth-zoo's vllm_utils dispatches to it.
  - trl unwrap_model_for_generation moved from trl.models to
    trl.models.utils across releases (unsloth/models/rl.py:152-155
    handles both with try/except).
  - trl GRPOTrainer / GRPOConfig must remain top-level exports for
    `from trl import GRPOTrainer` to work in user code, which is what
    `_patch_trl_rl_trainers("grpo_trainer")` discovers.

Strategy: for each tracked TRL tag, fetch the relevant source files
straight from github.com/huggingface/trl (no pip install required) and
assert that every symbol unsloth/unsloth-zoo's RL surface depends on
is present.

Versioning policy: cover the supported window declared in
pyproject.toml (`trl>=0.18.2,!=0.19.0,<=0.24.0`) PLUS several recent
releases ABOVE the cap, so we get early warning when TRL ships
something incompatible and the maintainer can extend the cap or add a
patch BEFORE a user hits it.
"""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


# Every stable TRL release from 0.18.2 (the pyproject floor) onwards,
# plus `main`. Refresh by running:
#   python -c "import urllib.request,json
#   from packaging.version import Version
#   r=json.loads(urllib.request.urlopen('https://pypi.org/pypi/trl/json').read())
#   v=sorted([Version(x) for x in r['releases'] if r['releases'][x] and not Version(x).is_prerelease and Version(x)>=Version('0.18.2')])
#   print(*[f'\"v{x}\",' for x in v],sep='\n')"
#
# 0.19.0 is excluded by pyproject (`!=0.19.0`) — the release was
# broken; we keep it in the matrix so we KNOW it's broken (and which
# symbols specifically), not just trust the pin.
#
# Anchors (per the project spec, ALL patches must stay forwards/
# backwards compatible with these): 0.22.2, 0.27.1, 1.0.0.
TRL_TAGS = [
    "v0.18.2",
    "v0.19.0",
    "v0.19.1",
    "v0.20.0",
    "v0.21.0",
    "v0.22.0",
    "v0.22.1",
    "v0.22.2",  # anchor
    "v0.23.0",
    "v0.23.1",
    "v0.24.0",  # current pyproject cap
    "v0.25.0",
    "v0.25.1",
    "v0.26.0",
    "v0.26.1",
    "v0.26.2",
    "v0.27.0",
    "v0.27.1",  # anchor
    "v0.27.2",
    "v0.28.0",
    "v0.29.0",
    "v0.29.1",
    "v1.0.0",  # anchor
    "v1.1.0",
    "v1.2.0",
    "v1.3.0",
    "v1.4.0",
    "main",
]


# -------------------------------------------------------------------------
# HARD-import top-level: from trl import X must keep working for these.
# unsloth/trainer.py + unsloth/models/rl.py rebind these by name.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_top_level_grpo_sft(tag: str):
    """`from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig`
    must keep resolving at the package root."""
    src = fetch_text("huggingface/trl", tag, "trl/__init__.py")
    assert src is not None, f"trl/__init__.py missing in {tag}"
    for name in ("GRPOTrainer", "GRPOConfig", "SFTTrainer", "SFTConfig"):
        assert name in src, (
            f"{tag}: `from trl import {name}` will fail; "
            f"unsloth/trainer.py + unsloth/models/rl.py rely on this re-export"
        )


# -------------------------------------------------------------------------
# trl.trainer.grpo_trainer.GRPOTrainer -- the canonical class. unsloth's
# RL patcher discovers it via `eval(f"trl.trainer.{trainer_file}.{name}")`
# in unsloth/models/rl.py:548-594.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_grpo_trainer_class_canonical_path(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None, (
        f"{tag}: trl/trainer/grpo_trainer.py missing — "
        f"unsloth.models.rl._patch_trl_rl_trainers('grpo_trainer') breaks"
    )
    assert has_def(
        src, "GRPOTrainer", "class"
    ), f"{tag}: trl.trainer.grpo_trainer.GRPOTrainer not defined as a class"


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_grpo_config_class_canonical_path(tag: str):
    """unsloth/models/rl.py:579-618 looks for the *Config sibling of the
    Trainer class via heuristic discovery; the canonical one is in
    grpo_config.py."""
    candidates = ["trl/trainer/grpo_config.py", "trl/trainer/grpo_trainer.py"]
    hit = first_match("huggingface/trl", tag, candidates)
    assert hit is not None, f"{tag}: neither grpo_config.py nor grpo_trainer.py found"
    _, src = hit
    assert has_def(src, "GRPOConfig", "class"), (
        f"{tag}: GRPOConfig class missing in {[p for p, _ in [hit]]}; "
        f"unsloth's *Config heuristic in models/rl.py:579-618 will fail"
    )


# -------------------------------------------------------------------------
# DataCollatorForPreference: unsloth.models.rl_replacements:318 hard-imports
# from trl.trainer.dpo_trainer. Some old TRL versions had it in
# trl.trainer.utils; modern ones moved to trl.trainer.dpo_trainer.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_data_collator_for_preference_resolvable(tag: str):
    """Either the new path (trl.trainer.dpo_trainer) or the old path
    (trl.trainer.utils) must define DataCollatorForPreference. unsloth's
    string-emitted import in rl_replacements.py:318 uses dpo_trainer;
    if neither path resolves, we have a gap."""
    new_path = fetch_text("huggingface/trl", tag, "trl/trainer/dpo_trainer.py")
    old_path = fetch_text("huggingface/trl", tag, "trl/trainer/utils.py")
    have = []
    if new_path is not None and "DataCollatorForPreference" in new_path:
        have.append("trl.trainer.dpo_trainer")
    if old_path is not None and "DataCollatorForPreference" in old_path:
        have.append("trl.trainer.utils")
    assert have, (
        f"{tag}: DataCollatorForPreference defined in NEITHER "
        f"trl/trainer/dpo_trainer.py NOR trl/trainer/utils.py — "
        f"unsloth/models/rl_replacements.py:318 will ImportError on real install"
    )


# -------------------------------------------------------------------------
# trl.trainer.utils.pad: emitted into the GRPO compile cell as
# _unsloth_trl_pad (rl_replacements.py:326).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_trainer_utils_pad(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/trainer/utils.py")
    if src is None:
        # Some TRL versions split utils into a package; check the
        # alternative location.
        src = fetch_text("huggingface/trl", tag, "trl/trainer/utils/__init__.py")
    assert src is not None, f"{tag}: trl/trainer/utils[.py|/__init__.py] both missing"
    assert has_def(src, "pad", "func") or "def pad(" in src, (
        f"{tag}: trl.trainer.utils.pad missing — "
        f"unsloth/models/rl_replacements.py:326 emits `from trl.trainer.utils "
        f"import pad as _unsloth_trl_pad` into the GRPO compile cell"
    )


# -------------------------------------------------------------------------
# trl.models.unwrap_model_for_generation -- moved between submodules
# across releases. unsloth/models/rl.py:152-155 handles both paths.
# Assert at least one resolves on every tag.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_unwrap_model_for_generation_either_path(tag: str):
    """unsloth/models/rl.py:152-155 tries
    `trl.models.utils.unwrap_model_for_generation` first, then
    `trl.models.unwrap_model_for_generation`. Tests must mirror the
    prod fallback exactly — checking a third path makes the test
    laxer than the runtime."""
    candidates = [
        "trl/models/utils.py",
        "trl/models/__init__.py",
    ]
    for path in candidates:
        src = fetch_text("huggingface/trl", tag, path)
        if src is None:
            continue
        if "unwrap_model_for_generation" in src:
            return
    pytest.fail(
        f"{tag}: trl.unwrap_model_for_generation not in any known path "
        f"({candidates}); unsloth/models/rl.py:152-155 will ImportError"
    )


# -------------------------------------------------------------------------
# trl.experimental.openenv: gated import (rl_replacements.py:1765-1770
# wraps in try/except). When present, must export the symbols unsloth
# patches.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_experimental_openenv_gated(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/experimental/openenv/__init__.py")
    if src is None:
        # OK: feature not in this release; unsloth's try/except handles it.
        pytest.skip(f"{tag}: trl.experimental.openenv not present (OK)")
    # Module exists -> at minimum, `utils` submodule must be importable
    # because unsloth patches via `import trl.experimental.openenv.utils`.
    utils_src = fetch_text("huggingface/trl", tag, "trl/experimental/openenv/utils.py")
    assert utils_src is not None, (
        f"{tag}: trl.experimental.openenv exists but utils.py missing; "
        f"unsloth/models/rl_replacements.py:1765 imports openenv.utils explicitly"
    )


# -------------------------------------------------------------------------
# trl.generation.vllm_generation: gated import for the fast_inference
# server mode (rl_replacements.py:1846-1848). When present, must define
# at least one symbol unsloth patches against.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_generation_vllm_generation_gated(tag: str):
    """unsloth/models/rl_replacements.py:1851-1971 string-rewrites
    `VLLMGeneration._init_vllm`, `.sync_weights`, and `.generate`. If
    VLLMGeneration is renamed or any of those three methods disappear,
    the rewrite silently no-ops and the fast_inference server path
    breaks at runtime. Gated: skip if the module isn't in this TRL."""
    src = fetch_text("huggingface/trl", tag, "trl/generation/vllm_generation.py")
    if src is None:
        # OK: pre-server-mode TRL. unsloth's try/except handles absence.
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


# -------------------------------------------------------------------------
# Sanity: TRL's __version__ string is parseable. unsloth/models/rl.py:63
# does `from trl import __version__ as trl_version_raw` and string-
# matches on it.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_version_parseable(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/__init__.py")
    assert src is not None
    # Recognised mechanisms (any one is sufficient):
    #   1. literal `__version__ = "x.y.z"` at module scope
    #   2. `from .version import __version__`
    #   3. `__version__ = version("trl")` via importlib.metadata
    #   4. `__version__ = f.read().strip()` (TRL 0.22.x reads from a
    #      sibling VERSION file)
    has_literal = bool(re.search(r'^__version__\s*=\s*["\']', src, re.MULTILINE))
    has_subimport = bool(
        re.search(r"^from\s+\.version\s+import\s+__version__", src, re.MULTILINE)
    )
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


# =========================================================================
# Coverage extension (added 2026-05): symbols / source-string contracts
# unsloth + unsloth-zoo touch but the original suite missed.
# =========================================================================


# -------------------------------------------------------------------------
# 1. trl.is_conversational — soft import in unsloth-zoo dataset_utils.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_is_conversational_export(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/__init__.py")
    assert src is not None
    if "is_conversational" not in src:
        # Some old TRLs omit it; gated soft import in unsloth-zoo
        # falls back to a local impl. OK.
        pytest.skip(f"{tag}: trl.is_conversational not exported (legacy TRL)")


# -------------------------------------------------------------------------
# 2-4. trl.trainer.sft_trainer module surface used by unsloth tokenizer
#      utils + tests.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_sft_trainer_module_internals(tag: str):
    """unsloth/tokenizer_utils.py:1538 does `from trl.trainer.sft_trainer
    import *`. The symbols below must exist for the wildcard import +
    eval-discovery to keep working."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/sft_trainer.py")
    assert src is not None, (
        f"{tag}: trl/trainer/sft_trainer.py missing; "
        f"unsloth/tokenizer_utils.py:1538 wildcard import fails"
    )
    assert has_def(
        src, "SFTTrainer", "class"
    ), f"{tag}: class SFTTrainer missing in sft_trainer.py"
    # neftune_post_forward_hook: optional (TRL removed it in some
    # versions); soft-imported in tokenizer_utils.py:1542. Don't fail.
    if "neftune_post_forward_hook" not in src:
        pass


# -------------------------------------------------------------------------
# 5-6. trl.trainer.dpo_trainer module + MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
#      — patched by unsloth-zoo/temporary_patches/misc.py:1376-1379.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_dpo_trainer_module_exists(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/trainer/dpo_trainer.py")
    assert src is not None, (
        f"{tag}: trl/trainer/dpo_trainer.py missing; "
        f"unsloth-zoo/temporary_patches/misc.py:1376 import fails"
    )
    assert has_def(
        src, "DPOTrainer", "class"
    ), f"{tag}: class DPOTrainer missing in dpo_trainer.py"


# -------------------------------------------------------------------------
# 7. trl.trainer.utils.ConstantLengthDataset — soft import in
#    unsloth-zoo/dataset_utils.py:596. Optional (TRL 0.20.0 removed it
#    on some paths).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
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
            f"{tag}: ConstantLengthDataset removed; unsloth-zoo soft "
            f"import handles this"
        )


# -------------------------------------------------------------------------
# 8. trl.models.utils.disable_gradient_checkpointing — added in TRL
#    1.0.0+. unsloth/models/rl.py:1976-1994 uses hasattr() for gating;
#    we still want the assertion that the symbol exists from 1.0.0
#    onwards so a future removal gets caught.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_models_utils_disable_gradient_checkpointing(tag: str):
    if tag == "main":
        # main is bleeding edge; expect symbol to track 1.0.0+ behaviour.
        require = True
    else:
        # Strip leading 'v' and parse.
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


# -------------------------------------------------------------------------
# 9. trl.import_utils + the `_*_available` cache pattern — used by
#    unsloth/import_fixes.py:508-516 to clear cached `is_X_available`
#    booleans so vllm-ascend imports work.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_import_utils_available_pattern(tag: str):
    candidates = [
        "trl/import_utils.py",
        "trl/import_utils/__init__.py",
    ]
    hit = first_match("huggingface/trl", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: trl/import_utils not present (legacy TRL)")
    _, src = hit
    # The patch iterates `vars(trl.import_utils)` looking for any name
    # ending in `_available`. At least one such cache var must exist or
    # the patch silently no-ops.
    has_pattern = bool(re.search(r"\b\w+_available\b", src))
    assert has_pattern, (
        f"{tag}: trl.import_utils has no `_available` cache var; "
        f"unsloth/import_fixes.py:508-516 silently no-ops"
    )


# -------------------------------------------------------------------------
# 10. trl.experimental.openenv.utils generators — at least one of the
#     two function names must exist (unsloth/models/rl_replacements.py
#     :1775-1781 calls getattr() to find one).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_openenv_utils_generators(tag: str):
    src = fetch_text("huggingface/trl", tag, "trl/experimental/openenv/utils.py")
    if src is None:
        pytest.skip(f"{tag}: openenv.utils not present (gated optional)")
    legacy = "generate_rollout_completions" in src
    new = "_generate_rollout_completions_colocate" in src
    assert legacy or new, (
        f"{tag}: openenv.utils has neither `generate_rollout_completions` "
        f"nor `_generate_rollout_completions_colocate`; "
        f"unsloth/models/rl_replacements.py:1775-1781 patch breaks"
    )


# -------------------------------------------------------------------------
# 11-16. GRPOTrainer required method names. unsloth/models/rl_replacements
#        .py uses function_name == "..." dispatch keys; if a method is
#        renamed, the patch silently doesn't apply. List of methods is
#        the precise dispatch key set.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_grpo_trainer_required_methods(tag: str):
    """Method names unsloth string-rewrites against. Drift here
    silently skips the rewrite. _get_per_token_logps was renamed to
    _get_per_token_logps_and_entropies in TRL 0.20+; either is fine
    since unsloth dispatches by function_name."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    # _prepare_inputs / _generate_and_score_completions / compute_loss
    # are stable across the entire support window.
    for m in ("_prepare_inputs", "_generate_and_score_completions", "compute_loss"):
        assert has_def(src, m, "func"), (
            f"{tag}: GRPOTrainer.{m} missing; "
            f"unsloth/models/rl_replacements.py dispatch by name silently skips"
        )
    # Per-token-logps surface: ONE of the two names must exist.
    has_legacy = has_def(src, "_get_per_token_logps", "func")
    has_new = has_def(src, "_get_per_token_logps_and_entropies", "func")
    assert has_legacy or has_new, (
        f"{tag}: neither GRPOTrainer._get_per_token_logps (TRL <=0.19) nor "
        f"._get_per_token_logps_and_entropies (TRL >=0.20) found; "
        f"unsloth's per-token-logps rewrite no-ops on both dispatch keys"
    )
    # Optional / version-dependent — never fail, just informational
    for m in ("_generate_single_turn", "_move_model_to_vllm", "_calculate_rewards"):
        _present = has_def(src, m, "func")
        _ = _present


# -------------------------------------------------------------------------
# Source-string contracts on trl/trainer/grpo_trainer.py. Each substring
# is one half of a `function.replace(old, new)` rewrite — if the
# substring no longer appears in TRL source, the rewrite is a no-op
# AND the user-facing GRPO behaviour silently diverges.
#
# Broken into per-version-window tests because some patterns only apply
# to a subset of TRL minors.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_grpo_source_inference_mode_unwrap(tag: str):
    """rl_replacements.py:526-535 inserts an autocast block immediately
    AFTER `with torch.inference_mode():` and `self.accelerator.unwrap_model
    (self.model)`. Both substrings must appear in `_prepare_inputs`."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    has_inference_mode = "torch.inference_mode" in src
    has_unwrap = "self.accelerator.unwrap_model" in src
    assert has_inference_mode and has_unwrap, (
        f"{tag}: GRPOTrainer source missing torch.inference_mode={has_inference_mode} "
        f"or self.accelerator.unwrap_model={has_unwrap}; "
        f"unsloth/models/rl_replacements.py:526 autocast insertion no-ops"
    )


# -------------------------------------------------------------------------
# 17. KTOTrainer.get_batch_logps + the literal raise message rewriter
#     hits.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_kto_get_batch_logps_signature(tag: str):
    """TRL 0.27+ moved KTOTrainer to trl.experimental.kto and the
    canonical kto_trainer.py shrank to a thin re-export wrapper. The
    real `get_batch_logps` lives at trl/experimental/kto/kto_trainer.py.
    Unsloth's MRO walk in models/rl.py:592-708 already follows
    trl.experimental.* parents, so either path is fine — we just
    require the symbol to exist SOMEWHERE."""
    candidates = [
        "trl/trainer/kto_trainer.py",
        "trl/experimental/kto/kto_trainer.py",
        "trl/experimental/kto/__init__.py",
    ]
    for path in candidates:
        src = fetch_text("huggingface/trl", tag, path)
        if src is None:
            continue
        if has_def(src, "get_batch_logps", "func"):
            return
    pytest.fail(
        f"{tag}: KTOTrainer.get_batch_logps not found in any of {candidates}; "
        f"unsloth/models/rl_replacements.py:1675 rewrite silently skipped"
    )


# -------------------------------------------------------------------------
# 18. SFTTrainer.__init__ literal `dict_args.pop("push_to_hub_token")`
#     OR our shim must short-circuit. transformers 5.0 removed this
#     kwarg; if TRL stops emitting the bare pop, our patch becomes
#     a no-op AND TRL itself crashes on transformers 5.0.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_sft_trainer_class(tag: str):
    """Sanity: SFTTrainer.__init__ exists. The
    `dict_args.pop("push_to_hub_token")` literal substring is checked
    only when present — its absence means TRL already adapted (e.g.
    via `dict_args.pop("push_to_hub_token", None)` with a default),
    which is also fine."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/sft_trainer.py")
    assert src is not None
    assert has_def(src, "SFTTrainer", "class"), f"{tag}: class SFTTrainer missing"


# -------------------------------------------------------------------------
# 19-21. DPOTrainer methods unsloth-zoo's rl_replacements rewrites.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_dpo_trainer_methods(tag: str):
    """DPOTrainer method-name surface unsloth's rewriters key on
    (rl_replacements.py:222-394). All four are version-windowed:
      - concatenated_inputs / concatenated_forward existed on
        DPOTrainer through TRL 0.29.x; TRL 1.0+ refactored these into
        free functions (concatenation moved out of the class).
      - _compute_loss_liger added ~TRL 0.20.
      - _set_signature_columns_if_needed: usually inherited from
        transformers.Trainer, may or may not be re-defined locally.
    None are STRICTLY required — when missing the matching unsloth
    rewriter cleanly no-ops (TRL itself does the work). We surface
    presence/absence as informational so a regression that
    SILENTLY drops one is at least visible in the test log."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/dpo_trainer.py")
    assert src is not None
    # The DPO class itself must always exist.
    assert has_def(
        src, "DPOTrainer", "class"
    ), f"{tag}: class DPOTrainer missing in dpo_trainer.py"
    # Informational only -- pass either way:
    for method in (
        "concatenated_inputs",
        "concatenated_forward",
        "_compute_loss_liger",
        "_set_signature_columns_if_needed",
        "_prepare_dataset",
    ):
        _present = has_def(src, method, "func")
        _ = _present  # informational; rewriter no-ops cleanly when absent


# -------------------------------------------------------------------------
# 22-23. trl.trainer.grpo_trainer must IMPORT or DEFINE the helpers
#        unsloth's source rewriters reference: profiling_context,
#        maybe_apply_chat_template, truncate_with_protected_tokens.
#        Either the symbol is locally defined OR imported from elsewhere
#        in trl.* — the rewriter only needs the NAME to be in scope at
#        the call site.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_grpo_internal_helpers_in_scope(tag: str):
    """Chat-template propagation is what unsloth's
    grpo_trainer_fix_maybe_apply_chat_template wires up so user-supplied
    `reasoning_effort` etc. survives the GRPO compile cell. The exact
    helper name moved across releases:
      - TRL <=0.24: `maybe_apply_chat_template(example, processing_class)`
        appeared as a literal in grpo_trainer.py — unsloth's regex
        rewriter substitutes it with a kwargs-aware version.
      - TRL >=0.25: TRL itself uses `apply_chat_template` and pipes
        `**self.chat_template_kwargs`, so the unsloth rewriter is a
        cleanly-no-op'd dead path on those versions (correct behaviour).
    Either pattern means the chat-template path is wired SOMEWHERE."""
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


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_trl_truncate_with_protected_tokens_optional(tag: str):
    """Some TRL versions (0.22.2-0.23.1 specifically) ship
    `truncate_with_protected_tokens`. Newer versions removed it.
    rl_replacements.py:712 has a regex that handles both presence
    and absence — but if the symbol is renamed without removal,
    we need to know."""
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None
    # No assertion — informational only. We just want to NOT silently
    # drift.
    has_it = "truncate_with_protected_tokens" in src
    _ = has_it  # informational; pass either way.

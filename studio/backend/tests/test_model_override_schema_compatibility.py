# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an install running two different builds against one override map loses.

The override map is a single unversioned JSON blob in ``app_settings``, and
``set_model_override`` REPLACES the entry it writes rather than merging into it. That
was harmless while every field the row could hold also had a control in every build
that wrote it. The llama-server tuning group broke the symmetry: four fields the
loader has always applied are now written by the settings route, so a client that
predates the route change sends a payload that simply lacks them, and the replace
takes them out.

The rule the suite encodes: a field a build does not KNOW ABOUT is dropped harmlessly
on read (a row is a whitelist rebuild, never a schema contract), but a field a build
does not SEND is deleted on write, and nothing on the server puts it back. The first
is forward compatibility working. The second is the exposure this schema shape
creates, and it is pinned here so that a later fix has something to change.

The identity rules a hydrating panel depends on are pinned too: the panel now asks
this module which row its model resolves to, so the browser's own fold and this one
have to agree on every path shape or the panel hydrates from another model's row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import utils.openai_auto_switch_settings as settings

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_openai_auto_switch import _put, override_store  # noqa: E402, F401

MODEL = "unsloth/Repo-GGUF:Q4_K_M"

SERVER_TUNING_FIELDS = ("load_mode", "spec_draft_cache_type", "ctx_checkpoints", "cache_ram")

PRE_TUNING_PAYLOAD = dict(
    custom_context_length = 8192,
    kv_cache_dtype = "q8_0",
    speculative_type = "dspark",
    spec_draft_n_max = 4,
    n_batch = 4096,
    tensor_parallel = True,
)

# Falsy-but-meaningful values: catches a route that stores on truth instead of 'is not None'.
TUNING_PAYLOAD = dict(
    load_mode = "mmap",
    spec_draft_cache_type = "q8_0",
    ctx_checkpoints = 0,
    cache_ram = -1,
)


def test_a_row_written_before_the_tuning_group_still_loads(override_store):
    legacy_row = {
        "llama_extra_args": ["--numa", "distribute"],
        "max_seq_length": 4096,
        "kv_cache_dtype": "q8_0",
    }

    assert settings.normalize_model_override(legacy_row) == legacy_row
    kwargs = settings.model_override_load_kwargs(legacy_row, is_gguf = True)
    assert kwargs == {
        "max_seq_length": 4096,
        "llama_extra_args": ["--numa", "distribute"],
        "cache_type_kv": "q8_0",
    }
    for field in SERVER_TUNING_FIELDS:
        assert field not in kwargs


def test_a_field_from_a_newer_build_is_ignored_rather_than_fatal(override_store):
    # No version stamp: forward compat relies on readers iterating only keys they know.
    from_the_future = {
        "custom_context_length": 8192,
        "load_mode": "mmap",
        "future_knob_from_a_newer_build": {"nested": [1, 2]},
    }

    assert settings.normalize_model_override(from_the_future) == {
        "custom_context_length": 8192,
        "load_mode": "mmap",
    }
    # The load path reads the raw row, so it would raise if a reader enumerated the dict.
    assert settings.model_override_load_kwargs(from_the_future, is_gguf = True) == {
        "max_seq_length": 8192,
        "load_mode": "mmap",
    }


def test_a_client_that_does_not_know_the_tuning_group_cannot_erase_it(override_store):
    """P3 and P4: the cell this whole file exists for.

    A frontend from before the settings route forwarded the four sends a payload that
    omits them, and ``set_model_override`` replaces the entry rather than merging. The
    row carries no version stamp, so the route cannot tell that omission apart from a
    user clearing the fields -- except that a build which knows them says so.

    Reachable two ways, and neither needs a deliberate downgrade: a second machine on
    the LAN still running the old build, and a browser holding a cached bundle against
    a server that has been upgraded under it.
    """
    _put(MODEL, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    before = settings.get_model_override(MODEL)
    for field, value in TUNING_PAYLOAD.items():
        assert before[field] == value

    _put(MODEL, **PRE_TUNING_PAYLOAD)
    after = settings.get_model_override(MODEL)

    for field, value in TUNING_PAYLOAD.items():
        assert after[field] == value, f"{field} was deleted by a save that never mentioned it"
    assert after == before
    kwargs = settings.model_override_load_kwargs(after, is_gguf = True)
    for field, value in TUNING_PAYLOAD.items():
        assert kwargs[field] == value


def test_a_client_that_does_know_them_still_clears_by_omission(override_store):
    # Blanket carry-over is wrong: the panel clears a field by omitting it.
    _put(MODEL, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    assert settings.get_model_override(MODEL)["load_mode"] == TUNING_PAYLOAD["load_mode"]

    _put(MODEL, **PRE_TUNING_PAYLOAD, mirrors_server_tuning = True)
    after = settings.get_model_override(MODEL)
    for field in SERVER_TUNING_FIELDS:
        assert field not in after, f"{field} survived an explicit clear"


def test_the_preservation_flag_is_not_itself_a_saved_field(override_store):
    # Write-mode bools survive exclude_none; left in saved_fields they break 'no fields means remove'.
    _put(MODEL, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    assert settings.get_model_override(MODEL)

    _put(MODEL, mirrors_server_tuning = True)
    assert settings.get_model_override(MODEL) == {}


def test_a_legacy_model_id_only_clear_is_not_undone_by_preservation(override_store):
    # Model-id-only payload: remove is None but is_removal is true; gate on the verdict, not the field.
    _put(MODEL, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    assert settings.get_model_override(MODEL)

    _put(MODEL)
    assert settings.get_model_override(MODEL) == {}


def test_tuning_carries_over_from_the_bare_repo_entry(override_store):
    # A repo:QUANT save retires the bare `repo` row, so preservation must walk the same spellings.
    bare_id = "unsloth/Repo-GGUF"
    _put(bare_id, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    assert settings.get_model_override(bare_id)["load_mode"] == TUNING_PAYLOAD["load_mode"]

    _put(MODEL, **PRE_TUNING_PAYLOAD)
    kept = settings.get_model_override(MODEL)
    for field, value in TUNING_PAYLOAD.items():
        assert kept[field] == value, f"{field} was lost saving under the qualified key"


def test_carry_over_does_not_activate_tuning_from_a_row_no_load_reads(override_store):
    # Load stops at the first non-empty row, so tuning in a shadowed row is dormant; do not pull it up.
    bare_id = "unsloth/Repo-GGUF"
    _put(bare_id, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    _put(MODEL, **PRE_TUNING_PAYLOAD, mirrors_server_tuning = True)
    _, active = settings.resolve_override_for_load(MODEL)
    assert active, "precondition: the qualified row is what a load resolves to"
    assert not any(field in active for field in SERVER_TUNING_FIELDS)

    _put(MODEL, **PRE_TUNING_PAYLOAD)

    kept = settings.get_model_override(MODEL)
    for field in SERVER_TUNING_FIELDS:
        assert field not in kept, f"{field} was promoted out of a row no load reads"
    for field, value in TUNING_PAYLOAD.items():
        assert settings.get_model_override(bare_id)[field] == value


def test_carry_over_reads_the_cached_spelling_a_load_would_resolve_to(override_store):
    # Lookup tries the snapshot path first, so carry-over must take tuning from it, not the repo row.
    snapshot_id = "/cache/models--org--Repo-GGUF/snapshots/abc:Q4_K_M"
    repo_id = "org/Repo-GGUF:Q4_K_M"
    live_tuning = dict(TUNING_PAYLOAD, load_mode = "mmap", cache_ram = -1)
    dormant_tuning = dict(TUNING_PAYLOAD, load_mode = "direct", cache_ram = 4096)

    # Written to the store directly: the route would retire the other spelling.
    settings.set_model_override(snapshot_id, **PRE_TUNING_PAYLOAD, **live_tuning)
    settings.set_model_override(repo_id, **PRE_TUNING_PAYLOAD, **dormant_tuning)
    assert snapshot_id in settings.cached_repo_alias_keys(repo_id)
    assert settings.is_cache_load_path_key(snapshot_id)
    assert not settings.is_cache_load_path_key(repo_id)

    _put(repo_id, **PRE_TUNING_PAYLOAD)

    kept = settings.get_model_override(repo_id)
    assert kept["load_mode"] == "mmap", "took the dormant row's tuning, not the live one"
    assert kept["cache_ram"] == -1


def test_a_fill_pass_adds_the_tuning_group_without_disturbing_the_row(override_store):
    _put(MODEL, **TUNING_PAYLOAD, speculative_type = "dspark")
    _put(MODEL, custom_context_length = 8192, fill_absent_fields = True)

    stored = settings.get_model_override(MODEL)
    assert stored["custom_context_length"] == 8192
    for field, value in TUNING_PAYLOAD.items():
        assert stored[field] == value


@pytest.mark.parametrize(
    "speculative_type",
    ["ngram", "mtp", "mtp+ngram", "off", None],
)
def test_the_draft_cache_dtype_is_dropped_under_a_mode_with_no_drafter(
    override_store, speculative_type
):
    """P10: the dtype needs a mode that loads a separate drafter, or it goes.

    Pre-existing in the normalizer, but this PR is what carries the field to the
    server at all, so the drop is now visible as a server row that silently lacks a
    value the panel showed. Storing it anyway would show an edit for a draft context
    that never exists, so the drop is right; the point of pinning it is that it is
    SILENT, and a user who sets the dtype and then changes the mode is not told.
    """
    _put(MODEL, spec_draft_cache_type = "q8_0", speculative_type = speculative_type)

    assert "spec_draft_cache_type" not in settings.get_model_override(MODEL)


def test_the_draft_cache_dtype_survives_a_mode_that_does_load_a_drafter(override_store):
    for mode in sorted(settings.SEPARATE_DRAFT_MODEL_SPEC_TYPES):
        _put(MODEL, spec_draft_cache_type = "q8_0", speculative_type = mode)
        assert settings.get_model_override(MODEL)["spec_draft_cache_type"] == "q8_0"


# Must mirror foldOverrideKey in features/model-picker/api/model-overrides.ts.
OVERRIDE_KEY_FOLDS = [
    ("C:\\models\\Foo.gguf", "c:/models/foo.gguf", True),
    ("C:\\models\\Foo.gguf", "C:\\models\\Foo.gguf\\", True),
    ("//share/models/Foo.gguf", "\\\\SHARE\\models\\foo.gguf", True),
    ("/mnt/c/models/Foo.gguf", "/mnt/C/models/foo.gguf", True),
    # POSIX paths are case-sensitive: two files can differ only in case.
    ("/models/Foo.gguf", "/models/foo.gguf", False),
    ("unsloth/Repo-GGUF", "UNSLOTH/repo-gguf", True),
    ("unsloth/Repo-GGUF:Q4_K_M", "unsloth/repo-gguf:q4_k_m", True),
    ("/models/foo.gguf", "models/foo.gguf", False),
    ("models/foo.gguf", "/models/foo.gguf", False),
]


@pytest.mark.parametrize(("stored_key", "lookup_key", "same_model"), OVERRIDE_KEY_FOLDS)
def test_an_override_key_resolves_the_way_the_browser_folds_it(
    override_store, stored_key, lookup_key, same_model
):
    settings.set_model_override(stored_key, max_seq_length = 4096)

    resolved = settings.resolve_model_override_key(lookup_key)
    if same_model:
        assert resolved == stored_key
        assert settings.get_model_override(lookup_key) == {"max_seq_length": 4096}
    else:
        assert resolved is None
        assert settings.get_model_override(lookup_key) == {}


def test_two_keys_that_fold_together_resolve_to_nothing(override_store):
    # An ambiguous fold matches nothing rather than picking by enumeration order.
    settings.set_model_override("C:\\models\\Foo.gguf", max_seq_length = 4096)
    settings.set_model_override("C:\\models\\FOO.gguf", max_seq_length = 8192)

    assert settings.resolve_model_override_key("c:/models/foo.gguf") is None


def test_the_disable_aliases_survive_override_normalization():
    """llama.cpp's own "none", plus "disable" / "disabled", reach /load as off.

    ``_clean_str`` drops anything outside the whitelist, so leaving them out filed an
    explicit disable as no override at all and the model followed the global
    preference, which enables a drafter whenever that preference is Auto.
    """
    for spelling in ("none", "None", "  DISABLE  ", "disabled"):
        normalized = settings.normalize_model_override({"speculative_type": spelling})
        assert normalized.get("speculative_type") == spelling.strip().lower(), spelling

    assert "speculative_type" not in settings.normalize_model_override(
        {"speculative_type": "bogus"}
    )


REASONING_PAYLOAD = dict(reasoning_budget = 512, reasoning_budget_message = "Wrap up.")


def test_a_client_that_does_not_know_the_reasoning_pair_cannot_erase_it(override_store):
    _put(
        MODEL,
        **PRE_TUNING_PAYLOAD,
        **REASONING_PAYLOAD,
        mirrors_server_tuning = True,
        mirrors_reasoning_budget = True,
    )
    before = settings.get_model_override(MODEL)
    for field, value in REASONING_PAYLOAD.items():
        assert before[field] == value

    _put(MODEL, **PRE_TUNING_PAYLOAD, mirrors_server_tuning = True)
    after = settings.get_model_override(MODEL)

    for field, value in REASONING_PAYLOAD.items():
        assert after[field] == value, f"{field} was deleted by a save that never mentioned it"
    assert after == before
    kwargs = settings.model_override_load_kwargs(after, is_gguf = True)
    for field, value in REASONING_PAYLOAD.items():
        assert kwargs[field] == value


def test_a_client_that_does_know_the_reasoning_pair_still_clears_by_omission(override_store):
    _put(
        MODEL,
        **PRE_TUNING_PAYLOAD,
        **REASONING_PAYLOAD,
        mirrors_server_tuning = True,
        mirrors_reasoning_budget = True,
    )
    assert settings.get_model_override(MODEL)["reasoning_budget"] == 512

    _put(MODEL, **PRE_TUNING_PAYLOAD, mirrors_server_tuning = True, mirrors_reasoning_budget = True)
    after = settings.get_model_override(MODEL)
    for field in REASONING_PAYLOAD:
        assert field not in after, f"{field} survived an explicit clear"


def test_the_reasoning_flag_is_not_itself_a_saved_field(override_store):
    _put(MODEL, **PRE_TUNING_PAYLOAD, mirrors_reasoning_budget = True)
    assert settings.get_model_override(MODEL)

    _put(MODEL, mirrors_reasoning_budget = True)
    assert settings.get_model_override(MODEL) == {}


@pytest.mark.parametrize(
    "fallback",
    [
        {"llama_extra_args": ["--reasoning-budget", "512"]},
        {"reasoning_budget": 512},
        {"reasoning_budget": 0},
        {"reasoning_budget_message": "Wrap up."},
    ],
)
def test_a_reset_tombstone_outlives_a_later_default_save(override_store, fallback):
    """The -1/"" pair is stored so a qualified row shadows a reasoning flag on a broader entry.
    A later save with the controls at their defaults omits the pair, and the row must not empty
    out and be deleted, or the load falls back and hands the reset value straight back."""
    bare = "unsloth/Repo-GGUF"
    settings.set_model_override(bare, **fallback)

    _put(MODEL, reasoning_budget = -1, mirrors_server_tuning = True, mirrors_reasoning_budget = True)
    assert settings.get_model_override(MODEL).get("reasoning_budget") == -1

    _put(MODEL, mirrors_server_tuning = True, mirrors_reasoning_budget = True)
    after = settings.get_model_override(MODEL)
    assert (
        after.get("reasoning_budget") == -1
    ), "the tombstone was dropped, so the bare row's --reasoning-budget 512 applies again"

    kwargs = settings.model_override_load_kwargs(after, is_gguf = True)
    assert kwargs["reasoning_budget"] == -1
    key, resolved = settings.resolve_override_for_load(bare, variant = "Q4_K_M")
    assert key == MODEL
    assert resolved.get("reasoning_budget") == -1
    assert resolved.get("reasoning_budget_message", "") == ""
    _, sibling = settings.resolve_override_for_load(bare, variant = "Q8_0")
    assert sibling == fallback


@pytest.mark.parametrize(
    "fallback", [{"reasoning_budget": 512}, {"reasoning_budget_message": "Stop"}]
)
def test_standalone_reasoning_reset_survives_a_later_default_save(override_store, fallback):
    path = "/srv/models/model-Q4_K_M.gguf"
    settings.set_model_override(f"{path}:Q4_K_M", **fallback)
    _put(path, reasoning_budget = -1, reasoning_budget_message = "", mirrors_reasoning_budget = True)
    _put(path, mirrors_reasoning_budget = True)
    key, resolved = settings.resolve_override_for_load(path)
    assert key == path
    assert resolved.get("reasoning_budget") == -1
    assert resolved.get("reasoning_budget_message") == ""


def test_no_tombstone_is_invented_without_a_fallback_to_shadow(override_store):
    _put(MODEL, reasoning_budget = -1, mirrors_server_tuning = True, mirrors_reasoning_budget = True)
    _put(MODEL, mirrors_server_tuning = True, mirrors_reasoning_budget = True)
    assert settings.get_model_override(MODEL) == {}

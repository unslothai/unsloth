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

# Reuse the in-memory override store and the route-level PUT helper.
from test_openai_auto_switch import _put, override_store  # noqa: E402, F401

MODEL = "unsloth/Repo-GGUF:Q4_K_M"

# The group the settings route learned to forward. Named once, because every case
# below is about the difference between a build that sends these and one that does not.
SERVER_TUNING_FIELDS = ("load_mode", "spec_draft_cache_type", "ctx_checkpoints", "cache_ram")

# Everything a build from before the group already sent, at values it would send.
# spec_draft_cache_type needs a mode that loads a separate drafter, so dspark is
# part of the shared payload rather than part of the tuning group.
PRE_TUNING_PAYLOAD = dict(
    custom_context_length = 8192,
    kv_cache_dtype = "q8_0",
    speculative_type = "dspark",
    spec_draft_n_max = 4,
    n_batch = 4096,
    tensor_parallel = True,
)

# The four at values that are falsy but meaningful: no checkpoints kept, no cache
# limit. A test that used only truthy values would pass against a route that stored
# them on truth rather than on "is not None".
TUNING_PAYLOAD = dict(
    load_mode = "mmap",
    spec_draft_cache_type = "q8_0",
    ctx_checkpoints = 0,
    cache_ram = -1,
)


def test_a_row_written_before_the_tuning_group_still_loads(override_store):
    # P1, the base case every other row here is measured against. The normalizer is a
    # whitelist rebuild, so a row from an install that never held the four is not
    # "missing" anything: it produces exactly the load it always did.
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
    # And none of the four is invented at a default, which would pin a knob the user
    # never touched for a model that has been loading without it.
    for field in SERVER_TUNING_FIELDS:
        assert field not in kwargs


def test_a_field_from_a_newer_build_is_ignored_rather_than_fatal(override_store):
    # P2. There is no version stamp to check, so forward compatibility rests entirely
    # on both readers iterating keys they know. A row written by a build that has
    # learned a fifth knob has to load on this one, minus that knob.
    from_the_future = {
        "custom_context_length": 8192,
        "load_mode": "mmap",
        "future_knob_from_a_newer_build": {"nested": [1, 2]},
    }

    assert settings.normalize_model_override(from_the_future) == {
        "custom_context_length": 8192,
        "load_mode": "mmap",
    }
    # The load path reads the raw stored row, not the normalized one, so it is the
    # one that would raise on an unexpected key if either reader enumerated the dict.
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

    # The old client: every field its build knows, and nothing it does not. It cannot
    # set the flag, which is why the flag defaults to the safe answer.
    _put(MODEL, **PRE_TUNING_PAYLOAD)
    after = settings.get_model_override(MODEL)

    for field, value in TUNING_PAYLOAD.items():
        assert after[field] == value, f"{field} was deleted by a save that never mentioned it"
    assert after == before
    # And they still reach the command line.
    kwargs = settings.model_override_load_kwargs(after, is_gguf = True)
    for field, value in TUNING_PAYLOAD.items():
        assert kwargs[field] == value


def test_a_client_that_does_know_them_still_clears_by_omission(override_store):
    # The other side of the trade. Preserving on every omission would be simpler and
    # wrong: the panel clears one of these by sending nothing for it, so a blanket
    # carry-over would swap a mixed-version window for a field no one can ever unset.
    _put(MODEL, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    assert settings.get_model_override(MODEL)["load_mode"] == TUNING_PAYLOAD["load_mode"]

    _put(MODEL, **PRE_TUNING_PAYLOAD, mirrors_server_tuning = True)
    after = settings.get_model_override(MODEL)
    for field in SERVER_TUNING_FIELDS:
        assert field not in after, f"{field} survived an explicit clear"


def test_the_preservation_flag_is_not_itself_a_saved_field(override_store):
    # It is a write mode, like fill_absent_fields. Both are bools, so exclude_none does
    # not drop them, and either one left in saved_fields would make every payload look
    # non-empty and break the legacy "no fields means remove".
    _put(MODEL, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    assert settings.get_model_override(MODEL)

    _put(MODEL, mirrors_server_tuning = True)
    assert settings.get_model_override(MODEL) == {}


def test_a_legacy_model_id_only_clear_is_not_undone_by_preservation(override_store):
    # The documented pre-`remove` contract: a payload carrying only model_id clears the
    # override. That leaves payload.remove None while is_removal is true, so a gate on
    # the field rather than the verdict would carry the tuning forward and rebuild a
    # non-empty row -- the request succeeds and the settings keep applying.
    _put(MODEL, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    assert settings.get_model_override(MODEL)

    _put(MODEL)
    assert settings.get_model_override(MODEL) == {}


def test_tuning_carries_over_from_the_bare_repo_entry(override_store):
    # A save under repo:QUANT reads flags off a bare `repo` row and retires it, so the
    # preservation has to reach the same spellings the extra-args carry-over does. A
    # lookup on the sent id alone finds nothing here and the tuning goes down with the
    # row. Same shape for the snapshot-path and cached-alias spellings that walk beside
    # it, which need HF cache state to reach and are covered by the alias sweep tests.
    bare_id = "unsloth/Repo-GGUF"
    _put(bare_id, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    assert settings.get_model_override(bare_id)["load_mode"] == TUNING_PAYLOAD["load_mode"]

    # The older client saving the qualified key: it never sends the group, and its row
    # does not exist yet, so everything it keeps has to come off the bare entry.
    _put(MODEL, **PRE_TUNING_PAYLOAD)
    kept = settings.get_model_override(MODEL)
    for field, value in TUNING_PAYLOAD.items():
        assert kept[field] == value, f"{field} was lost saving under the qualified key"


def test_carry_over_does_not_activate_tuning_from_a_row_no_load_reads(override_store):
    # The carry-over above walks a list of spellings, but a load does not: it stops at
    # the first non-empty row (resolve_override_for_load) rather than merging across
    # them. So tuning sitting in a row the qualified key shadows is dormant, and pulling
    # it up into the winning row would switch it on -- from a save that was about
    # something else entirely, on a client with no control for these fields to show it.
    bare_id = "unsloth/Repo-GGUF"
    _put(bare_id, **PRE_TUNING_PAYLOAD, **TUNING_PAYLOAD, mirrors_server_tuning = True)
    # The qualified row exists and wins, and has none of the group.
    _put(MODEL, **PRE_TUNING_PAYLOAD, mirrors_server_tuning = True)
    _, active = settings.resolve_override_for_load(MODEL)
    assert active, "precondition: the qualified row is what a load resolves to"
    assert not any(field in active for field in SERVER_TUNING_FIELDS)

    # The older client saves the qualified key, sending none of the group.
    _put(MODEL, **PRE_TUNING_PAYLOAD)

    kept = settings.get_model_override(MODEL)
    for field in SERVER_TUNING_FIELDS:
        assert field not in kept, f"{field} was promoted out of a row no load reads"
    # And the row it came from is left exactly as it was, still dormant behind the
    # qualified key rather than emptied by having been read.
    for field, value in TUNING_PAYLOAD.items():
        assert settings.get_model_override(bare_id)[field] == value


def test_carry_over_reads_the_cached_spelling_a_load_would_resolve_to(override_store):
    # An upgraded cache holds both spellings of one quant: the snapshot path an older
    # build keyed rows by, and the repo id a Settings save writes. A lookup tries the
    # load path FIRST, so the snapshot row is the live one and the repo row is dormant.
    # The carry-over walks its own list, and if that list leads with the sent id it
    # takes the unit off the dormant row -- and then the retirement block clears the
    # snapshot row, so the tuning that was actually applying is gone and tuning that
    # never applied takes its place. Neither row is visibly wrong afterwards, which is
    # what makes it worth pinning.
    snapshot_id = "/cache/models--org--Repo-GGUF/snapshots/abc:Q4_K_M"
    repo_id = "org/Repo-GGUF:Q4_K_M"
    live_tuning = dict(TUNING_PAYLOAD, load_mode = "mmap", cache_ram = -1)
    dormant_tuning = dict(TUNING_PAYLOAD, load_mode = "direct", cache_ram = 4096)

    # Written straight to the store rather than through the route: a save under either
    # spelling retires the other, which is the very cleanup this test is about, so the
    # route cannot be used to build the two-row state an upgrade leaves behind.
    settings.set_model_override(snapshot_id, **PRE_TUNING_PAYLOAD, **live_tuning)
    settings.set_model_override(repo_id, **PRE_TUNING_PAYLOAD, **dormant_tuning)
    # Precondition: the two really are the pair, and the path really is the winner.
    assert snapshot_id in settings.cached_repo_alias_keys(repo_id)
    assert settings.is_cache_load_path_key(snapshot_id)
    assert not settings.is_cache_load_path_key(repo_id)

    # The older client saves the repo-id spelling, sending none of the group.
    _put(repo_id, **PRE_TUNING_PAYLOAD)

    kept = settings.get_model_override(repo_id)
    assert kept["load_mode"] == "mmap", "took the dormant row's tuning, not the live one"
    assert kept["cache_ram"] == -1


def test_a_fill_pass_adds_the_tuning_group_without_disturbing_the_row(override_store):
    # The one merge path there is, and the one the backfill uses. A fill must not be
    # able to cause the erasure above, or the upgrade pass itself would strip the row
    # it was run to complete.
    _put(MODEL, **TUNING_PAYLOAD, speculative_type = "dspark")
    _put(MODEL, custom_context_length = 8192, fill_absent_fields = True)

    stored = settings.get_model_override(MODEL)
    assert stored["custom_context_length"] == 8192
    for field, value in TUNING_PAYLOAD.items():
        assert stored[field] == value


@pytest.mark.parametrize(
    "speculative_type",
    # Every mode that does NOT load a separate drafter, plus the unset case.
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
    # The other side of P10, so the parametrize above cannot pass by dropping the
    # field unconditionally.
    for mode in sorted(settings.SEPARATE_DRAFT_MODEL_SPEC_TYPES):
        _put(MODEL, spec_draft_cache_type = "q8_0", speculative_type = mode)
        assert settings.get_model_override(MODEL)["spec_draft_cache_type"] == "q8_0"


# P13. Each row is (stored key, key a load asks under, whether they name one model).
# The browser folds before it stores and this module folds before it reads, so a
# disagreement is a panel hydrating from another model's row. Mirrors foldOverrideKey
# in features/model-picker/api/model-overrides.ts, which is pinned from the other end
# in tests/llama-extra-args-override-lookup.test.ts.
OVERRIDE_KEY_FOLDS = [
    # A Windows volume is case-insensitive and its separators interchange.
    ("C:\\models\\Foo.gguf", "c:/models/foo.gguf", True),
    ("C:\\models\\Foo.gguf", "C:\\models\\Foo.gguf\\", True),
    # A UNC share is the same volume however it is spelled.
    ("//share/models/Foo.gguf", "\\\\SHARE\\models\\foo.gguf", True),
    # A WSL drive mount IS the Windows volume it exposes.
    ("/mnt/c/models/Foo.gguf", "/mnt/C/models/foo.gguf", True),
    # A POSIX path is not: two files can differ only in case, and replaying one
    # model's context and GPU pin onto the other is worse than finding nothing.
    ("/models/Foo.gguf", "/models/foo.gguf", False),
    # A repo id folds whole; the browser lowercases it before storing.
    ("unsloth/Repo-GGUF", "UNSLOTH/repo-gguf", True),
    # And the quant suffix folds with it.
    ("unsloth/Repo-GGUF:Q4_K_M", "unsloth/repo-gguf:q4_k_m", True),
    # A path never folds onto a repo id, in either direction: "models/foo.gguf" is a
    # legal repo id and "/models/foo.gguf" is a file, and one is not the other.
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
    # An upgrade can leave both casings behind. Picking one at enumeration order is
    # another model's settings half the time, so an ambiguous fold matches nothing.
    settings.set_model_override("C:\\models\\Foo.gguf", max_seq_length = 4096)
    settings.set_model_override("C:\\models\\FOO.gguf", max_seq_length = 8192)

    assert settings.resolve_model_override_key("c:/models/foo.gguf") is None

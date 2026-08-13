# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What happens to an install that predates this denylist.

Widening ``_DENYLIST_GROUPS`` is the one change here that can act on data already on
disk: an override saved before a flag was denied still holds it. Every path that
reads such an entry is pinned here, because the failure mode is a user who never
typed the flag being unable to load or to save.

The rule the suite encodes: an argument the CALLER just sent is refused loudly (400,
naming the flag), and an argument merely CARRIED OVER from storage is dropped
quietly. The first is a mistake being made now; the second is history.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_LSA_PATH = _BACKEND / "core" / "inference" / "llama_server_args.py"
_spec = importlib.util.spec_from_file_location("_lsa_compat_test", _LSA_PATH)
_lsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lsa)

# Saved by a build where each of these was still allowed. Not hypothetical: the
# first three appear verbatim in llama-server command lines people copy.
LEGACY_STORED = [
    ["--log-file", "/var/log/llama.log"],
    ["--slot-save-path", "/tmp/slots"],
    ["--media-path", "/srv/media"],
    ["--cors-origins", "*"],
    ["--agent"],
    ["--mcp-servers-json", "{}"],
]


@pytest.mark.parametrize("stored", LEGACY_STORED)
def test_a_stored_flag_denied_after_the_fact_is_dropped_not_kept(stored):
    # The validator itself still refuses: it is the boundary, and it has no idea
    # whether its caller is a request or a stored row.
    with pytest.raises(ValueError, match = "managed by Unsloth Studio"):
        _lsa.validate_extra_args(stored)


@pytest.mark.parametrize("stored", LEGACY_STORED)
def test_the_drop_helper_keeps_everything_else(stored):
    # What the carry-over paths use instead of a refusal. The user's other flags
    # are not collateral: only the denied names go.
    kept, dropped = _lsa.drop_managed_flags([*stored, "--numa", "distribute"])

    assert kept == ["--numa", "distribute"]
    assert dropped == [_lsa._flag_name(stored[0])]
    # And what survives is loadable, or the drop would have moved the failure
    # rather than removed it.
    assert _lsa.validate_extra_args(kept) == kept


def test_dropping_takes_the_flags_value_with_it():
    # Leaving "/var/log/llama.log" behind would hand llama.cpp a bare positional,
    # which it reads as the model path.
    kept, dropped = _lsa.drop_managed_flags(
        ["--top-k", "20", "--log-file", "/var/log/llama.log", "--seed", "1"]
    )

    assert kept == ["--top-k", "20", "--seed", "1"]
    assert dropped == ["--log-file"]


def test_an_attached_value_form_is_dropped_whole():
    kept, dropped = _lsa.drop_managed_flags(["--log-file=/x", "--top-k=20"])

    assert kept == ["--top-k=20"]
    assert dropped == ["--log-file"]


def test_nothing_to_drop_returns_the_list_unchanged():
    args = ["--numa", "distribute", "--top-k", "20"]
    kept, dropped = _lsa.drop_managed_flags(args)

    assert kept == args
    assert dropped == []


def test_an_empty_or_missing_list_is_handled():
    assert _lsa.drop_managed_flags(None) == ([], [])
    assert _lsa.drop_managed_flags([]) == ([], [])


def test_a_bound_breaking_stored_list_is_also_dropped_to_something_loadable():
    # The bounds are new too, so a stored list can be over them. A drop that
    # returned an unloadable list would leave the load failing anyway.
    kept, dropped = _lsa.drop_managed_flags(["--verbose"] * (_lsa.MAX_EXTRA_ARG_TOKENS + 10))

    assert _lsa.validate_extra_args(kept) == kept
    assert len(dropped) > 0


def test_a_poisoned_value_is_never_echoed_into_the_dropped_list():
    # Every caller joins this list into a warning log. A stored value carrying ANSI
    # escapes would then rewrite whatever is reading that log, and the value itself
    # is not the operator's business either: the flag name is what identifies it.
    kept, dropped = _lsa.drop_managed_flags(["--grammar", "\x1b[2Jroot ::= [0-9]", "--top-k", "20"])

    assert kept == ["--top-k", "20"]
    assert dropped == ["--grammar", "<value>"]
    assert not any("\x1b" in name for name in dropped)


def test_a_control_character_in_a_stored_value_is_dropped_too():
    kept, _ = _lsa.drop_managed_flags(["--chat-template", "a\x00b", "--top-k", "20"])

    assert _lsa.validate_extra_args(kept) == kept
    assert "--top-k" in kept


# --- the paths that actually carry a stored value over --------------------------
# The helper is only half of it: the two call sites reach it through module globals
# populated by an import list, so a wiring mistake is invisible until a load runs.


def test_the_inherited_load_path_drops_only_the_denied_flag(monkeypatch):
    import routes.inference as inference_route

    assert hasattr(inference_route, "drop_managed_flags"), (
        "the resolver reads this from module globals; an unlisted import NameErrors "
        "only when a model with stored flags is loaded"
    )

    class _Backend:
        extra_args = ["--log-file", "/var/log/llama.log", "--numa", "distribute"]
        # Same model and variant, or the resolver refuses the pickup before it ever
        # reaches the drop and the test proves nothing.
        extra_args_source = ("local/x", "")

    class _Config:
        is_gguf = True
        gguf_variant = ""

    class _Request:
        llama_extra_args = None
        gguf_variant = ""
        gpu_memory_mode = "auto"
        model_fields_set: set = set()

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    resolved = inference_route._resolve_inherited_extra_args(_Request(), _Config(), "local/x", None)

    # --numa surviving is the point: the previous behaviour returned [] on any
    # refusal, so one name added to the denylist took every other flag with it.
    assert resolved == ["--numa", "distribute"]


def test_the_override_save_carries_over_without_refusing(monkeypatch):
    # A user changing Context Length on a model whose stored flags predate the
    # denylist must not get a 400 about a flag they are not editing.
    import routes.settings as settings_route

    saved: dict = {}
    stored = {"llama_extra_args": ["--slot-save-path", "/tmp/slots", "--numa", "distribute"]}
    monkeypatch.setattr(
        settings_route, "get_model_override", lambda _id: dict(stored), raising = False
    )
    import utils.openai_auto_switch_settings as oas

    monkeypatch.setattr(oas, "get_model_override", lambda _id: dict(stored))
    monkeypatch.setattr(
        oas,
        "set_model_override",
        lambda model_id, **kwargs: saved.update({model_id: kwargs}),
    )
    monkeypatch.setattr(settings_route, "set_model_override", oas.set_model_override, raising = False)
    monkeypatch.setattr(
        settings_route, "resolve_model_override_keys", lambda _id: ["local/x"], raising = False
    )
    monkeypatch.setattr(settings_route, "cached_repo_alias_keys", lambda _id: [], raising = False)

    payload = settings_route.ModelOverridePayload(model_id = "local/x", max_seq_length = 4096)
    response = settings_route.update_openai_auto_switch_override(payload, current_subject = "t")

    assert response is not None
    written = saved.get("local/x", {})
    kept = written.get("llama_extra_args")
    if kept is not None:
        assert "--slot-save-path" not in kept
        assert "--numa" in kept


def test_the_auto_switch_path_sanitizes_a_legacy_override(monkeypatch):
    # The third carry-over path, and the one with no user in front of it: an OpenAI
    # auto-switch or an idle reload builds a LoadRequest from the stored override.
    # An explicit list is refused with a 400, so a flag denied after it was saved
    # would break that model's automatic loads until someone rewrote the entry.
    from utils.openai_auto_switch_settings import model_override_load_kwargs

    kwargs = model_override_load_kwargs(
        {"llama_extra_args": ["--agent", "--numa", "distribute"], "n_parallel": 4},
        is_gguf = True,
    )

    assert kwargs["llama_extra_args"] == ["--numa", "distribute"]
    assert kwargs["n_parallel"] == 4


def test_the_auto_switch_path_leaves_a_clean_override_alone():
    from utils.openai_auto_switch_settings import model_override_load_kwargs
    kwargs = model_override_load_kwargs(
        {"llama_extra_args": ["--numa", "distribute"]}, is_gguf = True
    )

    assert kwargs["llama_extra_args"] == ["--numa", "distribute"]


def test_trimming_to_the_bounds_never_leaves_a_flag_without_its_value():
    # validate_extra_args knows the arity of only a few flags, so a dangling
    # --grammar passes it and llama-server refuses the launch instead.
    kept, dropped = _lsa.drop_managed_flags(["--top-k", "20", "--grammar", "a" * 40_000])

    assert kept == ["--top-k", "20"]
    assert "--grammar" in dropped
    # And a log line does not carry the 40 KB value that broke the bound.
    assert all(len(name) < 100 for name in dropped)


def test_a_token_that_cannot_be_spawned_is_refused_at_the_boundary():
    # An unpaired surrogate survives JSON and the browser, passes every other check
    # here, and then makes subprocess.Popen raise while it encodes argv, after the
    # load has already begun switching models. A 400 is the honest answer.
    with pytest.raises(ValueError, match = "surrogate"):
        _lsa.validate_extra_args(["--chat-template", "\ud800"])


def test_a_stored_surrogate_is_dropped_like_any_other_unusable_value():
    kept, dropped = _lsa.drop_managed_flags(["--grammar", "\ud800", "--top-k", "20"])

    assert kept == ["--top-k", "20"]
    assert _lsa.validate_extra_args(kept) == kept
    assert "--grammar" in dropped


def test_validate_sizes_itself_with_the_arguments_the_caller_sent():
    # /validate estimates the memory that approves the follow-up /load, and a
    # --ctx-size in the extras changes that estimate. The resolver hands back its
    # fourth argument unchanged for an explicit list, so passing None there meant the
    # preflight approved a different command from the one that runs.
    import inspect

    import routes.inference as inference_route

    source = inspect.getsource(inference_route)
    assert (
        "_resolve_inherited_extra_args(\n            request, config, model_identifier, None\n        )"
        not in source
    )
    assert 'model_identifier, getattr(request, "llama_extra_args", None)' in source

    class _Request:
        llama_extra_args = ["--ctx-size", "8192"]

    class _Config:
        is_gguf = True
        gguf_variant = ""

    # The helper's own contract: an explicit list is returned as given.
    assert inference_route._resolve_inherited_extra_args(
        _Request(), _Config(), "local/x", _Request.llama_extra_args
    ) == ["--ctx-size", "8192"]

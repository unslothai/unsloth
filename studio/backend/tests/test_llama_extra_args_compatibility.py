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
    # The denied one for its name, the other because llama-server refuses the
    # attached spelling itself. Both self-contained, so neither takes a following
    # token with it.
    kept, dropped = _lsa.drop_managed_flags(["--log-file=/x", "--top-k=20", "--numa", "distribute"])

    assert kept == ["--numa", "distribute"]
    assert dropped == ["--log-file", "--top-k"]


def test_an_attached_value_in_the_middle_does_not_take_the_rest_with_it():
    # The trimming loop sheds the TAIL, so a stored "--top-k=20" left for it would
    # cost every flag written after it. Dropped in the walk instead, beside the
    # denied names.
    kept, _dropped = _lsa.drop_managed_flags(
        ["--top-k=20", "--numa", "distribute", "--grammar", "root ::= [0-9]"]
    )
    assert kept == ["--numa", "distribute", "--grammar", "root ::= [0-9]"]


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


def _inherit_with_ctx_flag(monkeypatch, stored, fields_set, max_seq_length):
    """Drive the real resolver for a same-model reload that inherits its extras."""
    import routes.inference as inference_route

    class _Backend:
        extra_args = list(stored)
        extra_args_source = ("local/x", "")

    class _Config:
        is_gguf = True
        gguf_variant = ""

    class _Request:
        llama_extra_args = None
        gguf_variant = ""
        gpu_memory_mode = "auto"
        model_fields_set = set(fields_set)

    _Request.max_seq_length = max_seq_length
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    return inference_route._resolve_inherited_extra_args(_Request(), _Config(), "local/x", None)


def test_a_matching_inherited_ctx_flag_survives_an_apply(monkeypatch):
    """The opt-in has to be durable, or the PR's own fix undoes itself.

    An Apply that re-sends the SAME Context Length is not a fresh save that the
    stored flag would outrank -- it is the same decision, and stripping it here
    relaunched at the VRAM-fit estimate while the stored override still said
    otherwise. Mirrors model_override_load_kwargs on the API auto-switch path;
    both ask matches_explicit_ctx_override so the two cannot drift.
    """
    stored = ["--ctx-size", "100352", "--top-k", "40"]

    assert _inherit_with_ctx_flag(monkeypatch, stored, {"max_seq_length"}, 100352) == [
        "--ctx-size",
        "100352",
        "--top-k",
        "40",
    ]

    # And still stripped alongside another set field, which was the reachable gap.
    assert _inherit_with_ctx_flag(
        monkeypatch, stored, {"max_seq_length", "cache_type_kv"}, 100352
    ) == ["--ctx-size", "100352", "--top-k", "40"]


def test_a_stale_inherited_ctx_flag_still_loses_to_a_fresh_context(monkeypatch):
    """Only a MATCHING value is the opt-in; a different one is a stale shadow."""
    stored = ["--ctx-size", "8192", "--top-k", "40"]

    assert _inherit_with_ctx_flag(monkeypatch, stored, {"max_seq_length"}, 32768) == [
        "--top-k",
        "40",
    ]


def test_a_malformed_inherited_ctx_flag_is_stripped_not_raised(monkeypatch):
    """parse_ctx_override raises on a flag with no value; a load must not."""
    stored = ["--ctx-size", "--top-k", "40"]

    assert _inherit_with_ctx_flag(monkeypatch, stored, {"max_seq_length"}, 32768) == [
        "--top-k",
        "40",
    ]


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
    assert (
        "_public_model_identifier(request.model_path, model_identifier),\n"
        '            getattr(request, "llama_extra_args", None),' in source
    )

    class _Request:
        llama_extra_args = ["--ctx-size", "8192"]

    class _Config:
        is_gguf = True
        gguf_variant = ""

    # The helper's own contract: an explicit list is returned as given.
    assert inference_route._resolve_inherited_extra_args(
        _Request(), _Config(), "local/x", _Request.llama_extra_args
    ) == ["--ctx-size", "8192"]


def test_a_poisoned_flag_takes_its_value_with_it():
    # The control characters are in the FLAG here. Dropping only that token leaves
    # "root ::= ..." as a bare positional, which validate_extra_args accepts and
    # llama-server reads as the model path.
    kept, dropped = _lsa.drop_managed_flags(["--grammar\x1b[2J", "root ::= [0-9]", "--top-k", "20"])

    assert kept == ["--top-k", "20"]
    assert _lsa.validate_extra_args(kept) == kept
    # And the name is not echoed into the log line either: the escape is IN it.
    assert dropped == ["<flag>"]
    assert all("\x1b" not in name for name in dropped)


def test_a_poisoned_flag_with_an_attached_value_drops_alone():
    kept, dropped = _lsa.drop_managed_flags(["--grammar\x1b=root", "--top-k", "20"])

    assert kept == ["--top-k", "20"]
    # The control character is judged first: its name is never echoed into a log,
    # and the attached-value rule would have named it.
    assert dropped == ["<flag>"]


def test_a_bare_value_with_no_flag_is_refused():
    # llama-server answers "invalid argument" and refuses to start, so this is a
    # failed load rather than a 400; and a build that DID take a positional would
    # read it as the model path, which is what denying -m / --model exists to stop.
    for bad in (
        ["/private/models/other.gguf"],
        ["--top-k", "20", "/models/other.gguf"],
    ):
        with pytest.raises(ValueError, match = "bare value"):
            _lsa.validate_extra_args(bad)
    # The attached spelling is refused before anything can be said about what
    # follows it, since llama-server never reads it as a flag at all.
    with pytest.raises(ValueError, match = "two separate arguments"):
        _lsa.validate_extra_args(["--top-k=20", "stray"])


def test_a_value_that_belongs_to_a_flag_is_still_fine():
    assert _lsa.validate_extra_args(["--numa", "distribute"])
    assert _lsa.validate_extra_args(["--grammar", "root ::= [0-9]"])
    # The one flag in llama-server's help that takes two values.
    assert _lsa.validate_extra_args(["--control-vector-layer-range", "1", "10"])


def test_the_underscore_spelling_keeps_its_detached_value():
    # llama.cpp takes both spellings (measured on b10360: `llama-server --ctx_size
    # 4096 --help` prints its help), and _flag_name folds one onto the other. Deciding
    # attachment from that fold read "--ctx_size 4096" as a flag carrying its own
    # value and then refused the 4096 as a bare token, which is a list the CLI has
    # passed through for as long as the field has existed.
    for good in (
        ["--ctx_size", "4096"],
        ["--n_gpu_layers", "5"],
        ["--rope_scaling", "yarn"],
        ["--top-k", "20", "--ctx_size", "4096"],
    ):
        assert _lsa.validate_extra_args(good) == good
    # The value is still consumed exactly once: a second bare token has no owner.
    with pytest.raises(ValueError, match = "bare value"):
        _lsa.validate_extra_args(["--ctx_size", "4096", "stray"])
    # The underscore spelling folds, the attached one does not exist for llama.cpp
    # whichever way it is spelled.
    with pytest.raises(ValueError, match = "two separate arguments"):
        _lsa.validate_extra_args(["--ctx_size=4096"])


def test_a_batch_below_the_floor_is_refused_before_the_launch():
    # llama-server aborts on a batch of 1 at any slot count, and on a batch below the
    # slots it serves (measured, and recorded beside the launcher that raises its own
    # --batch-size for it). Extras are appended after that flag and win, so this is a
    # server that dies during startup, by which time the previous model is unloaded.
    for args, slots in (
        (["-b", "1"], 1),
        (["--batch-size", "0"], 1),
        (["--batch-size=1"], 1),
        (["-b", "2"], 4),
        (["--top-k", "20", "-b", "3"], 4),
    ):
        with pytest.raises(ValueError, match = "aborts on --batch-size"):
            _lsa.check_batch_floor(args, slots)
    # At or above the floor, and anything this side cannot read, is left alone:
    # llama-server names an unreadable value better than a guess here would.
    for args, slots in (
        (["-b", "2"], 1),
        (["-b", "4"], 4),
        (["-b", "8"], 4),
        (["-b", "abc"], 4),
        (["--top-k", "20"], 4),
        ([], 4),
    ):
        assert _lsa.check_batch_floor(args, slots) is None


def test_a_scaled_sidecar_may_take_its_scale_separately():
    # Today's llama.cpp writes it into the value ("--lora-scaled FNAME:SCALE"), older
    # builds took it as its own token, and _sidecar_weight_files reads both. Allowed
    # and never required: demanding the second token would refuse the current syntax,
    # and refusing it broke a list that loaded before the positional check existed.
    for good in (
        ["--lora-scaled", "/a.gguf", "0.5"],
        ["--lora-scaled", "/a.gguf:0.5"],
        ["--lora-scaled", "/a.gguf"],
        ["--control-vector-scaled", "/v.gguf", "0.8"],
        ["--control-vector-scaled", "/v.gguf:0.8", "--top-k", "20"],
        ["--lora-scaled", "/a.gguf", "0.5", "--top-k", "20"],
    ):
        assert _lsa.validate_extra_args(good) == good
    # A third bare token still belongs to nothing.
    with pytest.raises(ValueError, match = "bare value"):
        _lsa.validate_extra_args(["--lora-scaled", "/a.gguf", "0.5", "stray"])


def test_a_two_value_flag_is_kept_whole():
    # Half of this option is not a smaller version of it: llama-server refuses
    # "--control-vector-layer-range 1" on the command line, so a list that carries it
    # is a load that fails at spawn rather than a request that fails at the boundary.
    for bad in (
        ["--control-vector-layer-range"],
        ["--control-vector-layer-range", "1"],
        ["--control-vector-layer-range", "1", "--numa", "distribute"],
        ["--top-k", "20", "--control-vector-layer-range", "1"],
    ):
        with pytest.raises(ValueError, match = "takes two values"):
            _lsa.validate_extra_args(bad)


def test_the_attached_form_of_a_two_value_flag_is_refused_like_any_other():
    # It used to be read as START owing an END, which was a guess about a spelling
    # llama.cpp does not have: the whole token is looked up in its option map, so
    # "--control-vector-layer-range=1" is an argument it has never heard of.
    for bad in (
        ["--control-vector-layer-range=1"],
        ["--control-vector-layer-range=1", "10"],
        ["--control-vector-layer-range=1", "--numa", "distribute"],
    ):
        with pytest.raises(ValueError, match = "two separate arguments"):
            _lsa.validate_extra_args(bad)
    # Detached, it is the one option here whose arity is known for certain.
    assert _lsa.validate_extra_args(["--control-vector-layer-range", "1", "10"])
    with pytest.raises(ValueError, match = "takes two values"):
        _lsa.validate_extra_args(["--control-vector-layer-range", "1"])


def test_trimming_sheds_a_two_value_flag_whole():
    # A list stored by an older build can exceed today's bound, and the trim walks in
    # from the tail. Shedding END alone leaves START looking like an ordinary value,
    # which nothing downstream would object to and llama-server would then reject.
    kept, dropped = _lsa.drop_managed_flags(
        ["--top-k", "20", "--control-vector-layer-range", "1", "10", "--grammar", "x" * 40000]
    )
    assert kept == ["--top-k", "20", "--control-vector-layer-range", "1", "10"]
    kept, dropped = _lsa.drop_managed_flags(
        ["--top-k", "20", "--control-vector-layer-range", "1", "x" * 40000]
    )
    assert kept == ["--top-k", "20"]
    assert "--control-vector-layer-range" in dropped
    # And whatever survives is a list the validator would take.
    assert _lsa.validate_extra_args(kept) == kept


def test_the_windows_check_measures_what_popen_would_write(monkeypatch):
    # list2cmdline is not a sum of lengths: backslashes before a quote double, so an
    # escape-heavy grammar can pass a byte cap and still blow CreateProcess's 32767
    # character limit once quoted, inside Popen, after the switch has begun.
    monkeypatch.setattr(_lsa.sys, "platform", "win32", raising = False)
    value = ("\\" * 10 + '"') * 2000
    assert len(value) < _lsa.MAX_EXTRA_ARGS_BYTES_WINDOWS

    with pytest.raises(ValueError, match = "Windows command line"):
        _lsa.validate_extra_args(["--grammar", value])

    # Same list, other platforms: no such limit, so no refusal.
    monkeypatch.setattr(_lsa.sys, "platform", "linux", raising = False)
    assert _lsa.validate_extra_args(["--grammar", value])


def test_the_loader_and_the_panel_resolve_the_same_row(monkeypatch):
    # The panel used to mirror these rules in the browser, where casefold is only
    # approximated by toLowerCase and an ambiguous fold is easy to get wrong. Both
    # sides now go through this, so what a panel shows and what a load applies cannot
    # disagree.
    import utils.openai_auto_switch_settings as oas

    stored = {
        "/models/Foo.gguf:Q4_K_M": {"llama_extra_args": ["--numa", "distribute"]},
        "unsloth/model-gguf": {"llama_extra_args": ["--top-k", "20"]},
    }
    monkeypatch.setattr(oas, "get_model_overrides", lambda: dict(stored))

    # The load path before the advertised alias, and variant-qualified first. The
    # path keeps its case (POSIX), while the quant folds, which is the rule a browser
    # mirroring this kept getting subtly wrong.
    key, override = oas.resolve_override_for_load(
        "/models/Foo.gguf", "unsloth/model-gguf", "q4_k_m"
    )
    assert override["llama_extra_args"] == ["--numa", "distribute"]
    assert key == "/models/Foo.gguf:Q4_K_M"

    # With no such row, the alias answers.
    key, override = oas.resolve_override_for_load("/models/other.gguf", "unsloth/model-gguf", None)
    assert override["llama_extra_args"] == ["--top-k", "20"]


def test_the_candidate_order_is_the_loaders_own():
    from utils.openai_auto_switch_settings import override_lookup_candidates

    assert override_lookup_candidates("local/x", "alias/x", "Q4") == [
        "local/x:Q4",
        "alias/x:Q4",
        "local/x",
        "alias/x",
    ]
    # A loose .gguf also answers to the filename-label key an early build wrote.
    candidates = override_lookup_candidates("/models/gemma-3-270m-it-Q4_K_M.gguf")
    assert candidates[0] == "/models/gemma-3-270m-it-Q4_K_M.gguf"
    assert any(key.endswith(":Q4_K_M") for key in candidates), candidates


def test_a_flag_padded_with_spaces_is_refused():
    # _flag_name strips before it looks anything up, so a quoted "--top-k " passed the
    # denylist and the arity walk as --top-k and then went to the child with the space
    # still on it. llama.cpp looks the WHOLE token up: measured on b10342, it answers
    # "error: invalid argument: --top-k", naming a flag that reads as correct.
    for bad in (["--top-k ", "20"], [" --top-k", "20"], ["--verbose "]):
        with pytest.raises(ValueError, match = "spaces around"):
            _lsa.validate_extra_args(bad)
    # A VALUE may legitimately carry whitespace: a grammar or a chat template does,
    # and quoting one into a single token is what the box is for.
    assert _lsa.validate_extra_args(["--grammar", "root ::= [0-9] "]) == [
        "--grammar",
        "root ::= [0-9] ",
    ]


def test_a_padded_flag_is_carried_over_by_dropping_it_with_its_value():
    # Dropped in the walk, like the denied names and the attached spelling: left to
    # the trimming loop it would shed the whole tail after it. Its value goes too,
    # since the flag never arrives and an orphan is a bare positional.
    kept, dropped = _lsa.drop_managed_flags(["--top-k ", "20", "--numa", "distribute"])
    assert kept == ["--numa", "distribute"]
    assert dropped == ["--top-k"]
    kept, _dropped = _lsa.drop_managed_flags(["--verbose ", "--numa", "distribute"])
    assert kept == ["--numa", "distribute"]


@pytest.mark.parametrize("flag", ["--parallel", "-np", "--n-parallel"])
def test_parallel_denials_point_at_the_supported_knob(flag):
    # Why (#9510): the parallel slot count IS user-settable, just not through extra args --
    # refusing `--parallel 1` without naming n_parallel sent users to undocumented env hacks.
    with pytest.raises(ValueError, match = "managed by Unsloth Studio.*n_parallel"):
        _lsa.validate_extra_args([flag, "1"])


def test_other_denials_stay_terse():
    with pytest.raises(ValueError, match = "cannot be passed as an extra arg$"):
        _lsa.validate_extra_args(["--model", "/etc/passwd"])

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A flag the caller sends is refused with a 400; one carried over from storage is dropped quietly."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


class _Config:
    is_gguf = True
    gguf_variant = ""


_BACKEND = Path(__file__).resolve().parent.parent
_LSA_PATH = _BACKEND / "core" / "inference" / "llama_server_args.py"
_spec = importlib.util.spec_from_file_location("_lsa_compat_test", _LSA_PATH)
_lsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lsa)

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
    with pytest.raises(ValueError, match = "managed by Unsloth Studio"):
        _lsa.validate_extra_args(stored)


@pytest.mark.parametrize("stored", LEGACY_STORED)
def test_the_drop_helper_keeps_everything_else(stored):
    kept, dropped = _lsa.drop_managed_flags([*stored, "--numa", "distribute"])

    assert kept == ["--numa", "distribute"]
    assert dropped == [_lsa._flag_name(stored[0])]
    assert _lsa.validate_extra_args(kept) == kept


def test_dropping_takes_the_flags_value_with_it():
    # A leftover value becomes a bare positional, which llama.cpp reads as the model path.
    kept, dropped = _lsa.drop_managed_flags(
        ["--top-k", "20", "--log-file", "/var/log/llama.log", "--seed", "1"]
    )

    assert kept == ["--top-k", "20", "--seed", "1"]
    assert dropped == ["--log-file"]


def test_an_attached_value_form_is_dropped_whole():
    kept, dropped = _lsa.drop_managed_flags(["--log-file=/x", "--top-k=20", "--numa", "distribute"])

    assert kept == ["--numa", "distribute"]
    assert dropped == ["--log-file", "--top-k"]


def test_an_attached_value_in_the_middle_does_not_take_the_rest_with_it():
    # The trimming loop sheds the tail, so attached forms are dropped in the walk instead.
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
    kept, dropped = _lsa.drop_managed_flags(["--verbose"] * (_lsa.MAX_EXTRA_ARG_TOKENS + 10))

    assert _lsa.validate_extra_args(kept) == kept
    assert len(dropped) > 0


def test_a_poisoned_value_is_never_echoed_into_the_dropped_list():
    # Stored values may carry ANSI escapes; only the flag name is logged.
    kept, dropped = _lsa.drop_managed_flags(["--grammar", "\x1b[2Jroot ::= [0-9]", "--top-k", "20"])

    assert kept == ["--top-k", "20"]
    assert dropped == ["--grammar", "<value>"]
    assert not any("\x1b" in name for name in dropped)


def test_a_control_character_in_a_stored_value_is_dropped_too():
    kept, _ = _lsa.drop_managed_flags(["--chat-template", "a\x00b", "--top-k", "20"])

    assert _lsa.validate_extra_args(kept) == kept
    assert "--top-k" in kept


def test_the_inherited_load_path_drops_only_the_denied_flag(monkeypatch):
    import routes.inference as inference_route

    assert hasattr(inference_route, "drop_managed_flags"), (
        "the resolver reads this from module globals; an unlisted import NameErrors "
        "only when a model with stored flags is loaded"
    )

    class _Backend:
        extra_args = ["--log-file", "/var/log/llama.log", "--numa", "distribute"]
        # Same model and variant, or the resolver refuses before reaching the drop.
        extra_args_source = ("local/x", "")

    class _Request:
        llama_extra_args = None
        gguf_variant = ""
        gpu_memory_mode = "auto"
        model_fields_set: set = set()

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    resolved = inference_route._resolve_inherited_extra_args(_Request(), _Config(), "local/x", None)

    assert resolved == ["--numa", "distribute"]


def _inherit_with_ctx_flag(monkeypatch, stored, fields_set, max_seq_length):
    """Drive the real resolver for a same-model reload that inherits its extras."""
    import routes.inference as inference_route

    class _Backend:
        extra_args = list(stored)
        extra_args_source = ("local/x", "")

    class _Request:
        llama_extra_args = None
        gguf_variant = ""
        gpu_memory_mode = "auto"
        model_fields_set = set(fields_set)

    _Request.max_seq_length = max_seq_length
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    return inference_route._resolve_inherited_extra_args(_Request(), _Config(), "local/x", None)


def test_a_matching_inherited_ctx_flag_survives_an_apply(monkeypatch):
    """A re-sent matching ctx flag must survive Apply, or relaunch ignores the stored override."""
    stored = ["--ctx-size", "100352", "--top-k", "40"]

    assert _inherit_with_ctx_flag(monkeypatch, stored, {"max_seq_length"}, 100352) == [
        "--ctx-size",
        "100352",
        "--top-k",
        "40",
    ]

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
    kept, dropped = _lsa.drop_managed_flags(["--top-k", "20", "--grammar", "a" * 40_000])

    assert kept == ["--top-k", "20"]
    assert "--grammar" in dropped
    assert all(len(name) < 100 for name in dropped)


def test_a_token_that_cannot_be_spawned_is_refused_at_the_boundary():
    # An unpaired surrogate makes Popen raise while encoding argv, mid model switch.
    with pytest.raises(ValueError, match = "surrogate"):
        _lsa.validate_extra_args(["--chat-template", "\ud800"])


def test_a_stored_surrogate_is_dropped_like_any_other_unusable_value():
    kept, dropped = _lsa.drop_managed_flags(["--grammar", "\ud800", "--top-k", "20"])

    assert kept == ["--top-k", "20"]
    assert _lsa.validate_extra_args(kept) == kept
    assert "--grammar" in dropped


def test_validate_sizes_itself_with_the_arguments_the_caller_sent():
    # --ctx-size in extras changes the /validate estimate, so it must be passed through.
    import routes.inference as inference_route

    import inspect

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

    assert inference_route._resolve_inherited_extra_args(
        _Request(), _Config(), "local/x", _Request.llama_extra_args
    ) == ["--ctx-size", "8192"]


def test_a_poisoned_flag_takes_its_value_with_it():
    # Dropping only the flag would leave its value as a bare positional (read as model path).
    kept, dropped = _lsa.drop_managed_flags(["--grammar\x1b[2J", "root ::= [0-9]", "--top-k", "20"])

    assert kept == ["--top-k", "20"]
    assert _lsa.validate_extra_args(kept) == kept
    assert dropped == ["<flag>"]
    assert all("\x1b" not in name for name in dropped)


def test_a_poisoned_flag_with_an_attached_value_drops_alone():
    kept, dropped = _lsa.drop_managed_flags(["--grammar\x1b=root", "--top-k", "20"])

    assert kept == ["--top-k", "20"]
    assert dropped == ["<flag>"]


def test_a_bare_value_with_no_flag_is_refused():
    for bad in (
        ["/private/models/other.gguf"],
        ["--top-k", "20", "/models/other.gguf"],
    ):
        with pytest.raises(ValueError, match = "bare value"):
            _lsa.validate_extra_args(bad)
    with pytest.raises(ValueError, match = "two separate arguments"):
        _lsa.validate_extra_args(["--top-k=20", "stray"])


def test_a_value_that_belongs_to_a_flag_is_still_fine():
    assert _lsa.validate_extra_args(["--numa", "distribute"])
    assert _lsa.validate_extra_args(["--grammar", "root ::= [0-9]"])
    assert _lsa.validate_extra_args(["--control-vector-layer-range", "1", "10"])


def test_the_underscore_spelling_keeps_its_detached_value():
    # llama.cpp accepts --ctx_size too; attachment must not be decided from the folded name.
    for good in (
        ["--ctx_size", "4096"],
        ["--n_gpu_layers", "5"],
        ["--rope_scaling", "yarn"],
        ["--top-k", "20", "--ctx_size", "4096"],
    ):
        assert _lsa.validate_extra_args(good) == good
    with pytest.raises(ValueError, match = "bare value"):
        _lsa.validate_extra_args(["--ctx_size", "4096", "stray"])
    with pytest.raises(ValueError, match = "two separate arguments"):
        _lsa.validate_extra_args(["--ctx_size=4096"])


def test_a_batch_below_the_floor_is_refused_before_the_launch():
    # llama-server aborts on batch 1 or batch below slots; extras win over our --batch-size.
    for args, slots in (
        (["-b", "1"], 1),
        (["--batch-size", "0"], 1),
        (["--batch-size=1"], 1),
        (["-b", "2"], 4),
        (["--top-k", "20", "-b", "3"], 4),
    ):
        with pytest.raises(ValueError, match = "aborts on --batch-size"):
            _lsa.check_batch_floor(args, slots)
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
    for good in (
        ["--lora-scaled", "/a.gguf", "0.5"],
        ["--lora-scaled", "/a.gguf:0.5"],
        ["--lora-scaled", "/a.gguf"],
        ["--control-vector-scaled", "/v.gguf", "0.8"],
        ["--control-vector-scaled", "/v.gguf:0.8", "--top-k", "20"],
        ["--lora-scaled", "/a.gguf", "0.5", "--top-k", "20"],
    ):
        assert _lsa.validate_extra_args(good) == good
    with pytest.raises(ValueError, match = "bare value"):
        _lsa.validate_extra_args(["--lora-scaled", "/a.gguf", "0.5", "stray"])


def test_a_two_value_flag_is_kept_whole():
    for bad in (
        ["--control-vector-layer-range"],
        ["--control-vector-layer-range", "1"],
        ["--control-vector-layer-range", "1", "--numa", "distribute"],
        ["--top-k", "20", "--control-vector-layer-range", "1"],
    ):
        with pytest.raises(ValueError, match = "takes two values"):
            _lsa.validate_extra_args(bad)


def test_the_attached_form_of_a_two_value_flag_is_refused_like_any_other():
    for bad in (
        ["--control-vector-layer-range=1"],
        ["--control-vector-layer-range=1", "10"],
        ["--control-vector-layer-range=1", "--numa", "distribute"],
    ):
        with pytest.raises(ValueError, match = "two separate arguments"):
            _lsa.validate_extra_args(bad)
    assert _lsa.validate_extra_args(["--control-vector-layer-range", "1", "10"])
    with pytest.raises(ValueError, match = "takes two values"):
        _lsa.validate_extra_args(["--control-vector-layer-range", "1"])


def test_trimming_sheds_a_two_value_flag_whole():
    kept, dropped = _lsa.drop_managed_flags(
        ["--top-k", "20", "--control-vector-layer-range", "1", "10", "--grammar", "x" * 40000]
    )
    assert kept == ["--top-k", "20", "--control-vector-layer-range", "1", "10"]
    kept, dropped = _lsa.drop_managed_flags(
        ["--top-k", "20", "--control-vector-layer-range", "1", "x" * 40000]
    )
    assert kept == ["--top-k", "20"]
    assert "--control-vector-layer-range" in dropped
    assert _lsa.validate_extra_args(kept) == kept


def test_the_windows_check_measures_what_popen_would_write(monkeypatch):
    # list2cmdline doubles backslashes before quotes, so quoting can exceed 32767 chars.
    monkeypatch.setattr(_lsa.sys, "platform", "win32", raising = False)
    value = ("\\" * 10 + '"') * 2000
    assert len(value) < _lsa.MAX_EXTRA_ARGS_BYTES_WINDOWS

    with pytest.raises(ValueError, match = "Windows command line"):
        _lsa.validate_extra_args(["--grammar", value])

    monkeypatch.setattr(_lsa.sys, "platform", "linux", raising = False)
    assert _lsa.validate_extra_args(["--grammar", value])


def test_the_loader_and_the_panel_resolve_the_same_row(monkeypatch):
    import utils.openai_auto_switch_settings as oas

    stored = {
        "/models/Foo.gguf:Q4_K_M": {"llama_extra_args": ["--numa", "distribute"]},
        "unsloth/model-gguf": {"llama_extra_args": ["--top-k", "20"]},
    }
    monkeypatch.setattr(oas, "get_model_overrides", lambda: dict(stored))

    key, override = oas.resolve_override_for_load(
        "/models/Foo.gguf", "unsloth/model-gguf", "q4_k_m"
    )
    assert override["llama_extra_args"] == ["--numa", "distribute"]
    assert key == "/models/Foo.gguf:Q4_K_M"

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
    candidates = override_lookup_candidates("/models/gemma-3-270m-it-Q4_K_M.gguf")
    assert candidates[0] == "/models/gemma-3-270m-it-Q4_K_M.gguf"
    assert any(key.endswith(":Q4_K_M") for key in candidates), candidates


def test_a_flag_padded_with_spaces_is_refused():
    # llama.cpp looks up the whole token, so a flag with a trailing space is invalid.
    for bad in (["--top-k ", "20"], [" --top-k", "20"], ["--verbose "]):
        with pytest.raises(ValueError, match = "spaces around"):
            _lsa.validate_extra_args(bad)
    assert _lsa.validate_extra_args(["--grammar", "root ::= [0-9] "]) == [
        "--grammar",
        "root ::= [0-9] ",
    ]


def test_a_padded_flag_is_carried_over_by_dropping_it_with_its_value():
    kept, dropped = _lsa.drop_managed_flags(["--top-k ", "20", "--numa", "distribute"])
    assert kept == ["--numa", "distribute"]
    assert dropped == ["--top-k"]
    kept, _dropped = _lsa.drop_managed_flags(["--verbose ", "--numa", "distribute"])
    assert kept == ["--numa", "distribute"]


@pytest.mark.parametrize("flag", ["--parallel", "-np", "--n-parallel"])
def test_parallel_denials_point_at_the_supported_knob(flag):
    with pytest.raises(ValueError, match = "managed by Unsloth Studio.*n_parallel"):
        _lsa.validate_extra_args([flag, "1"])


def test_other_denials_stay_terse():
    with pytest.raises(ValueError, match = "cannot be passed as an extra arg$"):
        _lsa.validate_extra_args(["--model", "/etc/passwd"])

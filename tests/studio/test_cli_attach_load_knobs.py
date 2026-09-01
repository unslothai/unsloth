# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import pytest
import typer

import unsloth_cli.commands.start as start_cli


BASE = "http://127.0.0.1:8888"
KEY = "k"
RESIDENT = {"id": "unsloth/Qwen3-8B", "loaded": True}


class FakeServer:
    """/v1/models advertises the sanitized public id; only the status identifier is
    what the load endpoint dedupes against."""

    def __init__(self, models, status):
        self.models = models
        self.status = status
        self.loads = []
        self.requests = []

    def http_json(
        self,
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        self.requests.append((method, url))
        if url.endswith("/v1/models"):
            return {"data": [dict(m) for m in self.models]}
        if url.endswith("/api/inference/status"):
            return dict(self.status)
        raise AssertionError(f"unexpected request: {method} {url}")

    def load(self, base, key, requested, load, payload):
        self.loads.append(payload)
        public = start_cli._public_model_id(requested) or requested
        if not any(m.get("id") == public for m in self.models):
            self.models.append({"id": public, "loaded": True})
        return {"status": "already_loaded", "model": requested}

    def install(self, monkeypatch):
        monkeypatch.setattr(start_cli, "_http_json", self.http_json)
        monkeypatch.setattr(start_cli, "_load_model_with_progress", self.load)
        return self


@pytest.fixture
def loads(monkeypatch):
    calls = []

    def _load(base, key, requested, load, payload):
        calls.append(payload)
        return {"status": "already_loaded", "model": requested}

    monkeypatch.setattr(start_cli, "_loaded_models", lambda base, key: [dict(RESIDENT)])
    monkeypatch.setattr(start_cli, "_load_model_with_progress", _load)
    monkeypatch.setattr(start_cli, "_http_json", lambda *a, **k: {})
    return calls


def sent(server, index = 0):
    """The load payload without force_reload.

    Every inferred attach whose settings status PROVED to differ carries it, so the
    tests that care assert it on its own rather than repeating it in each payload.
    """
    return {k: v for k, v in server.loads[index].items() if k != "force_reload"}


def test_context_length_without_model_reloads_the_resident_model(loads):
    entry = start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(max_seq_length = 32768),
    )
    assert loads == [{"model_path": RESIDENT["id"], "max_seq_length": 32768}]
    assert entry["id"] == RESIDENT["id"]


def test_gguf_variant_without_model_is_forwarded(loads):
    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gguf_variant = "UD-Q8_K_XL", max_seq_length = 32768),
    )
    assert loads == [
        {
            "model_path": RESIDENT["id"],
            "gguf_variant": "UD-Q8_K_XL",
            "max_seq_length": 32768,
        }
    ]


def test_bare_attach_does_not_load(loads):
    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions())
    assert loads == []
    assert entry["id"] == RESIDENT["id"]


def test_bare_attach_does_not_query_status(monkeypatch):
    """A bare attach stays a pure read of /v1/models."""
    server = FakeServer(
        [dict(RESIDENT)],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions())

    assert server.loads == []
    assert entry["id"] == RESIDENT["id"]
    assert not any(url.endswith("/api/inference/status") for _, url in server.requests)


def test_path_loaded_resident_is_reloaded_by_its_real_path(monkeypatch):
    """_same_loaded_identifier compares resident paths exactly, so the load must carry
    the identifier from status; a basename would resolve as a brand new model."""
    path = "/srv/models/Foo-Q4_K_M.gguf"
    server = FakeServer(
        [{"id": "Foo-Q4_K_M", "loaded": True}],
        {
            "is_gguf": True,
            "active_model": "Foo-Q4_K_M",
            "model_identifier": path,
            "gguf_variant": "Q4_K_M",
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)

    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert sent(server) == {"model_path": path, "max_seq_length": 32768}
    assert entry["id"] == "Foo-Q4_K_M"


def test_inferred_target_still_runs_the_preload_check(monkeypatch):
    """preload_check is the only gate before the load; _require_gguf_for_codex runs
    after _connect returns, i.e. after the shared model is already evicted."""
    server = FakeServer(
        [dict(RESIDENT)],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    def gate(
        base,
        key,
        model,
        variant = None,
    ):
        raise typer.Exit(code = 1)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(
            BASE,
            KEY,
            None,
            start_cli.LoadOptions(max_seq_length = 4096),
            preload_check = gate,
        )

    assert server.loads == []


def test_inferred_reload_warns_that_it_unloads_for_every_session(monkeypatch, capsys):
    """A changed context restarts llama-server for every attached session."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": True,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "gguf_variant": "Q4_K_M",
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert len(server.loads) == 1
    assert "unloads the current model for every attached session" in capsys.readouterr().out


def test_active_model_decides_the_resident_not_list_order(monkeypatch):
    """Cached, speech and chat entries coexist; order does not name the resident."""
    server = FakeServer(
        [
            {"id": "unsloth/whisper-large", "loaded": True},
            {"id": RESIDENT["id"], "loaded": True},
        ],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert sent(server) == {"model_path": RESIDENT["id"], "max_seq_length": 32768}


def test_unreloadable_resident_fails_before_loading(monkeypatch):
    """A native lease redacts model_identifier; guessing the basename is wrong."""
    server = FakeServer(
        [{"id": "Foo-Q4_K_M", "loaded": True}],
        {"is_gguf": True, "active_model": "Foo-Q4_K_M", "model_identifier": None},
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads == []


def test_explicit_flags_matching_defaults_still_reload(monkeypatch):
    """--context-length 0 and --no-tensor-parallel are resets, not omissions."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": True,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 32768,
            "tensor_parallel": True,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(
            max_seq_length = 0,
            tensor_parallel = False,
            supplied = frozenset({"max_seq_length", "tensor_parallel"}),
        ),
    )

    assert sent(server) == {
        "model_path": RESIDENT["id"],
        "max_seq_length": 0,
        "tensor_parallel": False,
    }


def test_inferred_attach_pins_the_resident_quant(monkeypatch):
    """A repo-id GGUF must be re-sent with the quant it is running.

    The load endpoint re-resolves a repo id, and ModelConfig.from_identifier auto-picks
    the preferred variant when none is sent, so posting model_path alone would evict a
    UI-chosen Q8_0 and download UD-Q4_K_XL instead of only changing the context.
    """
    server = FakeServer(
        [{"id": "unsloth/Qwen3-30B-A3B-GGUF", "loaded": True}],
        {
            "is_gguf": True,
            "active_model": "unsloth/Qwen3-30B-A3B-GGUF",
            "model_identifier": "unsloth/Qwen3-30B-A3B-GGUF",
            "gguf_variant": "Q8_0",
            "requested_context_length": 0,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(max_seq_length = 8192, supplied = frozenset({"max_seq_length"})),
    )

    assert sent(server) == {
        "model_path": "unsloth/Qwen3-30B-A3B-GGUF",
        "gguf_variant": "Q8_0",
        "max_seq_length": 8192,
    }


def test_inferred_attach_at_the_default_context_does_not_reresolve_the_quant(monkeypatch):
    """`--context-length 0` is a reset, so it now posts a load where it once posted none.

    That load must still name the running quant, or the reset silently swaps the weights.
    """
    server = FakeServer(
        [{"id": "unsloth/Qwen3-30B-A3B-GGUF", "loaded": True}],
        {
            "is_gguf": True,
            "active_model": "unsloth/Qwen3-30B-A3B-GGUF",
            "model_identifier": "unsloth/Qwen3-30B-A3B-GGUF",
            "gguf_variant": "Q8_0",
            "requested_context_length": 0,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(max_seq_length = 0, supplied = frozenset({"max_seq_length"})),
    )

    assert sent(server) == {
        "model_path": "unsloth/Qwen3-30B-A3B-GGUF",
        "gguf_variant": "Q8_0",
        "max_seq_length": 0,
    }


def test_inferred_attach_does_not_pin_a_variant_onto_a_direct_gguf_file(monkeypatch):
    """A direct .gguf path is loaded as itself; the server never redirects it by variant."""
    path = "/srv/models/Foo-Q4_K_M.gguf"
    server = FakeServer(
        [{"id": "Foo-Q4_K_M", "loaded": True}],
        {
            "is_gguf": True,
            "active_model": "Foo-Q4_K_M",
            "model_identifier": path,
            "gguf_variant": "Q4_K_M",
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert sent(server) == {"model_path": path, "max_seq_length": 32768}


def test_inferred_attach_to_a_non_gguf_resident_sends_no_variant(monkeypatch):
    """status.gguf_variant is absent for safetensors/MLX; nothing may be invented."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert sent(server) == {"model_path": RESIDENT["id"], "max_seq_length": 32768}


def test_hf_cache_resident_matches_the_advertised_repo_id(monkeypatch):
    """The server maps a cache path to its repo id; our basename helper does not."""
    cache_path = (
        "/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-8B-GGUF"
        "/snapshots/abc123/qwen3-8b-Q4_K_M.gguf"
    )
    server = FakeServer(
        [{"id": "unsloth/Qwen3-8B-GGUF", "loaded": True}],
        {
            "is_gguf": True,
            "active_model": "unsloth/Qwen3-8B-GGUF",
            "model_identifier": cache_path,
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)

    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert sent(server) == {"model_path": cache_path, "max_seq_length": 32768}
    assert entry["id"] == "unsloth/Qwen3-8B-GGUF"


def test_status_names_the_resident_even_when_the_catalog_lags(monkeypatch):
    """A status id absent from /v1/models must not fall back to list order."""
    server = FakeServer(
        [{"id": "unsloth/whisper-large", "loaded": True}],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
        },
    ).install(monkeypatch)

    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert sent(server) == {"model_path": RESIDENT["id"], "max_seq_length": 32768}
    assert entry["id"] == RESIDENT["id"]


def test_a_freshly_started_server_is_not_reloaded(monkeypatch):
    """_connect passes requested=None after auto-starting a server FROM these knobs."""
    server = FakeServer(
        [dict(RESIDENT)],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    entry = start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(max_seq_length = 32768),
        infer_resident = False,
    )

    assert server.loads == []
    assert entry["id"] == RESIDENT["id"]


def test_omitted_default_flags_are_not_forwarded(monkeypatch):
    """A bare --model load still sends only model_path."""
    server = FakeServer(
        [dict(RESIDENT)],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, "unsloth/Qwen3-14B", start_cli.LoadOptions())

    assert sent(server) == {"model_path": "unsloth/Qwen3-14B"}


class TestExplicitFlagsThroughTheRealCli:
    """Typer vendors its own click (so a ParameterSource identity test matches nothing)
    and invokes callbacks with no active click context (so reading it globally fails
    too). Both regressions silently look like a bare attach."""

    @staticmethod
    def _load_for(argv):
        from typer.testing import CliRunner

        captured = {}
        real_connect = start_cli._connect

        def fake_connect(api_key, model, load, *a, **k):
            captured["load"] = load
            raise SystemExit(0)

        start_cli._connect = fake_connect
        try:
            CliRunner().invoke(start_cli.start_app, argv)
        finally:
            start_cli._connect = real_connect
        return captured.get("load")

    @pytest.mark.parametrize(
        "flag, expected",
        [
            (["--context-length", "0"], "max_seq_length"),
            (["--load-in-4bit"], "load_in_4bit"),
            (["--no-tensor-parallel"], "tensor_parallel"),
            (["--gpu-memory-mode", "auto"], "gpu_memory_mode"),
        ],
    )
    def test_flags_equal_to_their_default_are_recorded(self, flag, expected):
        load = self._load_for(["codex", "--no-launch", *flag])
        assert load is not None, "the command never reached _connect"
        # Value equals the declared default, so only `supplied` carries the intent.
        assert expected in load.supplied
        assert expected in load.overrides()

    def test_a_bare_invocation_records_nothing(self):
        load = self._load_for(["codex", "--no-launch"])
        assert load is not None
        assert load.supplied == frozenset()
        assert load.overrides() == frozenset()

    @pytest.mark.parametrize("command", ["codex", "claude", "opencode", "hermes", "pi", "openclaw"])
    def test_every_agent_command_tracks_flags_identically(self, command):
        load = self._load_for([command, "--no-launch", "--context-length", "0"])
        assert load is not None, f"{command} never reached _connect"
        assert "max_seq_length" in load.supplied


def test_inferred_reload_carries_the_resident_runtime_settings(monkeypatch):
    """A reload is not a PATCH: unnamed knobs are reset unless we resend them."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": True,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "gguf_variant": "Q8_0",
            "requested_context_length": 4096,
            "cache_type_kv": "q8_0",
            "requested_parallel_slots": 4,
            "requested_n_batch": 1024,
            "requested_llama_extra_args": ["--foo"],
            "tensor_split": [0.5, 0.5],
            # Never set, so it must be omitted rather than sent as null.
            "requested_load_mode": None,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    sent = server.loads[0]
    assert sent["cache_type_kv"] == "q8_0"
    assert sent["n_parallel"] == 4
    assert sent["n_batch"] == 1024
    assert sent["llama_extra_args"] == ["--foo"]
    assert sent["tensor_split"] == [0.5, 0.5]
    assert "load_mode" not in sent


def test_user_supplied_knobs_beat_the_resident_values(monkeypatch):
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": True,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
            "tensor_parallel": True,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(tensor_parallel = False, supplied = frozenset({"tensor_parallel"})),
    )

    assert server.loads[0]["tensor_parallel"] is False


def test_explicit_zero_context_forces_the_reload(monkeypatch):
    """The server reads a bare 0 as "no preference", so say outright this is a reload."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 32768,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(max_seq_length = 0, supplied = frozenset({"max_seq_length"})),
    )

    assert server.loads[0]["max_seq_length"] == 0
    assert server.loads[0]["force_reload"] is True


def test_a_provable_no_op_is_not_forced(monkeypatch):
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 32768,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert "force_reload" not in server.loads[0]


def test_an_older_server_is_never_force_reloaded(monkeypatch):
    """_load_settings_differ cannot prove anything without status; forcing would evict."""
    server = FakeServer([dict(RESIDENT)], {}).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert "force_reload" not in server.loads[0]


def test_four_bit_flag_does_not_warn_about_a_gguf_resident(monkeypatch, capsys):
    """GGUF reports load_in_4bit as null because it has none; that is not a difference."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": True,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "gguf_variant": "Q8_0",
            "requested_context_length": 4096,
            "load_in_4bit": None,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(load_in_4bit = True, supplied = frozenset({"load_in_4bit"})),
    )

    assert "unloads the current model" not in capsys.readouterr().out


def test_status_reporting_no_chat_resident_does_not_pick_a_speech_sidecar(monkeypatch):
    """active_model null is a definitive "nothing is resident", not a missing endpoint."""
    server = FakeServer(
        [{"id": "unsloth/whisper-large-v3", "loaded": True}],
        {"is_gguf": False, "active_model": None, "model_identifier": None},
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads == []


def test_failed_inferred_load_names_the_inferred_resident(monkeypatch, capsys):
    """Catalog order can name a speech sidecar; the survivor probe must use the target."""
    server = FakeServer(
        [
            {"id": "unsloth/whisper-large-v3", "loaded": True},
            {"id": RESIDENT["id"], "loaded": True},
        ],
        {
            "is_gguf": True,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "gguf_variant": "Q8_0",
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("load refused")

    monkeypatch.setattr(start_cli, "_load_model_with_progress", boom)
    monkeypatch.setattr(start_cli, "_model_still_loaded", lambda *a: False)

    with pytest.raises(RuntimeError):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    # The chat model did NOT survive, so no reassuring message may be printed.
    assert "Nothing was unloaded" not in capsys.readouterr().err


def test_inferred_reload_keeps_a_full_precision_resident(monkeypatch):
    """LoadRequest defaults load_in_4bit to True, so omitting it quantizes the model.

    A non-GGUF resident loaded with load_in_4bit False, attached to with only a context
    change, must come back at the same precision.
    """
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
            "load_in_4bit": False,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads[0]["load_in_4bit"] is False


def test_an_explicit_precision_flag_beats_the_resident(monkeypatch):
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
            "load_in_4bit": False,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(load_in_4bit = True, supplied = frozenset({"load_in_4bit"})),
    )

    assert server.loads[0]["load_in_4bit"] is True


def test_a_gguf_resident_is_sent_no_precision_flag(monkeypatch):
    """GGUF reports load_in_4bit null because it has none, and nulls are dropped."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": True,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "gguf_variant": "Q8_0",
            "requested_context_length": 4096,
            "load_in_4bit": None,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert "load_in_4bit" not in server.loads[0]


def _gguf_status(**extra):
    base = {
        "is_gguf": True,
        "active_model": RESIDENT["id"],
        "model_identifier": RESIDENT["id"],
        "gguf_variant": "Q4_K_M",
        "requested_context_length": 8192,
    }
    base.update(extra)
    return base


def test_changing_one_knob_keeps_the_custom_context(monkeypatch):
    """max_seq_length defaults to 0, which the intent copies into n_ctx."""
    server = FakeServer([dict(RESIDENT)], _gguf_status()).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(tensor_parallel = True, supplied = frozenset({"tensor_parallel"})),
    )

    assert server.loads[0]["max_seq_length"] == 8192


def test_remote_code_resident_is_refused_before_the_load(monkeypatch):
    """The payload cannot carry the consent, and the worker dies before the retry."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
            "requires_trust_remote_code": True,
        },
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads == []


def test_tensor_parallel_does_not_restart_a_non_gguf_resident(monkeypatch, capsys):
    """The standard load never forwards it, so a restart would apply nothing."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
            "tensor_parallel": False,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(tensor_parallel = True, supplied = frozenset({"tensor_parallel"})),
    )

    assert "force_reload" not in server.loads[0]
    assert "unloads the current model" not in capsys.readouterr().out


def test_a_differently_spelled_quant_still_counts_as_a_change(monkeypatch, capsys):
    """Q4KM really reloads on the server, so stripping separators under-warns."""
    server = FakeServer([dict(RESIDENT)], _gguf_status()).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(gguf_variant = "Q4KM"))

    assert server.loads[0]["force_reload"] is True
    assert "unloads the current model" in capsys.readouterr().out


def test_manual_mode_keeps_a_pinned_layer_count(monkeypatch, capsys):
    """gpu_layers = -1 means "pick them", which would discard the pinned placement."""
    server = FakeServer(
        [dict(RESIDENT)],
        _gguf_status(gpu_memory_mode = "manual", gpu_layers = 20),
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gpu_memory_mode = "manual", supplied = frozenset({"gpu_memory_mode"})),
    )

    assert server.loads[0]["gpu_layers"] == 20
    # Preserving the count makes this a real no-op, so it must NOT restart the server.
    assert "force_reload" not in server.loads[0]
    assert "unloads the current model" not in capsys.readouterr().out


def test_switching_into_manual_still_asks_for_automatic_layers(monkeypatch):
    server = FakeServer([dict(RESIDENT)], _gguf_status(gpu_memory_mode = "auto")).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gpu_memory_mode = "manual", supplied = frozenset({"gpu_memory_mode"})),
    )

    assert server.loads[0]["gpu_layers"] == -1


def test_arch_gated_tensor_request_is_not_restarted(monkeypatch):
    """status says false because the gate normalized it, not because it was not asked."""
    server = FakeServer(
        [dict(RESIDENT)],
        _gguf_status(tensor_parallel = False, tensor_parallel_dropped_by_arch_gate = True),
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(tensor_parallel = True, supplied = frozenset({"tensor_parallel"})),
    )

    assert "force_reload" not in server.loads[0]


def test_paravirtual_placement_is_not_restarted(monkeypatch):
    """Every placement request normalizes to the same runtime on such a host."""
    server = FakeServer(
        [dict(RESIDENT)],
        _gguf_status(gpu_memory_mode = "manual", gpu_placement_paravirtual = True),
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gpu_memory_mode = "auto", supplied = frozenset({"gpu_memory_mode"})),
    )

    assert "force_reload" not in server.loads[0]


def test_a_no_op_attach_to_a_custom_code_resident_is_allowed(monkeypatch):
    """The refusal exists to protect a reload; a no-op has no reload to protect."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
            "load_in_4bit": True,
            "requires_trust_remote_code": True,
        },
    ).install(monkeypatch)

    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 4096))

    assert "force_reload" not in server.loads[0]
    assert entry["id"] == RESIDENT["id"]


def test_cpu_fallback_placement_is_not_restarted(monkeypatch):
    """_preserve_cpu_fallback_intent keeps this runtime across the reload anyway."""
    server = FakeServer(
        [dict(RESIDENT)],
        _gguf_status(
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            cpu_fallback_reason = "vulkan_startup_crash",
        ),
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gpu_memory_mode = "auto", supplied = frozenset({"gpu_memory_mode"})),
    )

    assert "force_reload" not in server.loads[0]


def test_the_requested_mlx_kv_width_survives_a_reload(monkeypatch):
    """The applied value is null when the runtime refused it; that must not round-trip."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
            "load_in_4bit": True,
            "mlx_kv_bits": None,
            "mlx_kv_bits_requested": 4,
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads[0]["mlx_kv_bits"] == 4


def test_a_proven_no_op_skips_the_preload_gate(monkeypatch):
    """Nothing is evicted, so the gate protects nothing and would reject a live attach."""
    server = FakeServer([dict(RESIDENT)], _gguf_status()).install(monkeypatch)
    checked = []

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(max_seq_length = 8192),
        preload_check = lambda *a: checked.append(a),
    )

    assert checked == []


def test_a_real_change_still_runs_the_preload_gate(monkeypatch):
    server = FakeServer([dict(RESIDENT)], _gguf_status()).install(monkeypatch)

    def gate(
        base,
        key,
        model,
        variant = None,
    ):
        raise typer.Exit(code = 1)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(
            BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768), preload_check = gate
        )

    assert server.loads == []


def test_a_quant_override_on_a_direct_file_is_refused(monkeypatch):
    """from_identifier consults a variant only for a directory, so this cannot apply."""
    path = "/srv/models/Foo-Q4_K_M.gguf"
    server = FakeServer(
        [{"id": "Foo-Q4_K_M", "loaded": True}],
        {
            "is_gguf": True,
            "active_model": "Foo-Q4_K_M",
            "model_identifier": path,
            "gguf_variant": "Q4_K_M",
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(gguf_variant = "UD-Q8_K_XL"))

    assert server.loads == []


def test_explicit_tensor_disable_clears_an_arch_gated_fallback(monkeypatch, capsys):
    """The backend keeps the tensor intent behind the fallback; only a reload clears it."""
    server = FakeServer(
        [dict(RESIDENT)],
        _gguf_status(tensor_parallel = False, tensor_parallel_dropped_by_arch_gate = True),
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(tensor_parallel = False, supplied = frozenset({"tensor_parallel"})),
    )

    assert server.loads[0]["force_reload"] is True
    assert "unloads the current model" in capsys.readouterr().out


def test_switching_into_manual_is_still_a_change(monkeypatch, capsys):
    """Only manual-to-manual is the no-op; auto-to-manual really does reload."""
    server = FakeServer([dict(RESIDENT)], _gguf_status(gpu_memory_mode = "auto")).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gpu_memory_mode = "manual", supplied = frozenset({"gpu_memory_mode"})),
    )

    assert server.loads[0]["force_reload"] is True
    assert "unloads the current model" in capsys.readouterr().out


def test_no_status_and_one_loaded_model_still_works(monkeypatch):
    """An older server with an unambiguous catalog is answerable."""
    server = FakeServer([dict(RESIDENT)], {}).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads[0]["model_path"] == RESIDENT["id"]


def test_no_status_and_several_loaded_models_refuses_to_guess(monkeypatch):
    """/v1/models lists loaded speech sidecars, so order is not evidence."""
    server = FakeServer(
        [
            {"id": "unsloth/whisper-large-v3", "loaded": True},
            dict(RESIDENT),
        ],
        {},
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads == []


def test_non_gguf_gpu_selection_survives_a_reload(monkeypatch):
    """Without this the replacement load falls back to automatic GPU selection."""
    server = FakeServer(
        [dict(RESIDENT)],
        {
            "is_gguf": False,
            "active_model": RESIDENT["id"],
            "model_identifier": RESIDENT["id"],
            "requested_context_length": 4096,
            "load_in_4bit": True,
            "requested_gpu_ids": [2, 3],
        },
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads[0]["gpu_ids"] == [2, 3]


def _direct_gguf_server(monkeypatch, path = "/srv/models/Foo-Q4_K_M.gguf"):
    return FakeServer(
        [{"id": "Foo-Q4_K_M", "loaded": True}],
        {
            "is_gguf": True,
            "active_model": "Foo-Q4_K_M",
            "model_identifier": path,
            "gguf_variant": "Q4_K_M",
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)


def test_restating_the_running_quant_applies_the_other_overrides(monkeypatch):
    """A matching variant asks for no change, so it must not block the context change."""
    path = "/srv/models/Foo-Q4_K_M.gguf"
    server = _direct_gguf_server(monkeypatch, path)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gguf_variant = "Q4_K_M", max_seq_length = 32768),
    )

    sent_payload = server.loads[0]
    assert sent_payload["model_path"] == path
    assert sent_payload["max_seq_length"] == 32768
    # The field is inapplicable to a direct file, so it is dropped rather than posted.
    assert "gguf_variant" not in sent_payload


def test_a_matching_quant_is_compared_case_insensitively(monkeypatch):
    server = _direct_gguf_server(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gguf_variant = "q4_k_m", max_seq_length = 32768),
    )

    assert server.loads[0]["max_seq_length"] == 32768


def test_a_differing_quant_on_a_direct_file_is_still_refused(monkeypatch):
    server = _direct_gguf_server(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(
            BASE,
            KEY,
            None,
            start_cli.LoadOptions(gguf_variant = "UD-Q8_K_XL", max_seq_length = 32768),
        )

    assert server.loads == []

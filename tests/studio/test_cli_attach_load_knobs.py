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
    """Serves /v1/models and /api/inference/status; records every load payload."""

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
    """The load payload without force_reload, which the tests that care assert alone."""
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
    """A bare attach must not even ask for status."""
    server = FakeServer(
        [dict(RESIDENT)],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions())

    assert server.loads == []
    assert entry["id"] == RESIDENT["id"]
    assert not any(url.endswith("/api/inference/status") for _, url in server.requests)


def test_path_loaded_resident_is_reloaded_by_its_real_path(monkeypatch):
    """The load carries the identifier from status, not the advertised basename."""
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
    """preload_check still runs on an inferred target."""
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
    """A settings change is announced as an unload."""
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
    """active_model, not catalog order, names the resident."""
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
    """A redacted model_identifier is refused, not guessed from the basename."""
    server = FakeServer(
        [{"id": "Foo-Q4_K_M", "loaded": True}],
        {"is_gguf": True, "active_model": "Foo-Q4_K_M", "model_identifier": None},
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads == []


def test_explicit_flags_matching_defaults_still_reload(monkeypatch):
    """A flag typed at its default value is still sent."""
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
    """A repo-id GGUF is re-sent with the quant it is running."""
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
    """The --context-length 0 reset still names the running quant."""
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
    """A direct .gguf path is sent with no variant."""
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
    """A non-GGUF resident is sent no variant."""
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
    """A cache-path resident resolves to the repo id the server advertises."""
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
    """A status id absent from /v1/models still names the resident."""
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
    """An auto-started server passes infer_resident False."""
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
    """A bare --model load sends only model_path."""
    server = FakeServer(
        [dict(RESIDENT)],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, "unsloth/Qwen3-14B", start_cli.LoadOptions())

    assert sent(server) == {"model_path": "unsloth/Qwen3-14B"}


class TestExplicitFlagsThroughTheRealCli:
    """`supplied` tracking through the real Typer and Click stack."""

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
    """Knobs the user did not name are carried over from the resident."""
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
    """An explicit 0 is force_reloaded."""
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
    """No status means no proof, so no force_reload."""
    server = FakeServer([dict(RESIDENT)], {}).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert "force_reload" not in server.loads[0]


def test_four_bit_flag_does_not_warn_about_a_gguf_resident(monkeypatch, capsys):
    """A null load_in_4bit on GGUF is not a difference."""
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
    """A null active_model is refused, not resolved from the catalog."""
    server = FakeServer(
        [{"id": "unsloth/whisper-large-v3", "loaded": True}],
        {"is_gguf": False, "active_model": None, "model_identifier": None},
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads == []


def test_failed_inferred_load_names_the_inferred_resident(monkeypatch, capsys):
    """The survivor probe uses the inferred target."""
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

    assert "Nothing was unloaded" not in capsys.readouterr().err


def test_inferred_reload_keeps_a_full_precision_resident(monkeypatch):
    """A full-precision resident comes back at the same precision."""
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
    """A GGUF resident is sent no load_in_4bit."""
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
    """A custom context survives a change to another knob."""
    server = FakeServer([dict(RESIDENT)], _gguf_status()).install(monkeypatch)

    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(tensor_parallel = True, supplied = frozenset({"tensor_parallel"})),
    )

    assert server.loads[0]["max_seq_length"] == 8192


def test_remote_code_resident_is_refused_before_the_load(monkeypatch):
    """A trust_remote_code resident is refused before anything is evicted."""
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
    """tensor_parallel is GGUF-only."""
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
    """Q4KM and Q4_K_M are compared as typed."""
    server = FakeServer([dict(RESIDENT)], _gguf_status()).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(gguf_variant = "Q4KM"))

    assert server.loads[0]["force_reload"] is True
    assert "unloads the current model" in capsys.readouterr().out


def test_manual_mode_keeps_a_pinned_layer_count(monkeypatch, capsys):
    """A resident already in manual keeps its pinned layer count."""
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
    """Re-asking for an arch-gated tensor request is a no-op."""
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
    """Placement cannot differ on a paravirtual host."""
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
    """Placement cannot differ under a CPU fallback."""
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
    """The requested MLX KV width round-trips, not the refused applied one."""
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
    """A proven no-op skips the gate."""
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
    """A different quant on a direct .gguf file is refused."""
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
    """Turning tensor_parallel off clears an arch-gated fallback."""
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
    """auto to manual is a real change."""
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
    """One loaded model is unambiguous without status."""
    server = FakeServer([dict(RESIDENT)], {}).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads[0]["model_path"] == RESIDENT["id"]


def test_no_status_and_several_loaded_models_refuses_to_guess(monkeypatch):
    """Several loaded models without status is refused."""
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
    """GPU placement is carried over."""
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
    """Restating the running quant does not block the other overrides."""
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


def test_a_diffusion_resident_is_never_an_attach_target(monkeypatch):
    """An image runtime answers with an active_model but can never serve chat."""
    server = FakeServer(
        [{"id": "unsloth/FLUX.1-dev-GGUF", "loaded": True}],
        {
            "is_gguf": True,
            "is_diffusion": True,
            "active_model": "unsloth/FLUX.1-dev-GGUF",
            "model_identifier": "unsloth/FLUX.1-dev-GGUF",
            "requested_context_length": 4096,
        },
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768))

    assert server.loads == []


@pytest.mark.parametrize(
    "field",
    ["spec_probe_retry_pending", "spec_dflash_retry_pending", "spec_fallback_binary_changed"],
)
def test_a_pending_retry_is_not_a_no_op(monkeypatch, capsys, field):
    """_runtime_matches_intent reloads on an identical intent while a retry is pending."""
    server = FakeServer([dict(RESIDENT)], _gguf_status(**{field: True})).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 8192))

    assert server.loads[0]["force_reload"] is True
    assert "unloads the current model" in capsys.readouterr().out


def test_no_pending_retry_is_still_a_no_op(monkeypatch):
    server = FakeServer([dict(RESIDENT)], _gguf_status()).install(monkeypatch)

    start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 8192))

    assert "force_reload" not in server.loads[0]

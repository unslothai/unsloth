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
    """A /v1/models + /api/inference/status pair that answers as the real server does.

    /v1/models advertises the sanitized public id (public_model_id strips a directory
    and the .gguf suffix), while /api/inference/status reports the identifier the load
    endpoint actually dedupes against. Keeping the two distinct is the whole point:
    a resident loaded by path is reachable only through the status field.
    """

    def __init__(self, models, status):
        self.models = models
        self.status = status
        self.loads = []
        self.requests = []

    def http_json(self, method, url, token, payload = None, timeout = 30, error = None):
        self.requests.append((method, url))
        if url.endswith("/v1/models"):
            return {"data": [dict(m) for m in self.models]}
        if url.endswith("/api/inference/status"):
            return dict(self.status)
        raise AssertionError(f"unexpected request: {method} {url}")

    def load(self, base, key, requested, load, payload):
        self.loads.append(payload)
        # The server registers what it loaded, so the post-load catalog match finds it.
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
    """Attaching with no knobs stays a pure read of /v1/models."""
    server = FakeServer(
        [dict(RESIDENT)],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions())

    assert server.loads == []
    assert entry["id"] == RESIDENT["id"]
    assert not any(url.endswith("/api/inference/status") for _, url in server.requests)


def test_path_loaded_resident_is_reloaded_by_its_real_path(monkeypatch):
    """/v1/models shows the basename; the load must carry the path status reports.

    _same_loaded_identifier compares a resident local path with os.path.normcase
    equality, so posting "Foo-Q4_K_M" can never dedupe against
    /srv/models/Foo-Q4_K_M.gguf -- and the server would try to resolve the basename
    as a brand new model.
    """
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

    entry = start_cli._resolve_model(
        BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768)
    )

    assert server.loads == [{"model_path": path, "max_seq_length": 32768}]
    # The agent is still pointed at the public id, never the server's filesystem path.
    assert entry["id"] == "Foo-Q4_K_M"


def test_inferred_target_still_runs_the_preload_check(monkeypatch):
    """Codex's pre-eviction GGUF gate is the only check that runs before the load.

    _require_gguf_for_codex runs after _connect returns, i.e. after the resident model
    has already been evicted for every attached session, so dropping the preload check
    turns a rejected launch into a destructive one.
    """
    server = FakeServer(
        [dict(RESIDENT)],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    def gate(base, key, model, variant = None):
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
    """A changed context restarts llama-server for everyone; say so before doing it."""
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

    start_cli._resolve_model(
        BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768)
    )

    assert len(server.loads) == 1
    assert "unloads the current model for every attached session" in capsys.readouterr().out


def test_active_model_decides_the_resident_not_list_order(monkeypatch):
    """Cached, speech and chat entries coexist; list order does not name the resident."""
    server = FakeServer(
        [
            {"id": "unsloth/whisper-large", "loaded": True},
            {"id": RESIDENT["id"], "loaded": True},
        ],
        {"is_gguf": False, "active_model": RESIDENT["id"], "model_identifier": RESIDENT["id"]},
    ).install(monkeypatch)

    start_cli._resolve_model(
        BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768)
    )

    assert server.loads == [{"model_path": RESIDENT["id"], "max_seq_length": 32768}]


def test_unreloadable_resident_fails_before_loading(monkeypatch):
    """A native lease-backed load redacts model_identifier; guessing the basename is wrong."""
    server = FakeServer(
        [{"id": "Foo-Q4_K_M", "loaded": True}],
        {"is_gguf": True, "active_model": "Foo-Q4_K_M", "model_identifier": None},
    ).install(monkeypatch)

    with pytest.raises(typer.Exit):
        start_cli._resolve_model(
            BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768)
        )

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

    assert server.loads == [
        {"model_path": RESIDENT["id"], "max_seq_length": 0, "tensor_parallel": False}
    ]


def test_hf_cache_resident_matches_the_advertised_repo_id(monkeypatch):
    """The server maps a cache path to its repo id; our basename helper does not.

    /v1/models advertises `unsloth/Qwen3-8B-GGUF` for a snapshot under
    models--unsloth--Qwen3-8B-GGUF, while stripping the basename would give
    `qwen3-8b-Q4_K_M`. Matching on that alone would report a successful load as
    "Unsloth didn't report it as loaded".
    """
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

    entry = start_cli._resolve_model(
        BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768)
    )

    assert server.loads == [{"model_path": cache_path, "max_seq_length": 32768}]
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

    entry = start_cli._resolve_model(
        BASE, KEY, None, start_cli.LoadOptions(max_seq_length = 32768)
    )

    assert server.loads == [{"model_path": RESIDENT["id"], "max_seq_length": 32768}]
    assert entry["id"] == RESIDENT["id"]


def test_a_freshly_started_server_is_not_reloaded(monkeypatch):
    """_connect passes requested=None after auto-starting a server FROM these knobs.

    They are already in effect there, so inferring a target would reload the model
    the CLI just finished loading.
    """
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

    assert server.loads == [{"model_path": "unsloth/Qwen3-14B"}]


class TestExplicitFlagsThroughTheRealCli:
    """`supplied` must be populated by an actual command invocation.

    This is not covered by calling _resolve_model directly. Typer vendors its own
    copy of click, so the context hands back a typer._click ParameterSource, a
    different enum class from click.core's -- an identity test against either
    silently matches nothing and every explicit-default flag is lost with no error.
    Typer also invokes the callback with no ACTIVE click context, so reading the
    context globally instead of taking the command's own `ctx` fails the same
    silent way. Both regressions look exactly like a bare attach.
    """

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
        # The value matches the declared default, so only `supplied` can carry the
        # user's intent; overrides() has nothing else to go on.
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

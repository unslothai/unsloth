# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `unsloth studio run --parallel` CLI flag.

Pre-PR `llama_parallel_slots` was hardcoded to 4. These tests pin
the typer Option (aliases, default 4, 1..64 range), the
typer/denylist subset invariant, and re-exec forwarding.

See ``test_studio_run_short_alias_clashes.py`` for the argv
canonicaliser and the legacy `-m` / `-hfr` / `-f` shim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_run_command():
    """Import `studio` without triggering server start; backend imports
    are lazy inside run()."""
    from unsloth_cli.commands import studio as _studio
    return _studio


def test_parallel_option_is_registered():
    """The `--parallel` flag (with aliases) must be on the `run` command."""
    studio_mod = _load_run_command()
    import inspect

    run_fn = studio_mod.run
    sig = inspect.signature(run_fn)
    assert "parallel" in sig.parameters, "missing `parallel` parameter on run()"

    param = sig.parameters["parallel"]
    opt = param.default  # typer.OptionInfo
    flags = set()
    decls = getattr(opt, "param_decls", None) or []
    for d in decls:
        flags.add(d)
    for required in ("--parallel", "--n-parallel", "-np"):
        assert required in flags, f"flag {required!r} missing from --parallel option"


def test_context_length_alias_is_registered():
    """`--context-length` is an operator-facing alias for --max-seq-length."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["max_seq_length"].default
    flags = set(getattr(opt, "param_decls", None) or [])
    assert "--max-seq-length" in flags
    assert "--context-length" in flags


def test_parallel_default_is_four():
    """Default must stay at 4 so plain `unsloth studio run` is unchanged."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["parallel"].default
    default = getattr(opt, "default", None)
    assert default == 4, f"default changed to {default}; would silently alter existing deployments"


def test_parallel_range_guards_are_set():
    """Range guards: 1 <= N <= 64. Outside this is a hard reject."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["parallel"].default
    assert getattr(opt, "min", None) == 1, "min must be 1 (0 = no decode possible)"
    assert getattr(opt, "max", None) == 64, "max must be 64 (KV split sanity cap)"


def test_typer_parallel_aliases_are_subset_of_backend_denylist():
    """Every typer alias for --parallel must be denied on the backend
    too; otherwise HTTP /load could smuggle the value via
    `llama_extra_args` and desync llama_parallel_slots from the
    running llama-server."""
    studio_mod = _load_run_command()
    import inspect
    import importlib.util

    # Load llama_server_args.py directly so the test doesn't need the
    # backend's full runtime chain (fastapi / structlog / loggers /
    # utils.hardware) installed -- the invariant is just about the
    # _DENYLIST_GROUPS tuple.
    lsa_path = (
        Path(__file__).resolve().parents[2]
        / "studio"
        / "backend"
        / "core"
        / "inference"
        / "llama_server_args.py"
    )
    spec = importlib.util.spec_from_file_location("_lsa_for_subset_test", lsa_path)
    lsa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lsa)
    _DENYLIST_GROUPS = lsa._DENYLIST_GROUPS

    parallel_group = next((g for g in _DENYLIST_GROUPS if "--parallel" in g), None)
    assert parallel_group is not None, "denylist must include a --parallel group"

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["parallel"].default
    typer_aliases = set(getattr(opt, "param_decls", []) or [])
    missing = typer_aliases - parallel_group
    assert not missing, (
        f"typer aliases {missing!r} are not in the backend denylist; "
        f"add them to _DENYLIST_GROUPS to keep /load from desyncing "
        f"llama_parallel_slots."
    )


# test_in_venv_path_passes_parallel_to_run_server (below) is the runtime
# equivalent of the retired source-text guard for hardcoded
# `llama_parallel_slots = 4`.


# Re-exec arg-builder coverage. run() re-execs into the studio venv
# (execvp on POSIX, Popen on Windows). Without explicit forwarding the
# child reverts to typer defaults and silently drops the user's value.


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _install_reexec_capture(monkeypatch, *, platform):
    studio_mod = _load_run_command()
    captured = []

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")

    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    fake_python = fake_venv / "bin" / "python"
    fake_bin = fake_venv / "bin" / "unsloth"
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_python)

    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if str(self) == str(fake_bin) else real_is_file(self),
    )

    # resolve_tool_policy is imported lazily inside run(); patch the source.
    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )

    monkeypatch.setattr(sys, "platform", platform)

    def capture(kind, argv):
        captured.append(
            {
                "kind": kind,
                "argv": list(argv),
                "start_api_key_marker": studio_mod.os.environ.get(
                    studio_mod._START_API_KEY_MARKER_ENV
                ),
            }
        )

    def fake_execvp(file, argv):
        capture("execvp", argv)
        raise _ExecCaptured(argv)

    class _FakePopen:
        def __init__(self, argv, *a, **kw):
            capture("popen", argv)
            self._argv = argv

        def wait(self):
            raise _ExecCaptured(self._argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)
    monkeypatch.setattr(studio_mod.subprocess, "Popen", _FakePopen)

    return captured


def _invoke_run(
    monkeypatch,
    args,
    *,
    platform = "linux",
):
    import typer as _typer

    studio_mod = _load_run_command()
    captured = _install_reexec_capture(monkeypatch, platform = platform)
    app = _typer.Typer()
    app.command(
        context_settings = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        },
    )(studio_mod.run)
    result = CliRunner().invoke(app, args, catch_exceptions = True)
    return result, captured


def _value_after(argv, flag):
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


_BASE = ["--model", "unsloth/Qwen3-1.7B-GGUF"]


@pytest.mark.parametrize(
    "flag,value",
    [("--parallel", "8"), ("--n-parallel", "16"), ("-np", "32")],
)
def test_reexec_forwards_parallel_all_aliases(monkeypatch, flag, value):
    """Every alias the user can type must reach the re-exec'd child."""
    result, captured = _invoke_run(monkeypatch, _BASE + [flag, value])
    assert (
        len(captured) == 1
    ), f"expected one launch via re-exec, got {captured}; output={result.output!r}"
    argv = captured[0]["argv"]
    assert (
        _value_after(argv, "--parallel") == value
    ), f"{flag} {value} was dropped on re-exec; argv = {argv}"


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_reexec_hands_off_start_api_key_marker_out_of_band(monkeypatch, platform):
    """A new child receives the marker while an old child sees no unknown flag."""
    result, captured = _invoke_run(
        monkeypatch,
        _BASE + ["--start-api-key-marker"],
        platform = platform,
    )
    assert len(captured) == 1, result.output
    assert "--start-api-key-marker" not in captured[0]["argv"]
    assert captured[0]["start_api_key_marker"] == "1"


def test_reexeced_child_consumes_start_api_key_marker_env(monkeypatch):
    """A supported child consumes the handoff before starting descendants."""
    studio_mod = _load_run_command()
    monkeypatch.setenv(studio_mod._START_API_KEY_MARKER_ENV, "1")

    inherited = studio_mod._consume_start_api_key_marker_env()

    assert inherited is True
    assert studio_mod._START_API_KEY_MARKER_ENV not in studio_mod.os.environ


def test_run_default_sets_tool_call_env(monkeypatch):
    """Plain `unsloth run` enables healing and nudging via the inherited env
    (written before the re-exec so the child server picks them up at import)."""
    studio_mod = _load_run_command()
    monkeypatch.delenv("UNSLOTH_DISABLE_TOOL_CALL_HEALING", raising = False)
    monkeypatch.delenv("UNSLOTH_TOOL_CALL_NUDGE", raising = False)
    _invoke_run(monkeypatch, _BASE)
    assert studio_mod.os.environ["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "0"
    assert studio_mod.os.environ["UNSLOTH_TOOL_CALL_NUDGE"] == "1"


def test_run_disable_flags_set_tool_call_env(monkeypatch):
    """`--disable-tool-call-healing --disable-tool-call-nudging` flips both env vars."""
    studio_mod = _load_run_command()
    monkeypatch.delenv("UNSLOTH_DISABLE_TOOL_CALL_HEALING", raising = False)
    monkeypatch.delenv("UNSLOTH_TOOL_CALL_NUDGE", raising = False)
    _invoke_run(
        monkeypatch,
        _BASE + ["--disable-tool-call-healing", "--disable-tool-call-nudging"],
    )
    assert studio_mod.os.environ["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] == "1"
    assert studio_mod.os.environ["UNSLOTH_TOOL_CALL_NUDGE"] == "0"


@pytest.mark.parametrize("inherited", ["0", "false", "False", "no", ""])
def test_run_omitted_flag_respects_inherited_env(monkeypatch, inherited):
    """When the flag is omitted, a value the parent set (e.g. `unsloth start`) wins
    instead of being reset to the default."""
    studio_mod = _load_run_command()
    monkeypatch.setenv("UNSLOTH_TOOL_CALL_NUDGE", inherited)
    monkeypatch.delenv("UNSLOTH_DISABLE_TOOL_CALL_HEALING", raising = False)
    _invoke_run(monkeypatch, _BASE)
    assert studio_mod.os.environ["UNSLOTH_TOOL_CALL_NUDGE"] == inherited


_SAMPLING_ENV_SUFFIXES = (
    "TEMPERATURE",
    "TOP_P",
    "TOP_K",
    "MIN_P",
    "REPETITION_PENALTY",
    "PRESENCE_PENALTY",
)


def test_run_sampling_flags_set_env(monkeypatch):
    """`--temperature`/`--top-k` write UNSLOTH_SAMPLING_* (a hard override the backend applies);
    an omitted sampling flag leaves its env unset so the per-model recommendation stays."""
    studio_mod = _load_run_command()
    for _v in _SAMPLING_ENV_SUFFIXES:
        monkeypatch.delenv(f"UNSLOTH_SAMPLING_{_v}", raising = False)
    _invoke_run(monkeypatch, _BASE + ["--temperature", "0.3", "--top-k", "40"])
    assert studio_mod.os.environ["UNSLOTH_SAMPLING_TEMPERATURE"] == "0.3"
    assert studio_mod.os.environ["UNSLOTH_SAMPLING_TOP_K"] == "40"
    assert "UNSLOTH_SAMPLING_TOP_P" not in studio_mod.os.environ


def test_run_no_sampling_flags_leaves_env_unset(monkeypatch):
    """Plain `unsloth run` writes no UNSLOTH_SAMPLING_*; the server keeps the recommendation."""
    studio_mod = _load_run_command()
    for _v in _SAMPLING_ENV_SUFFIXES:
        monkeypatch.delenv(f"UNSLOTH_SAMPLING_{_v}", raising = False)
    _invoke_run(monkeypatch, _BASE)
    assert not any(k.startswith("UNSLOTH_SAMPLING_") for k in studio_mod.os.environ)


def test_run_rejects_out_of_range_sampling_flag(monkeypatch):
    """typer enforces the documented ranges before a value can reach the server."""
    result, _captured = _invoke_run(monkeypatch, _BASE + ["--temperature", "9"])
    assert result.exit_code != 0


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_reexec_argv_is_consistent_across_platforms(monkeypatch, platform):
    """Linux/Darwin (execvp) and Windows (Popen) must build the same argv."""
    result, captured = _invoke_run(monkeypatch, _BASE + ["--parallel", "12"], platform = platform)
    assert len(captured) == 1
    expected_kind = "popen" if platform == "win32" else "execvp"
    assert (
        captured[0]["kind"] == expected_kind
    ), f"{platform}: expected launcher {expected_kind}, got {captured[0]['kind']}"
    assert _value_after(captured[0]["argv"], "--parallel") == "12"


def test_reexec_np_is_first_class_alias(monkeypatch):
    """`-np` must reach the child as --parallel <N>. Pre-PR Click
    clustered `-np 8` as `-p 8` (port=8) + stray `-n`; also pin that
    --port is no longer collateral damage."""
    result, captured = _invoke_run(monkeypatch, _BASE + ["-np", "8"])
    assert len(captured) == 1
    argv = captured[0]["argv"]
    assert (
        _value_after(argv, "--parallel") == "8"
    ), f"-np 8 silently became 4 after re-exec; argv = {argv}"
    # `-np 8` must not clobber --port (default 8888).
    assert _value_after(argv, "--port") == "8888", argv


def test_reexec_forwards_context_length_alias(monkeypatch):
    """Alias should normalize to the existing child --max-seq-length flag."""
    result, captured = _invoke_run(monkeypatch, _BASE + ["--context-length", "8192"])
    assert len(captured) == 1, result.output
    argv = captured[0]["argv"]
    assert _value_after(argv, "--max-seq-length") == "8192", argv
    assert "--context-length" not in argv, argv


def test_reexec_mixed_parallel_with_passthrough(monkeypatch):
    """--parallel + llama-server pass-through flags must all reach the child."""
    result, captured = _invoke_run(
        monkeypatch,
        # --top-k is now a first-class sampling flag (routed via UNSLOTH_SAMPLING_*), so use
        # --seed / --temp here, which remain genuine llama-server pass-through flags.
        _BASE + ["--parallel", "8", "--seed", "42", "--temp", "0.7"],
    )
    assert len(captured) == 1
    argv = captured[0]["argv"]
    assert _value_after(argv, "--parallel") == "8", argv
    assert _value_after(argv, "--seed") == "42", argv
    assert _value_after(argv, "--temp") == "0.7", argv


def test_context_length_banner_line_formats_ints():
    studio_mod = _load_run_command()
    assert studio_mod._format_context_length_line({"context_length": 4096}) == (
        "  Context length: 4096 tokens"
    )
    assert studio_mod._format_context_length_line({"context_length": "8192"}) == (
        "  Context length: 8192 tokens"
    )


@pytest.mark.parametrize("value", [None, 0, -1, True, ""])
def test_context_length_banner_line_omits_unknown_values(value):
    studio_mod = _load_run_command()
    assert studio_mod._format_context_length_line({"context_length": value}) is None


@pytest.mark.parametrize(
    "user_flag,expected_in_child",
    [
        ("--load-in-4bit", "--load-in-4bit"),
        ("--no-load-in-4bit", "--no-load-in-4bit"),
        (None, "--load-in-4bit"),  # default True
    ],
)
def test_reexec_forwards_load_in_4bit_in_both_directions(monkeypatch, user_flag, expected_in_child):
    """Re-exec must emit the chosen polarity (or the typer default),
    so a future default flip on one layer can't silently invert
    behaviour for users who never typed the flag."""
    extras = [user_flag] if user_flag else []
    result, captured = _invoke_run(monkeypatch, _BASE + extras)
    assert len(captured) == 1
    argv = captured[0]["argv"]
    other_polarity = (
        "--no-load-in-4bit" if expected_in_child == "--load-in-4bit" else "--load-in-4bit"
    )
    assert expected_in_child in argv, f"expected {expected_in_child} in child argv; got {argv}"
    assert other_polarity not in argv, f"unexpected {other_polarity} in child argv; got {argv}"


# Runtime check: fake sys.prefix into the studio venv to bypass
# re-exec, then assert run_server receives --parallel as
# llama_parallel_slots.


class _RunServerCaptured(SystemExit):
    def __init__(self, kwargs):
        super().__init__(0)
        self.kwargs = dict(kwargs)


def _types_module(name):
    import types as _types
    return _types.ModuleType(name)


def test_studio_default_rejects_parallel_when_subcommand_invoked():
    """`unsloth studio --parallel 8 run ...` would silently drop the 8
    (typer doesn't forward parent options to subcommands). The
    callback rejects with exit 2 and points at the subcommand flag."""
    studio_mod = _load_run_command()
    import typer as _typer

    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")

    runner = CliRunner()
    result = runner.invoke(app, ["studio", "--parallel", "8", "run", "--model", "X"])
    assert result.exit_code == 2, (
        f"expected exit 2 when --parallel is on studio group with a "
        f"subcommand invoked; got {result.exit_code}; output={result.output!r}"
    )
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--parallel" in combined, combined
    assert (
        "run --parallel 8" in combined
    ), f"error message must show the corrected invocation; got: {combined}"


def test_studio_default_rejects_api_only_when_subcommand_invoked():
    """`unsloth studio --api-only run ...` would silently serve the UI (the
    parent's --api-only never reaches run). The callback rejects with exit 2
    and points at the subcommand flag."""
    studio_mod = _load_run_command()
    import typer as _typer

    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")

    runner = CliRunner()
    result = runner.invoke(app, ["studio", "--api-only", "run", "--model", "X"])
    assert result.exit_code == 2, (
        f"expected exit 2 when --api-only is on studio group with a "
        f"subcommand invoked; got {result.exit_code}; output={result.output!r}"
    )
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--api-only" in combined, combined
    assert (
        "run --api-only" in combined
    ), f"error message must show the corrected invocation; got: {combined}"


def test_studio_default_default_parallel_with_subcommand_does_not_error():
    """Omitting --parallel on the group must still let subcommands
    run; the group's default 1 is benign."""
    studio_mod = _load_run_command()
    import typer as _typer

    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    runner = CliRunner()
    result = runner.invoke(app, ["studio", "--help"])
    assert result.exit_code == 0, result.output


def test_studio_default_exposes_parallel_option():
    """Plain `unsloth studio` exposes --parallel too so the API-only
    path can raise concurrency without going through the denied
    pass-through. Default stays at 1 (pre-PR); `run` keeps its 4."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.studio_default)
    assert (
        "parallel" in sig.parameters
    ), "studio_default missing `parallel`; API-only path can't set llama_parallel_slots"
    opt = sig.parameters["parallel"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--parallel" in decls
    assert "--n-parallel" in decls
    assert (
        getattr(opt, "default", None) == 1
    ), "studio_default --parallel must default to 1 (pre-PR); `run` is 4"
    assert getattr(opt, "min", None) == 1
    assert getattr(opt, "max", None) == 64


@pytest.mark.parametrize("value", [1, 4, 8, 64])
def test_in_venv_path_passes_parallel_to_run_server(monkeypatch, value):
    """In-venv path must forward --parallel to
    run_server(llama_parallel_slots=N), not the old hardcoded 4."""
    studio_mod = _load_run_command()

    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    # Pin STUDIO_HOME so sys.prefix.startswith() picks the in-venv branch.
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", fake_venv.parent)

    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )

    captured: dict = {}

    def fake_run_server(**kwargs):
        captured.update(kwargs)
        raise _RunServerCaptured(kwargs)

    fake_backend_run = sys.modules.setdefault(
        "studio.backend.run", _types_module("studio.backend.run")
    )
    fake_backend_run.run_server = fake_run_server
    fake_backend_run._resolve_external_ip = lambda: "127.0.0.1"
    # run() loads the backend via _load_run_module() (by file path), which
    # ignores a sys.modules mock with no matching __file__; inject it as the
    # cached run module so the stubbed run_server is used.
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", fake_backend_run)

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        },
    )(studio_mod.run)
    CliRunner().invoke(app, _BASE + ["--parallel", str(value)], catch_exceptions = True)

    assert (
        captured.get("llama_parallel_slots") == value
    ), f"run_server got llama_parallel_slots={captured.get('llama_parallel_slots')!r}, expected {value}"


# --api-only: serve API only (no UI). Both re-exec and in-venv paths must carry it.


def test_api_only_option_is_registered():
    studio_mod = _load_run_command()
    import inspect

    opt = inspect.signature(studio_mod.run).parameters["api_only"].default
    assert "--api-only" in set(getattr(opt, "param_decls", []) or [])
    assert getattr(opt, "default", None) is False  # opt-in; plain run keeps the UI


@pytest.mark.parametrize(
    "extra,present",
    [
        (["--api-only"], True),
        (["--secure", "--api-only"], True),  # secure headless path
        ([], False),
    ],
)
def test_reexec_forwards_api_only(monkeypatch, extra, present):
    """`--api-only` (and only when typed) must reach the re-exec'd child."""
    result, captured = _invoke_run(monkeypatch, _BASE + extra)
    assert len(captured) == 1, result.output
    argv = captured[0]["argv"]
    assert ("--api-only" in argv) is present, argv


@pytest.mark.parametrize("extra,expected", [(["--api-only"], True), ([], False)])
def test_in_venv_path_passes_api_only_to_run_server(monkeypatch, extra, expected):
    """In-venv path must forward --api-only to run_server(api_only=...)."""
    studio_mod = _load_run_command()

    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", fake_venv.parent)

    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )

    captured: dict = {}

    def fake_run_server(**kwargs):
        captured.update(kwargs)
        raise _RunServerCaptured(kwargs)

    fake_backend_run = sys.modules.setdefault(
        "studio.backend.run", _types_module("studio.backend.run")
    )
    fake_backend_run.run_server = fake_run_server
    fake_backend_run._resolve_external_ip = lambda: "127.0.0.1"
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", fake_backend_run)

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        },
    )(studio_mod.run)
    CliRunner().invoke(app, _BASE + extra, catch_exceptions = True)

    assert (
        captured.get("api_only") is expected
    ), f"run_server got api_only={captured.get('api_only')!r}, expected {expected}"
    # Headless serving must suppress the Tauri-only TAURI_PORT line.
    assert (
        captured.get("emit_tauri_port") is False
    ), f"run_server got emit_tauri_port={captured.get('emit_tauri_port')!r}, expected False"

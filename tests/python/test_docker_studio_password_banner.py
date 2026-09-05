# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""`docker logs` must show how to sign in to Studio, and a preset password must
survive restarts.

Studio writes its generated first-boot password to a file and prints only the
path, so a launcher banner saying "password below" left users with nothing to
type. The one-shot `studio-password` supervisord program prints the credential
once the file exists, or says why there is none, then a ready block once both
services answer. `unsloth studio` exits 1 when handed an initial password after
one is stored, so `unsloth-studio-run` applies UNSLOTH_STUDIO_PASSWORD only while
nothing is stored and the launcher keeps it out of supervisord's environment.
These run the shipped scripts against a fake Studio home.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
SCRIPT = DOCKER / "studio_password.sh"
RUN = DOCKER / "studio_run.sh"
LAUNCH = DOCKER / "studio_launch.sh"
SUPERVISORD = DOCKER / "supervisord.conf"
DOCKERFILE = DOCKER / "Dockerfile.studio"

behavioural = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _stub(bin_dir: Path, name: str, body: str) -> None:
    bin_dir.mkdir(exist_ok = True)
    path = bin_dir / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding = "utf-8")
    path.chmod(0o755)


def _clean_env(bin_dir: Path) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_STUDIO")}
    e["PATH"] = f"{bin_dir}{os.pathsep}" + e["PATH"]
    return e


def _auth_db(
    home: Path,
    *,
    must_change: int | None,
    legacy: bool = False,
) -> None:
    """auth.db as the backend creates it; None = the file exists but no admin row;
    legacy = the schema from before must_change_password existed."""
    auth = home / "auth"
    auth.mkdir(exist_ok = True)
    conn = sqlite3.connect(auth / "auth.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS auth_user (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, "
        "password_salt TEXT NOT NULL, password_hash TEXT NOT NULL, jwt_secret TEXT NOT NULL"
        + (")" if legacy else ", must_change_password INTEGER NOT NULL DEFAULT 0)")
    )
    if legacy:
        conn.execute(
            "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret) "
            "VALUES ('unsloth', 's', 'h', 'j')"
        )
    elif must_change is not None:
        conn.execute(
            "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret, must_change_password) "
            "VALUES ('unsloth', 's', 'h', 'j', ?)",
            (must_change,),
        )
    conn.commit()
    conn.close()


# --- studio-password: the login line and the ready summary -----------------------


def _run(
    home: Path,
    *,
    env: dict | None = None,
    wait: str = "3",
    services_up: bool = True,
) -> subprocess.CompletedProcess:
    """Run the shipped script against *home*, with curl stubbed: the ready probes
    must never reach a real Studio on the test host."""
    bin_dir = home / "stub-bin"
    _stub(bin_dir, "curl", "exit 0\n" if services_up else "exit 7\n")
    e = _clean_env(bin_dir)
    e.update(
        UNSLOTH_STUDIO_HOME = str(home),
        UNSLOTH_STUDIO_PASSWORD_WAIT = wait,
        UNSLOTH_STUDIO_READY_WAIT = "2",
    )
    e.update(env or {})
    return subprocess.run(["bash", str(SCRIPT)], capture_output = True, text = True, env = e, timeout = 60)


@behavioural
def test_the_generated_password_is_printed_once_studio_writes_it(tmp_path: Path):
    auth = tmp_path / "auth"
    auth.mkdir()

    def _studio_writes_it_later():
        time.sleep(1.0)
        (auth / ".bootstrap_password").write_bytes(b"s3cret pass\n")

    threading.Thread(target = _studio_writes_it_later).start()
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_PASSWORD_STATE": "generated"})
    assert res.returncode == 0, res.stderr
    assert "username: unsloth" in res.stdout
    assert "password: s3cret pass" in res.stdout, res.stdout
    assert "60 minutes" in res.stdout, "the change-it-or-shut-down window is not explained"
    assert "Unsloth container ready" in res.stdout
    assert (
        "Studio      http://localhost:8000   username: unsloth   password: s3cret pass"
        in res.stdout
    )


@behavioural
def test_the_summary_carries_the_jupyter_note_and_port(tmp_path: Path):
    res = _run(
        tmp_path,
        env = {
            "UNSLOTH_STUDIO_PASSWORD_STATE": "initial",
            "JUPYTER_PORT": "9999",
            "UNSLOTH_JUPYTER_NOTE": "generated password: abc123",
        },
    )
    assert (
        "JupyterLab  http://localhost:9999   generated password: abc123" in res.stdout
    ), res.stdout
    assert res.stdout.rstrip().endswith("=" * 72)


@behavioural
def test_services_that_never_answer_are_reported_not_hidden(tmp_path: Path):
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_PASSWORD_STATE": "initial"}, services_up = False)
    assert res.returncode == 0
    assert "startup incomplete" in res.stdout, res.stdout
    assert res.stdout.count("not answering") == 2


@behavioural
def test_a_crlf_file_does_not_leak_the_cr_into_the_credential(tmp_path: Path):
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / ".bootstrap_password").write_bytes(b"abc\r\n")
    res = _run(tmp_path)
    assert "password: abc   (" in res.stdout, repr(res.stdout)


@behavioural
@pytest.mark.parametrize("value", ["0", "-5", " -1 "])
def test_a_disabled_timeout_drops_the_shutdown_note(tmp_path: Path, value: str):
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / ".bootstrap_password").write_bytes(b"abc\n")
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": value})
    assert "Unsloth Studio login -> username: unsloth   password: abc\n" in res.stdout, repr(
        res.stdout
    )
    assert "change it on first sign-in" not in res.stdout


@behavioural
@pytest.mark.parametrize("value", ["abc", "", "  ", "1.5", "- 5", "1 000"])
def test_a_malformed_timeout_keeps_the_note_like_the_backend_does(tmp_path: Path, value: str):
    """bootstrap_timeout.py strips only the surrounding whitespace and falls back
    to 3600 on a typo rather than disabling, so Studio still stops after an hour;
    the note must not vanish. "- 5" and "1 000" are typos, not numbers."""
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / ".bootstrap_password").write_bytes(b"abc\n")
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": value})
    assert "60 minutes" in res.stdout, repr(res.stdout)


@behavioural
@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("+0900", "15 minutes"),
        ("30", "30 seconds"),
        ("90", "1 minute 30 seconds"),
        ("61", "1 minute 1 second"),
        ("1_000", "16 minutes 40 seconds"),
    ],
)
def test_the_timeout_is_reported_like_the_backend_formats_it(tmp_path: Path, value: str, text: str):
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / ".bootstrap_password").write_bytes(b"abc\n")
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": value})
    assert f"stops after {text} with" in res.stdout, repr(res.stdout)


@behavioural
def test_an_initial_password_is_never_echoed(tmp_path: Path):
    res = _run(
        tmp_path,
        env = {
            "UNSLOTH_STUDIO_PASSWORD_STATE": "initial",
            "UNSLOTH_STUDIO_PASSWORD": "hunter22hunter",
        },
    )
    assert res.returncode == 0
    assert "hunter22hunter" not in res.stdout
    assert "from UNSLOTH_STUDIO_PASSWORD" in res.stdout


@behavioural
def test_a_stored_password_is_reported_at_once(tmp_path: Path):
    started = time.monotonic()
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_PASSWORD_STATE": "stored"}, wait = "600")
    assert time.monotonic() - started < 20
    assert "already set on an earlier boot" in res.stdout, res.stdout
    assert "reset-password" in res.stdout


# --- unsloth-studio-run: the initial password is applied only while none is stored


def _run_wrapper(
    home: Path,
    *,
    args: list[str] = (),
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    bin_dir = home / "bin"
    # what supervisord would spawn: prints what the CLI would have been handed
    _stub(bin_dir, "unsloth", 'printf "%s|%s\\n" "${UNSLOTH_STUDIO_PASSWORD:-<unset>}" "$*"\n')
    e = _clean_env(home / "stub-bin")
    (home / "stub-bin").mkdir(exist_ok = True)
    e["UNSLOTH_STUDIO_HOME"] = str(home)
    e["UNSLOTH_STUDIO_INITIAL_PASSWORD_FILE"] = str(home / "initial")
    e.update(env or {})
    return subprocess.run(
        ["bash", str(RUN), *args], capture_output = True, text = True, env = e, timeout = 60
    )


@behavioural
@pytest.mark.parametrize(
    ("db", "stored"),
    [("none", False), ("empty", False), ("seeded", False), ("changed", True), ("legacy", True)],
)
def test_stored_means_an_admin_row_whose_password_was_changed(
    tmp_path: Path, db: str, stored: bool
):
    """A bare auth.db from an interrupted first launch, or a seeded row nobody
    changed, still accepts an initial password; only must_change_password=0 is
    "stored"."""
    if db == "empty":
        _auth_db(tmp_path, must_change = None)
    elif db == "seeded":
        _auth_db(tmp_path, must_change = 1)
    elif db == "changed":
        _auth_db(tmp_path, must_change = 0)
    elif db == "legacy":
        # the CLI migrates this row with default 0 and then rejects an initial password
        _auth_db(tmp_path, must_change = None, legacy = True)
    res = _run_wrapper(tmp_path, args = ["--stored"])
    assert (res.returncode == 0) is stored, res.stderr


@behavioural
def test_the_first_spawn_applies_the_initial_password(tmp_path: Path):
    (tmp_path / "initial").write_text("hunter22hunter", encoding = "utf-8")
    res = _run_wrapper(tmp_path)
    assert res.stdout.strip() == "hunter22hunter|studio -H 0.0.0.0 -p 8000", res.stdout + res.stderr


@behavioural
def test_a_respawn_after_the_password_is_stored_gets_no_initial_password(tmp_path: Path):
    """The file is still there (the launcher writes it once per boot) and even a
    stray UNSLOTH_STUDIO_PASSWORD in the environment must not reach the CLI."""
    (tmp_path / "initial").write_text("hunter22hunter", encoding = "utf-8")
    _auth_db(tmp_path, must_change = 0)
    res = _run_wrapper(tmp_path, env = {"UNSLOTH_STUDIO_PASSWORD": "hunter22hunter"})
    assert res.stdout.strip() == "<unset>|studio -H 0.0.0.0 -p 8000", res.stdout + res.stderr


@behavioural
def test_a_respawn_while_the_seeded_password_is_still_active_retries_it(tmp_path: Path):
    (tmp_path / "initial").write_text("hunter22hunter", encoding = "utf-8")
    _auth_db(tmp_path, must_change = 1)
    res = _run_wrapper(tmp_path)
    assert res.stdout.startswith("hunter22hunter|"), res.stdout + res.stderr


@behavioural
def test_the_staged_password_reaches_the_cli_byte_for_byte(tmp_path: Path):
    """A secret injector may append a newline; the CLI, not the wrapper, decides what
    to make of it."""
    (tmp_path / "initial").write_text("hunter22\n", encoding = "utf-8")
    res = _run_wrapper(tmp_path)
    assert res.stdout == "hunter22\n|studio -H 0.0.0.0 -p 8000\n", repr(res.stdout)


@behavioural
def test_no_file_means_no_initial_password(tmp_path: Path):
    res = _run_wrapper(tmp_path, env = {"UNSLOTH_STUDIO_PASSWORD": "hunter22hunter"})
    assert res.stdout.startswith("<unset>|"), res.stdout + res.stderr


# --- the launcher: where the initial password goes -------------------------------


def _launcher_banner_block() -> str:
    """The banner decision, verbatim from studio_launch.sh."""
    source = LAUNCH.read_text(encoding = "utf-8")
    start = source.index("INITIAL_FILE=")
    end = source.index('echo "Unsloth Studio  ->', start)
    return source[start:end]


def _banner(tmp_path: Path, *, env_password: str | None, stored: bool) -> dict:
    bin_dir = tmp_path / "stub-bin"
    _stub(
        bin_dir,
        "unsloth-studio-run",
        f'[[ "$1" == --stored ]] && exit {0 if stored else 1}\nexit 0\n',
    )
    initial = tmp_path / "run" / "initial"
    script = (
        "set -euo pipefail\n"
        + _launcher_banner_block()
        + 'printf "%s\\n%s\\n%s\\n" "$STUDIO_NOTE" "${UNSLOTH_STUDIO_PASSWORD:-<unset>}" "$UNSLOTH_STUDIO_PASSWORD_STATE"\n'
    )
    env = _clean_env(bin_dir)
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path)
    env["UNSLOTH_STUDIO_INITIAL_PASSWORD_FILE"] = str(initial)
    if env_password is not None:
        env["UNSLOTH_STUDIO_PASSWORD"] = env_password
    res = subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, env = env, timeout = 30
    )
    assert res.returncode == 0, res.stderr
    note, inherited, state = res.stdout.rstrip("\n").split("\n")
    return {
        "note": note,
        "inherited": inherited,
        "state": state,
        "file": initial.read_text(encoding = "utf-8") if initial.exists() else None,
        "mode": (initial.stat().st_mode & 0o777) if initial.exists() else None,
    }


@behavioural
def test_the_initial_password_goes_to_a_private_file_not_supervisord(tmp_path: Path):
    """supervisord keeps its environment for every respawn of the studio program,
    and `unsloth studio` exits 1 when handed an initial password after one is
    stored; a crash or unsloth-studio-update restart then parked Studio in FATAL."""
    got = _banner(tmp_path, env_password = "hunter22hunter", stored = False)
    assert got["inherited"] == "<unset>"
    assert got["file"] == "hunter22hunter"
    assert got["mode"] == 0o600
    assert got["state"] == "initial"
    assert "from UNSLOTH_STUDIO_PASSWORD" in got["note"]


@behavioural
def test_a_stored_password_wins_over_the_env(tmp_path: Path):
    got = _banner(tmp_path, env_password = "hunter22hunter", stored = True)
    assert got["inherited"] == "<unset>"
    assert got["file"] is None
    assert got["state"] == "stored"
    assert "set on an earlier boot" in got["note"]


@behavioural
def test_a_fresh_home_without_the_env_generates(tmp_path: Path):
    got = _banner(tmp_path, env_password = None, stored = False)
    assert got["inherited"] == "<unset>"
    assert got["file"] is None
    assert got["state"] == "generated"
    assert "generated password printed below" in got["note"]


def test_the_image_wires_the_scripts_in():
    conf = SUPERVISORD.read_text(encoding = "utf-8")
    assert "[program:studio-password]" in conf
    block = conf.split("[program:studio-password]", 1)[1].split("[program:", 1)[0]
    assert "autorestart=false" in block and "stdout_logfile=/dev/stdout" in block
    studio = conf.split("[program:studio]", 1)[1].split("[program:", 1)[0]
    assert "command=/usr/local/bin/unsloth-studio-run" in studio
    # the bootstrap timeout ends Studio with exit 0; autorestart=true would bring it
    # straight back with the same default credential and a fresh timer
    assert "autorestart=unexpected" in studio and "exitcodes=0" in studio
    dockerfile = DOCKERFILE.read_text(encoding = "utf-8")
    # the Studio build uses context ./docker behind a deny-all .dockerignore
    allow = (DOCKER / ".dockerignore").read_text(encoding = "utf-8").splitlines()
    for script, target in (
        ("studio_password.sh", "unsloth-studio-password"),
        ("studio_run.sh", "unsloth-studio-run"),
    ):
        assert f"COPY {script} /usr/local/bin/{target}" in dockerfile
        assert (
            f"/usr/local/bin/{target}" in dockerfile.split("RUN chmod +x", 1)[1].split("\n\n", 1)[0]
        )
        assert f"!{script}" in allow, f"COPY {script} has no source in the build context"
    launch = LAUNCH.read_text(encoding = "utf-8")
    assert (
        "first-boot password below" not in launch
    ), "the banner promises what Studio no longer prints"
    assert "unset UNSLOTH_STUDIO_PASSWORD" in launch

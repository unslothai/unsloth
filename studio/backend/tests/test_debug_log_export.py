# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The "Download all logs" bundle. Two things have to hold whatever the host
looks like: nothing in the archive names a path on this machine, and nothing in
it is a credential. Everything else -- a log that rotates, is swapped for a
link, or holds one record bigger than the budget -- is a warning, not a failed
download."""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.settings as settings_route
from utils import debug_log_export, debug_log_sources


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[settings_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings_route._require_ui_session] = lambda: None
    return TestClient(app, raise_server_exceptions = False)


def _seed_server_log(body: str = "hello\n") -> Path:
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "server"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / f"server-20260813-120000-pid{os.getpid()}.log"
    path.write_text(body, encoding = "utf-8")
    return path


def _seed_llama_log(body: str = "llama runner line\n") -> Path:
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "llama-server"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / "llama-1786000000.log"
    path.write_text(body, encoding = "utf-8")
    return path


def _members() -> dict[str, bytes]:
    """Build a bundle and read it back. testzip() on every call, so a member
    left half written by a mid-read failure cannot pass unnoticed."""
    handle = debug_log_export.build_log_archive()
    try:
        with zipfile.ZipFile(handle) as archive:
            assert archive.testzip() is None
            return {name: archive.read(name) for name in archive.namelist()}
    finally:
        handle.close()


def test_member_names_are_relative_and_name_no_directory():
    server = _seed_server_log()
    llama = _seed_llama_log()
    members = _members()
    assert set(members) == {f"server/{server.name}", f"llama-server/{llama.name}"}
    home = str(Path(os.environ["UNSLOTH_STUDIO_HOME"]))
    for name in members:
        assert not name.startswith("/")
        assert ".." not in name
        assert home not in name
        assert "\\" not in name
        # family + basename, and nothing else.
        assert name.count("/") == 1


def test_the_bundle_is_an_archive_the_stdlib_accepts():
    _seed_server_log("a line\n")
    handle = debug_log_export.build_log_archive()
    try:
        assert handle.tell() == 0
        with zipfile.ZipFile(handle) as archive:
            assert archive.testzip() is None
            info = archive.infolist()[0]
            assert info.compress_type == zipfile.ZIP_DEFLATED
    finally:
        handle.close()


def test_a_planted_token_is_masked_in_the_archive():
    """The whole point of reusing the viewer's redactor: what the tab would not
    show you, the download does not carry either."""
    path = _seed_server_log(
        "loading with HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg\n"
    )
    members = _members()
    body = members[f"server/{path.name}"].decode("utf-8")
    assert "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345" not in body
    assert "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg" not in body
    assert "<redacted>" in body
    # Not just that member: every member. Scanning the raw archive bytes instead
    # would prove nothing, since ZIP_DEFLATED means a cleartext token does not
    # appear in them either way.
    for name, content in members.items():
        assert b"hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345" not in content, name


def test_a_source_swapped_for_a_symlink_after_enumeration_is_refused(monkeypatch, tmp_path):
    """The enumeration walk refuses an escaping link, but that ran earlier.
    Anything can replace the entry before the export opens it."""
    path = _seed_server_log("ordinary line\n")
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY MATERIAL\n", encoding = "utf-8")
    enumerate_sources = debug_log_sources.list_sources

    def swap_then_return():
        sources = enumerate_sources()
        path.unlink()
        path.symlink_to(secret)
        return sources

    monkeypatch.setattr(debug_log_sources, "list_sources", swap_then_return)

    members = _members()
    assert f"server/{path.name}" not in members
    for body in members.values():
        assert b"PRIVATE KEY MATERIAL" not in body
    warnings = members[debug_log_export.WARNINGS_MEMBER].decode("utf-8")
    assert warnings.startswith(f"server/{path.name}: OSError (errno ")
    assert str(tmp_path) not in warnings


def test_a_file_replaced_between_the_stat_and_the_open_is_refused(monkeypatch):
    """The narrow window the fstat compare exists for.

    lstat says regular file, the entry is replaced, and the open lands on
    something else. Comparing the descriptor's device and inode against the
    stat we just took is what catches it.
    """
    path = _seed_server_log("ordinary line\n")
    real_open = os.open
    swapped: list[bool] = []

    def open_after_a_swap(*args, **kwargs):
        if not swapped and args and str(args[0]).endswith(path.name):
            swapped.append(True)
            path.unlink()
            path.write_text("SOMEONE ELSE'S FILE\n", encoding = "utf-8")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(os, "open", open_after_a_swap)

    members = _members()
    assert swapped
    for body in members.values():
        assert b"SOMEONE ELSE'S FILE" not in body
    warnings = members[debug_log_export.WARNINGS_MEMBER].decode("utf-8")
    assert warnings.startswith(f"server/{path.name}: OSError (errno ")


def test_a_file_truncated_mid_export_warns_without_naming_a_directory(monkeypatch):
    """A log that rotates under the read costs its own tail, not the bundle.

    The warning names the member, never `str(exc)`: OSError.__str__ appends the
    filename, which would put a host path in the archive.
    """
    path = _seed_server_log("first line\nsecond line\n")
    real_redact = debug_log_export.redact_log_text
    truncated = []

    def redact_then_truncate(text: str) -> str:
        if not truncated:
            truncated.append(True)
            os.truncate(path, 0)
        return real_redact(text)

    monkeypatch.setattr(debug_log_export, "redact_log_text", redact_then_truncate)

    members = _members()
    warnings = members[debug_log_export.WARNINGS_MEMBER].decode("utf-8").strip()
    assert warnings.startswith(f"server/{path.name}: OSError (errno ")
    assert str(path.parent) not in warnings
    # family + basename, so the only separator in the line is the family's.
    assert warnings.count("/") == 1
    # What was copied before the failure is kept.
    assert members[f"server/{path.name}"].decode("utf-8").startswith("first line")


def test_an_oversized_record_is_omitted_whole_rather_than_split():
    """Splitting could cut a key away from its value, and the redactor is
    anchored on the two being adjacent."""
    filler = "X" * (debug_log_export.MAX_RECORD_BYTES + 5_000)
    path = _seed_server_log(f"before\n{filler}\nafter\n")
    body = _members()[f"server/{path.name}"].decode("utf-8")
    assert body == f"before\n{debug_log_export.OVERSIZED_MARKER}\nafter\n"


def test_an_oversized_record_at_end_of_file_is_still_omitted():
    filler = "Y" * (debug_log_export.MAX_RECORD_BYTES + 5_000)
    path = _seed_server_log(f"before\n{filler}")
    body = _members()[f"server/{path.name}"].decode("utf-8")
    assert body == f"before\n{debug_log_export.OVERSIZED_MARKER}\n"


def test_a_record_with_no_trailing_newline_is_kept():
    path = _seed_server_log("no trailing newline")
    assert _members()[f"server/{path.name}"] == b"no trailing newline\n"


def test_an_empty_export_is_still_a_valid_archive():
    assert _members() == {}


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("server-1.log", "server-1.log"),
        ("../../etc/passwd", "passwd"),
        ("/var/log/syslog", "syslog"),
        (r"C:\Users\me\tauri.log", "tauri.log"),
        ("..", "log"),
        ("", "log"),
    ],
)
def test_a_label_can_never_become_a_path(label, expected):
    assert debug_log_export._safe_basename(label) == expected


def test_duplicate_labels_are_uniquified_by_a_loop():
    """One studio home per spelling means two families can hold the same
    filename, and a duplicated member extracts as whichever entry the tool
    reaches last."""
    used: set[str] = set()
    assert debug_log_export._member_name("server", "a.log", used) == "server/a.log"
    assert debug_log_export._member_name("server", "a.log", used) == "server/a-2.log"
    assert debug_log_export._member_name("server", "a.log", used) == "server/a-3.log"
    # The first candidate being taken by a real file is why it re-checks.
    used.add("server/b-2.log")
    assert debug_log_export._member_name("server", "b.log", used) == "server/b.log"
    assert debug_log_export._member_name("server", "b.log", used) == "server/b-3.log"
    # A label with no extension keeps its suffix at the end.
    assert debug_log_export._member_name("desktop-shell", "tauri", used) == "desktop-shell/tauri"
    assert debug_log_export._member_name("desktop-shell", "tauri", used) == "desktop-shell/tauri-2"


def test_the_route_streams_an_attachment(client):
    path = _seed_server_log("route line\n")
    response = client.get("/api/settings/debug/logs/export")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith('attachment; filename="unsloth-logs-')
    assert disposition.endswith('.zip"')
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.testzip() is None
        assert archive.read(f"server/{path.name}") == b"route line\n"


def test_an_api_key_session_cannot_download_the_logs():
    """A bundle of every log on the host is UI-operator material, exactly like
    the single-file read next to it."""
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[settings_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings_route.authenticated_via_api_key] = lambda: True
    api_client = TestClient(app, raise_server_exceptions = False)
    _seed_server_log()
    assert api_client.get("/api/settings/debug/logs/export").status_code == 403


def test_a_log_larger_than_the_tail_keeps_its_end(monkeypatch):
    """The 10-per-family cap bounds FILES, not bytes: the session log is never
    rotated and runs to gigabytes, so the tail is the part worth carrying."""
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 512)
    path = _seed_server_log("".join(f"line {index}\n" for index in range(400)))
    assert path.stat().st_size > 512
    body = _members()[f"server/{path.name}"].decode()
    assert "skipped the first" in body.splitlines()[0]
    # The end survived and the beginning is what went.
    assert "line 399" in body
    assert "line 0\n" not in body


def test_a_truncated_log_never_starts_in_the_middle_of_a_record(monkeypatch):
    """A read that begins mid-record hands out a credential whose key was left
    behind in the skipped part, which is exactly what redaction relies on."""
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 300)
    # Sized so the cut lands inside the secret-bearing record rather than on a
    # boundary: the key is before the seek point, the value after it.
    filler = "".join(f"padding line {index}\n" for index in range(40))
    path = _seed_server_log(
        f"{filler}HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz012345\n" + "tail line\n" * 12
    )
    body = _members()[f"server/{path.name}"].decode()
    assert "hf_abcdefghijklmnopqrstuvwxyz012345" not in body
    # Whatever survived is whole records only.
    for line in body.splitlines()[1:]:
        assert line == "" or line.startswith(("padding line", "tail line", "HF_TOKEN=", "["))


def test_the_total_budget_names_what_it_left_out(monkeypatch):
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 64)
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 64)
    _seed_server_log("server line\n" * 20)
    _seed_llama_log("llama line\n" * 20)
    members = _members()
    warnings = members[debug_log_export.WARNINGS_MEMBER].decode()
    assert "budget" in warnings or "no complete record" in warnings


def test_a_log_inside_the_tail_is_not_truncated_or_warned_about():
    path = _seed_server_log("short enough\n")
    members = _members()
    assert members[f"server/{path.name}"] == b"short enough\n"
    assert debug_log_export.WARNINGS_MEMBER not in members


def test_the_budget_is_spread_across_families_not_drained_by_one(monkeypatch):
    """list_sources groups by family, so consuming it in order spends the whole
    budget on the server logs and ships a bundle with no runner logs at all."""
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 200)
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 100)
    _seed_server_log("server line\n" * 40)
    _seed_llama_log("llama line\n" * 40)
    members = _members()
    families = {name.split("/")[0] for name in members if name != debug_log_export.WARNINGS_MEMBER}
    assert families == {"server", "llama-server"}


def test_round_robin_keeps_the_newest_of_each_family_first():
    def source(family, label):
        return debug_log_sources.LogSource(
            id = f"{family}:{label}", family = family, label = label, realpath = label,
            size_bytes = 0, modified_at = 0.0, is_current = False,
        )
    grouped = [
        source("server", "s1"), source("server", "s2"), source("server", "s3"),
        source("llama-server", "l1"), source("llama-server", "l2"),
    ]
    assert [s.label for s in debug_log_export._newest_first_across_families(grouped)] == [
        "s1", "l1", "s2", "l2", "s3",
    ]
    assert debug_log_export._newest_first_across_families([]) == []


def test_one_log_with_no_record_boundary_does_not_abandon_the_rest(monkeypatch):
    """A tail window holding no newline is a property of THAT file. Spending the
    whole budget on it drops every later source, including ones that fit."""
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 1000)
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 1000)
    _seed_server_log("header line\n" + "a" * 4000)
    llama = _seed_llama_log("llama line\n")
    members = _members()
    # The unusable one is named...
    assert "no complete record" in members[debug_log_export.WARNINGS_MEMBER].decode()
    # ...and the one that plainly fits is still in the bundle.
    assert members[f"llama-server/{llama.name}"] == b"llama line\n"


def test_a_log_being_appended_to_is_still_bounded(monkeypatch):
    """The seek bounds where the read STARTS. A live log has no end, so without
    a ceiling the loop follows the writer -- on the one file both caps exist for."""
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 4096)
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 8192)
    path = _seed_server_log("start\n")

    real_redact = debug_log_export.redact_log_text
    state = {"appended": False}

    def grow_once(text):
        # Append far more than the allowance while the export is mid-read.
        if not state["appended"]:
            state["appended"] = True
            with path.open("a", encoding = "utf-8") as handle:
                handle.write("appended line\n" * 20000)
        return real_redact(text)

    monkeypatch.setattr(debug_log_export, "redact_log_text", grow_once)
    body = _members()[f"server/{path.name}"]
    assert path.stat().st_size > 200000, "the log really did grow during the export"
    assert len(body) <= 4096 + 128, f"read ran past the allowance: {len(body)} bytes"


def test_a_failed_read_still_spends_its_budget(monkeypatch):
    """Bytes already redacted and written are spent whether or not the file
    survived to EOF."""
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 1 << 20)
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 1 << 20)
    path = _seed_server_log("kept line\n" * 200)
    real_redact = debug_log_export.redact_log_text
    state = {"hits": 0}

    def truncate_midway(text):
        state["hits"] += 1
        if state["hits"] == 50:
            os.truncate(path, 0)
        return real_redact(text)

    monkeypatch.setattr(debug_log_export, "redact_log_text", truncate_midway)
    members = _members()
    warnings = members[debug_log_export.WARNINGS_MEMBER].decode()
    assert "OSError" in warnings or "errno" in warnings


def test_a_record_straddling_the_seek_never_leaks_its_credential(monkeypatch):
    """The seek must resync FORWARD to a real newline however far that is.

    Giving up after one probe starts the read at an arbitrary mid-record offset,
    and `redact_log_text` is anchored on the key: a record holding an AWS secret
    whose KEY falls before that offset and whose VALUE falls after it arrives
    masked of nothing. An AWS secret has no prefix for a shape rule to catch, so
    the anchor is the only defence. The offsets below are laid out so the cut
    lands exactly between the key and its value.
    """
    record_cap = 4096
    monkeypatch.setattr(debug_log_export, "MAX_RECORD_BYTES", record_cap)
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 2 * record_cap)
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 1 << 20)

    secret = "wJalrXUtnFEMIKSECRETDENGbPxRfiCYEXAMPLEKEY"
    key = 'aws_secret_access_key="'
    head = "x" * 200
    # Solving len(head) + 1 + lead + len(key) == size - record_cap for the trailer.
    trailer = record_cap - 1 - len(secret) - 1
    lead = 5000
    straddler = "a" * lead + key + secret + '"' + "b" * trailer
    path = _seed_server_log(f"{head}\n{straddler}\n")

    size = path.stat().st_size
    cut = size - record_cap
    key_end = len(head) + 1 + lead + len(key)
    # The geometry this test exists to exercise: one record spanning the seek
    # point, cut between its key and its value.
    assert key_end == cut, (key_end, cut)
    assert len(head) + 1 < size - 2 * record_cap

    # Either the record survives whole and is masked, or it is refused for
    # having no boundary. What must never happen is the value arriving without
    # its key, which is what a mid-record start produces.
    for name, body in _members().items():
        assert secret.encode() not in body, f"credential leaked into {name}"


def test_a_tail_with_no_record_boundary_is_refused_not_read_midway(monkeypatch):
    monkeypatch.setattr(debug_log_export, "MAX_RECORD_BYTES", 512)
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 2048)
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 1 << 20)
    secret = "wJalrXUtnFEMIKSECRETDENGbPxRfiCYEXAMPLEKEY"
    path = _seed_server_log("head\n" + "c" * 5000 + f'aws_secret_access_key="{secret}"')
    members = _members()
    assert f"server/{path.name}" not in members
    assert "no complete record" in members[debug_log_export.WARNINGS_MEMBER].decode()


def test_an_undecodable_filename_does_not_take_the_export_down():
    """Path.name hands back lone surrogates via surrogateescape; zipfile cannot
    encode them, and the raise is not an OSError so both handlers miss it."""
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "server"
    directory.mkdir(parents = True, exist_ok = True)
    raw = os.fsdecode(b"server-\xff\xfe.log")
    (directory / raw).write_bytes(b"a line\n")
    members = _members()
    assert any(name.startswith("server/") for name in members)


def test_a_label_cannot_forge_a_line_in_the_warnings_member():
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "server"
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "server-a\nforged: nothing was omitted\nb.log").write_bytes(b"x\n")
    for name in _members():
        assert "\n" not in name


def test_the_warnings_member_does_not_stamp_the_host_clock(monkeypatch):
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 8)
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 64)
    _seed_server_log("a line that is definitely longer than eight bytes\n")
    handle = debug_log_export.build_log_archive()
    try:
        with zipfile.ZipFile(handle) as archive:
            for info in archive.infolist():
                assert info.date_time == (1980, 1, 1, 0, 0, 0), info.filename
                assert info.comment == b"" and info.extra == b""
    finally:
        handle.close()


def test_a_pathological_log_cannot_run_past_the_time_budget(monkeypatch):
    """The byte budget assumes a throughput the redactor does not guarantee:
    its ANSI rules backtrack quadratically, so a log of unterminated C1
    introducers costs thousands of times what the same bytes of text do."""
    monkeypatch.setattr(debug_log_export, "MAX_BUILD_SECONDS", 0.5)
    real_redact = debug_log_export.redact_log_text

    def slow(text):
        time.sleep(0.05)
        return real_redact(text)

    monkeypatch.setattr(debug_log_export, "redact_log_text", slow)
    _seed_server_log("a line\n" * 500)
    _seed_llama_log("another line\n" * 500)

    started = time.monotonic()
    members = _members()
    elapsed = time.monotonic() - started

    assert elapsed < 10, f"build ran for {elapsed:.1f}s despite the budget"
    warnings = members[debug_log_export.WARNINGS_MEMBER].decode()
    truncated = any(
        debug_log_export.TRUNCATED_MARKER.encode() in body for body in members.values()
    )
    assert truncated or "time budget" in warnings


def test_a_fifo_in_the_log_directory_does_not_hang_the_export(monkeypatch):
    """O_NOFOLLOW refuses a symlink but not a FIFO, and opening a FIFO with no
    writer blocks forever -- before the fstat check gets a turn to reject it."""
    path = _seed_server_log("ordinary line\n")
    enumerate_sources = debug_log_sources.list_sources

    def swap_for_a_fifo():
        sources = enumerate_sources()
        path.unlink()
        os.mkfifo(path)
        return sources

    monkeypatch.setattr(debug_log_sources, "list_sources", swap_for_a_fifo)
    started = time.monotonic()
    members = _members()
    assert time.monotonic() - started < 10, "the open blocked on the FIFO"
    assert f"server/{path.name}" not in members
    path.unlink()


def test_a_second_export_is_refused_rather_than_queued(client, monkeypatch):
    """The route is a sync def on a 40-thread pool shared with every other sync
    endpoint, and a build is seconds of CPU."""
    _seed_server_log("a line\n")
    settings_route._DEBUG_LOG_EXPORT_LOCK.acquire()
    try:
        assert client.get("/api/settings/debug/logs/export").status_code == 429
    finally:
        settings_route._DEBUG_LOG_EXPORT_LOCK.release()
    # Released again, so the next caller is served normally.
    assert client.get("/api/settings/debug/logs/export").status_code == 200


def test_a_utf16_log_never_ships_its_credentials():
    """UTF-8 replacement decoding of UTF-16 leaves a NUL between every
    character, so every masking rule stops matching and the record is copied
    through with the credential still perfectly readable."""
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "server"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / "server-20260813-120000-utf16.log"
    path.write_bytes("HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUv012345\n".encode("utf-16-le"))
    for name, body in _members().items():
        assert b"hf_AbCdEfGhIjKlMnOpQrStUv012345" not in body, name
        # And not readable with the NULs stripped either, which is all `strings`
        # does.
        assert b"hf_AbCdEfGhIjKlMnOpQrStUv012345" not in body.replace(b"\x00", b""), name


def test_an_invalid_utf8_record_is_refused_rather_than_replaced():
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "server"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / "server-20260813-120000-binary.log"
    path.write_bytes(b"ordinary line\n" + b"\xff\xfe token=abcdef123456\n")
    body = _members()[f"server/{path.name}"]
    assert b"ordinary line" in body
    assert debug_log_export.UNREADABLE_MARKER.encode() in body


def test_valid_non_ascii_utf8_is_kept():
    """The refusal is for what the redactor cannot read, not for anything that
    is merely not ASCII."""
    path = _seed_server_log("connexion établie vers le modèle 模型\n")
    body = _members()[f"server/{path.name}"].decode("utf-8")
    assert "établie" in body and "模型" in body


def test_an_empty_log_is_an_empty_member_not_a_warning():
    """(0, 0) satisfies `skipped >= size`, which blamed the export for a file
    that simply has nothing in it yet."""
    path = _seed_server_log("")
    members = _members()
    assert members[f"server/{path.name}"] == b""
    assert debug_log_export.WARNINGS_MEMBER not in members


def test_a_tail_landing_exactly_on_a_boundary_keeps_that_record(monkeypatch):
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 1 << 20)
    # "keep me\n" is 8 bytes, and the allowance is exactly that, so `start`
    # lands on the newline before it.
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 8)
    path = _seed_server_log("dropped\nkeep me\n")
    assert _members()[f"server/{path.name}"].decode().endswith("keep me\n")


def test_the_tail_scan_does_not_follow_a_growing_log(monkeypatch):
    """The forward scan must stop at the size taken from the descriptor. A live
    log has no EOF: scanning to it follows the writer, which is the same
    unbounded read the record loop is careful to avoid."""
    monkeypatch.setattr(debug_log_export, "MAX_RECORD_BYTES", 512)
    monkeypatch.setattr(debug_log_export, "MAX_SOURCE_TAIL_BYTES", 2048)
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", 1 << 20)
    # No newline anywhere, so the scan runs the whole tail looking for one.
    path = _seed_server_log("z" * 6000)
    real_stat = os.fstat
    grew = {"count": 0}

    def grow_on_stat(fd):
        result = real_stat(fd)
        if grew["count"] < 200:
            grew["count"] += 1
            with path.open("a", encoding = "utf-8") as handle:
                handle.write("y" * 4000)
        return result

    monkeypatch.setattr(os, "fstat", grow_on_stat)
    started = time.monotonic()
    _members()
    assert time.monotonic() - started < 20, "the scan followed the writer"

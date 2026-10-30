# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The download worker fetches the commit its metadata read resolved: a branch moving in between cannot swap the
files under the manifest, refs/main still names the snapshot, and huggingface_hub >= 1.x skips its own repo_info."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_WORKER = _BACKEND / "hub" / "workers" / "hf_download.py"
_REPO = "fake/repo"
_OLD = "a" * 40
_NEW = "b" * 40
_TOKEN = "hf_fakegatedtoken0123456789"
# A full snapshot leaves GGUF files to the variant downloads.
_FULL = ("config.json", "model.safetensors")


def _hub_major() -> int:
    import huggingface_hub
    return int(huggingface_hub.__version__.split(".")[0])


def _commits() -> dict:
    old = {
        "config.json": b'{"model_type": "gpt2"}' * 3,
        "model.safetensors": hashlib.sha256(b"old").digest() * 2048,
        "model-Q4_K_M.gguf": b"GGUF" + hashlib.sha256(b"q4old").digest() * 1024,
    }
    # The moved branch changes sizes and content, so files from it cannot pass the old manifest.
    new = {
        "config.json": b'{"model_type": "gpt2", "n_layer": 2}' * 5,
        "model.safetensors": hashlib.sha256(b"new").digest() * 4096,
        "model-Q4_K_M.gguf": b"GGUF" + hashlib.sha256(b"q4new").digest() * 2048,
    }
    return {_OLD: old, _NEW: new}


def _is_lfs(name: str) -> bool:
    return name.endswith((".safetensors", ".gguf"))


def _etag(name: str, data: bytes) -> str:
    if _is_lfs(name):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class _FakeHub:
    """A Hub whose ``main`` moves from _OLD to _NEW right after the first metadata read when ``move`` is set."""

    def __init__(
        self,
        *,
        move: bool = False,
        gated: bool = False,
        fail_get: str | None = None,
    ):
        # down: every request after the next model_info answers 503 (the network dropping right after it).
        self.down = False
        self.info_served = False
        self.commits = _commits()
        self.head = _OLD
        self.move = move
        self.gated = gated
        self.fail_get = fail_get
        self.dropped = False
        self.requests: list[tuple[str, str]] = []
        self.lock = threading.Lock()
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(
                self,
                code: int,
                body: bytes = b"",
                headers: dict | None = None,
                head = False,
            ):
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body and not head:
                    self.wfile.write(body)

            def _json(
                self,
                obj,
                head = False,
            ):
                self._send(
                    200, json.dumps(obj).encode(), {"Content-Type": "application/json"}, head
                )

            def _rev(self, rev: str) -> str | None:
                return hub.head if rev == "main" else (rev if rev in hub.commits else None)

            def _reply(self, head: bool):
                path = self.path.split("?", 1)[0]
                with hub.lock:
                    hub.requests.append((self.command, self.path))
                if hub.down and hub.info_served:
                    self._send(503, b"down", {}, head)
                    return
                if hub.gated and self.headers.get("Authorization") != f"Bearer {_TOKEN}":
                    self._send(
                        401,
                        b'{"error": "Access to this repo is restricted."}',
                        {"X-Error-Code": "GatedRepo", "Content-Type": "application/json"},
                        head,
                    )
                    return
                prefix = f"/api/models/{_REPO}"
                if path.startswith(prefix):
                    rest = path[len(prefix) :]
                    if rest.startswith("/tree/"):
                        commit = self._rev(rest[len("/tree/") :].split("/", 1)[0])
                        if commit is None:
                            self._send(404, head = head)
                            return
                        tree = []
                        for name, data in hub.commits[commit].items():
                            entry = {
                                "type": "file",
                                "path": name,
                                "size": len(data),
                                "oid": _etag(name, data),
                            }
                            if _is_lfs(name):
                                entry["lfs"] = {
                                    "oid": _etag(name, data),
                                    "size": len(data),
                                    "pointerSize": 130,
                                }
                            tree.append(entry)
                        self._json(tree, head)
                        return
                    rev = rest[len("/revision/") :] if rest.startswith("/revision/") else "main"
                    commit = self._rev(rev)
                    if commit is None:
                        self._send(404, head = head)
                        return
                    siblings = []
                    for name, data in hub.commits[commit].items():
                        entry = {"rfilename": name, "size": len(data), "blobId": _etag(name, data)}
                        if _is_lfs(name):
                            entry["lfs"] = {
                                "sha256": _etag(name, data),
                                "size": len(data),
                                "pointerSize": 130,
                            }
                        siblings.append(entry)
                    self._json(
                        {"id": _REPO, "sha": commit, "siblings": siblings, "private": False}, head
                    )
                    if not rest:
                        hub.info_served = True
                    if hub.move and not rest:
                        hub.head = _NEW
                        hub.move = False
                    return
                marker = f"/{_REPO}/resolve/"
                if not path.startswith(marker):
                    self._send(404, head = head)
                    return
                rev, _, name = path[len(marker) :].partition("/")
                commit = self._rev(rev)
                data = hub.commits.get(commit, {}).get(name) if commit else None
                if data is None:
                    self._send(404, head = head)
                    return
                headers = {"X-Repo-Commit": commit, "ETag": f'"{_etag(name, data)}"'}
                if _is_lfs(name):
                    headers["X-Linked-Etag"] = f'"{_etag(name, data)}"'
                    headers["X-Linked-Size"] = str(len(data))
                start = 0
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes=") and not head:
                    start = int(rng[len("bytes=") :].split("-", 1)[0] or 0)
                if not head and hub.fail_get == name and hub.dropped:
                    # Still failing: the first attempt gives up instead of resuming in-process.
                    self._send(404, head = head)
                    return
                if not head and hub.fail_get == name:
                    hub.dropped = True
                    # Half the body, then the connection drops: leaves a resumable .incomplete.
                    self.send_response(200)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data[: len(data) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                if start:
                    headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
                    self._send(206, data[start:], headers, head)
                else:
                    self._send(200, data, headers, head)

            def do_GET(self):
                self._reply(False)

            def do_HEAD(self):
                self._reply(True)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target = self.server.serve_forever, daemon = True).start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def metadata_calls(self) -> list[str]:
        return [p for m, p in self.requests if p.startswith("/api/")]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _run(
    hub: _FakeHub,
    tmp_path: Path,
    *extra: str,
    token: str | None = None,
    offline = False,
):
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env.update(
        HF_ENDPOINT = hub.endpoint,
        HF_HOME = str(tmp_path / "hf"),
        HF_HUB_CACHE = str(tmp_path / "hf" / "hub"),
        UNSLOTH_STUDIO_HOME = str(tmp_path / "studio"),
        HF_HUB_DISABLE_XET = "1",
        HF_HUB_ENABLE_HF_TRANSFER = "0",
        HF_HUB_DISABLE_TELEMETRY = "1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN = "1",
        HF_HUB_OFFLINE = "1" if offline else "0",
        HF_HUB_ETAG_TIMEOUT = "5",
        HF_HUB_DOWNLOAD_TIMEOUT = "5",
    )
    if token:
        env["HF_TOKEN"] = token
    hub.requests.clear()
    return subprocess.run(
        [sys.executable, str(_WORKER), "--repo-id", _REPO, "--transport", "http", *extra],
        cwd = str(_BACKEND),
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )


def _repo_dir(tmp_path: Path) -> Path:
    return tmp_path / "hf" / "hub" / "models--fake--repo"


def _assert_snapshot(tmp_path: Path, commit: str, names) -> None:
    repo = _repo_dir(tmp_path)
    snapshot = repo / "snapshots" / commit
    for name in names:
        assert (snapshot / name).read_bytes() == _commits()[commit][name], name
    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == commit
    assert not (repo / "snapshots" / _NEW).exists() or commit == _NEW


@pytest.fixture
def hub_factory():
    hubs = []

    def make(**kw):
        hub = _FakeHub(**kw)
        hubs.append(hub)
        return hub

    yield make
    for hub in hubs:
        hub.close()


def test_branch_moving_after_the_metadata_read_still_fetches_the_planned_commit(
    tmp_path, hub_factory
):
    hub = hub_factory(move = True)
    proc = _run(hub, tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    _assert_snapshot(tmp_path, _OLD, _FULL)
    assert hub.head == _NEW  # the move really happened mid-job
    later = [p for _, p in hub.requests[1:]]
    assert later and all(_OLD in p for p in later), later


def test_gguf_variant_and_scoped_downloads_fetch_the_planned_commit(tmp_path, hub_factory):
    hub = hub_factory(move = True)
    proc = _run(hub, tmp_path, "--variant", "Q4_K_M")
    assert proc.returncode == 0, proc.stderr[-2000:]
    _assert_snapshot(tmp_path, _OLD, ["model-Q4_K_M.gguf"])

    files = tmp_path / "files.json"
    files.write_text(json.dumps(["config.json"]), encoding = "utf-8")
    hub2 = hub_factory(move = True)
    proc = _run(hub2, tmp_path / "scoped", "--variant", "scope1", "--files-json", str(files))
    assert proc.returncode == 0, proc.stderr[-2000:]
    _assert_snapshot(tmp_path / "scoped", _OLD, ["config.json"])


def test_one_listing_per_download_with_no_second_resolution(tmp_path, hub_factory):
    hub = hub_factory()
    proc = _run(hub, tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    calls = hub.metadata_calls()
    assert sum(1 for p in calls if p.split("?")[0] == f"/api/models/{_REPO}") == 1
    assert not [p for p in calls if "/revision/main" in p or "/tree/main" in p], calls
    if _hub_major() >= 1:
        # huggingface_hub >= 1.x skips repo_info for a sha: model_info + the tree listing only.
        assert len(calls) == 2, calls
    else:
        assert len(calls) == 2 and f"/revision/{_OLD}" in calls[1], calls


def test_gated_repo_with_a_token_pins_and_without_one_still_fails(tmp_path, hub_factory):
    hub = hub_factory(gated = True)
    proc = _run(hub, tmp_path, token = _TOKEN)
    assert proc.returncode == 0, proc.stderr[-2000:]
    _assert_snapshot(tmp_path, _OLD, _FULL)

    hub2 = hub_factory(gated = True)
    proc = _run(hub2, tmp_path / "anon")
    assert proc.returncode != 0
    assert not (_repo_dir(tmp_path / "anon") / "snapshots").exists()


def test_resume_after_a_dropped_transfer_and_offline_rerun(tmp_path, hub_factory):
    hub = hub_factory(fail_get = "model.safetensors")
    first = _run(hub, tmp_path)
    assert first.returncode != 0
    assert (_repo_dir(tmp_path) / "refs" / "main").read_text(encoding = "utf-8") == _OLD
    hub.fail_get = None
    hub.move = True  # the branch moves before the resume's metadata read is answered
    second = _run(hub, tmp_path)
    assert second.returncode == 0, second.stderr[-2000:]
    _assert_snapshot(tmp_path, _OLD, _FULL)
    # Offline: no metadata, so no pin; the cached snapshot refs/main names is verified as before.
    third = _run(hub, tmp_path, offline = True)
    assert third.returncode == 0, third.stderr[-2000:]
    assert hub.requests == []


def test_network_drop_after_the_metadata_read_keeps_the_older_snapshot_loadable(
    tmp_path, hub_factory
):
    hub = hub_factory()
    first = _run(hub, tmp_path)
    assert first.returncode == 0, first.stderr[-2000:]
    hub.head = _NEW
    hub.down = True
    hub.info_served = False
    second = _run(hub, tmp_path)
    assert second.returncode != 0
    # Nothing of _NEW landed, so refs/main still names the complete older snapshot, as an unpinned download left it.
    repo = _repo_dir(tmp_path)
    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == _OLD
    assert [p.name for p in (repo / "refs").iterdir()] == ["main"]
    assert "retrying unpinned" in second.stderr
    env = dict(os.environ, HF_HUB_CACHE = str(tmp_path / "hf" / "hub"), HF_HUB_OFFLINE = "1")
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "from huggingface_hub import hf_hub_download as d; print(open(d('fake/repo', 'model.safetensors'), 'rb').read().hex())",
        ],
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert loaded.returncode == 0, loaded.stderr[-2000:]
    assert bytes.fromhex(loaded.stdout.strip()) == _commits()[_OLD]["model.safetensors"]


def test_no_pin_without_a_full_sha(monkeypatch, tmp_path):
    from hub.workers import hf_download

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    assert hf_download._pinned_revision("model", _REPO, None) == {}
    assert hf_download._pinned_revision("model", _REPO, "main") == {}
    assert hf_download._pinned_revision("model", _REPO, "A" * 40) == {}
    assert not (tmp_path / "models--fake--repo").exists()
    assert hf_download._pinned_revision("model", _REPO, _OLD) == {"revision": _OLD}
    assert (tmp_path / "models--fake--repo" / "refs" / "main").read_text(encoding = "utf-8") == _OLD


def test_pinned_call_rolls_the_ref_back_only_when_nothing_landed(monkeypatch, tmp_path):
    from hub.workers import hf_download
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--fake--repo"
    ref = repo / "refs" / "main"
    ref.parent.mkdir(parents = True)
    calls = []

    def fake(behaviour):
        def snapshot_download(**kw):
            calls.append(kw.get("revision"))
            if kw.get("revision") is None:
                return "unpinned"
            if behaviour == "land":
                (repo / "snapshots" / _NEW).mkdir(parents = True, exist_ok = True)
                (repo / "snapshots" / _NEW / "config.json").write_text("{}")
            if behaviour == "peer_rollback":
                ref.write_text(_OLD)
                return "pinned"
            if behaviour == "interrupt":
                raise KeyboardInterrupt
            raise OSError("network down")

        return snapshot_download

    # Nothing landed: the ref goes back and the unpinned call runs (main's offline fallback).
    ref.write_text(_OLD)
    assert hf_download._pinned_snapshot_download(fake("drop"), _NEW, repo_id = _REPO) == "unpinned"
    assert calls == [_NEW, None] and ref.read_text() == _OLD
    # A file landed: the ref stays on the new commit and the error stands, as an unpinned download leaves it.
    calls.clear()
    with pytest.raises(OSError):
        hf_download._pinned_snapshot_download(fake("land"), _NEW, repo_id = _REPO)
    assert calls == [_NEW] and ref.read_text() == _NEW
    # A cancel before anything landed restores the ref and is not retried.
    ref.write_text(_OLD)
    calls.clear()
    with pytest.raises(KeyboardInterrupt):
        hf_download._pinned_snapshot_download(fake("interrupt"), "c" * 40, repo_id = _REPO)
    assert calls == ["c" * 40] and ref.read_text() == _OLD
    # A peer's rollback during a successful download is undone: the fetched commit is named.
    assert (
        hf_download._pinned_snapshot_download(fake("peer_rollback"), "d" * 40, repo_id = _REPO)
        == "pinned"
    )
    assert ref.read_text() == "d" * 40
    # Same when a peer had already published the commit before this download started.
    assert (
        hf_download._pinned_snapshot_download(fake("peer_rollback"), "d" * 40, repo_id = _REPO)
        == "pinned"
    )
    assert ref.read_text() == "d" * 40
    # No ref before and nothing landed: no ref after.
    ref.unlink()
    assert (
        hf_download._pinned_snapshot_download(fake("drop"), "e" * 40, repo_id = _REPO) == "unpinned"
    )
    assert not ref.exists()
    # A failed ref write leaves no temp file under refs/ (every file there reads as a ref).
    monkeypatch.setattr(
        hf_download.os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError("locked"))
    )
    assert hf_download._pinned_revision("model", _REPO, "f" * 40) == {}
    assert [p.name for p in ref.parent.iterdir()] == []

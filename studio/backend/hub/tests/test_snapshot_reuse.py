# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A new Hub commit must not re-download files it did not change (no-symlink cache)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from hub.utils import snapshot_reuse
from hub.utils.download_manifest import ExpectedFile

REPO = "Org/Model"
OLD = "1" * 40
NEW = "2" * 40


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _blob(seed: int, size: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(b"%d-%d" % (seed, counter)).digest()
        counter += 1
    return bytes(out[:size])


def _copy_layout(
    tmp_path: Path,
    files: dict[str, bytes],
    commit: str = OLD,
) -> Path:
    repo_dir = tmp_path / "hub" / "models--Org--Model"
    snap = repo_dir / "snapshots" / commit
    for rel, data in files.items():
        (snap / rel).parent.mkdir(parents = True, exist_ok = True)
        (snap / rel).write_bytes(data)
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(commit)
    return repo_dir


def _reuse(tmp_path, expected, **kwargs):
    return snapshot_reuse.reuse_unchanged_snapshot_files(
        "model", REPO, NEW, expected, hub_cache = tmp_path / "hub", **kwargs
    )


def test_unchanged_file_is_linked_and_a_same_size_changed_file_is_not(tmp_path):
    same = _blob(1, 4096)
    old_changed, new_changed = _blob(2, 4096), _blob(3, 4096)
    repo_dir = _copy_layout(
        tmp_path, {"model.safetensors": same, "vae/vae.safetensors": old_changed}
    )

    result = _reuse(
        tmp_path,
        [
            ExpectedFile("model.safetensors", len(same), _sha256(same)),
            ExpectedFile("vae/vae.safetensors", len(new_changed), _sha256(new_changed)),
        ],
    )

    assert result.reused == ("model.safetensors",)
    assert result.reused_bytes == len(same)
    target = repo_dir / "snapshots" / NEW / "model.safetensors"
    assert target.read_bytes() == same
    assert not target.is_symlink()
    assert (
        os.stat(target).st_ino == os.stat(repo_dir / "snapshots" / OLD / "model.safetensors").st_ino
    )
    assert not (repo_dir / "snapshots" / NEW / "vae" / "vae.safetensors").exists()


def test_symlink_layout_is_left_to_huggingface_hub(tmp_path):
    data = _blob(4, 2048)
    digest = _sha256(data)
    repo_dir = tmp_path / "hub" / "models--Org--Model"
    (repo_dir / "blobs").mkdir(parents = True)
    (repo_dir / "blobs" / digest).write_bytes(data)
    (repo_dir / "snapshots" / OLD).mkdir(parents = True)
    try:
        (repo_dir / "snapshots" / OLD / "model.safetensors").symlink_to(
            Path("..", "..", "blobs", digest)
        )
    except OSError:
        pytest.skip("symlinks unavailable")

    result = _reuse(tmp_path, [ExpectedFile("model.safetensors", len(data), digest)])

    assert result.reused == ()
    assert not (repo_dir / "snapshots" / NEW).exists()


def test_a_symlinked_candidate_whose_blob_is_gone_is_not_used(tmp_path):
    data = _blob(5, 2048)
    repo_dir = _copy_layout(tmp_path, {})
    (repo_dir / "snapshots" / OLD).mkdir(parents = True)
    try:
        (repo_dir / "snapshots" / OLD / "model.safetensors").symlink_to(tmp_path / "elsewhere.bin")
    except OSError:
        pytest.skip("symlinks unavailable")
    (tmp_path / "elsewhere.bin").write_bytes(data)

    result = _reuse(tmp_path, [ExpectedFile("model.safetensors", len(data), _sha256(data))])

    assert result.reused == ()


def test_hub_digest_for_the_old_commit_skips_hashing(tmp_path, monkeypatch):
    data = _blob(6, 8192)
    digest = _sha256(data)
    _copy_layout(tmp_path, {"model.safetensors": data})
    calls = []

    def remote(commit, paths):
        calls.append((commit, tuple(paths)))
        return {p: digest for p in paths}

    monkeypatch.setattr(
        snapshot_reuse,
        "file_digest",
        lambda *a, **k: pytest.fail("hashed although the Hub vouched for the old commit"),
    )
    result = _reuse(
        tmp_path, [ExpectedFile("model.safetensors", len(data), digest)], remote_digests = remote
    )

    assert result.reused == ("model.safetensors",)
    assert result.hashed_bytes == 0
    assert calls == [(OLD, ("model.safetensors",))]


def test_hub_digest_mismatch_falls_back_to_hashing_and_still_refuses_changed_bytes(tmp_path):
    old, new = _blob(7, 1024), _blob(8, 1024)
    _copy_layout(tmp_path, {"model.safetensors": old})

    result = _reuse(
        tmp_path,
        [ExpectedFile("model.safetensors", len(new), _sha256(new))],
        remote_digests = lambda commit, paths: {p: _sha256(old) for p in paths},
    )

    assert result.reused == ()
    assert result.hashed_bytes == len(old)


def test_unreachable_old_commit_is_verified_by_hashing(tmp_path):
    data = _blob(9, 4096)
    _copy_layout(tmp_path, {"model.safetensors": data})

    def remote(commit, paths):
        raise OSError("404 revision not found")

    result = _reuse(
        tmp_path,
        [ExpectedFile("model.safetensors", len(data), _sha256(data))],
        remote_digests = remote,
    )

    assert result.reused == ("model.safetensors",)
    assert result.hashed_bytes == len(data)


def _linked_snapshots(tmp_path: Path, data: bytes, n: int) -> list[str]:
    commits = [f"{i + 3:x}" * 40 for i in range(n)]
    repo_dir = _copy_layout(tmp_path, {"model.safetensors": data}, commit = commits[0])
    for commit in commits[1:]:
        (repo_dir / "snapshots" / commit).mkdir(parents = True)
        os.link(
            repo_dir / "snapshots" / commits[0] / "model.safetensors",
            repo_dir / "snapshots" / commit / "model.safetensors",
        )
    return commits


def test_one_file_linked_into_many_snapshots_is_hashed_once(tmp_path):
    old, new = _blob(20, 8192), _blob(21, 8192)
    _linked_snapshots(tmp_path, old, 5)

    result = _reuse(tmp_path, [ExpectedFile("model.safetensors", len(new), _sha256(new))])

    assert result.reused == ()
    assert result.hashed_bytes == len(old)


def test_hub_digests_are_asked_newest_first_and_only_until_matched(tmp_path):
    data = _blob(22, 8192)
    commits = []
    for i in range(4):
        commit = f"{i + 3:x}" * 40
        commits.append(commit)
        _copy_layout(tmp_path, {"model.safetensors": data}, commit = commit)
    calls = []

    def remote(commit, paths):
        calls.append(commit)
        return {p: _sha256(data) for p in paths}

    found = snapshot_reuse.reusable_paths(
        "model",
        REPO,
        NEW,
        {"model.safetensors": len(data)},
        hub_cache = tmp_path / "hub",
        remote_digests = remote,
    )

    assert found == {"model.safetensors"}
    assert len(calls) == 2
    assert calls[0] == NEW and calls[1] in commits


def test_a_copy_cut_short_by_a_cancel_is_removed_by_the_next_reuse(tmp_path):
    data = _blob(23, 8192)
    repo_dir = _copy_layout(tmp_path, {"model.safetensors": data})
    target = repo_dir / "snapshots" / NEW
    target.mkdir(parents = True)
    (target / ".model.safetensors.reuse-99999-deadbeef").write_bytes(data[:100])

    result = _reuse(tmp_path, [ExpectedFile("model.safetensors", len(data), _sha256(data))])

    assert result.reused == ("model.safetensors",)
    assert sorted(p.name for p in target.iterdir()) == ["model.safetensors"]


def test_a_retry_does_not_preflight_files_an_earlier_attempt_already_placed(tmp_path, monkeypatch):
    # Preflight must not count a file an earlier cancelled attempt already placed.
    from huggingface_hub import constants

    from hub.workers import hf_download

    data, missing = _blob(24, 8192), _blob(25, 4096)
    repo_dir = _copy_layout(tmp_path, {"model.safetensors": data})
    target = repo_dir / "snapshots" / NEW
    target.mkdir(parents = True)
    os.link(repo_dir / "snapshots" / OLD / "model.safetensors", target / "model.safetensors")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hf_download, "_protected_blob_hashes", lambda: frozenset())
    expected = [
        ExpectedFile("model.safetensors", len(data), _sha256(data)),
        ExpectedFile("vae/vae.safetensors", len(missing), _sha256(missing)),
    ]

    left = hf_download._reuse_unchanged_files("model", REPO, NEW, expected, None)

    assert [f.path for f in left] == ["vae/vae.safetensors"]
    assert hf_download._reuse_unchanged_files("model", REPO, None, expected, None) == expected


def test_size_mismatch_is_rejected_without_reading_the_file(tmp_path, monkeypatch):
    _copy_layout(tmp_path, {"model.safetensors": _blob(10, 1000)})
    monkeypatch.setattr(
        snapshot_reuse, "file_digest", lambda *a, **k: pytest.fail("hashed a wrong-size file")
    )

    result = _reuse(tmp_path, [ExpectedFile("model.safetensors", 1001, "a" * 64)])

    assert result.reused == ()


def test_cached_digest_feeds_the_plan_but_the_worker_rehashes(tmp_path, monkeypatch):
    data = _blob(11, 4096)
    repo_dir = _copy_layout(tmp_path, {"model.safetensors": data})
    expected = [ExpectedFile("model.safetensors", len(data), _sha256(data))]

    first = snapshot_reuse.find_reusable_copies(
        repo_dir, NEW, {"model.safetensors": (len(data), _sha256(data))}
    )
    assert first[1] == len(data)
    estimate = snapshot_reuse.find_reusable_copies(
        repo_dir, NEW, {"model.safetensors": (len(data), _sha256(data))}, allow_hashing = False
    )
    assert estimate == ({"model.safetensors": first[0]["model.safetensors"]}, 0)

    result = _reuse(tmp_path, expected)

    assert result.reused == ("model.safetensors",)
    assert result.hashed_bytes == len(data)


def test_a_stale_cached_digest_does_not_carry_changed_bytes_forward(tmp_path):
    """Same size, same mtime (restored, or FAT/exFAT 2 s granularity), different bytes."""
    good, bad = _blob(12, 4096), _blob(13, 4096)
    repo_dir = _copy_layout(tmp_path, {"model.safetensors": bad})
    candidate = repo_dir / "snapshots" / OLD / "model.safetensors"
    key = snapshot_reuse._digest_cache_key(candidate, "sha256", os.stat(candidate))
    snapshot_reuse._remember_digests({key: _sha256(good)})

    result = _reuse(tmp_path, [ExpectedFile("model.safetensors", len(good), _sha256(good))])

    assert result.reused == ()
    assert not (repo_dir / "snapshots" / NEW / "model.safetensors").exists()


def test_small_non_lfs_file_is_verified_by_git_blob_id(tmp_path):
    config = b'{"hidden_size": 8}\n'
    _copy_layout(tmp_path, {"config.json": config})

    result = _reuse(tmp_path, [ExpectedFile("config.json", len(config), _git_sha1(config))])

    assert result.reused == ("config.json",)


def test_copy_when_hard_links_are_unavailable(tmp_path, monkeypatch):
    data = _blob(12, 4096)
    repo_dir = _copy_layout(tmp_path, {"model.safetensors": data})

    def no_link(*a, **k):
        raise OSError("hard links not supported")

    monkeypatch.setattr(snapshot_reuse.os, "link", no_link)
    result = _reuse(tmp_path, [ExpectedFile("model.safetensors", len(data), _sha256(data))])

    assert result.copied == 1 and result.linked == 0
    target = repo_dir / "snapshots" / NEW / "model.safetensors"
    assert target.read_bytes() == data
    assert os.stat(target).st_nlink == 1


def test_existing_target_and_existing_blob_are_left_alone(tmp_path):
    data = _blob(13, 2048)
    other = _blob(14, 2048)
    repo_dir = _copy_layout(tmp_path, {"a.bin": data, "b.bin": other})
    (repo_dir / "snapshots" / NEW).mkdir(parents = True)
    (repo_dir / "snapshots" / NEW / "a.bin").write_bytes(b"already here")
    (repo_dir / "blobs" / _sha256(other)).write_bytes(other)

    result = _reuse(
        tmp_path,
        [
            ExpectedFile("a.bin", len(data), _sha256(data)),
            ExpectedFile("b.bin", len(other), _sha256(other)),
        ],
    )

    assert result.reused == ()
    assert (repo_dir / "snapshots" / NEW / "a.bin").read_bytes() == b"already here"
    assert not (repo_dir / "snapshots" / NEW / "b.bin").exists()


def test_superseded_partial_is_removed_unless_protected(tmp_path):
    data = _blob(15, 4096)
    digest = _sha256(data)
    repo_dir = _copy_layout(tmp_path, {"model.safetensors": data})
    partial = repo_dir / "blobs" / f"{digest}.incomplete"
    partial.write_bytes(data[:100])

    protected = _reuse(
        tmp_path,
        [ExpectedFile("model.safetensors", len(data), digest)],
        protected_blob_hashes = frozenset({digest}),
    )
    # A peer is downloading it: no pointer, or its finished blob would stay behind as a copy.
    assert protected.reused == ()
    assert not (repo_dir / "snapshots" / NEW / "model.safetensors").exists()
    assert partial.exists()

    _reuse(tmp_path, [ExpectedFile("model.safetensors", len(data), digest)])
    assert (repo_dir / "snapshots" / NEW / "model.safetensors").exists()
    assert not partial.exists()


def test_unsafe_paths_and_unknown_digests_are_ignored(tmp_path):
    data = _blob(16, 512)
    _copy_layout(tmp_path, {"model.safetensors": data})

    result = _reuse(
        tmp_path,
        [
            ExpectedFile("../escape.bin", len(data), _sha256(data)),
            ExpectedFile("model.safetensors", len(data), None),
            ExpectedFile("model.safetensors", len(data), "not-a-digest"),
        ],
    )

    assert result.reused == ()


def test_reusable_paths_is_read_only_and_never_hashes(tmp_path, monkeypatch):
    same, changed = _blob(17, 2048), _blob(18, 2048)
    repo_dir = _copy_layout(tmp_path, {"te.safetensors": same, "vae.safetensors": changed})
    monkeypatch.setattr(snapshot_reuse, "file_digest", lambda *a, **k: pytest.fail("plan hashed"))
    digests = {
        OLD: {"te.safetensors": _sha256(same), "vae.safetensors": _sha256(changed)},
        NEW: {"te.safetensors": _sha256(same), "vae.safetensors": _sha256(b"new" + changed[3:])},
    }

    found = snapshot_reuse.reusable_paths(
        "model",
        REPO,
        NEW,
        {"te.safetensors": len(same), "vae.safetensors": len(changed), "absent.bin": 5},
        hub_cache = tmp_path / "hub",
        remote_digests = lambda commit, paths: {p: digests[commit][p] for p in paths},
    )

    assert found == {"te.safetensors"}
    assert not (repo_dir / "snapshots" / NEW).exists()


def test_reusable_paths_makes_no_request_without_a_local_candidate(tmp_path):
    _copy_layout(tmp_path, {"other.bin": b"x"})

    found = snapshot_reuse.reusable_paths(
        "model",
        REPO,
        NEW,
        {"te.safetensors": 10},
        hub_cache = tmp_path / "hub",
        remote_digests = lambda commit, paths: pytest.fail("network without a candidate"),
    )

    assert found == set()


_UNCHANGED = _blob(100, 3 * 1024 * 1024)
_VAE = _blob(101, 256 * 1024)
_CHANGED_OLD = _blob(102, 512 * 1024)
_CHANGED_NEW = _blob(103, 512 * 1024)
_REVISIONS = {
    OLD: {
        "README.md": b"# v1\n",
        "text_encoder.safetensors": _UNCHANGED,
        "vae/vae.safetensors": _VAE,
        "changed.safetensors": _CHANGED_OLD,
    },
    NEW: {
        "README.md": b"# v2, a card edit\n",
        "text_encoder.safetensors": _UNCHANGED,
        "vae/vae.safetensors": _VAE,
        "changed.safetensors": _CHANGED_NEW,
    },
}
_LFS_MIN = 1024


class _FakeHub:
    def __init__(self):
        self.head = OLD
        self.downloaded: dict[tuple[str, str], int] = {}
        self.lock = threading.Lock()
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _commit(self, rev):
                rev = unquote(rev)
                return hub.head if rev == "main" else rev if rev in _REVISIONS else None

            def _resolve(self, head_only):
                parts = urlsplit(self.path).path.split("/")
                commit = self._commit(parts[4])
                rel = unquote("/".join(parts[5:]))
                data = _REVISIONS.get(commit, {}).get(rel)
                if data is None:
                    self.send_response(404)
                    self.send_header("X-Error-Code", "EntryNotFound")
                    self.end_headers()
                    return
                lfs = len(data) >= _LFS_MIN
                self.send_response(200)
                self.send_header("X-Repo-Commit", commit)
                self.send_header("ETag", f'"{_git_sha1(data)}"')
                if lfs:
                    self.send_header("X-Linked-Etag", f'"{_sha256(data)}"')
                    self.send_header("X-Linked-Size", str(len(data)))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(data)
                    with hub.lock:
                        key = (commit, rel)
                        hub.downloaded[key] = hub.downloaded.get(key, 0) + len(data)

            def do_HEAD(self):
                self._resolve(head_only = True)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path.startswith(f"/api/models/{REPO}"):
                    rest = path[len(f"/api/models/{REPO}") :]
                    if rest in ("", "/"):
                        return self._json(_model_info(hub.head))
                    if rest.startswith("/revision/"):
                        commit = self._commit(rest[len("/revision/") :])
                        return self._json(_model_info(commit))
                    if rest.startswith("/tree/"):
                        commit = self._commit(rest[len("/tree/") :].split("/")[0])
                        return self._json(
                            [_tree_entry(p, d) for p, d in _REVISIONS[commit].items()]
                        )
                if "/resolve/" in path:
                    return self._resolve(head_only = False)
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                path = urlsplit(self.path).path
                prefix = f"/api/models/{REPO}/paths-info/"
                if not path.startswith(prefix):
                    self.send_response(404)
                    self.end_headers()
                    return
                commit = self._commit(path[len(prefix) :])
                length = int(self.headers.get("Content-Length") or 0)
                form = parse_qs(self.rfile.read(length).decode())
                files = _REVISIONS.get(commit, {})
                self._json([_tree_entry(p, files[p]) for p in form.get("paths", []) if p in files])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target = self.server.serve_forever, daemon = True).start()

    def close(self):
        self.server.shutdown()


def _tree_entry(path, data):
    entry = {"type": "file", "path": path, "size": len(data), "oid": _git_sha1(data)}
    if len(data) >= _LFS_MIN:
        entry["lfs"] = {"oid": _sha256(data), "size": len(data), "pointerSize": 134}
    return entry


def _model_info(commit):
    siblings = []
    for path, data in _REVISIONS[commit].items():
        sibling = {"rfilename": path, "size": len(data), "blobId": _git_sha1(data)}
        if len(data) >= _LFS_MIN:
            sibling["lfs"] = {"sha256": _sha256(data), "size": len(data), "pointerSize": 134}
        siblings.append(sibling)
    return {"id": REPO, "modelId": REPO, "sha": commit, "siblings": siblings, "private": False}


_WORKER_SCRIPT = textwrap.dedent(
    """
    import sys
    backend, mode, symlinks = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    sys.path.insert(0, backend)
    from huggingface_hub import file_download
    if not symlinks:
        # What a Windows machine without Developer Mode answers: blobs are moved into snapshots.
        file_download.are_symlinks_supported = lambda cache_dir = None: False
    from hub.workers import hf_download
    if mode == "scoped":
        hf_download._download_scoped_snapshot(
            "Org/Model", "@diffusion",
            ["text_encoder.safetensors", "vae/vae.safetensors", "changed.safetensors"],
            None, "http",
        )
    else:
        hf_download._download_snapshot("Org/Model", None, "http")
    """
)


@pytest.fixture()
def fake_hub():
    hub = _FakeHub()
    yield hub
    hub.close()


def _run_worker(tmp_path, hub, mode, symlinks):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HF_", "HUGGING_FACE_", "UNSLOTH_PROTECTED"))
    }
    env.update(
        HF_ENDPOINT = hub.url,
        HF_HOME = str(tmp_path / "hf_home"),
        HF_HUB_CACHE = str(tmp_path / "hub"),
        HF_HUB_DISABLE_XET = "1",
        HF_HUB_ENABLE_HF_TRANSFER = "0",
        HF_HUB_DISABLE_TELEMETRY = "1",
        HF_HUB_DISABLE_SYMLINKS_WARNING = "1",
        HF_TOKEN = "",
        UNSLOTH_STUDIO_HOME = os.environ.get("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio")),
        NO_PROXY = "127.0.0.1,localhost",
        no_proxy = "127.0.0.1,localhost",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT, str(_BACKEND_DIR), mode, "1" if symlinks else "0"],
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc


def _symlinks_available(tmp_path) -> bool:
    try:
        (tmp_path / "probe_link").symlink_to(tmp_path)
        return True
    except OSError:
        return False


@pytest.mark.parametrize("mode", ["scoped", "snapshot"])
@pytest.mark.parametrize("symlinks", [False, True])
def test_new_revision_downloads_only_what_changed(tmp_path, fake_hub, mode, symlinks):
    if symlinks and not _symlinks_available(tmp_path):
        pytest.skip("symlinks unavailable")
    _run_worker(tmp_path, fake_hub, mode, symlinks)
    first = dict(fake_hub.downloaded)
    assert first[(OLD, "text_encoder.safetensors")] == len(_UNCHANGED)

    fake_hub.head = NEW
    fake_hub.downloaded.clear()
    proc = _run_worker(tmp_path, fake_hub, mode, symlinks)

    second = fake_hub.downloaded
    assert (NEW, "text_encoder.safetensors") not in second, proc.stderr[-2000:]
    assert (NEW, "vae/vae.safetensors") not in second
    assert second[(NEW, "changed.safetensors")] == len(_CHANGED_NEW)

    snap = tmp_path / "hub" / "models--Org--Model" / "snapshots" / NEW
    assert (snap / "text_encoder.safetensors").read_bytes() == _UNCHANGED
    assert (snap / "vae" / "vae.safetensors").read_bytes() == _VAE
    assert (snap / "changed.safetensors").read_bytes() == _CHANGED_NEW
    old_snap = snap.parent / OLD
    assert (old_snap / "changed.safetensors").read_bytes() == _CHANGED_OLD
    assert (tmp_path / "hub" / "models--Org--Model" / "refs" / "main").read_text() == NEW
    if symlinks:
        assert (snap / "text_encoder.safetensors").is_symlink()
        assert "Reused" not in proc.stderr
    else:
        assert not (snap / "text_encoder.safetensors").is_symlink()
        assert "Reused" in proc.stderr


def test_a_corrupted_same_size_copy_is_downloaded_again_not_carried_forward(tmp_path, fake_hub):
    """The Hub vouching for the OLD commit's digest says nothing about the bytes on disk now."""
    _run_worker(tmp_path, fake_hub, "scoped", symlinks = False)
    old_encoder = (
        tmp_path / "hub" / "models--Org--Model" / "snapshots" / OLD / "text_encoder.safetensors"
    )
    damaged = bytearray(old_encoder.read_bytes())
    damaged[1000:1010] = b"\x00" * 10
    old_encoder.write_bytes(bytes(damaged))

    fake_hub.head = NEW
    fake_hub.downloaded.clear()
    _run_worker(tmp_path, fake_hub, "scoped", symlinks = False)

    assert fake_hub.downloaded[(NEW, "text_encoder.safetensors")] == len(_UNCHANGED)
    assert (NEW, "vae/vae.safetensors") not in fake_hub.downloaded
    snap = old_encoder.parent.parent / NEW
    assert (snap / "text_encoder.safetensors").read_bytes() == _UNCHANGED


def test_a_reuse_pass_persists_its_digests_in_one_write(tmp_path, monkeypatch):
    files = {f"shard-{i}.bin": _blob(40 + i, 2048) for i in range(5)}
    repo_dir = _copy_layout(tmp_path, files)
    writes = []
    monkeypatch.setattr(
        snapshot_reuse, "_remember_digests", lambda entries: writes.append(dict(entries))
    )

    matches, _ = snapshot_reuse.find_reusable_copies(
        repo_dir, NEW, {p: (len(d), _sha256(d)) for p, d in files.items()}
    )

    assert set(matches) == set(files)
    assert len(writes) == 1 and len(writes[0]) == len(files)


def test_a_blob_a_live_peer_is_downloading_gets_no_pointer(tmp_path):
    from filelock import FileLock

    data = _blob(50, 4096)
    digest = _sha256(data)
    repo_dir = _copy_layout(tmp_path, {"model.safetensors": data})
    lock = tmp_path / "hub" / ".locks" / repo_dir.name / f"{digest}.lock"
    lock.parent.mkdir(parents = True)
    expected = [ExpectedFile("model.safetensors", len(data), digest)]

    with FileLock(str(lock)):
        # A peer that started after launch is not in protected_blob_hashes; its lock still counts.
        held = _reuse(tmp_path, expected)
    assert held.reused == ()
    assert not (repo_dir / "snapshots" / NEW / "model.safetensors").exists()

    assert _reuse(tmp_path, expected).reused == ("model.safetensors",)


def test_hard_linked_revisions_are_counted_once_in_cache_usage(tmp_path):
    from huggingface_hub import scan_cache_dir

    from hub.services.models.cache_inventory import _repo_gguf_size_bytes, repo_unique_size_bytes

    gguf = _blob(60, 8192)
    repo_dir = _copy_layout(tmp_path, {"model-Q4_K_M.gguf": gguf, "README.md": b"v1"})
    new = repo_dir / "snapshots" / NEW
    new.mkdir(parents = True)
    os.link(repo_dir / "snapshots" / OLD / "model-Q4_K_M.gguf", new / "model-Q4_K_M.gguf")
    (new / "README.md").write_bytes(b"v2")

    (repo,) = scan_cache_dir(tmp_path / "hub").repos
    # Without symlinks each snapshot path is its own blob_path; the inode is what is stored once.
    assert (
        len(
            {
                str(f.blob_path)
                for r in repo.revisions
                for f in r.files
                if f.file_name.endswith(".gguf")
            }
        )
        == 2
    )
    assert _repo_gguf_size_bytes(repo) == len(gguf)
    # The model and dataset listings total every file: the README differs, the weights do not.
    assert repo_unique_size_bytes(repo) == len(gguf) + len(b"v1") + len(b"v2")

    (new / "model-Q4_K_M.gguf").unlink()
    (new / "model-Q4_K_M.gguf").write_bytes(gguf)  # a real second copy still counts twice
    (repo,) = scan_cache_dir(tmp_path / "hub").repos
    assert _repo_gguf_size_bytes(repo) == 2 * len(gguf)
    assert repo_unique_size_bytes(repo) == 2 * len(gguf) + len(b"v1") + len(b"v2")


def test_a_plan_asks_the_hub_about_at_most_a_few_old_commits(tmp_path):
    repo_dir = tmp_path / "hub" / "models--Org--Model"
    for i in range(6):
        snap = repo_dir / "snapshots" / (f"{i:x}" * 40)[:40]
        snap.mkdir(parents = True)
        (snap / "w.safetensors").write_bytes(_blob(70 + i, 1024))  # same size, new bytes each time
        os.utime(snap, (1_000 + i, 1_000 + i))
    asked = []

    matches, _ = snapshot_reuse.find_reusable_copies(
        repo_dir,
        NEW,
        {"w.safetensors": (1024, "f" * 64)},
        remote_digests = lambda commit, paths: asked.append(commit) or {p: "e" * 64 for p in paths},
        allow_hashing = False,
    )

    assert matches == {}
    assert len(asked) == snapshot_reuse._REMOTE_DIGEST_COMMITS


def test_a_partial_is_only_removed_while_holding_the_blob_lock(tmp_path):
    from filelock import FileLock

    repo_dir = _copy_layout(tmp_path, {"x.bin": b"x"})
    digest = "d" * 64
    partial = repo_dir / "blobs" / f"{digest}.incomplete"
    partial.write_bytes(b"partial")
    lock = tmp_path / "hub" / ".locks" / repo_dir.name / f"{digest}.lock"
    lock.parent.mkdir(parents = True)

    with FileLock(str(lock)):  # a peer's download in progress
        snapshot_reuse._drop_superseded_partial(repo_dir, digest, frozenset())
        assert partial.exists()
    snapshot_reuse._drop_superseded_partial(repo_dir, digest, frozenset())
    assert not partial.exists()


def test_companion_cleanup_counts_hard_linked_revisions_once(tmp_path):
    from huggingface_hub import scan_cache_dir

    from hub.services.models.companion_cleanup import _repo_blob_bytes

    weights = _blob(80, 8192)
    repo_dir = _copy_layout(tmp_path, {"vae/vae.safetensors": weights})
    (repo_dir / "snapshots" / NEW / "vae").mkdir(parents = True)
    os.link(
        repo_dir / "snapshots" / OLD / "vae" / "vae.safetensors",
        repo_dir / "snapshots" / NEW / "vae" / "vae.safetensors",
    )

    (repo,) = scan_cache_dir(tmp_path / "hub").repos
    assert _repo_blob_bytes(repo) == len(weights)

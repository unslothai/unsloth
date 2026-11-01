# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cache hit after a network error must not publish an incomplete snapshot."""

import json
import subprocess
import sys

import pytest

from .test_hf_download_pinned_revision import _FakeHub, _FULL, _NEW, _OLD, _repo_dir, _run


@pytest.mark.parametrize("kind", ["full", "scoped", "gguf"])
def test_reused_file_and_failed_update_keep_old_weights_loadable(tmp_path, kind):
    hub = _FakeHub()
    try:
        hub.commits[_NEW]["config.json"] = hub.commits[_OLD]["config.json"]
        for commit in (_OLD, _NEW):
            hub.commits[commit]["mmproj-F16.gguf"] = b"GGUF" + b"shared" * 16
        hub.head, hub.down = _NEW, True
        repo = _repo_dir(tmp_path)
        old = repo / "snapshots" / _OLD
        old.mkdir(parents = True)
        for name in hub.commits[_OLD]:
            (old / name).write_bytes(hub.commits[_OLD][name])
        ref = repo / "refs" / "main"
        ref.parent.mkdir()
        ref.write_text(_OLD)
        extra = []
        if kind == "scoped":
            files = tmp_path / "files.json"
            files.write_text(json.dumps(list(_FULL)))
            extra = ["--variant", "test-scope", "--files-json", str(files)]
        elif kind == "gguf":
            extra = ["--variant", "Q4_K_M"]
        reused = "mmproj-F16.gguf" if kind == "gguf" else "config.json"
        weights = "model-Q4_K_M.gguf" if kind == "gguf" else "model.safetensors"
        result = _run(hub, tmp_path, *extra)
        assert result.returncode != 0
        assert (repo / "snapshots" / _NEW / reused).is_file(), result.stderr
        assert not (repo / "snapshots" / _NEW / weights).exists()
        assert ref.read_text() == _OLD, result.stderr
        # Exercise the real offline repo-ID resolver, not just the ref's contents.
        offline = subprocess.run(
            [
                sys.executable,
                "-c",
                "from huggingface_hub import hf_hub_download; import sys; "
                "print(hf_hub_download('fake/repo', sys.argv[2], "
                "cache_dir=sys.argv[1], local_files_only=True))",
                str(repo.parent),
                weights,
            ],
            text = True,
            capture_output = True,
            timeout = 20,
        )
        assert offline.returncode == 0, offline.stderr
        assert _OLD in offline.stdout
    finally:
        hub.close()

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the malware / unsafe-file gate (utils.security.file_security).

The gate queries Hugging Face's own security scan
(``model_info(securityStatus=True).security_repo_status``) -- metadata only, it
never downloads the flagged files. Only the Hub call is stubbed (no network).
Policy: hard block on a non-"safe" level (unknown levels fail closed); fail open
when the scan is unavailable; skip local paths only (a remote *.gguf-named repo is
still scanned); no first-party exemption. The block is scoped to the load-path RCE
vector -- a root-level, code-executing file -- so flagged safetensors (inert) and
flagged pickles in subdirectories (which from_pretrained never reads) do not block.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from utils.security import evaluate_file_security


def _patch_status(status):
    """Patch huggingface_hub.model_info to return one fixed security_repo_status."""

    def _mi(*_args, **_kwargs):
        return SimpleNamespace(security_repo_status = status)

    return patch("huggingface_hub.model_info", side_effect = _mi)


def _patch_raises(exc = RuntimeError("offline")):
    return patch("huggingface_hub.model_info", side_effect = exc)


def _patch_no_index():
    """Make the weight-index lookup find no index files (definitive: nothing sharded)."""
    from huggingface_hub.utils import EntryNotFoundError

    def _dl(
        repo_id = None,
        filename = None,
        token = None,
        **kw,
    ):
        raise EntryNotFoundError(filename or "")

    return patch("huggingface_hub.hf_hub_download", side_effect = _dl)


def _patch_index(weight_map, index_filename = "pytorch_model.bin.index.json"):
    """Serve a root weight index mapping tensor names -> shard paths; others 404."""
    import json
    import tempfile
    from pathlib import Path

    from huggingface_hub.utils import EntryNotFoundError

    def _dl(
        repo_id = None,
        filename = None,
        token = None,
        **kw,
    ):
        if filename == index_filename:
            p = Path(tempfile.mkdtemp()) / filename
            p.write_text(json.dumps({"weight_map": weight_map}))
            return str(p)
        raise EntryNotFoundError(filename or "")

    return patch("huggingface_hub.hf_hub_download", side_effect = _dl)


def _patch_index_unreadable():
    """Make every index fetch fail transiently (inconclusive lookup -> fail closed)."""

    def _dl(
        repo_id = None,
        filename = None,
        token = None,
        **kw,
    ):
        raise RuntimeError("transient network error")

    return patch("huggingface_hub.hf_hub_download", side_effect = _dl)


def _patch_index_mixed(weight_map, readable_index, failing_index):
    """Serve ONE index cleanly while a DIFFERENT index fails transiently.

    Models the dangerous case: one index reads fine (so a naive "did we read any
    index?" check would treat the result as definitive), but the flagged shard is
    listed only by the index we could not read.
    """
    import json
    import tempfile
    from pathlib import Path

    from huggingface_hub.utils import EntryNotFoundError

    def _dl(repo_id = None, filename = None, token = None, **kw):
        if filename == readable_index:
            p = Path(tempfile.mkdtemp()) / filename
            p.write_text(json.dumps({"weight_map": weight_map}))
            return str(p)
        if filename == failing_index:
            raise RuntimeError("transient network error")
        raise EntryNotFoundError(filename or "")

    return patch("huggingface_hub.hf_hub_download", side_effect = _dl)


@pytest.mark.parametrize("level", ["unsafe", "suspicious", "malicious"])
def test_blocks_each_blocking_level(level):
    status = {"scansDone": True, "filesWithIssues": [{"path": "pytorch_model.bin", "level": level}]}
    with _patch_status(status):
        d = evaluate_file_security("evil/repo")
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "pytorch_model.bin", "level": level}]
    assert d.response_payload()["security_blocked"] is True


def test_ignores_safe_only():
    status = {
        "scansDone": True,
        "filesWithIssues": [{"path": "model.safetensors", "level": "safe"}],
    }
    with _patch_status(status):
        d = evaluate_file_security("good/repo")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_blocks_unsafe_even_when_scans_not_done():
    # scansDone is frequently False even for clean repos; a file ALREADY flagged
    # unsafe must still block (do not gate blocking on scansDone).
    status = {"scansDone": False, "filesWithIssues": [{"path": "x.pkl", "level": "unsafe"}]}
    with _patch_status(status):
        d = evaluate_file_security("evil/repo")
    assert d.blocked is True


def test_fail_open_when_scan_unavailable():
    # model_info returns no security_repo_status -> unknown -> allow.
    with _patch_status(None):
        d = evaluate_file_security("unknown/repo")
    assert d.blocked is False


def test_fail_open_on_exception_offline():
    with _patch_raises():
        d = evaluate_file_security("offline/repo")
    assert d.blocked is False


def test_fail_open_scans_done_no_issues():
    with _patch_status({"scansDone": True, "filesWithIssues": []}):
        d = evaluate_file_security("clean/repo")
    assert d.blocked is False


def test_skips_local_path():
    # A local path has no Hub scan; must not even call model_info.
    with patch("huggingface_hub.model_info", side_effect = AssertionError("should not be called")):
        d = evaluate_file_security("/tmp/some/local/model")
    assert d.blocked is False
    assert "local" in d.reason


def test_remote_gguf_named_repo_is_still_scanned():
    # A repo must not evade the scan by ending its NAME in .gguf: only LOCAL paths
    # skip the Hub scan. A remote repo id (even "evil/model.gguf") is scanned, so a
    # poisoned pickle smuggled into such a repo is still blocked.
    status = {
        "scansDone": True,
        "filesWithIssues": [{"path": "pytorch_model.bin", "level": "unsafe"}],
    }
    with _patch_status(status):
        d = evaluate_file_security("evil/model.gguf")
    assert d.blocked is True
    assert d.unsafe_files


def test_skips_local_gguf_file():
    # A local .gguf path is caught by is_local_path -- no Hub call.
    with patch("huggingface_hub.model_info", side_effect = AssertionError("should not be called")):
        d = evaluate_file_security("/tmp/models/model.gguf")
    assert d.blocked is False
    assert "local" in d.reason


def test_no_first_party_exemption():
    # A poisoned pickle in a first-party repo still blocks (compromised-repo defense).
    status = {
        "scansDone": True,
        "filesWithIssues": [{"path": "pytorch_model.bin", "level": "unsafe"}],
    }
    with _patch_status(status):
        d = evaluate_file_security("unsloth/some-model")
    assert d.blocked is True


def test_malformed_entries_are_ignored():
    status = {
        "scansDone": True,
        "filesWithIssues": ["not-a-dict", {"path": "ok.pkl", "level": "unsafe"}],
    }
    with _patch_status(status):
        d = evaluate_file_security("evil/repo")
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "ok.pkl", "level": "unsafe"}]


def test_response_payload_shape():
    status = {"scansDone": True, "filesWithIssues": [{"path": "a.pkl", "level": "malicious"}]}
    with _patch_status(status):
        payload = evaluate_file_security("evil/repo").response_payload()
    assert set(payload) == {"unsafe_files", "security_blocked", "reason"}
    assert payload["security_blocked"] is True
    assert payload["unsafe_files"] == [{"path": "a.pkl", "level": "malicious"}]


# ── Load-path RCE scoping: block only files a load would actually deserialize ──


def test_flagged_safetensors_does_not_block():
    # safetensors is tensor-only and cannot execute code; a flag on one (Hub
    # sometimes marks a repo's safetensors "unsafe" when picklescan trips on a
    # sibling pickle) is not an RCE vector, so it must not block.
    status = {
        "scansDone": False,
        "filesWithIssues": [{"path": "model-00001-of-00004.safetensors", "level": "unsafe"}],
    }
    with _patch_status(status):
        d = evaluate_file_security("nvidia/some-model")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_flagged_subdirectory_pickle_does_not_block():
    # from_pretrained reads weight files at the repo ROOT; a flagged pickle in a
    # subdirectory that NO root index references (e.g. a NeMo checkpoint) is never
    # loaded, so it must not block.
    status = {
        "scansDone": False,
        "filesWithIssues": [
            {"path": "nemo/weights/common.pt", "level": "unsafe"},
            {"path": "nemo/weights/__0_0.distcp", "level": "unsafe"},
        ],
    }
    with _patch_status(status), _patch_no_index():
        d = evaluate_file_security("nvidia/some-model")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_nemotron_h_shaped_status_loads():
    # Real shape of nvidia/Nemotron-H-8B-Base-8K: root safetensors flagged "unsafe"
    # plus NeMo pickle checkpoints under nemo/ that no index references. None are a
    # load-path RCE vector, so a legitimate first-party model must LOAD, not block.
    status = {
        "scansDone": False,
        "filesWithIssues": [
            {"path": "nemo/weights/.metadata", "level": "unsafe"},
            {"path": "nemo/weights/__0_0.distcp", "level": "unsafe"},
            {"path": "nemo/weights/common.pt", "level": "unsafe"},
            {"path": "model-00001-of-00004.safetensors", "level": "unsafe"},
            {"path": "model-00002-of-00004.safetensors", "level": "unsafe"},
        ],
    }
    with _patch_status(status), _patch_no_index():
        d = evaluate_file_security("nvidia/Nemotron-H-8B-Base-8K")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_indexed_subdir_shard_blocks():
    # A flagged pickle shard in a SUBDIRECTORY that a root weight index references is
    # deserialized by from_pretrained, so it IS a load-path vector and must block.
    status = {
        "scansDone": False,
        "filesWithIssues": [
            {"path": "shards/pytorch_model-00001-of-00002.bin", "level": "unsafe"},
        ],
    }
    weight_map = {
        "layer.0.weight": "shards/pytorch_model-00001-of-00002.bin",
        "layer.1.weight": "shards/pytorch_model-00002-of-00002.bin",
    }
    with _patch_status(status), _patch_index(weight_map):
        d = evaluate_file_security("evil/sharded")
    assert d.blocked is True
    assert d.unsafe_files == [
        {"path": "shards/pytorch_model-00001-of-00002.bin", "level": "unsafe"}
    ]


def test_unindexed_subdir_pickle_does_not_block_when_index_present():
    # An index exists, but the flagged subdir pickle is NOT listed in it -> not a
    # load input -> must not block (the model still loads from the indexed shards).
    status = {
        "scansDone": False,
        "filesWithIssues": [{"path": "extras/notes.bin", "level": "unsafe"}],
    }
    weight_map = {"layer.0.weight": "pytorch_model-00001-of-00001.bin"}
    with _patch_status(status), _patch_index(weight_map):
        d = evaluate_file_security("org/has-index")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_inconclusive_index_lookup_blocks_subdir_pickle():
    # If the index lookup cannot complete (transient error) we cannot rule out that
    # the flagged subdir pickle is a loadable shard, so fail closed (block).
    status = {
        "scansDone": False,
        "filesWithIssues": [{"path": "weights/model_part.bin", "level": "unsafe"}],
    }
    with _patch_status(status), _patch_index_unreadable():
        d = evaluate_file_security("org/transient")
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "weights/model_part.bin", "level": "unsafe"}]


def test_partial_index_read_with_transient_failure_blocks_subdir_pickle():
    # One index (safetensors) reads cleanly, but the bin index fails transiently and
    # the flagged subdir pickle is a .bin shard the unread index would list. A partial
    # path set must not be treated as definitive: fail closed (block).
    status = {
        "scansDone": False,
        "filesWithIssues": [
            {"path": "shards/pytorch_model-00001-of-00002.bin", "level": "unsafe"},
        ],
    }
    # The readable safetensors index lists unrelated, benign shards; the flagged .bin
    # would only appear in the bin index that we could not fetch.
    safetensors_map = {"layer.0.weight": "model-00001-of-00001.safetensors"}
    with _patch_status(status), _patch_index_mixed(
        safetensors_map,
        readable_index = "model.safetensors.index.json",
        failing_index = "pytorch_model.bin.index.json",
    ):
        d = evaluate_file_security("evil/mixed-index")
    assert d.blocked is True
    assert d.unsafe_files == [
        {"path": "shards/pytorch_model-00001-of-00002.bin", "level": "unsafe"}
    ]


def test_root_pickle_alongside_safetensors_still_blocks():
    # A genuinely-dangerous ROOT pickle blocks even when the repo also ships a
    # (flagged) safetensors -- the pickle is a real load-path vector.
    status = {
        "scansDone": False,
        "filesWithIssues": [
            {"path": "model.safetensors", "level": "unsafe"},
            {"path": "pytorch_model.bin", "level": "unsafe"},
        ],
    }
    with _patch_status(status):
        d = evaluate_file_security("evil/repo")
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "pytorch_model.bin", "level": "unsafe"}]


def test_eicar_shaped_root_files_block():
    # mcpotato/42-eicar-street ships its dangerous files at the repo ROOT, so the
    # canonical malware repo stays blocked under the load-path scoping.
    status = {
        "scansDone": True,
        "filesWithIssues": [
            {"path": "model_broken_X.pkl", "level": "unsafe"},
            {"path": "danger.dat", "level": "unsafe"},
            {"path": "eicar_test_file", "level": "unsafe"},
        ],
    }
    with _patch_status(status):
        d = evaluate_file_security("mcpotato/42-eicar-street")
    assert d.blocked is True
    assert len(d.unsafe_files) == 3


def test_unknown_future_level_fails_closed():
    # Hub schema drift: a non-"safe" level we do not recognize (e.g. "infected") on
    # a root pickle must block, so a newly-named bad verdict is not silently allowed.
    status = {"scansDone": True, "filesWithIssues": [{"path": "weights.bin", "level": "infected"}]}
    with _patch_status(status):
        d = evaluate_file_security("evil/repo")
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "weights.bin", "level": "infected"}]


def test_pending_or_scanning_level_does_not_block():
    # A not-yet-finished per-file scan state must not false-block.
    for lvl in ("pending", "scanning", "queued", "unscanned", "error"):
        status = {
            "scansDone": False,
            "filesWithIssues": [{"path": "pytorch_model.bin", "level": lvl}],
        }
        with _patch_status(status):
            d = evaluate_file_security("some/repo")
        assert d.blocked is False, lvl

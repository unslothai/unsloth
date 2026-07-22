# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the malware / unsafe-file gate (utils.security.file_security).

The gate reads HF's security scan (model_info securityStatus) metadata-only and never
downloads flagged files; only the Hub call is stubbed. Policy: block a non-"safe" level
(unknown levels fail closed), fail open when the scan is unavailable, skip local paths
only, no first-party exemption. The block is scoped to the load-path RCE vector (a
root-level code-executing file), so flagged safetensors and subdir pickles do not block.
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
    """Serve one index cleanly while another fails transiently: the flagged shard is
    listed only by the index we could not read, so a naive "read any index?" check breaks."""
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
    # scansDone is often False for clean repos; an already-flagged file must still block.
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
    # Only LOCAL paths skip the Hub scan, so a remote .gguf repo is still scanned and a
    # poisoned pickle smuggled into it is blocked.
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
    # safetensors is tensor-only and cannot execute code, so a flag on one (often
    # picklescan tripping on a sibling pickle) is not an RCE vector and must not block.
    status = {
        "scansDone": False,
        "filesWithIssues": [{"path": "model-00001-of-00004.safetensors", "level": "unsafe"}],
    }
    with _patch_status(status):
        d = evaluate_file_security("nvidia/some-model")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_flagged_subdirectory_pickle_does_not_block():
    # from_pretrained reads only root weights; a flagged subdir pickle no root index
    # references (e.g. a NeMo checkpoint) is never loaded, so it must not block.
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
    # Real Nemotron-H-8B-Base-8K shape: flagged root safetensors + unreferenced nemo/
    # pickles. None is a load-path vector, so it must load.
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
    # A flagged subdir shard that a root index references IS deserialized, so it blocks.
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
    # An index exists but does not list the flagged subdir pickle -> not loaded -> no block.
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
    # An unreadable index can't rule out that the flagged subdir pickle is a shard -> block.
    status = {
        "scansDone": False,
        "filesWithIssues": [{"path": "weights/model_part.bin", "level": "unsafe"}],
    }
    with _patch_status(status), _patch_index_unreadable():
        d = evaluate_file_security("org/transient")
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "weights/model_part.bin", "level": "unsafe"}]


def test_partial_index_read_with_transient_failure_blocks_subdir_pickle():
    # The bin index (which would list the flagged shard) fails transiently; a partial
    # path set is not definitive, so fail closed.
    status = {
        "scansDone": False,
        "filesWithIssues": [
            {"path": "shards/pytorch_model-00001-of-00002.bin", "level": "unsafe"},
        ],
    }
    # The readable index lists only benign shards; the flagged .bin is in the unread index.
    safetensors_map = {"layer.0.weight": "model-00001-of-00001.safetensors"}
    with (
        _patch_status(status),
        _patch_index_mixed(
            safetensors_map,
            readable_index = "model.safetensors.index.json",
            failing_index = "pytorch_model.bin.index.json",
        ),
    ):
        d = evaluate_file_security("evil/mixed-index")
    assert d.blocked is True
    assert d.unsafe_files == [
        {"path": "shards/pytorch_model-00001-of-00002.bin", "level": "unsafe"}
    ]


def test_root_pickle_alongside_safetensors_still_blocks():
    # A real root pickle blocks even alongside a flagged safetensors; it is a load-path vector.
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
    # The canonical eicar repo ships its dangerous files at the ROOT, so it stays blocked.
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
    # Hub schema drift: an unrecognized non-"safe" level (e.g. "infected") on a root pickle must block.
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


# -- Subdir load roots: Spark-TTS / BiCodec load from_pretrained(<snapshot>/LLM) --


def test_flagged_pickle_under_load_subdir_blocks():
    # A flagged pickle directly under a declared load subdir is a root-level artifact there.
    status = {
        "scansDone": False,
        "filesWithIssues": [{"path": "LLM/pytorch_model.bin", "level": "unsafe"}],
    }
    with _patch_status(status), _patch_no_index():
        d = evaluate_file_security("org/spark-tts", load_subdirs = ("LLM",))
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "LLM/pytorch_model.bin", "level": "unsafe"}]


def test_flagged_pickle_under_subdir_without_load_root_does_not_block():
    # Same file, but NOT declared a load root and not indexed -> not deserialized.
    status = {
        "scansDone": False,
        "filesWithIssues": [{"path": "LLM/pytorch_model.bin", "level": "unsafe"}],
    }
    with _patch_status(status), _patch_no_index():
        d = evaluate_file_security("org/not-a-load-root")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_indexed_shard_under_load_subdir_blocks():
    # An index inside the load subdir referencing a flagged shard makes it a vector.
    status = {
        "scansDone": False,
        "filesWithIssues": [
            {"path": "LLM/shards/pytorch_model-00001-of-00002.bin", "level": "unsafe"}
        ],
    }
    weight_map = {
        "layer.0.weight": "shards/pytorch_model-00001-of-00002.bin",
        "layer.1.weight": "shards/pytorch_model-00002-of-00002.bin",
    }
    with (
        _patch_status(status),
        _patch_index(weight_map, index_filename = "LLM/pytorch_model.bin.index.json"),
    ):
        d = evaluate_file_security("org/spark-tts", load_subdirs = ("LLM",))
    assert d.blocked is True


# -- Source files are the consent gate's domain, not a deserialization vector --


def test_flagged_root_python_helper_does_not_block():
    # A root .py is never deserialized; repo code runs only via auto_map under the consent
    # gate, so flagging it here would false-block.
    status = {
        "scansDone": True,
        "filesWithIssues": [
            {"path": "build_pickles.py", "level": "unsafe"},
            {"path": "train.py", "level": "suspicious"},
        ],
    }
    with _patch_status(status):
        d = evaluate_file_security("org/has-helper-scripts")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_root_pickle_still_blocks_with_flagged_python_sibling():
    # The .py exemption must not mask a genuine root pickle in the same repo.
    status = {
        "scansDone": True,
        "filesWithIssues": [
            {"path": "convert.py", "level": "unsafe"},
            {"path": "pytorch_model.bin", "level": "unsafe"},
        ],
    }
    with _patch_status(status):
        d = evaluate_file_security("evil/mixed")
    assert d.blocked is True


# -- Alias resolution: scan the repo the loader actually fetches from --


def _patch_status_capture(status):
    """Like _patch_status, but records the repo id model_info was queried with."""
    seen = {}

    def _mi(repo, *_a, **_k):
        seen["repo"] = repo
        return SimpleNamespace(security_repo_status = status)

    return patch("huggingface_hub.model_info", side_effect = _mi), seen


def test_spark_tts_llm_alias_scans_real_repo():
    # "Spark-TTS-0.5B/LLM" loads as unsloth/Spark-TTS-0.5B with LLM as load root; scanning
    # the literal alias 404s and fails open, missing a flagged LLM/ pickle.
    status = {"filesWithIssues": [{"path": "LLM/pytorch_model.bin", "level": "unsafe"}]}
    cap, seen = _patch_status_capture(status)
    with cap, patch("utils.paths.is_local_path", return_value = False), _patch_no_index():
        d = evaluate_file_security("Spark-TTS-0.5B/LLM", load_subdirs = ())
    assert seen["repo"] == "unsloth/Spark-TTS-0.5B"  # scanned the real repo, not the alias
    assert d.model_name == "unsloth/Spark-TTS-0.5B"
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "LLM/pytorch_model.bin", "level": "unsafe"}]


def test_non_llm_alias_is_not_rewritten():
    # A normal repo id with one slash must be scanned as-is (no spurious rewrite).
    status = {"filesWithIssues": [{"path": "pytorch_model.bin", "level": "unsafe"}]}
    cap, seen = _patch_status_capture(status)
    with cap, patch("utils.paths.is_local_path", return_value = False):
        d = evaluate_file_security("org/model")
    assert seen["repo"] == "org/model"
    assert d.model_name == "org/model"


def test_generic_slash_llm_repo_is_scanned_as_itself():
    # A third-party repo merely ending in "/LLM" is not a bicodec alias, so it must be
    # scanned as itself; rewriting to unsloth/<parent> would scan the wrong repo.
    status = {"filesWithIssues": [{"path": "pytorch_model.bin", "level": "unsafe"}]}
    cap, seen = _patch_status_capture(status)
    with cap, patch("utils.paths.is_local_path", return_value = False):
        d = evaluate_file_security("evil/LLM")
    assert seen["repo"] == "evil/LLM"  # scanned the real repo, not unsloth/evil
    assert d.model_name == "evil/LLM"
    assert d.blocked is True


def test_security_load_subdirs_yaml_fallback(monkeypatch):
    # Tokenizer detection failed, but a YAML default of audio_type=bicodec still yields LLM.
    import utils.models.model_config as mc
    from utils.security import security_load_subdirs

    monkeypatch.setattr(mc, "detect_audio_type", lambda *_a, **_k: None)
    monkeypatch.setattr(mc, "load_model_defaults", lambda *_a, **_k: {"audio_type": "bicodec"})
    assert security_load_subdirs("unsloth/Spark-TTS-0.5B") == ("LLM",)

    # A non-bicodec default contributes no subdir.
    monkeypatch.setattr(mc, "load_model_defaults", lambda *_a, **_k: {"audio_type": None})
    assert security_load_subdirs("unsloth/Llama-3.2-1B") == ()

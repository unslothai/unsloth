# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.model_ids import model_id_matches, public_model_id  # noqa: E402


def test_local_gguf_path_becomes_clean_stem():
    assert public_model_id("/srv/models/Qwen3-30B-A3B-Q4_K_M.gguf") == "Qwen3-30B-A3B-Q4_K_M"
    assert public_model_id("/home/u/.cache/models/llama.gguf") == "llama"


def test_hf_repo_id_unchanged():
    assert public_model_id("unsloth/Qwen3-30B-A3B-GGUF") == "unsloth/Qwen3-30B-A3B-GGUF"
    assert public_model_id("Qwen3-30B-A3B") == "Qwen3-30B-A3B"


def test_none_and_empty_passthrough():
    assert public_model_id(None) is None
    assert public_model_id("") == ""


def test_windows_path():
    assert public_model_id("C:\\models\\foo.gguf") == "foo"
    assert public_model_id("models\\sub\\bar.gguf") == "bar"


def test_directory_path_uses_basename():
    assert public_model_id("/opt/models/MyModelDir") == "MyModelDir"
    # A 3+ segment relative path is a local path, not an org/model repo id.
    assert public_model_id("a/b/c") == "c"


def test_hf_cache_snapshot_recovers_the_repo_id():
    from core.inference.model_ids import hf_cache_repo_id

    # The snapshot basename is a commit sha, so recover org/name instead.
    snapshot = (
        "/home/u/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF"
        "/snapshots/c1ac76e99d5513b141e8adde7288b85c3f9c32ec"
    )
    assert public_model_id(snapshot) == "unsloth/gemma-4-31B-it-GGUF"
    # A file inside the snapshot resolves the same way, not to the file stem.
    assert public_model_id(snapshot + "/gemma-4-31B-it-UD-Q5_K_XL.gguf") == (
        "unsloth/gemma-4-31B-it-GGUF"
    )
    assert hf_cache_repo_id("/opt/models/plain.gguf") is None
    assert hf_cache_repo_id(None) is None


def test_relative_and_home_paths_are_sanitized():
    # ./ ../ ~ prefixed paths are local and must not be echoed raw.
    assert public_model_id("./model.gguf") == "model"
    assert public_model_id("../models/foo.gguf") == "foo"
    assert public_model_id("~/models/baz.gguf") == "baz"
    assert public_model_id("./mistral") == "mistral"
    assert public_model_id("~/mistral") == "mistral"
    assert public_model_id(".\\models\\foo.gguf") == "foo"


def test_dotted_repo_id_not_mistaken_for_relative_path():
    # A leading dot that is not ./ or ../ is an ordinary clean name.
    assert public_model_id(".hidden-model") == ".hidden-model"
    assert public_model_id("org/.config") == "org/.config"


def test_matches_clean_and_legacy():
    path = "/srv/models/Qwen3-Q4.gguf"
    assert model_id_matches("Qwen3-Q4", path)  # clean public id
    assert model_id_matches(path, path)  # legacy raw path
    assert not model_id_matches("other", path)
    assert not model_id_matches(None, path)
    assert not model_id_matches("x", None)


def test_display_model_name_uses_the_repo_leaf_not_the_snapshot_sha():
    from core.inference.model_ids import display_model_name

    posix = (
        "/home/u/.cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF"
        "/snapshots/57326b941c4603e24d1a5e71c22520c66e086eb8"
    )
    assert display_model_name(posix) == "DeepSeek-V4-Flash-0731-GGUF"
    # The reported case: a Windows cache path has no "/" to split on, so a raw
    # rsplit would label the model with the whole home directory.
    windows = (
        "C:\\Users\\An\\.cache\\huggingface\\hub"
        "\\models--unsloth--DeepSeek-V4-Flash-0731-GGUF"
        "\\snapshots\\57326b941c4603e24d1a5e71c22520c66e086eb8"
    )
    assert display_model_name(windows) == "DeepSeek-V4-Flash-0731-GGUF"


def test_display_model_name_leaves_ordinary_ids_alone():
    from core.inference.model_ids import display_model_name

    assert display_model_name("unsloth/Qwen3-30B-A3B-GGUF") == "Qwen3-30B-A3B-GGUF"
    assert display_model_name("Qwen3-30B-A3B") == "Qwen3-30B-A3B"
    assert display_model_name("/srv/models/Qwen3-30B-A3B-Q4_K_M.gguf") == (
        "Qwen3-30B-A3B-Q4_K_M"
    )
    assert display_model_name(None) is None
    assert display_model_name("") == ""

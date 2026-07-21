# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.model_ids import model_id_matches, public_model_id  # noqa: E402


def test_local_gguf_path_becomes_clean_stem():
    assert (
        public_model_id("/srv/models/Qwen3-30B-A3B-Q4_K_M.gguf")
        == "Qwen3-30B-A3B-Q4_K_M"
    )
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

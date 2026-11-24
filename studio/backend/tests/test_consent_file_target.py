# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The remote-code gate assumed a local target is a folder. A JSON file path loads as the config itself
(AutoConfig reads it and imports its auto_map, here from another Hub repo), and a folder named *.gguf
is a model folder, not a llama.cpp file: both must be read for auto_map."""

import json

from utils.security import consent, remote_code_scan

AUTO_MAP = {"auto_map": {"AutoConfig": "acme/hidden--configuration_helper.HelperConfig"}}


def test_a_json_file_target_is_read_as_the_config(tmp_path):
    blob = tmp_path / "blobs" / "0123abcd"
    blob.parent.mkdir()
    blob.write_text(json.dumps({"model_type": "dac", **AUTO_MAP}))
    assert consent._config_has_auto_map(str(blob)) is True
    assert remote_code_scan.external_auto_map_repos(str(blob)) == {"acme/hidden"}


def test_the_scan_reviews_the_external_code_a_file_target_names(tmp_path, monkeypatch):
    target = tmp_path / "codec.json"
    target.write_text(json.dumps(AUTO_MAP))
    seen = []

    def add_external(files, refs, hf_token, model_name):
        seen.extend(refs)
        return True

    monkeypatch.setattr(remote_code_scan, "_add_external_refs", add_external)
    remote_code_scan.repo_remote_code_files(str(target))
    assert ("acme/hidden", "configuration_helper.py") in seen


def test_a_folder_named_gguf_is_scanned(tmp_path):
    folder = tmp_path / "model.gguf"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(AUTO_MAP))
    assert consent._config_has_auto_map(str(folder)) is True


def test_a_gguf_file_is_still_skipped(tmp_path):
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    assert consent._config_has_auto_map(str(gguf)) is False
    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert consent._config_has_auto_map(str(folder)) is False

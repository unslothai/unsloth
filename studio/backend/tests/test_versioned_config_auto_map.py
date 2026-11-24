# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""transformers follows config.json's configuration_files list to a config.<version>.json and takes
auto_map from there. The remote-code scan, the consent gate and dictation's own-code check must read
that file too, or its code (here from another Hub repo) runs unseen."""

import json

from core.inference import stt_mlx
from utils.security import consent, remote_code_scan


def _model(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"model_type": "llama", "configuration_files": ["config.4.0.0.json"]})
    )
    (folder / "config.4.0.0.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "auto_map": {"AutoConfig": "acme/hidden--configuration_helper.HelperConfig"},
            }
        )
    )
    return folder


def test_consent_gate_sees_auto_map_in_a_versioned_config(tmp_path):
    assert consent._config_has_auto_map(str(_model(tmp_path))) is True


def test_scan_fetches_the_external_repo_a_versioned_config_names(tmp_path):
    assert remote_code_scan.external_auto_map_repos(str(_model(tmp_path))) == {"acme/hidden"}


def test_scan_adds_the_external_code_to_what_it_reviews(tmp_path, monkeypatch):
    seen = []

    def add_external(files, refs, hf_token, model_name):
        seen.extend(refs)
        return True

    monkeypatch.setattr(remote_code_scan, "_add_external_refs", add_external)
    remote_code_scan.repo_remote_code_files(str(_model(tmp_path)))
    assert ("acme/hidden", "configuration_helper.py") in seen


def test_dictation_counts_a_versioned_config_as_custom_code(tmp_path):
    assert stt_mlx.ships_custom_code(_model(tmp_path)) is True


def test_a_self_referencing_list_terminates(tmp_path):
    folder = tmp_path / "loop"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"configuration_files": ["config.json", "../x.json", "/abs.json"]})
    )
    assert consent._config_has_auto_map(str(folder)) is False
    assert (
        remote_code_scan.versioned_config_paths(
            "config.json", {"configuration_files": ["../x.json", "/a.json"]}
        )
        == ()
    )

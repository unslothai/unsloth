# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stray non-object JSON file under an Ollama ``manifests/`` tree must be skipped.

``rglob("*")`` accepts every file below ``manifests/``, so an interrupted pull, an editor
backup, or any unrelated JSON reaches the parser. Both readers used to call ``.get()`` on
whatever ``json.loads`` returned; on a list or a string that raises ``AttributeError``, which
neither reader's ``except OSError`` catches, so one such file 500'd ``GET /models/local`` and
``GET /v1/models`` and emptied the whole model picker.
"""

import json

import pytest

from routes.models import _dir_has_downloaded_model, _scan_ollama_dir

# Valid JSON, wrong shape. Each of these is something json.loads happily returns.
NON_OBJECT_MANIFESTS = ("[]", '["a"]', '"just a string"', "3", "null", "true")


def _manifest_dir(root, model = "foo"):
    d = root / "manifests" / "registry.ollama.ai" / "library" / model
    d.mkdir(parents = True, exist_ok = True)
    return d


def _write_good_model(
    root,
    model = "good",
    blob = "sha256-abc123",
):
    """A manifest whose model layer resolves to a real blob, i.e. one the scan must surface."""
    blobs = root / "blobs"
    blobs.mkdir(exist_ok = True)
    (blobs / blob.replace(":", "-")).write_bytes(b"GGUF fake weights")
    (_manifest_dir(root, model) / "latest").write_text(
        json.dumps(
            {
                "layers": [
                    {"mediaType": "application/vnd.ollama.image.model", "digest": blob},
                ],
            },
        ),
        encoding = "utf-8",
    )


@pytest.mark.parametrize("payload", NON_OBJECT_MANIFESTS)
def test_scan_skips_non_object_manifest(tmp_path, payload):
    (_manifest_dir(tmp_path, "bad") / "latest").write_text(payload, encoding = "utf-8")
    (tmp_path / "blobs").mkdir(exist_ok = True)

    assert _scan_ollama_dir(tmp_path) == []


@pytest.mark.parametrize("payload", NON_OBJECT_MANIFESTS)
def test_one_bad_manifest_does_not_hide_the_good_models(tmp_path, payload):
    _write_good_model(tmp_path)
    (_manifest_dir(tmp_path, "bad") / "latest").write_text(payload, encoding = "utf-8")

    found = _scan_ollama_dir(tmp_path)

    assert [m.model_id for m in found] == ["ollama/good:latest"]


@pytest.mark.parametrize(
    "manifest",
    [
        {"layers": "not-a-list"},
        {"layers": {"mediaType": "application/vnd.ollama.image.model"}},
        {"layers": ["not-a-dict", 7, None]},
        {"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": 42}]},
        {"config": "not-a-dict"},
        {"config": ["digest"]},
        # A dict config carrying a non-string digest: truthy, so it reaches
        # config_digest.replace(":", "-") once a blobs/ dir exists.
        {"config": {"digest": 42}},
    ],
)
def test_scan_survives_wrong_shapes_inside_the_manifest(tmp_path, manifest):
    (_manifest_dir(tmp_path, "bad") / "latest").write_text(json.dumps(manifest), encoding = "utf-8")
    (tmp_path / "blobs").mkdir(exist_ok = True)

    assert _scan_ollama_dir(tmp_path) == []


@pytest.mark.parametrize("payload", NON_OBJECT_MANIFESTS)
def test_folder_chip_probe_skips_non_object_manifest(tmp_path, payload):
    (_manifest_dir(tmp_path, "bad") / "latest").write_text(payload, encoding = "utf-8")
    (tmp_path / "blobs").mkdir(exist_ok = True)

    assert _dir_has_downloaded_model(tmp_path) is False


def test_folder_chip_probe_still_finds_a_real_model(tmp_path):
    _write_good_model(tmp_path)
    (_manifest_dir(tmp_path, "bad") / "latest").write_text("[]", encoding = "utf-8")

    assert _dir_has_downloaded_model(tmp_path) is True

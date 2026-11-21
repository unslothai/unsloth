# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""upload_to_huggingface must only ship tensorboard logs sitting next to the exported file:
an events file elsewhere in the working directory used to leak into the repo, and
path_in_repo was built by string-replacing file_location, which yields leading slashes or
backslashes. Hub calls are mocked: nothing reaches the network."""

import os

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from types import SimpleNamespace

import huggingface_hub
import pytest

import unsloth.save as save_mod


class _Hub:
    """Fake HfApi recording upload_file calls as (path_or_fileobj, path_in_repo)."""

    def __init__(self):
        self.uploads = []
        hub = self

        class FakeApi:
            def __init__(self, *a, **kw):
                pass

            def upload_file(self, path_or_fileobj, path_in_repo, **kw):
                hub.uploads.append((str(path_or_fileobj), path_in_repo))

        self.Api = FakeApi


class _Card:
    def __init__(self, content):
        self.data = SimpleNamespace(datasets = None)

    def push_to_hub(self, *a, **kw):
        pass


@pytest.fixture
def hub(monkeypatch):
    h = _Hub()
    monkeypatch.setattr(save_mod, "HfApi", h.Api)
    monkeypatch.setattr(huggingface_hub, "create_repo", lambda **kw: None)
    monkeypatch.setattr(huggingface_hub, "ModelCard", _Card)
    return h


def _model():
    return SimpleNamespace(config = SimpleNamespace(_name_or_path = "unsloth/tiny", model_type = "llama"))


def _tfevents(h):
    return [repo_path for _, repo_path in h.uploads if "tfevents" in repo_path]


def test_tfevents_outside_export_dir_are_not_uploaded(hub, tmp_path, monkeypatch):
    export = tmp_path / "export"
    export.mkdir()
    weights = export / "model.gguf"
    weights.write_bytes(b"GGUF")
    (tmp_path / "events.out.tfevents.unrelated").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    save_mod.upload_to_huggingface(
        _model(), "owner/repo", "token", "finetuned", file_location = str(weights)
    )

    assert _tfevents(hub) == []
    assert (str(weights), "model.gguf") in hub.uploads


def test_tfevents_inside_export_dir_upload_with_posix_path(hub, tmp_path, monkeypatch):
    export = tmp_path / "export"
    nested = export / "runs"
    nested.mkdir(parents = True)
    weights = export / "model.gguf"
    weights.write_bytes(b"GGUF")
    (export / "events.out.tfevents.1.host").write_bytes(b"x")
    (nested / "events.out.tfevents.2.host").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    save_mod.upload_to_huggingface(
        _model(), "owner/repo", "token", "finetuned", file_location = str(weights)
    )

    tfevents = _tfevents(hub)
    assert sorted(tfevents) == ["events.out.tfevents.1.host", "runs/events.out.tfevents.2.host"]
    for repo_path in tfevents:
        assert not repo_path.startswith("/")
        assert "\\" not in repo_path


def test_tfevents_next_to_bare_filename_in_cwd(hub, tmp_path, monkeypatch):
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    (tmp_path / "events.out.tfevents.4.host").write_bytes(b"x")
    nested = tmp_path / "nested_run"
    nested.mkdir()
    (nested / "events.out.tfevents.5.host").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    save_mod.upload_to_huggingface(
        _model(), "owner/repo", "token", "finetuned", file_location = "model.gguf"
    )

    assert _tfevents(hub) == ["events.out.tfevents.4.host"]


def test_tfevents_found_when_file_location_is_a_directory(hub, tmp_path, monkeypatch):
    export = tmp_path / "export"
    export.mkdir()
    (export / "events.out.tfevents.3.host").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    save_mod.upload_to_huggingface(
        _model(), "owner/repo", "token", "finetuned", file_location = str(export)
    )

    assert _tfevents(hub) == ["events.out.tfevents.3.host"]


def test_temp_config_removed_when_upload_fails(tmp_path, monkeypatch):
    uploads = []

    class FailingApi:
        def __init__(self, *a, **kw):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, **kw):
            uploads.append((str(path_or_fileobj), path_in_repo))
            if path_in_repo == "config.json":
                raise OSError("hub unreachable")

    monkeypatch.setattr(save_mod, "HfApi", FailingApi)
    monkeypatch.setattr(huggingface_hub, "create_repo", lambda **kw: None)
    monkeypatch.setattr(huggingface_hub, "ModelCard", _Card)
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"GGUF")

    with pytest.raises(OSError, match = "hub unreachable"):
        save_mod.upload_to_huggingface(
            _model(), "owner/repo", "token", "finetuned", file_location = str(weights)
        )

    config_uploads = [path for path, repo_path in uploads if repo_path == "config.json"]
    assert config_uploads, "config.json upload was never attempted"
    for path in config_uploads:
        assert not os.path.exists(path)
    assert not (tmp_path / "_temporary_unsloth_config.json").exists()
    assert os.listdir(tmp_path) == ["model.gguf"]


def test_tfevents_in_hidden_dirs_are_not_uploaded(hub, tmp_path, monkeypatch):
    export = tmp_path / "export"
    hidden = export / ".hidden"
    hidden.mkdir(parents = True)
    weights = export / "model.gguf"
    weights.write_bytes(b"GGUF")
    (export / "events.out.tfevents.1.host").write_bytes(b"x")
    (hidden / "events.out.tfevents.2.host").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    save_mod.upload_to_huggingface(
        _model(), "owner/repo", "token", "finetuned", file_location = str(weights)
    )

    assert _tfevents(hub) == ["events.out.tfevents.1.host"]


def test_tfevents_found_in_dir_with_glob_metacharacters(hub, tmp_path, monkeypatch):
    export = tmp_path / "export[final]"
    export.mkdir()
    weights = export / "model.gguf"
    weights.write_bytes(b"GGUF")
    (export / "events.out.tfevents.meta").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    save_mod.upload_to_huggingface(
        _model(), "owner/repo", "token", "finetuned", file_location = str(weights)
    )

    assert _tfevents(hub) == ["events.out.tfevents.meta"]


def test_tfevents_directories_are_skipped(hub, tmp_path, monkeypatch):
    export = tmp_path / "export"
    (export / "fake.out.tfevents.dir").mkdir(parents = True)
    weights = export / "model.gguf"
    weights.write_bytes(b"GGUF")
    (export / "events.out.tfevents.real").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    save_mod.upload_to_huggingface(
        _model(), "owner/repo", "token", "finetuned", file_location = str(weights)
    )

    assert _tfevents(hub) == ["events.out.tfevents.real"]

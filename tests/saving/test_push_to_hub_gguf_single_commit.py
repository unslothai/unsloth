import os
from types import SimpleNamespace

import huggingface_hub
import pytest

import unsloth.save as save_mod


class _RecordingApi:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def record(*args, **kwargs):
            _RecordingApi.calls.append((name, kwargs))
            if name == "create_commit" and kwargs.get("create_pr"):
                return SimpleNamespace(pr_url = "https://huggingface.co/u/my-model/discussions/1")
            return SimpleNamespace(pr_url = None)

        return record


def _fake_convert(**kwargs):
    directory = kwargs["save_directory"]
    os.makedirs(directory, exist_ok = True)
    stem = os.path.basename(directory)
    files = []
    for suffix in ("Q8_0.gguf", "BF16-mmproj.gguf"):
        path = os.path.join(directory, f"{stem}.{suffix}")
        open(path, "wb").write(b"GGUF")
        files.append(path)
    open(os.path.join(directory, "config.json"), "w").write("{}")
    modelfile = os.path.join(directory, "Modelfile")
    open(modelfile, "w").write("FROM x")
    return {
        "gguf_files": files,
        "modelfile_location": modelfile,
        "want_full_precision": False,
        "is_vlm": True,
        "fix_bos_token": False,
        "save_directory": directory,
    }


@pytest.fixture
def push(monkeypatch):
    _RecordingApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingApi)
    monkeypatch.setattr(save_mod, "unsloth_save_pretrained_gguf", _fake_convert)

    def run(**kwargs):
        save_mod.unsloth_push_to_hub_gguf(object(), "u/my-model", tokenizer = object(), **kwargs)
        return _RecordingApi.calls

    return run


EXPECTED_PATHS = {
    "my-model.Q8_0.gguf",
    "my-model.BF16-mmproj.gguf",
    "config.json",
    "Modelfile",
    "README.md",
}


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"create_pr": True}, {"revision": "new-branch"}, {"revision": "refs/pr/3"}],
)
def test_all_files_go_in_one_commit(push, kwargs):
    calls = push(**kwargs)
    names = [name for name, _ in calls]
    assert "upload_file" not in names
    assert names.count("create_commit") == 1
    commit = next(kw for name, kw in calls if name == "create_commit")
    assert {op.path_in_repo for op in commit["operations"]} == EXPECTED_PATHS
    assert commit["create_pr"] == kwargs.get("create_pr", False)
    assert commit["revision"] == kwargs.get("revision")


def test_new_revision_is_created_before_the_commit(push):
    names = [name for name, _ in push(revision = "new-branch")]
    assert names.index("create_branch") < names.index("create_commit")
    branch = next(kw for name, kw in _RecordingApi.calls if name == "create_branch")
    assert branch["branch"] == "new-branch" and branch["exist_ok"] is True


@pytest.mark.parametrize("kwargs", [{}, {"create_pr": True}, {"revision": "refs/pr/3"}])
def test_no_branch_is_created_without_a_new_revision(push, kwargs):
    assert "create_branch" not in [name for name, _ in push(**kwargs)]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import ModelCard


class Model:
    config = SimpleNamespace(_name_or_path = "base/model", model_type = "llama")


SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth/save.py"

LIFTED = ("upload_to_huggingface", "create_huggingface_repo")


@pytest.fixture
def uploading(monkeypatch):
    """The card-writing functions plus `MODEL_CARD`, lifted out of save.py.

    save.py cannot be imported here: it pulls in the accelerator at module scope. The
    definitions under test read nothing else from the module, so run them on their own.
    """
    source = SAVE_PY
    tree = ast.parse(source.read_text(encoding = "utf-8"))

    pushed = {}

    class RecordingCard(ModelCard):
        def push_to_hub(
            self,
            repo_id,
            token = None,
            **kwargs,
        ):
            pushed["repo_id"] = repo_id
            pushed["content"] = self.content

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "ModelCard", RecordingCard)
    monkeypatch.setattr(huggingface_hub, "create_repo", lambda **kwargs: None)

    env = {
        "HfApi": lambda token = None: SimpleNamespace(),
        "get_token": lambda: "env-token",
        "_determine_username": lambda directory, old, token: (directory, directory.split("/")[0]),
        "logger": SimpleNamespace(warning_once = lambda *args: None),
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in LIFTED)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "MODEL_CARD" for t in node.targets)
        )
    ]
    assert len(nodes) == len(LIFTED) + 1, f"expected MODEL_CARD and {LIFTED} in save.py"
    module = ast.Module(body = nodes, type_ignores = [])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), env)
    return env, pushed


def test_upload_model_card_names_the_method(uploading):
    # The heading is "# Uploaded {method} model" and every caller in save.py passes "finetuned",
    # so a card that does not name it came out as "Uploaded  model" with the gap left in.
    env, pushed = uploading

    env["upload_to_huggingface"](Model(), "owner/model", "token", "finetuned", "trl")

    assert pushed["repo_id"] == "owner/model"
    assert "# Uploaded finetuned model" in pushed["content"]
    assert "# Uploaded  model" not in pushed["content"]


def test_upload_model_card_still_carries_username_extra_and_datasets(uploading):
    env, pushed = uploading

    env["upload_to_huggingface"](
        Model(), "owner/model", "token", "finetuned", "trl", datasets = ["owner/data"]
    )

    card = ModelCard(pushed["content"])
    assert "**Developed by:** owner" in pushed["content"]
    assert card.data.base_model == "base/model"
    assert "trl" in card.data.tags and "unsloth" in card.data.tags
    assert card.data.datasets == ["owner/data"]


def test_create_huggingface_repo_also_names_the_method(uploading):
    # Same template, same heading, so the repo-creation card had the same hole in it.
    env, pushed = uploading

    env["create_huggingface_repo"](Model(), "owner/model", "token")

    assert "# Uploaded finetuned model" in pushed["content"]
    assert "# Uploaded  model" not in pushed["content"]


def test_no_card_in_save_py_is_formatted_without_a_method():
    """The guard for the whole file rather than one call site.

    `MODEL_CARD`'s heading is "# Uploaded {method} model". Any site that formats it
    with an empty `method` ships a card reading "Uploaded  model", gap and all, which
    is what this test exists to stop coming back.
    """
    tree = ast.parse(SAVE_PY.read_text(encoding = "utf-8"))

    formats = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "MODEL_CARD"
    ]
    assert formats, "no MODEL_CARD.format call found in save.py"

    for call in formats:
        method = next((kw.value for kw in call.keywords if kw.arg == "method"), None)
        where = f"save.py:{call.lineno}"
        assert method is not None, f"{where}: MODEL_CARD.format without a method"
        if isinstance(method, ast.Constant):
            assert isinstance(method.value, str), f"{where}: method is not a string"
            assert method.value.strip(), f"{where}: method is empty, card reads 'Uploaded  model'"

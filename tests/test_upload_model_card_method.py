# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import ModelCard


class Model:
    config = SimpleNamespace(_name_or_path = "base/model", model_type = "llama")


@pytest.fixture
def uploading(monkeypatch):
    """`upload_to_huggingface` plus `MODEL_CARD`, lifted out of save.py.

    save.py cannot be imported here: it pulls in the accelerator at module scope. The two
    definitions under test read nothing else from the module, so run them on their own.
    """
    source = Path(__file__).resolve().parents[1] / "unsloth/save.py"
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
        "_determine_username": lambda directory, old, token: (directory, directory.split("/")[0]),
        "logger": SimpleNamespace(warning_once = lambda *args: None),
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "upload_to_huggingface")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "MODEL_CARD" for t in node.targets)
        )
    ]
    assert len(nodes) == 2, "expected MODEL_CARD and upload_to_huggingface in save.py"
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

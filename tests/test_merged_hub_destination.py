import ast
import gc
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from huggingface_hub import ModelCard


class PeftModel:
    config = SimpleNamespace(_name_or_path = "base/model", model_type = "llama")


class FullModel:
    config = PeftModel.config

    def state_dict(self):
        return {}

    def save_pretrained(self, directory, **kwargs):
        (Path(directory) / "config.json").write_text('{"model_type": "llama"}')
        (Path(directory) / "model.safetensors").write_bytes(b"model weights")

    def push_to_hub(self, **kwargs):
        raise AssertionError("Model files must join the single staged commit")


@pytest.fixture
def saving(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "unsloth/save.py"
    tree = ast.parse(source.read_text())
    records = {"merges": [], "uploads": [], "repos": [], "directories": []}
    artifacts = {
        "config.json": '{"model_type": "llama"}',
        "generation_config.json": "{}",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                }
            }
        ),
        "model-00001-of-00002.safetensors": "first weights",
        "model-00002-of-00002.safetensors": "second weights",
    }

    def merge(*args, **kwargs):
        records["merges"].append(kwargs)
        if kwargs["push_to_hub"]:
            return
        directory = Path(kwargs["save_directory"])
        records["directories"].append(directory)
        for filename, content in artifacts.items():
            (directory / filename).write_text(content)
        if records.get("fail_merge"):
            raise OSError("merge failed")
        if records.get("existing_card"):
            (directory / "README.md").write_text(
                "---\nlicense: mit\ntags:\n- existing\n---\nOriginal card"
            )

    zoo = ModuleType("unsloth_zoo.saving_utils")
    zoo.merge_and_overwrite_lora = merge
    monkeypatch.setitem(sys.modules, "unsloth_zoo.saving_utils", zoo)

    class Api:
        def __init__(self, token):
            self.token = token

        def create_repo(self, **kwargs):
            records["repos"].append(kwargs)

        def upload_folder(self, **kwargs):
            files = {p.name: p.read_text() for p in Path(kwargs["folder_path"]).iterdir()}
            records["uploads"].append({**kwargs, "files": files, "token": self.token})
            if records.get("fail_upload"):
                raise OSError("upload failed")
            return "commit-info"

    env = dict(
        os = os,
        gc = gc,
        Path = Path,
        HfApi = Api,
        PeftModel = PeftModel,
        PreTrainedTokenizerBase = type("Tokenizer", (), {}),
        ProcessorMixin = type("Processor", (), {}),
        torch = SimpleNamespace(
            save = lambda: None,
            float16 = "float16",
            bfloat16 = "bfloat16",
            cuda = SimpleNamespace(is_bf16_supported = lambda: False),
        ),
        get_token = lambda: "cached-fixture",
        _determine_username = lambda repo, old, token: (repo, repo.split("/")[0]),
        _prewarm_base_model_hub_cache = lambda *args, **kwargs: None,
        get_model_name = lambda name: name,
        _normalize_compressed_method = lambda method: None,
        _normalize_torchao_method = lambda method: None,
        _is_qwen3_5_vlm = lambda model: False,
        logger = SimpleNamespace(warning_once = lambda *args: None),
    )
    names = {
        "_push_merged_to_hub_revision",
        "unsloth_generic_save",
        "unsloth_generic_push_to_hub_merged",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODEL_CARD" for t in node.targets
        ):
            nodes.append(node)
    module = ast.Module(
        body = [
            ast.ImportFrom(module = "__future__", names = [ast.alias(name = "annotations")], level = 0),
            *nodes,
        ],
        type_ignores = [],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), env)
    return env, records, artifacts


@pytest.mark.parametrize(
    "revision, create_pr", [("release-candidate", False), (None, True), ("release-candidate", True)]
)
def test_public_merged_push_commits_all_artifacts_to_one_destination(saving, revision, create_pr):
    env, records, artifacts = saving
    env["unsloth_generic_push_to_hub_merged"](
        PeftModel(),
        "owner/model",
        token = "explicit-fixture",
        private = True,
        revision = revision,
        create_pr = create_pr,
        datasets = ["owner/data"],
        tags = ["fine-tuned"],
        commit_message = "Merge result",
        commit_description = "Review all artifacts",
    )
    assert len(records["uploads"]) == 1
    upload = records["uploads"][0]
    assert upload["revision"] == revision and upload["create_pr"] is create_pr
    assert upload["token"] == "explicit-fixture"
    assert upload["repo_id"] == "owner/model"
    assert upload["commit_message"] == "Merge result"
    assert upload["commit_description"] == "Review all artifacts"
    assert set(upload["files"]) == {*artifacts, "README.md"}
    assert all(upload["files"][name] == content for name, content in artifacts.items())
    card = ModelCard(upload["files"]["README.md"])
    assert card.data.datasets == ["owner/data"]
    assert "fine-tuned" in card.data.tags and "unsloth" in card.data.tags
    assert records["repos"] == [
        {"repo_id": "owner/model", "repo_type": "model", "private": True, "exist_ok": True}
    ]
    assert all(not call["push_to_hub"] for call in records["merges"])
    assert not any(directory.exists() for directory in records["directories"])


def test_existing_local_card_metadata_and_body_survive(saving):
    env, records, _ = saving
    records["existing_card"] = True
    env["unsloth_generic_push_to_hub_merged"](PeftModel(), "owner/model", revision = "candidate")
    upload = records["uploads"][0]
    card = ModelCard(upload["files"]["README.md"])
    assert card.data.license == "mit"
    assert card.data.tags == ["existing", "unsloth"]
    assert "Original card" in card.content
    assert upload["token"] == "cached-fixture"


@pytest.mark.parametrize("failure", ["fail_merge", "fail_upload"])
def test_staging_is_cleaned_on_failure(saving, failure):
    env, records, _ = saving
    records[failure] = True
    with pytest.raises(OSError, match = "failed"):
        env["unsloth_generic_push_to_hub_merged"](PeftModel(), "owner/model", create_pr = True)
    if failure == "fail_merge":
        assert records["uploads"] == []
    assert not any(directory.exists() for directory in records["directories"])


def test_default_push_keeps_streaming_merger(saving):
    env, records, _ = saving
    env["unsloth_generic_push_to_hub_merged"](PeftModel(), "owner/model", token = "fixture")
    assert records["merges"][0]["push_to_hub"] is True
    assert records["uploads"] == []
    assert records["repos"] == []


def test_non_main_rank_does_not_create_or_upload(saving):
    env, records, _ = saving
    env["unsloth_generic_save"](
        PeftModel(), None, "owner/model", push_to_hub = True, create_pr = True, is_main_process = False
    )
    assert records["repos"] == records["uploads"] == records["merges"] == []


def test_full_finetune_stages_model_and_metadata_together(saving):
    env, records, _ = saving
    env["unsloth_generic_push_to_hub_merged"](
        FullModel(), "owner/model", create_pr = True, datasets = ["owner/data"]
    )
    assert len(records["uploads"]) == 1
    upload = records["uploads"][0]
    assert upload["create_pr"] is True
    assert set(upload["files"]) == {"config.json", "model.safetensors", "README.md"}
    assert ModelCard(upload["files"]["README.md"]).data.datasets == ["owner/data"]
    assert records["merges"] == []

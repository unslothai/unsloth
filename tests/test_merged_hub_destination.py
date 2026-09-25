# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import gc
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from huggingface_hub import HfApi, ModelCard
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RevisionNotFoundError,
)


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
def saving(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "unsloth/save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    records = {
        "merges": [],
        "adapter_saves": [],
        "uploads": [],
        "repos": [],
        "directories": [],
        "branches": [],
        "downloads": [],
        "revisions": {"main"},
    }
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
        for filename, content in records.get("extra_files", {}).items():
            path = directory / filename
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_text(content)
        if records.get("fail_merge"):
            raise OSError("merge failed")
        if records.get("existing_card"):
            (directory / "README.md").write_text(
                "---\nlicense: mit\ntags:\n- existing\n---\nOriginal card"
            )

    zoo = ModuleType("unsloth_zoo.saving_utils")
    zoo.merge_and_overwrite_lora = merge
    zoo.get_original_model_id = lambda path: records.get("original_model_id")
    monkeypatch.setitem(sys.modules, "unsloth_zoo.saving_utils", zoo)

    def download(
        repo_id,
        filename,
        *,
        revision = None,
        token = None,
        **kwargs,
    ):
        records["downloads"].append((repo_id, filename, revision, token))
        if records.get("download_error"):
            raise records["download_error"]
        content = records.get("remote_cards", {}).get(revision or "main")
        if content is None:
            raise EntryNotFoundError("No README at destination")
        path = tmp_path / "remote-card.md"
        path.write_text(content)
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    class Api:
        def __init__(self, token):
            self.token = token

        def create_repo(self, **kwargs):
            records["repos"].append(kwargs)

        def create_branch(self, **kwargs):
            records["branches"].append(kwargs)
            records["revisions"].add(kwargs["branch"])

        def create_commit(self, **kwargs):
            if (
                records.get("enforce_revision")
                and (kwargs["revision"] or "main") not in records["revisions"]
            ):
                raise RevisionNotFoundError("Requested branch does not exist")
            if kwargs["commit_message"] is None:
                HfApi(token = False).create_commit(
                    repo_id = kwargs["repo_id"], operations = [], commit_message = None
                )
            files = {
                operation.path_in_repo: Path(operation.path_or_fileobj).read_text(encoding = "utf-8")
                for operation in kwargs["operations"]
            }
            records["uploads"].append({**kwargs, "files": files, "token": self.token})
            if records.get("fail_upload"):
                raise OSError("upload failed")
            # Overridable so a test can hand back a real-shaped CommitInfo, whose `pr_url` is
            # the only place the pull request's own address exists.
            return records.get("commit_info", "commit-info")

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
        # save_method="lora" leaves this module for the adapter save rather than the merge,
        # so record the handover instead of re-implementing it.
        unsloth_save_model = lambda *args, **kwargs: (
            records["adapter_saves"].append({"args": args, "kwargs": kwargs}),
            (kwargs.get("save_directory"), None),
        )[1],
    )
    names = {
        "_push_merged_to_hub_revision",
        "unsloth_generic_save",
        "unsloth_generic_push_to_hub_merged",
        # Real code, not stubs: these two decide what save_method="lora" and
        # safe_serialization=None mean, which is what several tests below assert on.
        "_normalize_safe_serialization",
        "_is_adapter_save_method",
        "_honours_safe_serialization",
        "_refuse_unsaveable_text_core",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            # _normalize_safe_serialization must be defined before the functions that call
            # it, which the source order already gives; only the decorators go.
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


@pytest.mark.parametrize("original_id", ["upstream/base-model", None])
def test_local_base_model_card_uses_hub_identifier(saving, tmp_path, original_id):
    env, records, _ = saving
    records["original_model_id"] = original_id
    model = PeftModel()
    model.config = SimpleNamespace(_name_or_path = str(tmp_path), model_type = "llama")
    env["unsloth_generic_push_to_hub_merged"](model, "owner/model", revision = "candidate")
    card = ModelCard(records["uploads"][0]["files"]["README.md"])
    assert card.data.base_model == (original_id or "owner/model")


def test_staged_cache_metadata_is_excluded_but_nested_artifacts_survive(saving):
    env, records, artifacts = saving
    records["extra_files"] = {
        ".cache/huggingface/download/model.metadata": "download metadata",
        ".git/config": "local git metadata",
        "nested/.cache/huggingface/download/model.lock": "lock",
        "nested/tokenizer.json": "nested tokenizer",
    }
    env["unsloth_generic_push_to_hub_merged"](PeftModel(), "owner/model", create_pr = True)
    assert set(records["uploads"][0]["files"]) == {*artifacts, "README.md", "nested/tokenizer.json"}


def test_none_commit_message_uses_default(saving):
    env, records, _ = saving
    env["unsloth_generic_push_to_hub_merged"](
        FullModel(), "owner/model", create_pr = True, commit_message = None
    )
    assert records["uploads"][0]["commit_message"] == "Trained with Unsloth"


@pytest.mark.parametrize("options", [{"revision": "candidate"}, {"create_pr": True}, {}])
def test_unforced_4bit_rejection_does_not_create_hub_resources(saving, options):
    env, records, _ = saving
    with pytest.raises(RuntimeError, match = "merged_4bit_forced"):
        env["unsloth_generic_push_to_hub_merged"](
            PeftModel(), "owner/new-model", save_method = "merged_4bit", **options
        )
    assert records["repos"] == records["branches"] == records["merges"] == records["uploads"] == []


def test_forced_4bit_mode_still_reaches_staged_save(saving):
    env, records, _ = saving
    env["unsloth_generic_push_to_hub_merged"](
        PeftModel(), "owner/model", save_method = "merged_4bit_forced", revision = "candidate"
    )
    assert records["merges"][0]["save_method"] == "merged_4bit"
    assert len(records["uploads"]) == 1
    assert records["uploads"][0]["revision"] == "candidate"


@pytest.mark.parametrize("create_pr", [False, True])
def test_missing_destination_branch_is_created(saving, create_pr):
    env, records, _ = saving
    records["enforce_revision"] = True
    env["unsloth_generic_push_to_hub_merged"](
        FullModel(), "owner/model", revision = "candidate", create_pr = create_pr
    )
    assert records["branches"] == [
        {"repo_id": "owner/model", "repo_type": "model", "branch": "candidate", "exist_ok": True}
    ]
    assert records["uploads"][0]["revision"] == "candidate"


def test_existing_pull_request_ref_is_not_created_as_a_branch(saving):
    env, records, _ = saving
    records["enforce_revision"] = True
    records["revisions"].add("refs/pr/3")
    env["unsloth_generic_push_to_hub_merged"](FullModel(), "owner/model", revision = "refs/pr/3")
    assert records["branches"] == []
    assert records["uploads"][0]["revision"] == "refs/pr/3"


@pytest.mark.parametrize("create_pr", [False, True])
@pytest.mark.parametrize("status", [403, 500])
def test_branch_errors_allow_only_forbidden_pr_contributions(
    saving, monkeypatch, create_pr, status
):
    import huggingface_hub.hf_api as hf_api

    env, records, _ = saving
    error = HfHubHTTPError(
        "branch request failed",
        response = SimpleNamespace(status_code = status, headers = {}, request = None),
    )
    monkeypatch.setattr(
        hf_api, "get_session", lambda: SimpleNamespace(post = lambda **kwargs: object())
    )

    def fail_request(response):
        raise error

    monkeypatch.setattr(hf_api, "hf_raise_for_status", fail_request)
    api = HfApi(token = False)
    monkeypatch.setattr(
        api,
        "list_repo_refs",
        lambda **kwargs: SimpleNamespace(branches = [SimpleNamespace(name = "release/candidate")]),
    )
    monkeypatch.setattr(
        env["HfApi"], "create_branch", lambda self, **kwargs: api.create_branch(**kwargs)
    )
    kwargs = dict(revision = "release/candidate", create_pr = create_pr)
    if status == 403 and create_pr:
        env["unsloth_generic_push_to_hub_merged"](FullModel(), "owner/model", **kwargs)
        assert records["uploads"][0]["create_pr"] is True
        assert records["uploads"][0]["revision"] == "release/candidate"
    else:
        with pytest.raises(HfHubHTTPError, match = "branch request failed"):
            env["unsloth_generic_push_to_hub_merged"](FullModel(), "owner/model", **kwargs)
        assert records["uploads"] == []


@pytest.mark.parametrize("revision", [None, "candidate", "refs/pr/3"])
@pytest.mark.parametrize("model_class", [FullModel, PeftModel])
def test_remote_destination_card_survives(saving, revision, model_class):
    env, records, _ = saving
    records["existing_card"] = True
    records["remote_cards"] = {
        revision
        or "main": "---\nlicense: apache-2.0\ntags:\n- custom\ndatasets:\n- owner/original\n---\nCustom destination description"
    }
    env["unsloth_generic_push_to_hub_merged"](
        model_class(),
        "owner/model",
        revision = revision,
        create_pr = revision is None,
        token = "explicit-fixture",
        tags = ["fine-tuned"],
    )
    card = ModelCard(records["uploads"][0]["files"]["README.md"])
    assert card.data.license == "apache-2.0"
    assert card.data.datasets == ["owner/original"]
    assert card.data.tags == ["custom", "fine-tuned", "unsloth"]
    assert "Custom destination description" in card.content
    assert records["downloads"] == [("owner/model", "README.md", revision, "explicit-fixture")]


@pytest.mark.parametrize("revision", [None, "candidate", "refs/pr/3"])
def test_reused_destination_refreshes_merged_adapter_provenance(saving, revision):
    env, records, _ = saving
    records["remote_cards"] = {
        revision
        or "main": "---\nbase_model: previous/base\nlicense: mit\ncustom_field: retained\n---\nUser description"
    }
    env["unsloth_generic_push_to_hub_merged"](
        PeftModel(), "owner/model", revision = revision, create_pr = revision is None
    )
    card = ModelCard(records["uploads"][0]["files"]["README.md"])
    assert card.data.base_model == "base/model"
    assert card.data.license == "mit"
    assert card.data.to_dict()["custom_field"] == "retained"
    assert "User description" in card.content


@pytest.mark.parametrize("error", [OSError, LocalEntryNotFoundError])
def test_card_download_failure_does_not_overwrite_remote_card(saving, error):
    env, records, _ = saving
    records["download_error"] = error("connection failed")
    with pytest.raises(error, match = "connection failed"):
        env["unsloth_generic_push_to_hub_merged"](FullModel(), "owner/model", create_pr = True)
    assert records["uploads"] == []
    assert not any(directory.exists() for directory in records["directories"])


class Transformers5Model(FullModel):
    def __init__(self):
        self.saved = []

    def save_pretrained(self, directory, **kwargs):
        self.saved.append(kwargs)
        super().save_pretrained(directory)

    def push_to_hub(
        self,
        repo_id,
        *,
        commit_message = None,
        commit_description = None,
        private = None,
        token = None,
        revision = None,
        create_pr = False,
        max_shard_size = "50GB",
        tags = None,
    ):
        raise AssertionError("Model files must join the single staged commit")


class Tokenizer:
    padding_side = "right"

    def save_pretrained(self, directory):
        assert self.padding_side == "left"
        (Path(directory) / "tokenizer.json").write_text("{}")

    def push_to_hub(self, *args, **kwargs):
        raise AssertionError("Tokenizer files must join the single staged commit")


def test_default_full_finetune_push_stages_one_commit(saving):
    env, records, _ = saving
    model, tokenizer = Transformers5Model(), Tokenizer()
    env["unsloth_generic_push_to_hub_merged"](
        model, "owner/model", tokenizer, token = "fixture", datasets = ["owner/data"]
    )
    assert len(records["uploads"]) == 1
    upload = records["uploads"][0]
    assert upload["repo_id"] == "owner/model"
    assert upload["revision"] is None and upload["create_pr"] is False
    assert set(upload["files"]) == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "README.md",
    }
    assert ModelCard(upload["files"]["README.md"]).data.datasets == ["owner/data"]
    assert "state_dict" in model.saved[0] and model.saved[0]["safe_serialization"] is True
    assert tokenizer.padding_side == "right"
    assert records["merges"] == []


def test_default_full_finetune_push_uploads_16bit_safetensors(monkeypatch):
    pytest.importorskip("unsloth", reason = "unsloth is not importable on this runner")
    try:
        import unsloth.save as save
    except ImportError as error:
        pytest.skip(f"unsloth.save is not importable on this runner: {error}")
    import torch
    import transformers
    from safetensors.torch import load_file

    commits = []

    class Api:
        def __init__(self, token):
            self.token = token

        def create_repo(self, **kwargs):
            pass

        def create_commit(self, **kwargs):
            commits.append(
                {
                    operation.path_in_repo: (
                        load_file(operation.path_or_fileobj)
                        if operation.path_in_repo.endswith(".safetensors")
                        else None
                    )
                    for operation in kwargs["operations"]
                }
            )

    def download(*args, **kwargs):
        raise EntryNotFoundError("No README at destination")

    monkeypatch.setattr(save, "HfApi", Api)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    config = transformers.LlamaConfig(
        vocab_size = 32,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    model = transformers.LlamaForCausalLM(config).float()
    save.unsloth_generic_push_to_hub_merged(
        model, "owner/model", token = "fixture", tags = ["fine-tuned"]
    )
    assert len(commits) == 1
    assert {"config.json", "model.safetensors", "README.md"} <= set(commits[0])
    target = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    assert {tensor.dtype for tensor in commits[0]["model.safetensors"].values()} == {target}
    assert next(model.parameters()).dtype == torch.float32


def test_the_push_names_the_destination_and_not_the_staging_folder(saving, capsys):
    """The staged save reports the temp folder it wrote, which is deleted moments later.

    A caller asked for a repository, so the repository is what the output has to name; otherwise
    the last thing printed is a success line pointing at a path that no longer exists.
    """
    env, records, _ = saving
    env["unsloth_generic_push_to_hub_merged"](FullModel(), "owner/model", token = "fixture")
    printed = capsys.readouterr().out
    assert "owner/model" in printed
    assert "https://huggingface.co/owner/model" in printed


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "https://huggingface.co/owner/model"),
        # A branch upload is not on the repository's default branch.
        ({"revision": "my-branch"}, "https://huggingface.co/owner/model/tree/my-branch"),
        # An existing pull request, addressed the way the Hub spells it.
        ({"revision": "refs/pr/3"}, "https://huggingface.co/owner/model/discussions/3"),
        # A fresh pull request: the repository page can hold no model files at all.
        ({"create_pr": True}, "https://huggingface.co/owner/model/discussions"),
        # `create_pr` wins over the branch it was opened against.
        (
            {"create_pr": True, "revision": "my-branch"},
            "https://huggingface.co/owner/model/discussions",
        ),
    ],
)
def test_the_printed_destination_is_where_the_files_landed(saving, capsys, kwargs, expected):
    """A branch or pull-request upload does not appear on the repository page."""
    env, records, _ = saving
    env["unsloth_generic_push_to_hub_merged"](FullModel(), "owner/model", token = "fixture", **kwargs)
    printed = capsys.readouterr().out
    assert f"Saved model to {expected}\n" in printed, printed


def test_the_pull_requests_own_url_is_preferred_when_the_hub_returns_one(saving, capsys):
    """`CommitInfo.pr_url` names the exact pull request; nothing local can reconstruct it."""
    env, records, _ = saving
    records["commit_info"] = SimpleNamespace(
        pr_url = "https://huggingface.co/owner/model/discussions/7"
    )
    env["unsloth_generic_push_to_hub_merged"](
        FullModel(), "owner/model", token = "fixture", create_pr = True
    )
    printed = capsys.readouterr().out
    assert "Saved model to https://huggingface.co/owner/model/discussions/7\n" in printed, printed


def test_a_pickle_request_this_transformers_cannot_honour_is_reported(saving):
    """transformers 5 removed `safe_serialization`, so `False` silently yields safetensors."""
    env, records, _ = saving
    said = []
    env["logger"] = SimpleNamespace(warning_once = lambda message, *a, **kw: said.append(message))

    class NoSafeSerialization(FullModel):
        """transformers 5's shape: named parameters, but nothing that honours the request."""

        def save_pretrained(
            self,
            directory,
            max_shard_size = "50GB",
            variant = None,
            **kwargs,
        ):
            super().save_pretrained(directory)

    class HonoursIt(FullModel):
        def save_pretrained(
            self,
            directory,
            safe_serialization = True,
            **kwargs,
        ):
            super().save_pretrained(directory)

    class Patched(FullModel):
        """`patch_saving_functions` wraps the real method in a passthrough that tells us nothing,
        and keeps the original, which is what has to be probed."""

        def save_pretrained(self, *args, **kwargs):
            FullModel.save_pretrained(self, args[0])

        original_model_save_pretrained = HonoursIt.save_pretrained

    honours = env["_honours_safe_serialization"]
    assert honours(NoSafeSerialization().save_pretrained) is False
    assert honours(HonoursIt().save_pretrained) is True
    # Cannot tell, so it must not warn: a wrong warning is worse than none.
    assert honours(Patched().save_pretrained) is True

    def push(model, **kwargs):
        said.clear()
        env["unsloth_generic_push_to_hub_merged"](model, "owner/model", token = "fixture", **kwargs)
        return [message for message in said if "not a pickle" in message]

    assert push(NoSafeSerialization(), safe_serialization = False)
    # A transformers that still takes it, a patched model whose original does, and the default
    # `True`, are all silent.
    assert push(HonoursIt(), safe_serialization = False) == []
    assert push(Patched(), safe_serialization = False) == []
    assert push(NoSafeSerialization()) == []

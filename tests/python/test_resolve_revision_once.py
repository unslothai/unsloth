# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""One revision resolve per load: transformers >= 5.18 sends a repo_info request from every from_pretrained."""

import ast
from pathlib import Path

import pytest

import unsloth  # noqa: F401
from unsloth.models import loader_utils

REPO = Path(__file__).parents[2]
SHA = "0123456789abcdef0123456789abcdef01234567"

hub = pytest.importorskip("huggingface_hub")
ResolvedRevision = getattr(hub, "ResolvedRevision", None)
needs_resolved = pytest.mark.skipif(
    ResolvedRevision is None or loader_utils._hub_resolve_revision is None,
    reason = "transformers < 5.18 / huggingface_hub without ResolvedRevision",
)


def _in_load(fn):
    return loader_utils._load_probe_memo_scope(fn)()


@pytest.fixture
def resolver(monkeypatch):
    calls = []

    def _fake(
        name,
        revision = None,
        *,
        token = None,
        local_files_only = False,
        cache_dir = None,
    ):
        calls.append((name, revision, token, local_files_only, cache_dir))
        if name == "org/missing":
            return revision
        return ResolvedRevision(resolved = SHA, initial = revision, repo_id = name)

    monkeypatch.setattr(loader_utils, "_hub_resolve_revision", _fake)
    return calls


@needs_resolved
def test_one_resolve_per_repo_and_ref(resolver):
    def load():
        first = loader_utils._resolved_revision("org/model", None, None)
        again = [loader_utils._resolved_revision("org/model", None, None) for _ in range(4)]
        assert all(r is first for r in again)
        assert isinstance(first, ResolvedRevision) and first.resolved == SHA
        # Each distinct input of the transformers call resolves on its own.
        loader_utils._resolved_revision("org/model", "dev", None)
        loader_utils._resolved_revision("org/model", None, "hf_x")
        loader_utils._resolved_revision("org/model", None, None, True)
        loader_utils._resolved_revision("org/model", None, None, False, "/tmp/other_cache")
        loader_utils._resolved_revision("org/other", None, None)

    _in_load(load)
    assert len(resolver) == 6
    assert resolver[0] == ("org/model", None, None, False, None)


@needs_resolved
def test_next_load_resolves_again(resolver):
    _in_load(lambda: loader_utils._resolved_revision("org/model", None, None))
    _in_load(lambda: loader_utils._resolved_revision("org/model", None, None))
    assert len(resolver) == 2


@needs_resolved
def test_offline_env_is_part_of_the_key(resolver, monkeypatch):
    def load():
        loader_utils._resolved_revision("org/model", None, None)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        loader_utils._resolved_revision("org/model", None, None)

    _in_load(load)
    assert len(resolver) == 2


@needs_resolved
def test_outside_a_load_local_paths_and_resolved_input_pass_through(resolver, tmp_path):
    assert loader_utils._resolved_revision("org/model", "main", None) == "main"
    assert type(loader_utils._resolved_revision("org/model", None, None)) is type(None)

    pinned = ResolvedRevision(resolved = SHA, initial = "v1", repo_id = "org/model")

    def load():
        assert loader_utils._resolved_revision(str(tmp_path), "v1", None) == "v1"
        assert loader_utils._resolved_revision(str(tmp_path), None, None) is None
        assert loader_utils._resolved_revision("org/model", pinned, None) is pinned
        assert loader_utils._resolved_revision(None, None, None) is None

    _in_load(load)
    assert resolver == []


@needs_resolved
def test_failed_resolve_keeps_plain_revision_and_is_not_retried(resolver, monkeypatch):
    def load():
        assert loader_utils._resolved_revision("org/missing", "v1", None) == "v1"
        assert type(loader_utils._resolved_revision("org/missing", "v1", None)) is str

    _in_load(load)
    assert len(resolver) == 1

    def boom(*args, **kwargs):
        raise RuntimeError("hub down")

    monkeypatch.setattr(loader_utils, "_hub_resolve_revision", boom)
    assert _in_load(lambda: loader_utils._resolved_revision("org/model", None, None)) is None


def test_transformers_without_resolve_revision_is_unchanged(monkeypatch):
    monkeypatch.setattr(loader_utils, "_hub_resolve_revision", None)
    seen = []

    def config_loader(name, **kwargs):
        seen.append(kwargs)
        return {"name": name}

    config_loader.__module__ = "transformers.models.auto.configuration_auto"

    def load():
        assert loader_utils._resolved_revision("org/model", None, None) is None
        assert loader_utils._resolved_revision("org/model", "v1", None) == "v1"
        kwargs = {"revision": "v1", "token": None}
        assert loader_utils._kwargs_with_resolved_revision("org/model", kwargs) is kwargs
        loader_utils._memo_config_probe(config_loader, "org/model", revision = "v1")

    _in_load(load)
    assert seen == [{"revision": "v1"}]
    assert type(seen[0]["revision"]) is str


@needs_resolved
def test_config_probe_pins_transformers_loaders_only(resolver):
    seen = {"transformers": [], "peft": []}

    def config_loader(name, **kwargs):
        seen["transformers"].append(kwargs["revision"])
        return {"name": name}

    def peft_loader(name, **kwargs):
        seen["peft"].append(kwargs["revision"])
        return {"name": name}

    config_loader.__module__ = "transformers.models.auto.configuration_auto"
    peft_loader.__module__ = "peft.config"

    def load():
        for _ in range(2):  # the second probe is a memo hit
            loader_utils._memo_config_probe(config_loader, "org/model", token = None, revision = None)
        loader_utils._memo_config_probe(peft_loader, "org/model", token = None, revision = None)
        loader_utils._memo_config_probe(
            config_loader, "org/model", token = None, revision = None, trust_remote_code = True
        )

    _in_load(load)
    assert len(seen["transformers"]) == 2
    assert all(isinstance(r, ResolvedRevision) for r in seen["transformers"])
    assert seen["peft"] == [None] and type(seen["peft"][0]) is type(None)
    assert len(resolver) == 1


@needs_resolved
def test_config_probe_pins_a_wrapped_transformers_classmethod(resolver):
    """import_fixes replaces AutoConfig.from_pretrained with wrappers defined in unsloth.import_fixes."""
    seen = []

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            seen.append(kwargs["revision"])
            return {"name": name}

    FakeAutoConfig.__module__ = "transformers.models.auto.configuration_auto"
    assert FakeAutoConfig.from_pretrained.__module__ == __name__

    def load():
        loader_utils._memo_config_probe(
            FakeAutoConfig.from_pretrained, "org/model", token = None, revision = None
        )
        assert loader_utils._resolved_revision("org/model", None, None) is seen[0]

    _in_load(load)
    assert isinstance(seen[0], ResolvedRevision)
    assert len(resolver) == 1


@needs_resolved
def test_real_resolve_revision_signature(tmp_path, monkeypatch):
    """The real transformers helper with the cache-only path Unsloth passes through (no network)."""
    from huggingface_hub.file_download import repo_folder_name

    refs = tmp_path / repo_folder_name(repo_id = "org/model", repo_type = "model") / "refs"
    refs.mkdir(parents = True)
    (refs / "main").write_text(SHA)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    resolved = _in_load(
        lambda: loader_utils._resolved_revision("org/model", None, None, True, str(tmp_path))
    )
    assert isinstance(resolved, ResolvedRevision) and resolved.resolved == SHA
    assert resolved == "main" and resolved.initial is None


def _calls_named(tree, name):
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == name
    ]


def _parents(tree):
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


@pytest.mark.parametrize(
    "relpath",
    ["unsloth/models/llama.py", "unsloth/models/vision.py", "unsloth/tokenizer_utils.py"],
)
def test_resolved_revision_only_reaches_transformers_loads(relpath):
    """A ResolvedRevision equals "main", not None: it must not reach Unsloth's own revision checks,
    warnings or _mark_loaded_revision. Its only uses are `revision =` of a from_pretrained call."""
    tree = ast.parse((REPO / relpath).read_text(encoding = "utf-8"))
    parent = _parents(tree)
    holders = set()
    uses = 0
    for call in _calls_named(tree, "_resolved_revision") + _calls_named(
        tree, "_kwargs_with_resolved_revision"
    ):
        uses += 1
        up = parent[call]
        if isinstance(up, ast.Assign):
            assert len(up.targets) == 1 and isinstance(up.targets[0], ast.Name), ast.unparse(up)
            holders.add(up.targets[0].id)
            continue
        assert isinstance(up, ast.keyword), ast.unparse(up)
        assert up.arg in ("revision", None), ast.unparse(up)
        assert ast.unparse(parent[up].func).endswith("from_pretrained"), ast.unparse(parent[up])
    assert uses, f"{relpath}: no resolved revision use left"
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in holders and isinstance(node.ctx, ast.Load):
            kw = parent[node]
            assert isinstance(kw, ast.keyword) and kw.arg == "revision", ast.unparse(kw)
            assert ast.unparse(parent[kw].func).split(".")[-1] in (
                "from_pretrained",
                "_from_pretrained_left_padded",
            ), ast.unparse(parent[kw])

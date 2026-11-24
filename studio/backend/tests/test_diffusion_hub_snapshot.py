# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A returning load reads an unchanged, already validated Hub snapshot from the cache with one Hub call (CPU, fakes).

The record (repo, commit, components, library versions, loader code, the snapshot's file list) is written only after a
revalidating load whose commit the Hub still confirms; any change, a missing file, offline or a failed cache-only read
revalidates per file as before."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_hub_snapshot as hs

REPO = "org/model"
C1, C2 = "a" * 40, "b" * 40
COMP = {"loader": "flux3", "text_encoder": False}


class _Api:
    def __init__(self, sha):
        self.sha = sha
        self.calls = 0

    def model_info(self, repo_id, **kw):
        self.calls += 1
        if isinstance(self.sha, BaseException):
            raise self.sha
        return types.SimpleNamespace(sha = self.sha, siblings = [])


def _snapshot(
    cache,
    commit,
    files = ("model_index.json", "text_encoder/tokenizer.json", "vae/config.json"),
):
    root = cache / "models--org--model"
    (root / "refs").mkdir(parents = True, exist_ok = True)
    (root / "refs" / "main").write_text(commit)
    snap = root / "snapshots" / commit
    for f in files:
        (snap / f).parent.mkdir(parents = True, exist_ok = True)
        (snap / f).write_text("{}")
    return snap


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv(hs.SNAPSHOT_ENV, raising = False)
    store = tmp_path / "studio_cache"
    import utils.paths.storage_roots as roots

    monkeypatch.setattr(roots, "cache_root", lambda: store)
    from core.inference import hub_reachability as reach

    monkeypatch.setattr(reach, "offline_mode", lambda: False)
    reach._UNREACHABLE.clear()
    cache = tmp_path / "hub"
    _snapshot(cache, C1)
    return types.SimpleNamespace(cache = str(cache), cache_path = cache, store = store)


def test_no_record_no_shortcut_then_a_validated_load_enables_it(env):
    api = _Api(C1)
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = api) is None
    assert api.calls == 0  # nothing recorded: no Hub call spent on it
    assert hs.note_validated(REPO, COMP, cache_dir = env.cache, api = api) is True
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = api) == C1
    assert api.calls == 2


def test_a_moved_ref_offline_or_unreachable_revalidates(env, monkeypatch):
    hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(C1))
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C2)) is None
    assert (
        hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(ConnectionError("down")))
        is None
    )
    from core.inference import hub_reachability as reach

    monkeypatch.setattr(reach, "offline_mode", lambda: True)
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None


def test_an_offline_fallback_load_records_nothing(env):
    assert (
        hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(ConnectionError("down")))
        is False
    )
    assert hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(C2)) is False
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None


def test_other_components_versions_or_code_do_not_match(env, monkeypatch):
    hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(C1))
    assert (
        hs.current_snapshot(REPO, dict(COMP, text_encoder = True), cache_dir = env.cache, api = _Api(C1))
        is None
    )
    monkeypatch.setattr(hs, "_versions", lambda: {"transformers": "99.0"})
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None
    monkeypatch.undo()
    monkeypatch.setattr(hs, "_code_digest", lambda: "other")
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None


def test_a_file_added_or_gone_since_the_record_revalidates(env):
    hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(C1))
    snap = env.cache_path / "models--org--model" / "snapshots" / C1
    (snap / "vae" / "config.json").unlink()
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None
    (snap / "vae" / "config.json").write_text("{}")
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) == C1
    noexist = env.cache_path / "models--org--model" / ".no_exist" / C1 / "text_encoder"
    noexist.mkdir(parents = True)
    (noexist / "processor_config.json").write_text("")
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None


def test_a_link_whose_blob_is_gone_revalidates(env):
    import os

    snap = env.cache_path / "models--org--model" / "snapshots" / C1
    blob = env.cache_path / "models--org--model" / "blobs" / "e1"
    blob.parent.mkdir(parents = True)
    blob.write_text("{% chat %}")
    try:
        os.symlink(blob, snap / "text_encoder" / "chat_template.jinja")
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks on this host")
    hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(C1))
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) == C1
    blob.unlink()  # the snapshot keeps a dangling link: not cached, so no cache-only read
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None


def test_local_dirs_and_the_switch(env, monkeypatch, tmp_path):
    hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(C1))
    assert hs.current_snapshot(str(tmp_path), COMP, cache_dir = env.cache, api = _Api(C1)) is None
    monkeypatch.setenv(hs.SNAPSHOT_ENV, "0")
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None
    assert hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is False


def test_cache_first_falls_back_and_reports():
    seen, dropped = [], []

    def load(local):
        seen.append(local)
        if local:
            raise OSError("not in the cache")
        return "online"

    assert (
        hs.cache_first(load, cached = True, name = "x", on_fallback = lambda: dropped.append(1))
        == "online"
    )
    assert seen == [True, False] and dropped == [1]
    seen.clear()
    assert hs.cache_first(load, cached = False, name = "x") == "online" and seen == [False]
    # other errors are not a cache miss
    with pytest.raises(RuntimeError):
        hs.cache_first(
            lambda local: (_ for _ in ()).throw(RuntimeError("boom")), cached = True, name = "x"
        )


def test_forget_drops_the_record(env):
    hs.note_validated(REPO, COMP, cache_dir = env.cache, api = _Api(C1))
    hs.forget(REPO, COMP, cache_dir = env.cache)
    assert hs.current_snapshot(REPO, COMP, cache_dir = env.cache, api = _Api(C1)) is None


def _loader_env(
    monkeypatch,
    cached,
    fail = (),
):
    """load_flux3_pipeline with every component loader stubbed: records each read's local_files_only (and, in
    ``reads.revisions``, the revision it pinned). The loaders are stand-in classes, so a diffusers without the FLUX.3
    classes (any release before them still meets the declared floor) runs these too."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    from core.inference import diffusion_flux3 as f3

    class _Reads(list):
        revisions: dict

    reads = _Reads()
    reads.revisions = {}

    def stub(name, value):
        def from_pretrained(*a, **kw):
            reads.append((name, kw.get("local_files_only")))
            reads.revisions.setdefault(name, []).append(kw.get("revision"))
            if kw.get("local_files_only") and name in fail:
                raise OSError(f"{name} not cached")
            return value

        return types.SimpleNamespace(from_pretrained = from_pretrained)

    def read_index(
        repo,
        token,
        local,
        revision = None,
    ):
        reads.append(("index", local))
        reads.revisions.setdefault("index", []).append(revision)
        return {}

    module = torch.nn.Linear(2, 2)
    monkeypatch.setattr(f3, "flux3_available", lambda: True)
    monkeypatch.setattr(f3, "_live_cache_dir", lambda: "/nonexistent")
    monkeypatch.setattr(f3, "_read_index", read_index)
    for owner, attr, name, value in (
        (transformers, "AutoProcessor", "processor", object()),
        (diffusers, "Flux3Transformer2DModel", "transformer", module),
        (diffusers, "AutoencoderKLFlux2", "vae", object()),
    ):
        cls = getattr(owner, attr, None)
        if cls is None:
            monkeypatch.setattr(owner, attr, stub(name, value), raising = False)
        else:
            monkeypatch.setattr(cls, "from_pretrained", stub(name, value).from_pretrained)
    monkeypatch.setattr(f3, "transformer_load_storage", lambda *a, **k: None)
    monkeypatch.setattr(f3, "_pin_compute_dtype", lambda *a: None)
    monkeypatch.setattr(f3, "flux3_blocks_class", lambda index, tr: "blocks")
    monkeypatch.setattr(f3, "flux3_family_blocks_mismatch", lambda *a: None)
    monkeypatch.setattr(f3.cond_blocks, "prepare", lambda tr: None)
    monkeypatch.setattr(
        f3,
        "flux3_blocks",
        lambda b: types.SimpleNamespace(
            init_pipeline = lambda: types.SimpleNamespace(scheduler = None)
        ),
    )
    monkeypatch.setattr(f3, "flux3_pipeline_class", lambda: (lambda **kw: kw))
    noted, forgot = [], []
    monkeypatch.setattr(hs, "current_snapshot", lambda *a, **k: C1 if cached else None)
    monkeypatch.setattr(hs, "note_validated", lambda *a, **k: noted.append(a[1]) or True)
    monkeypatch.setattr(hs, "forget", lambda *a, **k: forgot.append(a[1]))
    return f3, reads, noted, forgot, torch


def test_a_validated_snapshot_reads_every_component_from_the_cache(monkeypatch):
    f3, reads, noted, forgot, torch = _loader_env(monkeypatch, cached = True)
    f3.load_flux3_pipeline(REPO, torch.bfloat16, text_encoder = object())
    assert reads == [("index", True), ("processor", True), ("transformer", True), ("vae", True)]
    assert noted == [] and forgot == []


def test_without_a_record_the_load_revalidates_and_records(monkeypatch):
    f3, reads, noted, forgot, torch = _loader_env(monkeypatch, cached = False)
    f3.load_flux3_pipeline(REPO, torch.bfloat16, text_encoder = object())
    assert reads == [("index", False), ("processor", False), ("transformer", False), ("vae", False)]
    assert len(noted) == 1 and noted[0]["text_encoder"] is False and noted[0]["transformer"] is True


def test_a_component_missing_from_the_cache_is_read_online_and_the_record_dropped(monkeypatch):
    f3, reads, noted, forgot, torch = _loader_env(monkeypatch, cached = True, fail = ("processor",))
    f3.load_flux3_pipeline(REPO, torch.bfloat16, text_encoder = object())
    assert ("processor", True) in reads and ("processor", False) in reads
    assert reads[-1] == ("vae", True) and len(forgot) == 1 and noted == []


def test_a_cache_only_load_never_asks_the_hub(monkeypatch):
    f3, reads, noted, forgot, torch = _loader_env(monkeypatch, cached = True)
    asked = []
    monkeypatch.setattr(hs, "current_snapshot", lambda *a, **k: asked.append(1) or C1)
    f3.load_flux3_pipeline(REPO, torch.bfloat16, text_encoder = object(), local_files_only = True)
    assert asked == [] and noted == [] and all(local for _n, local in reads)


def test_cache_only_reads_pin_the_validated_commit(monkeypatch):
    """A concurrent download can move the shared cache's refs/main between the check and a component read: every
    cache-only read pins the commit the check validated, so none can open another snapshot's files."""
    f3, reads, noted, forgot, torch = _loader_env(monkeypatch, cached = True)
    f3.load_flux3_pipeline(REPO, torch.bfloat16, text_encoder = object())
    assert reads.revisions == {"index": [C1], "processor": [C1], "transformer": [C1], "vae": [C1]}


def test_a_hub_repair_pins_the_validated_commit_and_a_revalidating_load_stays_unpinned(monkeypatch):
    """A component missing from the validated snapshot is fetched from the Hub at that same commit: an unpinned
    repair after main moved would pair the new commit's processor with the old commit's transformer and VAE."""
    f3, reads, noted, forgot, torch = _loader_env(monkeypatch, cached = True, fail = ("processor",))
    f3.load_flux3_pipeline(REPO, torch.bfloat16, text_encoder = object())
    assert reads.revisions == {
        "index": [C1],
        "processor": [C1, C1],
        "transformer": [C1],
        "vae": [C1],
    }
    assert ("processor", False) in reads and len(forgot) == 1
    f3, reads, noted, forgot, torch = _loader_env(monkeypatch, cached = False)
    f3.load_flux3_pipeline(REPO, torch.bfloat16, text_encoder = object())
    assert set(sum(reads.revisions.values(), [])) == {None}


def test_a_pinned_cache_read_opens_the_validated_snapshot_after_refs_main_moves(tmp_path):
    """The real huggingface_hub cache reader: refs/main moved to B after A was validated. Unpinned reads B; pinned A."""
    from huggingface_hub import hf_hub_download

    for commit in (C1, C2):
        _snapshot(tmp_path, commit, files = ("model_index.json",))
        path = tmp_path / "models--org--model" / "snapshots" / commit / "model_index.json"
        path.write_text(commit)
    (tmp_path / "models--org--model" / "refs" / "main").write_text(C2)
    read = lambda **kw: open(  # noqa: E731
        hf_hub_download(REPO, "model_index.json", cache_dir = tmp_path, local_files_only = True, **kw)
    ).read()
    assert read() == C2
    assert read(revision = C1) == C1

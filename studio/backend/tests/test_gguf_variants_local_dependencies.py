# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A downloaded GGUF must stay a one-click On Device row when the Hub cannot be read.

The picker collapses a repo holding one complete quant only when the variants answer says its
dependencies are resolved. Only a live Hub listing used to say so, so with the Hub offline,
unreachable or refusing the saved token, every downloaded sole-quant repo fell back to the
expander. A Hub-less answer now proves it from disk: the quant's shards, its snapshot's own
partial state, the download's recorded plan and any companion a Hub answer named earlier.
With the Hub up a local-first answer still never claims it, since a companion the current
revision added is only visible in the listing (#10243).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hub.services.models import gguf_variants
from hub.utils import download_manifest, inventory_scan
from utils import hf_cache_settings

REPO = "org/Sole-GGUF"
REV = "0123456789abcdef0123456789abcdef01234567"
QUANT = "Q4_K_M"
MAIN = f"Sole-{QUANT}.gguf"


def _snapshot():
    hub_cache = hf_cache_settings.get_hf_cache_paths().hub_cache
    repo = hub_cache / "models--org--Sole-GGUF"
    snapshot = repo / "snapshots" / REV
    snapshot.mkdir(parents = True, exist_ok = True)
    (repo / "refs").mkdir(exist_ok = True)
    (repo / "refs" / "main").write_text(REV)
    return repo, snapshot


def _write(path, size: int = 256):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))
    return path


def _record_plan(repo, files):
    assert download_manifest.write_manifest(
        "model",
        REPO,
        QUANT,
        [download_manifest.ExpectedFile(name, size) for name, size in files],
        hub_cache = repo.parent,
    )


@pytest.fixture()
def hub_calls(monkeypatch, tmp_path):
    # A cache of this test's own: the suite-wide empty cache is shared across tests.
    hub_cache = str(tmp_path / "hub")
    monkeypatch.setitem(hf_cache_settings._EXPLICIT_CACHE_ENV, "HF_HUB_CACHE", hub_cache)
    monkeypatch.setenv("HF_HUB_CACHE", hub_cache)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", hub_cache)
    calls = []

    def refused(*args, **kwargs):
        calls.append(args)
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(gguf_variants, "list_gguf_variants", refused)
    monkeypatch.setattr(gguf_variants, "_gguf_all_variant_requirements", refused)
    monkeypatch.setattr("huggingface_hub.model_info", refused)
    monkeypatch.setattr("huggingface_hub.list_repo_files", refused)
    inventory_scan.invalidate_hf_cache_scans()
    return calls


def _answer(**kwargs):
    inventory_scan.invalidate_hf_cache_scans()
    return asyncio.run(gguf_variants.get_gguf_variants_response(REPO, **kwargs))


# What the picker sends when it believes the Hub is offline, and a Hub that fails a request.
HUBLESS = [
    pytest.param({"prefer_local_cache": True, "offline": True}, id = "offline"),
    pytest.param({}, id = "hub_failed"),
]


@pytest.mark.parametrize("request_kwargs", HUBLESS)
def test_a_complete_download_is_resolved_without_the_hub(hub_calls, request_kwargs):
    repo, snapshot = _snapshot()
    _write(snapshot / MAIN)
    _record_plan(repo, [(MAIN, 256)])

    response = _answer(**request_kwargs)

    assert [v.quant for v in response.variants] == [QUANT]
    assert response.variants[0].downloaded is True
    assert response.dependencies_resolved is True
    if request_kwargs.get("offline"):
        assert hub_calls == [], "an offline answer must not reach the Hub"


@pytest.mark.parametrize("request_kwargs", HUBLESS)
def test_a_download_from_before_manifests_is_judged_on_its_shards(hub_calls, request_kwargs):
    _repo, snapshot = _snapshot()
    _write(snapshot / MAIN)

    assert _answer(**request_kwargs).dependencies_resolved is True


def test_a_local_first_answer_with_the_hub_up_still_defers_to_the_listing(hub_calls):
    """#10243: a companion the current revision added only shows in the Hub listing."""
    repo, snapshot = _snapshot()
    _write(snapshot / MAIN)
    _record_plan(repo, [(MAIN, 256)])

    response = _answer(prefer_local_cache = True)

    assert response.variants[0].downloaded is True
    assert response.dependencies_resolved is False


@pytest.mark.parametrize("request_kwargs", HUBLESS)
@pytest.mark.parametrize(
    "companion",
    ["mmproj-F16.gguf", f"mtp-Sole-{QUANT}.gguf"],
    ids = ["mmproj", "mtp_drafter"],
)
def test_a_planned_companion_that_is_missing_leaves_it_unresolved(
    hub_calls, request_kwargs, companion
):
    repo, snapshot = _snapshot()
    _write(snapshot / MAIN)
    _record_plan(repo, [(MAIN, 256), (companion, 128)])

    assert _answer(**request_kwargs).dependencies_resolved is False


@pytest.mark.parametrize("request_kwargs", HUBLESS)
def test_a_truncated_planned_file_leaves_it_unresolved(hub_calls, request_kwargs):
    repo, snapshot = _snapshot()
    _write(snapshot / MAIN, 200)
    _record_plan(repo, [(MAIN, 256)])

    assert _answer(**request_kwargs).dependencies_resolved is False


@pytest.mark.parametrize("request_kwargs", HUBLESS)
def test_a_split_quant_short_a_shard_leaves_it_unresolved(hub_calls, request_kwargs):
    _repo, snapshot = _snapshot()
    _write(snapshot / QUANT / f"Sole-{QUANT}-00001-of-00002.gguf")

    response = _answer(**request_kwargs)

    assert all(not v.downloaded for v in response.variants)
    assert response.dependencies_resolved is False


@pytest.mark.parametrize("request_kwargs", HUBLESS)
def test_a_cancelled_download_leaves_it_unresolved(hub_calls, request_kwargs):
    repo, snapshot = _snapshot()
    _write(snapshot / MAIN)
    _record_plan(repo, [(MAIN, 256)])
    download_manifest.write_cancel_marker("model", REPO, QUANT, hub_cache = repo.parent)

    assert _answer(**request_kwargs).dependencies_resolved is False


@pytest.mark.parametrize("request_kwargs", HUBLESS)
@pytest.mark.parametrize("drafter_on_disk", [False, True], ids = ["missing", "present"])
def test_a_companion_the_hub_named_earlier_must_be_on_disk(
    hub_calls, monkeypatch, request_kwargs, drafter_on_disk
):
    _repo, snapshot = _snapshot()
    _write(snapshot / MAIN)
    drafter = f"mtp-Sole-{QUANT}.gguf"
    if drafter_on_disk:
        _write(snapshot / drafter, 128)
    requirement = SimpleNamespace(
        expected_files = (
            download_manifest.ExpectedFile(MAIN, 256),
            download_manifest.ExpectedFile(drafter, 128),
        )
    )
    monkeypatch.setattr(gguf_variants, "_variant_requirement_cache_get", lambda key: requirement)

    assert _answer(**request_kwargs).dependencies_resolved is drafter_on_disk


def test_a_live_listing_still_resolves_dependencies(hub_calls, monkeypatch):
    _repo, snapshot = _snapshot()
    _write(snapshot / MAIN)
    info = SimpleNamespace(filename = MAIN, quant = QUANT, display_label = None, size_bytes = 256)
    sibling = SimpleNamespace(rfilename = MAIN, size = 256, blob_id = "b", lfs = None)
    monkeypatch.setattr(
        gguf_variants, "list_gguf_variants", lambda *a, **k: ([info], False, [sibling])
    )
    monkeypatch.setattr(gguf_variants, "_gguf_all_variant_requirements", lambda *a, **k: {})

    assert _answer().dependencies_resolved is True

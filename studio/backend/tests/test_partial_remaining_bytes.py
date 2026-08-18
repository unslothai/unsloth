# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A partial row is priced by what a resume still has to fetch.

The card used to print the variant total beside a resume button, so continuing a
sharded download that was 40 GB in still read "56 GB" and looked like the whole
model coming down again. Bytes reused are whole files: a finished shard is kept,
an unresumable partial is refetched, so a one-file quant really does read back
its full size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.services.models.gguf_variants import (
    variant_remaining_bytes,
    variant_remaining_bytes_from_state,
)
from hub.utils import download_manifest, download_registry, hf_cache_state, state_dir
from hub.utils.download_manifest import ExpectedFile
from hub.utils.gguf_plan import plan_from_expected_files


SHARD_A = "a" * 64
SHARD_B = "b" * 64
GB = 1024**3


@pytest.fixture
def blobs(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    blobs_dir = root / "models--Org--Model" / "blobs"
    blobs_dir.mkdir(parents = True)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setattr(download_registry, "hf_cache_root", lambda **_kw: root)
    monkeypatch.setattr(hf_cache_state, "hf_cache_root", lambda **_kw: root)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda *_a, **_k: [root])
    return blobs_dir


def _write(path: Path, size: int) -> Path:
    path.write_bytes(b"")
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


def _split_plan():
    """Two shards of one quant, 2 GB each."""
    return plan_from_expected_files(
        "Q4_K_M",
        [
            ExpectedFile(path = "model-Q4_K_M-00001-of-00002.gguf", size = 2 * GB, sha256 = SHARD_A),
            ExpectedFile(path = "model-Q4_K_M-00002-of-00002.gguf", size = 2 * GB, sha256 = SHARD_B),
        ],
    )


def test_a_finished_shard_is_subtracted(blobs):
    _write(blobs / SHARD_A, 2 * GB)
    assert variant_remaining_bytes("Org/Model", _split_plan()) == 2 * GB


def test_nothing_on_disk_still_prices_the_whole_variant(blobs):
    assert variant_remaining_bytes("Org/Model", _split_plan()) == 4 * GB


def test_an_unresumable_partial_is_priced_as_a_full_refetch(blobs):
    # 1.18+ writes <etag>.<nonce>.incomplete and never reopens it, so those bytes are gone.
    _write(blobs / f"{SHARD_A}.deadbeef{hf_cache_state.INCOMPLETE_SUFFIX}", 2 * GB)
    assert variant_remaining_bytes("Org/Model", _split_plan()) == 4 * GB


def test_a_one_file_quant_reads_back_its_full_size(blobs):
    """Nothing to keep, so a resume costs the whole quant. This is the case users report
    as the model downloading all over again, and the number has to say so."""
    plan = plan_from_expected_files(
        "Q4_K_M",
        [ExpectedFile(path = "model-Q4_K_M.gguf", size = 4 * GB, sha256 = SHARD_A)],
    )
    _write(blobs / f"{SHARD_A}.deadbeef{hf_cache_state.INCOMPLETE_SUFFIX}", 3 * GB)
    assert variant_remaining_bytes("Org/Model", plan) == 4 * GB


def test_an_unresolvable_plan_reports_nothing_rather_than_guessing(blobs):
    assert variant_remaining_bytes("Org/Model", None) is None
    empty = plan_from_expected_files(
        "Q4_K_M",
        [ExpectedFile(path = "model-Q4_K_M.gguf", size = 4 * GB, sha256 = None)],
    )
    assert variant_remaining_bytes("Org/Model", empty) is None


def test_a_complete_variant_has_nothing_left_to_fetch(blobs):
    _write(blobs / SHARD_A, 2 * GB)
    _write(blobs / SHARD_B, 2 * GB)
    assert variant_remaining_bytes("Org/Model", _split_plan()) == 0


# --------------------------------------------------------------------------------------------
# Offline and local-cache listings, which have no hub plan to price from. The on-device card
# asks for those (preferLocalCache), so leaving them unpriced showed the full total there.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    return tmp_path


def _write_manifest(files):
    assert download_manifest.write_manifest("model", "Org/Model", "Q4_K_M", files, "http")


def test_the_worker_manifest_prices_a_local_partial(blobs, state):
    _write_manifest(
        [
            ExpectedFile(path = "model-Q4_K_M-00001-of-00002.gguf", size = 2 * GB, sha256 = SHARD_A),
            ExpectedFile(path = "model-Q4_K_M-00002-of-00002.gguf", size = 2 * GB, sha256 = SHARD_B),
        ]
    )
    _write(blobs / SHARD_A, 2 * GB)

    assert variant_remaining_bytes_from_state("Org/Model", "Q4_K_M", None) == 2 * GB


def test_a_row_with_no_manifest_stays_unpriced(blobs, state):
    assert variant_remaining_bytes_from_state("Org/Model", "Q4_K_M", None) is None


def test_a_companion_the_row_does_not_count_is_still_priced(blobs, state):
    # The mmproj comes down with the quant, so it belongs in the transfer even though the
    # row's own size does not include it.
    _write_manifest(
        [
            ExpectedFile(path = "model-Q4_K_M.gguf", size = 4 * GB, sha256 = SHARD_A),
            ExpectedFile(path = "mmproj-F16.gguf", size = 1 * GB, sha256 = SHARD_B),
        ]
    )

    assert variant_remaining_bytes_from_state("Org/Model", "Q4_K_M", None) == 5 * GB


def test_an_unnamed_variant_is_not_priced(blobs, state):
    assert variant_remaining_bytes_from_state("Org/Model", "", None) is None


def test_a_local_row_is_not_capped_by_the_shards_it_already_has(blobs, state):
    """A local scan sizes a variant from the shards ON DISK, so an early interruption makes that
    total smaller than the transfer. Capping by it reported less left than must be fetched."""
    shard_c = "c" * 64
    _write_manifest(
        [
            ExpectedFile(path = "m-Q4_K_M-00001-of-00003.gguf", size = 2 * GB, sha256 = SHARD_A),
            ExpectedFile(path = "m-Q4_K_M-00002-of-00003.gguf", size = 2 * GB, sha256 = SHARD_B),
            ExpectedFile(path = "m-Q4_K_M-00003-of-00003.gguf", size = 2 * GB, sha256 = shard_c),
        ]
    )
    _write(blobs / SHARD_A, 2 * GB)

    # 2 GB on disk, so the local row advertises 2 GB, but 4 GB is still to fetch.
    assert variant_remaining_bytes_from_state("Org/Model", "Q4_K_M", None) == 4 * GB


# --------------------------------------------------------------------------------------------
# A partial is measured by the bytes really on disk, and one shard is credited once however
# many repo directories the cache holds for the same repo. Both were observed against real
# caches: an interrupted hf_transfer download and two real `hf_hub_download` calls that spelled
# one repo id in two casings.
# --------------------------------------------------------------------------------------------


MB = 1024**2


def _sparse(path: Path, written: int, logical: int) -> Path:
    """A real sparse file: *written* bytes allocated, *logical* bytes reported."""
    with path.open("wb") as handle:
        handle.write(b"\xa5" * written)
        handle.truncate(logical)
    return path


def test_a_sparse_partial_is_priced_by_the_bytes_it_actually_holds(blobs):
    """hf_transfer's parallel Range writer leaves a partial whose st_size runs ahead of what has
    been written. Crediting the logical size understated the transfer by the whole gap, and once
    st_size reached the declared size the card read "0 B left" for a file barely started."""
    from filelock import FileLock

    plan = plan_from_expected_files(
        "Q4_K_M",
        [ExpectedFile(path = "model-Q4_K_M.gguf", size = 64 * MB, sha256 = SHARD_A)],
    )
    partial = _sparse(blobs / f"{SHARD_A}.deadbeef{hf_cache_state.INCOMPLETE_SUFFIX}", 4 * MB, 64 * MB)
    assert partial.stat().st_size == 64 * MB
    assert partial.stat().st_blocks * 512 < 8 * MB

    # Held lock: the one state in which a partial no later attempt could reopen still counts,
    # because a live writer is finishing it. That is exactly when it is sparsest.
    lock_path = blobs.parent.parent / ".locks" / blobs.parent.name / f"{SHARD_A}.lock"
    lock_path.parent.mkdir(parents = True, exist_ok = True)
    with FileLock(str(lock_path), timeout = 5):
        remaining = variant_remaining_bytes("Org/Model", plan)

    assert remaining is not None
    assert remaining >= 64 * MB - 8 * MB, "credited the sparse file's logical size, not its bytes"


def test_one_shard_in_two_case_variant_repo_dirs_is_credited_once(blobs, monkeypatch):
    """The Hub resolves repo ids case-insensitively and huggingface_hub keeps the caller's
    casing in the folder name, so a case-sensitive filesystem holds models--Org--Model beside
    models--org--model. Summing the directories counted one shard twice and clamped a variant
    still missing a whole shard to "0 B left"."""
    root = blobs.parent.parent
    twin = root / "models--org--model" / "blobs"
    twin.mkdir(parents = True)
    _write(blobs / SHARD_A, 2 * GB)
    _write(twin / SHARD_A, 2 * GB)

    assert variant_remaining_bytes("Org/Model", _split_plan()) == 2 * GB

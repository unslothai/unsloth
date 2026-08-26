# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Route-level tests for ``GET /kv-cache-estimate``.

The estimate is only worth showing if it describes the load llama-server will
actually perform, so these drive the real handler and the real drafter resolver
rather than a replayed copy of their logic.

Two classes of defect are covered:

* The MTP reserve must follow the loader's own ``is_mtp_model`` precondition. A
  model with no embedded head, no MTP name and no separate drafter cannot run
  MTP, so the loader emits ``--spec-default`` and reserves nothing. Because
  ``_estimate_mtp_overhead_bytes`` defaults ``mtp_keeps_target_ctx=True`` (so an
  unsure caller over-reserves), skipping that precondition billed every MLA
  model a second full f16 copy of its own KV.

* Drafter discovery must resolve the same file ``_download_mtp`` opens, on every
  platform. The estimate runs on hosts whose filesystems disagree about case and
  whose ``Path`` ordering differs, and it is reached for both HF snapshots and
  plain local folders.

No GPU, no network. Cross-platform: the layout cases below construct real
directories under ``tmp_path`` and assert against the loader's own picker.
"""

import asyncio
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Installs the process-wide loggers/structlog/httpx stubs and the GGUF builder.
from test_kv_cache_estimation import _backend_from_gguf, _make_gguf_bytes  # noqa: E402

import routes.models as models_routes  # noqa: E402
from core.inference.llama_cpp import _pick_mtp  # noqa: E402

# The platform keys the repo already parametrises over elsewhere
# (test_diffusion_predownload_guard_platforms.py).
PLATFORMS = ("linux", "wsl", "win32", "darwin")

# MLA (kv_lora_rank set) with NO MTP head: no nextn_predict_layers. This is the
# shape that was charged a duplicate KV.
_MLA_NO_HEAD = {
    "context_length": 131072,
    "block_count": 62,
    "attention.head_count": 32,
    "attention.head_count_kv": 16,
    "embedding_length": 5376,
    "attention.key_length": 128,
    "attention.value_length": 128,
    "attention.kv_lora_rank": 512,
    "attention.key_length_mla": 256,
}


def _write_gguf(path: Path, fields: dict, arch: str = "testarch") -> Path:
    kv = {"general.architecture": arch}
    for k, v in fields.items():
        kv[f"{arch}.{k}"] = v
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(_make_gguf_bytes(arch, kv))
    return path


def _call_route(monkeypatch, *, path: Path, weights_bytes: int, repo_id: str,
                speculative_type: str | None, mtp_token: bool = True,
                is_local: bool = False):
    """Drive the real handler with the quant already resolved to *path*."""
    monkeypatch.setattr(
        models_routes, "_resolve_quant_gguf",
        lambda _repo, _quant, _local: (str(path), weights_bytes),
    )
    monkeypatch.setattr(
        models_routes, "is_local_path", lambda _p: is_local, raising = False
    )
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities",
        classmethod(lambda cls, *a, **k: {"mtp_token": mtp_token}),
    )
    return asyncio.run(
        models_routes.get_kv_cache_estimate(
            repo_id = repo_id,
            quant = "Q4_K_M",
            n_ctx = 32768,
            cache_type_kv = None,
            n_parallel = 1,
            speculative_type = speculative_type,
            request = None,
            current_subject = "test",
        )
    )


class TestMtpReserveFollowsTheLoader:
    """The reserve is charged only when the loader could run MTP."""

    @pytest.mark.parametrize("mode", ["mtp", "auto", "mtp+ngram"])
    def test_headless_mla_is_not_charged_a_duplicate_kv(
        self, monkeypatch, tmp_path, mode
    ):
        gguf = _write_gguf(tmp_path / "plain-model-Q4_K_M.gguf", _MLA_NO_HEAD)
        out = _call_route(
            monkeypatch, path = gguf, weights_bytes = 4096,
            repo_id = "org/plain-model", speculative_type = mode,
        )
        # The KV itself must still be reported; only the reserve goes away.
        assert out["kv_bytes"] and out["kv_bytes"] > 0
        assert out["spec_bytes"] is None, (
            f"{mode}: a model with no head, no MTP name and no drafter was charged "
            f"{out['spec_bytes']} bytes; the loader reserves nothing for it"
        )

    def test_embedded_head_is_still_charged(self, monkeypatch, tmp_path):
        """The gate must not suppress a model that genuinely runs MTP."""
        fields = dict(_MLA_NO_HEAD)
        fields["nextn_predict_layers"] = 1
        gguf = _write_gguf(tmp_path / "headed-Q4_K_M.gguf", fields)
        out = _call_route(
            monkeypatch, path = gguf, weights_bytes = 4096,
            repo_id = "org/headed", speculative_type = "mtp",
        )
        assert out["spec_bytes"] and out["spec_bytes"] > 0

    def test_ngram_costs_nothing(self, monkeypatch, tmp_path):
        """ngram drafts from generated text, so it holds no VRAM."""
        gguf = _write_gguf(tmp_path / "plain-Q4_K_M.gguf", _MLA_NO_HEAD)
        out = _call_route(
            monkeypatch, path = gguf, weights_bytes = 4096,
            repo_id = "org/plain", speculative_type = "ngram",
        )
        assert out["spec_bytes"] is None

    def test_binary_without_mtp_support_reserves_nothing(self, monkeypatch, tmp_path):
        fields = dict(_MLA_NO_HEAD)
        fields["nextn_predict_layers"] = 1
        gguf = _write_gguf(tmp_path / "headed-Q4_K_M.gguf", fields)
        out = _call_route(
            monkeypatch, path = gguf, weights_bytes = 4096,
            repo_id = "org/headed", speculative_type = "mtp", mtp_token = False,
        )
        assert out["spec_bytes"] is None


class TestDrafterDiscoveryMatchesTheLoader:
    """_resolve_mtp_drafter must name the file _pick_mtp names."""

    @staticmethod
    def _snapshot(tmp_path: Path) -> Path:
        """An HF cache snapshot, the layout _companion_snapshot_sibling expects."""
        snap = tmp_path / "models--org--repo" / "snapshots" / ("a" * 40)
        snap.mkdir(parents = True)
        return snap

    def test_finds_root_companion_when_weights_are_in_a_quant_subdir(self, tmp_path):
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "BF16" / "model.gguf", _MLA_NO_HEAD)
        companion = _write_gguf(snap / "mtp-model.gguf", _MLA_NO_HEAD)

        got, size = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(companion), (
            "the loader fetches the snapshot-root companion; an estimate that "
            "misses it prices the reserve at 0"
        )
        assert size == companion.stat().st_size

    def test_a_directory_named_mtp_does_not_make_every_sibling_a_drafter(self, tmp_path):
        """_drafter_path_kind matches ancestor segments; _pick_mtp does not."""
        root = tmp_path / "mtp"
        main = _write_gguf(root / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        _write_gguf(root / "another-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, size = models_routes._resolve_mtp_drafter(str(main), search_root = str(root))
        assert got is None, f"billed an unrelated sibling as a drafter: {got} ({size} bytes)"

    def test_the_main_weight_is_never_its_own_drafter(self, tmp_path):
        root = tmp_path / "mtp"
        main = _write_gguf(root / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        got, _ = models_routes._resolve_mtp_drafter(str(main), search_root = str(root))
        assert got != str(main)

    def test_nested_mtp_subdir_copies_are_skipped(self, tmp_path):
        """MTP/ copies are explicit-selection; the loader does not auto-fetch them."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got is None

    def test_root_companion_wins_over_a_nested_copy(self, tmp_path):
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        root_companion = _write_gguf(snap / "mtp-model.gguf", _MLA_NO_HEAD)
        _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(root_companion)

    def test_appledouble_shadow_is_not_billed(self, tmp_path):
        """macOS/exFAT/SMB write "._name" metadata beside every file."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        real = _write_gguf(snap / "mtp-model.gguf", _MLA_NO_HEAD)
        # Sorts ahead of "mtp-model.gguf" on a plain byte sort.
        (snap / "._mtp-model.gguf").write_bytes(b"\x00\x05\x16\x07" + b"x" * 4000)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(real), f"AppleDouble shadow billed as the drafter: {got}"

    def test_uppercase_extension_is_recognised(self, tmp_path):
        """Case-insensitive filesystems (Windows, APFS) can hand back .GGUF."""
        assert _pick_mtp(["MTP-Model.GGUF"]) == "MTP-Model.GGUF"

    def test_picker_ordering_is_platform_independent(self):
        """sorted() over Path is case-folded on Windows and not on POSIX;
        _pick_mtp sorts relative strings, so every host agrees."""
        listing = ["mtp-b.gguf", "MTP-a.gguf", "model.gguf"]
        assert _pick_mtp(listing) == _pick_mtp(list(reversed(listing)))

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_resolution_is_stable_across_platforms(self, monkeypatch, tmp_path, platform):
        monkeypatch.setattr(sys, "platform", platform)
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "BF16" / "model.gguf", _MLA_NO_HEAD)
        companion = _write_gguf(snap / "mtp-model.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(companion)

    def test_broken_symlink_is_skipped_not_billed(self, tmp_path):
        """HF snapshots are symlinks into blobs/; an interrupted download dangles."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        (snap / "mtp-model.gguf").symlink_to(snap / "blobs" / "missing")

        got, size = models_routes._resolve_mtp_drafter(str(main))
        assert (got, size) == (None, 0)

    def test_live_symlink_reports_the_target_size(self, tmp_path):
        """The real HF cache layout: snapshot entry -> blob."""
        snap = self._snapshot(tmp_path)
        blobs = snap.parent.parent / "blobs"
        blobs.mkdir(parents = True)
        blob = _write_gguf(blobs / "deadbeef", _MLA_NO_HEAD)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        (snap / "mtp-model.gguf").symlink_to(blob)

        got, size = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(snap / "mtp-model.gguf")
        assert size == blob.stat().st_size, "must follow the link to the blob"

    def test_missing_directory_is_survivable(self, tmp_path):
        got, size = models_routes._resolve_mtp_drafter(str(tmp_path / "gone" / "x.gguf"))
        assert (got, size) == (None, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

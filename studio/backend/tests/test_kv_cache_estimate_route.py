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

from __future__ import annotations

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


# An ordinary GQA model with nothing special about it, for asserting that the
# shared machinery ran at all.
_PLAIN_GQA = {
    "context_length": 32768,
    "block_count": 32,
    "attention.head_count": 32,
    "attention.head_count_kv": 8,
    "embedding_length": 4096,
    "attention.key_length": 128,
    "attention.value_length": 128,
}

# A sliding-window model, which is the only shape where --ctx-checkpoints costs
# anything: each checkpoint is an SWA snapshot per slot.
_SWA_MODEL = {
    "context_length": 131072,
    "block_count": 32,
    "attention.head_count": 32,
    "attention.head_count_kv": 8,
    "embedding_length": 4096,
    "attention.key_length": 128,
    "attention.value_length": 128,
    "attention.sliding_window": 4096,
    "attention.sliding_window_pattern": 6,
}


def _write_gguf(
    path: Path,
    fields: dict,
    arch: str = "testarch",
) -> Path:
    kv = {"general.architecture": arch}
    for k, v in fields.items():
        kv[f"{arch}.{k}"] = v
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(_make_gguf_bytes(arch, kv))
    return path


def _call_route(
    monkeypatch,
    *,
    path: Path,
    weights_bytes: int,
    repo_id: str,
    speculative_type: str | None,
    mtp_token: bool = True,
    is_local: bool = False,
    n_parallel: int = 1,
    spec_draft_n_max: int | None = None,
    ctx_checkpoints: int | None = None,
    disable_vision: bool = False,
    n_ctx: int | None = 32768,
):
    """Drive the real handler with the quant already resolved to *path*."""
    monkeypatch.setattr(
        models_routes,
        "_resolve_quant_gguf",
        lambda _repo, _quant, _local: (str(path), weights_bytes),
    )
    monkeypatch.setattr(models_routes, "is_local_path", lambda _p: is_local, raising = False)
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, *a, **k: {"mtp_token": mtp_token}),
    )
    return asyncio.run(
        models_routes.get_kv_cache_estimate(
            repo_id = repo_id,
            quant = "Q4_K_M",
            n_ctx = n_ctx,
            cache_type_kv = None,
            n_parallel = n_parallel,
            speculative_type = speculative_type,
            spec_draft_n_max = spec_draft_n_max,
            spec_draft_cache_type = None,
            ctx_checkpoints = ctx_checkpoints,
            disable_vision = disable_vision,
            n_batch = None,
            n_ubatch = None,
            tensor_parallel = False,
            request = None,
            current_subject = "test",
        )
    )


class TestMtpReserveFollowsTheLoader:
    """The reserve is charged only when the loader could run MTP."""

    @pytest.mark.parametrize("mode", ["mtp", "auto", "mtp+ngram"])
    def test_headless_mla_is_not_charged_a_duplicate_kv(self, monkeypatch, tmp_path, mode):
        gguf = _write_gguf(tmp_path / "plain-model-Q4_K_M.gguf", _MLA_NO_HEAD)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/plain-model",
            speculative_type = mode,
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
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/headed",
            speculative_type = "mtp",
        )
        assert out["spec_bytes"] and out["spec_bytes"] > 0

    def test_ngram_costs_nothing(self, monkeypatch, tmp_path):
        """ngram drafts from generated text, so it holds no VRAM."""
        gguf = _write_gguf(tmp_path / "plain-Q4_K_M.gguf", _MLA_NO_HEAD)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/plain",
            speculative_type = "ngram",
        )
        assert out["spec_bytes"] is None

    def test_binary_without_mtp_support_reserves_nothing(self, monkeypatch, tmp_path):
        fields = dict(_MLA_NO_HEAD)
        fields["nextn_predict_layers"] = 1
        gguf = _write_gguf(tmp_path / "headed-Q4_K_M.gguf", fields)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/headed",
            speculative_type = "mtp",
            mtp_token = False,
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

    def test_nested_mtp_subdir_copy_is_used_when_no_root_mirror(self, tmp_path):
        """A repo publishing heads only under MTP/, as Qwen3.8-Flash-Next and
        Qwen3.8-27B do, still gets a drafter. _cached_repo_mtp_drafter already
        reuses such a copy offline, so skipping it here gave a cached user
        speculation and a fresh one none."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        nested = _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(nested)

    def test_nested_fallback_prefers_q8_over_bf16(self, tmp_path):
        """bf16 sorts first by name and is the worst head: larger and slower,
        because a draft step is dominated by the LM head."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        _write_gguf(snap / "MTP" / "mtp-model-BF16.gguf", _MLA_NO_HEAD)
        q8 = _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(q8)

    def test_nested_fallback_prefers_the_shared_head(self, tmp_path):
        """A -shared- head borrows the target's token_embd/output instead of
        carrying its own: 1.35 GB smaller at Q8_0 and no worse, accepting
        identically to the full head (159 of 284) on the shipped prebuilt. Only
        qwen4exp reaches this path, and its MTP graph and the borrow ship in the
        same fork, so a build that can draft one carries the other."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)
        shared = _write_gguf(snap / "MTP" / "mtp-model-shared-Q8_0.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(shared)

    def test_precision_outranks_the_shared_preference(self, tmp_path):
        """Q8_0 first is the stronger rule: a shared bf16 head is both larger than
        a full Q8_0 one and slower, since a draft step is dominated by the LM head
        and that head is cheaper to execute at 8 bits."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        q8 = _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)
        _write_gguf(snap / "MTP" / "mtp-model-shared-BF16.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(q8)

    def test_nested_fallback_is_skipped_for_other_architectures(self, tmp_path):
        """The fallback is qwen4exp only. Qwen3.8-27B bakes the head into 20 of
        its 24 quants, and llama.cpp prefers a -md drafter over an embedded one, so
        pricing the sidecar would reserve for a file the load will not open."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen35")
        _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, size = models_routes._resolve_mtp_drafter(str(main))
        assert got is None, f"billed a sidecar the load will not fetch: {got} ({size} bytes)"

    def test_nested_fallback_is_skipped_for_a_qwen4exp_with_its_own_head(self, tmp_path):
        """Architecture is not the whole gate: a qwen4exp GGUF converted with the
        block kept needs no sidecar either. Nothing published does this, so this
        pins the intent."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(
            snap / "model-Q4_K_M.gguf",
            {**_MLA_NO_HEAD, "nextn_predict_layers": 1},
            arch = "qwen4exp",
        )
        _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got is None

    def test_a_zero_head_count_still_takes_the_nested_fallback(self, tmp_path):
        """Qwen3.8-27B writes the key on every quant and sets it to 0 on the four
        with no head, so presence of the key must not read as a head."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(
            snap / "model-UD-IQ1_S.gguf",
            {**_MLA_NO_HEAD, "nextn_predict_layers": 0},
            arch = "qwen4exp",
        )
        nested = _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(nested)

    def test_an_embedded_head_ignores_a_root_compatibility_mirror(self, tmp_path):
        snap = self._snapshot(tmp_path)
        main = _write_gguf(
            snap / "RVN-Q6_K-mtp.gguf",
            {**_MLA_NO_HEAD, "nextn_predict_layers": 1},
            arch = "qwen35",
        )
        _write_gguf(snap / "mtp-RVN.gguf", _MLA_NO_HEAD)

        got, size = models_routes._resolve_mtp_drafter(str(main))
        assert got is None
        assert size == 0

    def test_a_headless_model_still_takes_a_root_drafter(self, tmp_path):
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "gemma-4-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "gemma3")
        companion = _write_gguf(snap / "mtp-gemma-4.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(companion)

    def test_a_non_drafter_parked_under_mtp_is_not_launched(self, tmp_path):
        """Everything under MTP/ classifies as a drafter, which keeps companions
        out of variant menus and is too broad for choosing what to launch: an
        mmproj or an imatrix would be handed to --model-draft."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        _write_gguf(snap / "MTP" / "mmproj-BF16.gguf", _MLA_NO_HEAD)
        _write_gguf(snap / "MTP" / "imatrix_unsloth.gguf", _MLA_NO_HEAD)
        _write_gguf(snap / "MTP" / "Qwen3.8-Flash-Next-Q8_0.gguf", _MLA_NO_HEAD)

        got, size = models_routes._resolve_mtp_drafter(str(main))
        assert got is None, f"would launch a non-drafter as the draft model: {got} ({size} bytes)"

    def test_the_older_mtp_suffix_naming_still_resolves(self, tmp_path):
        """Gemma 4 published <model>-MTP.gguf before the mtp- prefix, and the
        local scan still accepts it, so the hub path must too."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        old = _write_gguf(snap / "MTP" / "model-Q8_0-MTP.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(old)

    def test_an_incomplete_nested_split_does_not_shadow_a_complete_head(self, tmp_path):
        """llama.cpp resolves sibling shards from the first one's directory, so
        half a set is unusable and _download_companion_gguf answers None to it.
        Ranked first and rejected after, it would disable speculation with a usable
        lower-ranked head present."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        _write_gguf(snap / "MTP" / "mtp-model-Q8_0-00001-of-00002.gguf", _MLA_NO_HEAD)
        complete = _write_gguf(snap / "MTP" / "mtp-model-Q4_K_M.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(complete)

    def test_a_complete_nested_split_is_still_chosen(self, tmp_path):
        """The filter drops incomplete sets, not split sets."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        first = _write_gguf(snap / "MTP" / "mtp-model-Q8_0-00001-of-00002.gguf", _MLA_NO_HEAD)
        _write_gguf(snap / "MTP" / "mtp-model-Q8_0-00002-of-00002.gguf", _MLA_NO_HEAD)
        _write_gguf(snap / "MTP" / "mtp-model-Q4_K_M.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(first)

    def test_an_incomplete_root_split_falls_through_to_the_nested_tier(self, tmp_path):
        """The root tier returns early, so an incomplete root set used to hide a
        usable nested head and leave the load with no drafter at all."""
        snap = self._snapshot(tmp_path)
        main = _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD, arch = "qwen4exp")
        _write_gguf(snap / "mtp-model-Q8_0-00001-of-00002.gguf", _MLA_NO_HEAD)
        nested = _write_gguf(snap / "MTP" / "mtp-model-Q8_0.gguf", _MLA_NO_HEAD)

        got, _ = models_routes._resolve_mtp_drafter(str(main))
        assert got == str(nested)

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


class TestTheEstimateMatchesTheConfiguredLoad:
    """Settings a user can save must reach the estimator that prices them.

    Each of these is a term the load planner forwards and this route did not, so
    the bar reported a comfortable fit for a launch that reserves more.
    """

    def test_saved_context_checkpoints_are_priced(self, monkeypatch, tmp_path):
        """Each checkpoint is an SWA snapshot per slot, so this is not small."""
        fields = {
            "context_length": 131072,
            "block_count": 48,
            "attention.head_count": 32,
            "attention.head_count_kv": 8,
            "embedding_length": 4096,
            "attention.key_length": 128,
            "attention.value_length": 128,
            "attention.sliding_window": 4096,
            "attention.sliding_window_pattern": [True, True, True, False],
        }
        gguf = _write_gguf(tmp_path / "swa-Q4_K_M.gguf", fields)
        none = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/swa",
            speculative_type = None,
            n_parallel = 4,
        )
        many = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/swa",
            speculative_type = None,
            n_parallel = 4,
            ctx_checkpoints = 32,
        )
        assert (
            many["kv_bytes"] > none["kv_bytes"]
        ), "saved ctx_checkpoints did not reach the KV estimator"

    def test_draft_depth_is_priced_on_a_hybrid_mamba_target(self, monkeypatch, tmp_path):
        """The rollback copies dominate the reserve; the zero default omits them."""
        fields = {
            "context_length": 131072,
            "block_count": 48,
            "attention.head_count": 32,
            "attention.head_count_kv": 8,
            "embedding_length": 4096,
            "attention.key_length": 128,
            "attention.value_length": 128,
            "ssm.inner_size": 6144,
            "ssm.state_size": 128,
            "ssm.group_count": 16,
            "ssm.conv_kernel": 4,
            "full_attention_interval": 4,
            "nextn_predict_layers": 1,
        }
        gguf = _write_gguf(tmp_path / "mamba-Q4_K_M.gguf", fields)

        def at(depth):
            return _call_route(
                monkeypatch,
                path = gguf,
                weights_bytes = 4096,
                repo_id = "org/mamba",
                speculative_type = "mtp",
                n_parallel = 4,
                spec_draft_n_max = depth,
            )

        none = at(None)
        zero = at(0)
        deep = at(16)
        assert (
            deep["spec_bytes"] > zero["spec_bytes"] * 10
        ), "spec_draft_n_max did not reach the MTP estimator"
        # A blank field is NOT zero: _build_speculative_flags emits its own
        # default (2 with a GPU, 3 without) and the rollback state is multiplied
        # by it, so pricing an unset field as zero dropped the dominant
        # allocation on this exact model shape. An explicit zero still means
        # zero, which is what separates the two calls below.
        assert (
            none["spec_bytes"] > zero["spec_bytes"]
        ), "an unset draft depth was priced as zero, dropping every rollback copy"
        assert (
            none["spec_bytes"] < deep["spec_bytes"]
        ), "the default depth should sit between drafting nothing and a depth of 16"

    def test_a_split_drafter_is_billed_for_every_shard(self, tmp_path):
        snap = TestDrafterDiscoveryMatchesTheLoader._snapshot(tmp_path)
        _write_gguf(snap / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        # A two-shard companion. llama-server opens shard 1 and reserves both.
        one = snap / "mtp-model-00001-of-00002.gguf"
        two = snap / "mtp-model-00002-of-00002.gguf"
        _write_gguf(one, _MLA_NO_HEAD)
        two.write_bytes(b"\x00" * 500_000)

        got, size = models_routes._resolve_mtp_drafter(str(snap / "model-Q4_K_M.gguf"))
        assert got == str(one), "llama-server is handed shard 1"
        assert (
            size >= one.stat().st_size + two.stat().st_size
        ), f"billed {size} bytes, which is shard 1 alone rather than the family"

    def test_a_dspark_or_dflash_mode_reports_its_reserve_unpriced(self, monkeypatch, tmp_path):
        """Rather than a fit that omits the launch's largest allocation."""
        gguf = _write_gguf(tmp_path / "plain-Q4_K_M.gguf", _MLA_NO_HEAD)
        for mode in ("dspark", "dflash"):
            out = _call_route(
                monkeypatch,
                path = gguf,
                weights_bytes = 4096,
                repo_id = "org/plain",
                speculative_type = mode,
            )
            assert out["spec_unpriced"] is True, f"{mode} claimed to be priced"
            assert out["spec_bytes"] is None

    def test_a_priceable_mode_is_not_marked_unpriced(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "plain-Q4_K_M.gguf", _MLA_NO_HEAD)
        for mode in (None, "ngram", "mtp", "auto"):
            out = _call_route(
                monkeypatch,
                path = gguf,
                weights_bytes = 4096,
                repo_id = "org/plain",
                speculative_type = mode,
            )
            assert out["spec_unpriced"] is False, f"{mode} wrongly marked unpriced"

    def test_the_vision_projector_is_charged_unless_vision_is_off(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "vision-Q4_K_M.gguf", _MLA_NO_HEAD)
        mmproj = tmp_path / "mmproj-F16.gguf"
        mmproj.write_bytes(b"\x00" * 800_000)

        on = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = str(tmp_path),
            speculative_type = None,
            is_local = True,
        )
        assert (
            on["projector_bytes"] and on["projector_bytes"] > mmproj.stat().st_size
        ), "the projector is charged above its file size (_MMPROJ_VRAM_SAFETY)"

        off = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = str(tmp_path),
            speculative_type = None,
            is_local = True,
            disable_vision = True,
        )
        assert off["projector_bytes"] is None, "vision off must free the projector"

    def test_a_model_with_no_projector_reports_none(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "text-Q4_K_M.gguf", _MLA_NO_HEAD)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = str(tmp_path),
            speculative_type = None,
            is_local = True,
        )
        assert out["projector_bytes"] is None


class TestHostMemoryIsNotChargedToTheCard:
    """Context checkpoints live in host heap, so a VRAM bar must be able to
    exclude them. The load planner's own GPU figure is
    ``kv_bytes - kv_checkpoint_bytes``; without the second term reported the bar
    warns OOM over memory that never reaches the GPU."""

    def test_checkpoints_are_reported_as_their_own_share(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "swa-model-Q4_K_M.gguf", _SWA_MODEL)
        with_checkpoints = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/swa",
            speculative_type = None,
            ctx_checkpoints = 8,
        )
        without = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/swa",
            speculative_type = None,
            ctx_checkpoints = 0,
        )
        share = with_checkpoints["kv_checkpoint_bytes"]
        assert share, "checkpoints were requested but no host share was reported"
        # By difference against the same call with none, which is how the load
        # planner derives it -- asking one function twice cannot drift from it.
        assert share == with_checkpoints["kv_bytes"] - without["kv_bytes"]
        # And it is a SHARE of kv_bytes, not a figure beside it: the field
        # shipped meaning the whole cache and an existing caller still reads it
        # that way.
        assert share < with_checkpoints["kv_bytes"]

    def test_no_checkpoints_means_no_host_share(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "swa-model-Q4_K_M.gguf", _SWA_MODEL)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/swa",
            speculative_type = None,
            ctx_checkpoints = None,
        )
        assert out["kv_checkpoint_bytes"] is None


class TestAutoAbstainsOverASidecar:
    """Auto promotes to DSpark or DFlash when the model ships one, so the route
    must say the reserve is unpriced rather than chart a total without it.

    These assert the OUTCOME, not that the block runs: the first version of this
    guard referenced two helpers that were only in another function's scope, and
    the resulting NameError was swallowed by the surrounding except, leaving
    spec_unpriced false with nothing in the response to show it had failed.
    """

    def _call(self, monkeypatch, tmp_path, *, sidecar: str | None, supports: bool):
        model_dir = tmp_path / "local-model"
        gguf = _write_gguf(model_dir / "model-Q4_K_M.gguf", _MLA_NO_HEAD)
        if sidecar:
            # A DFlash sidecar is identified by its header, not its name:
            # detect_dflash_file confirms general.architecture = dflash, which
            # settles the adversarial case of a real model merely CALLED DFlash.
            _write_gguf(
                model_dir / sidecar,
                _MLA_NO_HEAD,
                arch = "dflash" if sidecar.startswith("dflash") else "testarch",
            )
        from core.inference.llama_cpp import LlamaCppBackend

        monkeypatch.setattr(
            LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(
                lambda cls, *a, **k: {
                    "mtp_token": True,
                    "supports_dspark": supports,
                    "supports_dflash": supports,
                }
            ),
        )
        monkeypatch.setattr(
            models_routes,
            "_resolve_quant_gguf",
            lambda _repo, _quant, _local: (str(gguf), 4096),
        )
        monkeypatch.setattr(models_routes, "is_local_path", lambda _p: True, raising = False)
        return asyncio.run(
            models_routes.get_kv_cache_estimate(
                repo_id = str(model_dir),
                quant = "Q4_K_M",
                n_ctx = 32768,
                cache_type_kv = None,
                n_parallel = 1,
                speculative_type = "auto",
                spec_draft_n_max = None,
                spec_draft_cache_type = None,
                ctx_checkpoints = None,
                disable_vision = False,
                n_batch = None,
                n_ubatch = None,
                tensor_parallel = False,
                request = None,
                current_subject = "test",
            )
        )

    @pytest.mark.parametrize("sidecar", ["dspark-model-Q8_0.gguf", "dflash-model-Q8_0.gguf"])
    def test_a_local_sidecar_makes_the_reserve_unpriced(self, monkeypatch, tmp_path, sidecar):
        out = self._call(monkeypatch, tmp_path, sidecar = sidecar, supports = True)
        assert out["spec_unpriced"] is True, (
            "Auto charted a total with the sidecar missing; a DSpark drafter is "
            "about 11 GB, so this is the largest single allocation the route can drop"
        )

    def test_no_sidecar_leaves_the_estimate_priced(self, monkeypatch, tmp_path):
        out = self._call(monkeypatch, tmp_path, sidecar = None, supports = True)
        assert out["spec_unpriced"] is False

    def test_a_binary_that_cannot_run_one_still_prices_normally(self, monkeypatch, tmp_path):
        # The planner gates sidecar selection on the binary's own capability, so
        # abstaining where the launch would never open one blanks the bar for
        # nothing.
        out = self._call(monkeypatch, tmp_path, sidecar = "dspark-model-Q8_0.gguf", supports = False)
        assert out["spec_unpriced"] is False


class TestThePlannerFiguresArrive:
    """The route delegates to the load planner for the terms it cannot derive
    itself, inside a try/except so a planner that cannot size a model still
    leaves the KV bar drawn.

    That except is the hazard: every earlier version of this delegation failed
    into it silently and returned nulls that looked like "not sizeable" rather
    than "the call is broken". These assert the figures are actually present for
    a model the planner can size, so a signature drift or an unresolved
    parameter fails the suite instead of quietly blanking the bar.
    """

    def test_the_planner_terms_are_populated(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "planner-model"
        gguf = _write_gguf(model_dir / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.setattr(
            models_routes,
            "_resolve_quant_gguf",
            lambda _repo, _quant, _local: (str(gguf), 4_000_000_000),
        )
        monkeypatch.setattr(models_routes, "is_local_path", lambda _p: True, raising = False)
        out = asyncio.run(
            models_routes.get_kv_cache_estimate(
                repo_id = str(model_dir),
                quant = "Q4_K_M",
                n_ctx = 8192,
                cache_type_kv = None,
                n_parallel = 1,
                speculative_type = None,
                spec_draft_n_max = None,
                spec_draft_cache_type = None,
                ctx_checkpoints = None,
                disable_vision = False,
                n_batch = None,
                n_ubatch = None,
                tensor_parallel = False,
                request = None,
                current_subject = "test",
            )
        )
        assert out["compute_bytes"], (
            "the planner delegation failed into its except; every launch reserves "
            "compute buffers, so a null here means the call broke rather than that "
            "the model has none"
        )
        assert out["gpu_bytes"], "no authoritative GPU total"
        assert out["gpu_floor_bytes"], "no irreducible floor"
        # The floor is what survives shrinking the context, so it cannot exceed
        # the full plan, and for a context-sensitive model it should be smaller.
        assert out["gpu_floor_bytes"] <= out["gpu_bytes"]
        assert out["gpu_floor_bytes"] < out["gpu_bytes"], (
            "the floor equals the full plan, so the second pricing did not use a "
            "shorter context and cannot separate reducible from fixed"
        )


class TestADirectGgufFileResolves:
    """A custom or LM Studio entry whose path is the .gguf itself never goes
    through variant selection, so there is no quant label to match. The file
    names the weights on its own."""

    def test_a_direct_file_is_its_own_resolution(self, tmp_path):
        gguf = _write_gguf(tmp_path / "some-model-Q4_K_M.gguf", _PLAIN_GQA)
        path, size = models_routes._resolve_quant_gguf(str(gguf), "", True)
        assert path == str(gguf), "a direct file selection did not resolve to itself"
        assert size == gguf.stat().st_size

    def test_a_directory_still_takes_the_quant_scan(self, tmp_path):
        # The direct-file branch must not swallow the ordinary case: a folder is
        # still scanned for the quant that was asked for.
        model_dir = tmp_path / "folder"
        _write_gguf(model_dir / "model-Q4_K_M.gguf", _PLAIN_GQA)
        path, size = models_routes._resolve_quant_gguf(str(model_dir), "Q4_K_M", True)
        assert path and path.endswith("model-Q4_K_M.gguf")
        assert size > 0
        # And a quant it does not hold resolves to nothing rather than to
        # whatever happened to be there.
        missing, _ = models_routes._resolve_quant_gguf(str(model_dir), "Q2_K", True)
        assert missing is None


class TestTheInheritedEnvironmentIsPriced:
    """Studio can be launched with llama.cpp's own environment variables set, and
    the child inherits them. The estimate has to price what the child will run,
    not what the request said."""

    def test_an_inherited_context_beats_the_native_length(self, monkeypatch, tmp_path):
        # load_model drops an inherited context only when it is zero, so a
        # positive one is the legitimate way to set the window. Falling through
        # to native priced a 4k load at the header's length.
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "4096")
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/repo",
            speculative_type = None,
            n_ctx = None,
        )
        assert out["n_ctx"] == 4096, (
            "the estimate used the header's native window for a launch the "
            "environment pins to 4096"
        )
        assert out["native_context"] == _PLAIN_GQA["context_length"]

    def test_no_inherited_context_still_uses_native(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.delenv("LLAMA_ARG_CTX_SIZE", raising = False)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/repo",
            speculative_type = None,
            n_ctx = None,
        )
        assert out["n_ctx"] == _PLAIN_GQA["context_length"]

    def test_an_inherited_cache_type_sizes_the_cache(self, monkeypatch, tmp_path):
        # A q8_0 cache is roughly half an f16 one. Pricing f16 while the child
        # opens q8_0 makes the KV segment and the per-token readout contradict
        # the planner total drawn beside them.
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.delenv("LLAMA_ARG_CACHE_TYPE_K", raising = False)
        monkeypatch.delenv("LLAMA_ARG_CACHE_TYPE_V", raising = False)
        default = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/repo",
            speculative_type = None,
        )
        monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q8_0")
        monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")
        inherited = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4096,
            repo_id = "org/repo",
            speculative_type = None,
        )
        assert inherited["kv_bytes"] < default["kv_bytes"], (
            "an inherited q8_0 cache was priced as f16, so the KV segment was "
            "about twice the size the launch reserves"
        )

    def test_an_inherited_context_is_reported_as_pinned(self, monkeypatch, tmp_path):
        # The loader keeps a positive inherited context rather than fitting it,
        # so a caller must not soften its verdict for that launch. Saying
        # "auto-fitted" there suppressed the overage AND drew only the
        # irreducible floor, which is a comfortable fit for a load that OOMs.
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "4096")
        assert (
            _call_route(
                monkeypatch,
                path = gguf,
                weights_bytes = 4096,
                repo_id = "org/repo",
                speculative_type = None,
                n_ctx = None,
            )["context_is_pinned"]
            is True
        )

    def test_an_omitted_context_with_no_inheritance_is_not_pinned(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.delenv("LLAMA_ARG_CTX_SIZE", raising = False)
        assert (
            _call_route(
                monkeypatch,
                path = gguf,
                weights_bytes = 4096,
                repo_id = "org/repo",
                speculative_type = None,
                n_ctx = None,
            )["context_is_pinned"]
            is False
        )

    def test_an_explicit_context_is_pinned(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.delenv("LLAMA_ARG_CTX_SIZE", raising = False)
        assert (
            _call_route(
                monkeypatch,
                path = gguf,
                weights_bytes = 4096,
                repo_id = "org/repo",
                speculative_type = None,
                n_ctx = 8192,
            )["context_is_pinned"]
            is True
        )

    def test_an_inherited_device_pin_is_reported(self, monkeypatch, tmp_path):
        # The child is confined to the cards LLAMA_ARG_DEVICE names and an
        # automatic launch preserves the pin, so an aggregate VRAM budget
        # describes a pool the launch will not open. The caller cannot see the
        # environment, so the route has to say.
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0")
        assert (
            _call_route(
                monkeypatch,
                path = gguf,
                weights_bytes = 4096,
                repo_id = "org/repo",
                speculative_type = None,
            )["inherited_device_pin"]
            is True
        )

    @pytest.mark.parametrize("value", ["", "none", "NONE"])
    def test_no_usable_pin_is_not_reported_as_one(self, monkeypatch, tmp_path, value):
        # "none" is a CPU-only launch, which the planner already answers with
        # zero GPU bytes and which draws no bar on its own; reporting it as a pin
        # would blank the row for a second, unrelated reason.
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        monkeypatch.setenv("LLAMA_ARG_DEVICE", value)
        assert (
            _call_route(
                monkeypatch,
                path = gguf,
                weights_bytes = 4096,
                repo_id = "org/repo",
                speculative_type = None,
            )["inherited_device_pin"]
            is False
        )

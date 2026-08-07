# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Disposable A/B proof for the Windows image dependency download regression.

The test uses declared Hub metadata and a synthetic cache verdict, so CI proves the
planning behavior without downloading the 10.8 GB FLUX.2 Klein dependency set.
"""

from __future__ import annotations

from pathlib import Path
import types

from core.inference import diffusion
from core.inference.diffusion import DiffusionBackend


GB = 1_000_000_000
CHECKPOINT_REPO = "unsloth/FLUX.1-dev-GGUF"
CHECKPOINT_FILE = "flux1-dev-Q4_K_M.gguf"
BASE_REPO = "black-forest-labs/FLUX.1-dev"


def test_cached_checkpoint_is_not_staged_again(monkeypatch):
    repos = {
        CHECKPOINT_REPO: [types.SimpleNamespace(rfilename = CHECKPOINT_FILE, size = 7 * GB)],
        BASE_REPO: [
            types.SimpleNamespace(rfilename = "model_index.json", size = 1_000),
            types.SimpleNamespace(rfilename = "scheduler/scheduler_config.json", size = 2_000),
            types.SimpleNamespace(rfilename = "tokenizer/tokenizer.json", size = 3_000),
            types.SimpleNamespace(rfilename = "text_encoder/model.safetensors", size = 2 * GB),
            types.SimpleNamespace(
                rfilename = "vae/diffusion_pytorch_model.safetensors", size = 500_000_000
            ),
        ],
    }

    class FakeApi:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            del files_metadata, token
            return types.SimpleNamespace(siblings = repos[repo_id])

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *args, **kwargs: FakeApi())
    monkeypatch.setattr(diffusion, "_resolve_base_repo", lambda *args, **kwargs: BASE_REPO)
    monkeypatch.setattr(
        diffusion,
        "prefer_ungated_mirror",
        lambda repo_id, token, files = None: repo_id,
    )
    monkeypatch.setattr(diffusion, "_assert_base_repo_accessible", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        DiffusionBackend,
        "_te_prequant_plan_files",
        staticmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_dense_quant_prefetch_needed",
        lambda self, family, kwargs: False,
    )
    # ``raising=False`` deliberately makes the same probe runnable on the negative branch, where
    # the helper does not exist. The old planner ignores the synthetic cache verdict and fails.
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(
            lambda repo_id, filename: repo_id == CHECKPOINT_REPO and filename == CHECKPOINT_FILE
        ),
        raising = False,
    )

    plan = DiffusionBackend().download_plan(
        CHECKPOINT_REPO,
        gguf_filename = CHECKPOINT_FILE,
        model_kind = "gguf",
    )

    repos_to_download = [entry["repo_id"] for entry in plan["entries"]]
    assert repos_to_download == [BASE_REPO], (
        "cached Q4 must be omitted; only its missing companion repo should remain, "
        f"got {repos_to_download}"
    )
    assert plan["total_bytes"] == plan["entries"][0]["bytes"]
    print("PASS backend omitted cached Q4 and planned only missing companions")


def test_hub_selection_and_cancel_keep_one_staged_intent():
    repo_root = Path(__file__).resolve().parents[2]
    page = (repo_root / "studio/frontend/src/features/images/images-page.tsx").read_text(
        encoding = "utf-8"
    )

    assert 'source: ModelSelectorChangeMeta["source"] = "hub"' in page
    assert 'if (source !== "hub") return handleLoadRef.current(repoId, opts);' in page
    assert "if (isDownloaded !== false)" not in page
    assert "onCancelled: () => {" in page
    assert "stagedLoadDeferred.current = false;" in page
    print("PASS frontend plans every Hub pick and clears auto-load intent on cancel")

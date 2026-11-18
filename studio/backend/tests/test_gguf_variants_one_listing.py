# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A repo with no GGUF files is listed once per request: the MLX precision fallback reuses the
model_info the GGUF listing already fetched, with the same answer as a second fetch gave."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hub.services.models import gguf_variants
from hub.utils import hf_tokens

GB = 1_000_000_000


def _folder(folder: str, size: int) -> dict[str, int]:
    return {
        f"{folder}/config.json": 4_000,
        f"{folder}/model.safetensors": size,
    }


_MLX = (
    "mlx-community/Qwen3-8B-MLX",
    ("mlx", "safetensors"),
    "mlx",
    {**_folder("4-bit", 5 * GB), **_folder("8-bit", 9 * GB), "README.md": 1},
)
_SAFETENSORS = (
    "org/Qwen3-8B",
    ("safetensors",),
    "transformers",
    {"config.json": 4_000, "model.safetensors": 16 * GB, "README.md": 1},
)


def _fake_hub(
    monkeypatch,
    tmp_path,
    tags,
    library,
    sizes,
    refuse_token = False,
):
    calls = []

    class _Api:
        def __init__(self, token = None):
            self.token = token

        def model_info(
            self,
            repo_id,
            files_metadata = False,
            timeout = None,
            **kwargs,
        ):
            calls.append((repo_id, self.token, files_metadata, timeout, tuple(sorted(kwargs))))
            if refuse_token and self.token:
                raise RuntimeError("401 refused")
            return SimpleNamespace(
                tags = list(tags),
                library_name = library,
                pipeline_tag = "text-generation",
                siblings = [
                    SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": f"sha-{name}"})
                    for name, size in sizes.items()
                ],
            )

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: None)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    if refuse_token:
        monkeypatch.setattr(
            hf_tokens,
            "is_rejected_credential_error",
            lambda exc, token: "401" in str(exc) and bool(token),
        )
    return calls


@pytest.mark.parametrize("case", [_MLX, _SAFETENSORS], ids = ["mlx", "safetensors"])
@pytest.mark.parametrize("token", [None, "hf_test"], ids = ["anonymous", "token"])
def test_a_repo_without_gguf_is_listed_once(monkeypatch, tmp_path, case, token):
    repo, tags, library, sizes = case
    calls = _fake_hub(monkeypatch, tmp_path, tags, library, sizes)
    answer = asyncio.run(gguf_variants.get_gguf_variants_answer(repo, hf_token = token))
    assert len(calls) == 1, calls
    assert calls[0][2] is True
    if repo == _MLX[0]:
        assert answer.response.variant_format == "mlx"
        assert [v.quant for v in answer.response.variants] == ["4-bit", "8-bit"]
    else:
        assert not answer.response.variants


def test_a_refused_token_reuses_the_anonymous_listing(monkeypatch, tmp_path):
    repo, tags, library, sizes = _MLX
    calls = _fake_hub(monkeypatch, tmp_path, tags, library, sizes, refuse_token = True)
    answer = asyncio.run(gguf_variants.get_gguf_variants_answer(repo, hf_token = "hf_bad"))
    assert [token for _, token, *_ in calls] == ["hf_bad", False]
    assert answer.response.variant_format == "mlx"


def test_without_a_hub_listing_the_mlx_fallback_still_fetches(monkeypatch, tmp_path):
    repo, tags, library, sizes = _MLX
    calls = _fake_hub(monkeypatch, tmp_path, tags, library, sizes)
    empty = gguf_variants.VariantsAnswer(gguf_variants.GgufVariantsResponse(repo_id = repo), None)

    async def _scan(
        key,
        compute,
        offline = False,
    ):
        return empty

    monkeypatch.setattr(gguf_variants, "_shared_variants_scan", _scan)
    answer = asyncio.run(gguf_variants.get_gguf_variants_answer(repo))
    assert len(calls) == 1
    assert answer.response.variant_format == "mlx"


def test_a_failed_listing_is_not_reused(monkeypatch, tmp_path):
    repo, tags, library, sizes = _MLX
    calls = _fake_hub(monkeypatch, tmp_path, tags, library, sizes)

    def _boom(*args, **kwargs):
        raise RuntimeError("hub down")

    monkeypatch.setattr(gguf_variants, "list_gguf_variants", _boom)
    try:
        asyncio.run(gguf_variants.get_gguf_variants_answer(repo))
    except Exception:
        pass
    assert all(call[2] is True for call in calls)


def test_offline_never_reaches_the_hub(monkeypatch, tmp_path):
    repo, tags, library, sizes = _MLX
    calls = _fake_hub(monkeypatch, tmp_path, tags, library, sizes)
    try:
        asyncio.run(gguf_variants.get_gguf_variants_answer(repo, offline = True))
    except Exception:
        pass
    assert calls == []


def test_list_gguf_variants_hands_back_its_model_info(monkeypatch, tmp_path):
    from hub.utils.gguf import HUB_MODEL_INFO_SINK, list_gguf_variants

    repo, tags, library, sizes = _MLX
    calls = _fake_hub(monkeypatch, tmp_path, tags, library, sizes)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    sink: list = []
    reset = HUB_MODEL_INFO_SINK.set(sink)
    try:
        variants, _, siblings = list_gguf_variants(repo, hf_token = "hf_x")
    finally:
        HUB_MODEL_INFO_SINK.reset(reset)
    assert variants == [] and len(sink) == 1 and len(calls) == 1
    assert [s.rfilename for s in sink[0].siblings] == [s.rfilename for s in siblings]
    assert list_gguf_variants(repo, hf_token = "hf_x")[0] == [] and len(sink) == 1

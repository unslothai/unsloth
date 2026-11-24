# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Wan2.2 TI2V-5B Auto on a resident card: bf16 online, but an offline install whose cache holds only the hosted
INT8 DiT (staged by the older Auto) keeps loading that INT8 checkpoint instead of failing on missing dense shards."""

import sys

import pytest

from core.inference.diffusion import DiffusionBackend
from core.inference.video import VideoBackend
from .test_video_backend import (  # noqa: F401
    _assume_the_restricted_load_is_available,
    fake_runtime,
)
from .test_video_quantised_offload import _spoof

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _cache(monkeypatch, *, int8: bool, dense) -> None:
    """The hub cache as the loader sees it: the hosted INT8 DiT repo and/or the dense transformer shards."""
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo, name, *a, **k: int8 and repo != REPO),
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_released_transformer_cached",
        staticmethod(lambda base: base in dense if isinstance(dense, (set, tuple)) else dense),
    )


def _assemble(monkeypatch, *, dense: bool, offline: bool) -> None:
    pipeline = sys.modules["diffusers"].WanPipeline
    assemble = pipeline.from_pretrained

    def assemble_checked(source, **kwargs):
        if offline and "transformer" not in kwargs and not dense:
            raise OSError("Dense Wan transformer missing from cache")
        return assemble(source, **kwargs)

    monkeypatch.setattr(pipeline, "from_pretrained", assemble_checked)


@pytest.mark.parametrize("requested", ["auto", "int8"])
@pytest.mark.parametrize("via", ["local_files_only", "hf_offline_env"])
@pytest.mark.usefixtures("fake_runtime")
def test_offline_int8_only_cache_loads_int8(monkeypatch, requested, via):
    if via == "hf_offline_env":
        monkeypatch.setattr("core.inference.video._can_download", lambda: False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    spy = _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = True, dense = False)
    _assemble(monkeypatch, dense = False, offline = True)
    result = VideoBackend().load_pipeline(
        REPO, local_files_only = via == "local_files_only", transformer_quant = requested
    )
    assert result["loaded"] and result["transformer_quant"] == "int8"
    assert spy.seeded == ["int8"]


@pytest.mark.usefixtures("fake_runtime")
def test_offline_dense_cached_auto_keeps_bf16(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    spy = _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = True, dense = True)
    _assemble(monkeypatch, dense = True, offline = True)
    result = VideoBackend().load_pipeline(REPO, local_files_only = True, transformer_quant = "auto")
    assert result["loaded"]
    assert spy.seeded == []
    assert result["transformer_quant"] != "int8"


@pytest.mark.usefixtures("fake_runtime")
def test_offline_dense_cached_under_resolved_repo_auto_keeps_bf16(monkeypatch):
    # The dense shards sit only in the unsloth mirror the load resolves the base to, not under the upstream id.
    from core.inference.diffusion_families import mirror_repo

    resolved = mirror_repo(REPO)
    assert resolved and resolved != REPO
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    spy = _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = True, dense = {resolved})
    _assemble(monkeypatch, dense = True, offline = True)
    result = VideoBackend().load_pipeline(REPO, local_files_only = True, transformer_quant = "auto")
    assert result["loaded"]
    assert spy.seeded == []
    assert result["transformer_quant"] != "int8"


@pytest.mark.usefixtures("fake_runtime")
def test_online_no_cache_auto_plans_bf16(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr("core.inference.video._can_download", lambda: True)
    spy = _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = False, dense = False)
    from core.inference import video as V
    from core.inference.video_families import detect_video_family

    fam = detect_video_family(REPO)
    assert not V._auto_offline_keeps_cached_seed(fam, "int8", REPO, local_files_only = False)
    planned = VideoBackend()._video_planned_auto_denoiser_scheme(
        fam, base = REPO, kind = "pipeline", transformer_quant = "auto", speed_mode = None
    )
    assert planned is None
    result = VideoBackend().load_pipeline(REPO, transformer_quant = "auto")
    assert result["loaded"]
    assert spy.seeded == []

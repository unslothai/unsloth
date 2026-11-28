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

import faulthandler
import socket
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
MIRROR = "unsloth/Wan2.2-TI2V-5B-FP8"
_REAL_VIDEO_FETCH_REPO = __import__(
    "core.inference.video", fromlist = ["_video_fetch_repo"]
)._video_fetch_repo


@pytest.fixture(autouse = True)
def _hermetic(monkeypatch):
    """Hermetic: Hub lookups are pinned, sockets refuse, and a hang dumps stacks after 60 s."""
    import huggingface_hub
    from huggingface_hub import constants

    import core.inference.diffusion_families as families
    import core.inference.video as V

    faulthandler.dump_traceback_later(60, exit = False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True, raising = False)
    monkeypatch.setattr(V, "_can_download", lambda: False)
    monkeypatch.setattr(V, "_video_fetch_repo", lambda repo_id, *a, **k: repo_id)
    monkeypatch.setattr(
        families, "mirror_repo", lambda repo_id: MIRROR if repo_id == REPO else None
    )

    def _no_hub(*a, **k):
        raise OSError("hermetic test: no Hub access")

    monkeypatch.setattr(huggingface_hub, "HfApi", _no_hub)
    for name in ("hf_hub_download", "snapshot_download", "model_info", "list_repo_files"):
        if hasattr(huggingface_hub, name):
            monkeypatch.setattr(huggingface_hub, name, _no_hub)

    def _no_connect(self, *a, **k):
        raise OSError("hermetic test: network disabled")

    monkeypatch.setattr(socket.socket, "connect", _no_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_connect)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


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
    import core.inference.video as V

    monkeypatch.setattr(
        V, "_video_fetch_repo", lambda repo_id, *a, **k: MIRROR if repo_id == REPO else repo_id
    )
    spy = _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = True, dense = {MIRROR})
    _assemble(monkeypatch, dense = True, offline = True)
    result = VideoBackend().load_pipeline(REPO, local_files_only = True, transformer_quant = "auto")
    assert result["loaded"]
    assert spy.seeded == []
    assert result["transformer_quant"] != "int8"


@pytest.mark.usefixtures("fake_runtime")
def test_offline_no_mirror_dense_only_in_mirror_falls_back_to_int8(monkeypatch):
    # UNSLOTH_DIFFUSION_NO_MIRROR=1: dense shards cached only in the mirror are unusable, so the cached int8 loads.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    spy = _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = True, dense = {MIRROR})
    _assemble(monkeypatch, dense = False, offline = True)
    result = VideoBackend().load_pipeline(REPO, local_files_only = True, transformer_quant = "auto")
    assert result["loaded"] and result["transformer_quant"] == "int8"
    assert spy.seeded == ["int8"]


@pytest.mark.usefixtures("fake_runtime")
def test_offline_dense_only_in_unselected_mirror_falls_back_to_int8(monkeypatch):
    # Upstream companions cached without a transformer, the dense transformer only in the mirror: _video_fetch_repo
    # assembles from the more complete upstream, which has no dense weights, so the cached int8 loads.
    import core.inference.video as V

    monkeypatch.setattr(V, "_video_fetch_repo", _REAL_VIDEO_FETCH_REPO)
    monkeypatch.setattr(V, "_is_gated_video_upstream", lambda repo_id: False)
    monkeypatch.setattr(
        V,
        "_video_cached_completeness_detail",
        lambda repo, **k: (4, False) if repo == REPO else (-1, False),
    )
    monkeypatch.setattr(
        V, "_video_cached_completeness", lambda repo, **k: 1 if repo == MIRROR else -1
    )
    assert V._video_fetch_repo(REPO) == REPO
    spy = _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = True, dense = {MIRROR})
    _assemble(monkeypatch, dense = False, offline = True)
    result = VideoBackend().load_pipeline(REPO, local_files_only = True, transformer_quant = "auto")
    assert result["loaded"] and result["transformer_quant"] == "int8"
    assert spy.seeded == ["int8"]


@pytest.mark.usefixtures("fake_runtime")
def test_online_no_cache_auto_plans_bf16(monkeypatch):
    # Online is _can_download() True; the Hub itself stays unreachable, so only the plan is checked, not a load.
    from core.inference import video as V
    from core.inference.video_families import detect_video_family

    monkeypatch.setattr(V, "_can_download", lambda: True)
    _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = False, dense = False)
    fam = detect_video_family(REPO)
    assert not V._auto_offline_keeps_cached_seed(fam, "int8", REPO, local_files_only = False)
    planned = VideoBackend()._video_planned_auto_denoiser_scheme(
        fam, base = REPO, kind = "pipeline", transformer_quant = "auto", speed_mode = None
    )
    assert planned is None


@pytest.mark.usefixtures("fake_runtime")
def test_online_int8_cached_auto_still_plans_bf16(monkeypatch):
    from core.inference import video as V
    from core.inference.video_families import detect_video_family

    monkeypatch.setattr(V, "_can_download", lambda: True)
    _spoof(monkeypatch, tier_gib = 48)
    _cache(monkeypatch, int8 = True, dense = False)
    fam = detect_video_family(REPO)
    assert not V._auto_offline_keeps_cached_seed(fam, "int8", REPO, local_files_only = False)
    assert (
        VideoBackend()._video_planned_auto_denoiser_scheme(
            fam, base = REPO, kind = "pipeline", transformer_quant = "auto", speed_mode = None
        )
        is None
    )


A14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.usefixtures("fake_runtime")
def test_a14b_offline_auto_at_80_keeps_a_cached_int8_pair_only_without_dense(monkeypatch, dense):
    # Wan2.2 T2V-A14B on 80 GiB: Auto is bf16 where the pair stays resident, but an offline install whose cache
    # holds only the hosted INT8 pair (staged by the older Auto) keeps loading it instead of failing.
    spy = _spoof(monkeypatch, tier_gib = 80)
    _cache(monkeypatch, int8 = True, dense = dense)
    _assemble(monkeypatch, dense = dense, offline = True)
    result = VideoBackend().load_pipeline(A14B, local_files_only = True, transformer_quant = "auto")
    assert result["loaded"]
    if dense:
        assert spy.seeded == [] and result["transformer_quant"] != "int8"
    else:
        assert spy.seeded == ["int8"] and result["transformer_quant"] == "int8"


@pytest.mark.usefixtures("fake_runtime")
def test_a14b_online_auto_at_80_plans_bf16(monkeypatch):
    from core.inference import video as V
    from core.inference.video_families import detect_video_family

    monkeypatch.setattr(V, "_can_download", lambda: True)
    _spoof(monkeypatch, tier_gib = 80)
    _cache(monkeypatch, int8 = True, dense = False)
    fam = detect_video_family(A14B)
    assert (
        VideoBackend()._video_planned_auto_denoiser_scheme(
            fam, base = A14B, kind = "pipeline", transformer_quant = "auto", speed_mode = None
        )
        is None
    )

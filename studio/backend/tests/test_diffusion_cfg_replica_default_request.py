# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The default request each backend notes for the CFG replica placement. CPU only."""

import types

import pytest

pytest.importorskip("torch")

from core.inference import diffusion as image  # noqa: E402
from core.inference import diffusion_families as df  # noqa: E402
from core.inference import diffusion_multigpu as mg  # noqa: E402
from core.inference import video  # noqa: E402
from core.inference import video_families as vf  # noqa: E402


def _image_family(name):
    return next(f for f in df._FAMILIES if f.name == name)


class _Pipe:
    config = types.SimpleNamespace(is_distilled = False)


@pytest.mark.parametrize(
    "family, repo, guided",
    [
        ("qwen-image", "Qwen/Qwen-Image", True),  # true CFG 4.0 with a negative prompt
        ("z-image", "Tongyi-MAI/Z-Image", True),  # guidance 3.0, negative prompt from 0
        ("z-image", "Tongyi-MAI/Z-Image-Turbo", False),  # guidance 0
        ("flux.1", "black-forest-labs/FLUX.1-dev", False),  # embedded guidance, no negative branch
        ("flux.2-klein", "black-forest-labs/FLUX.2-klein-4B", False),  # distilled, guidance 1
        ("qwen-image-2.1", "Qwen/Qwen-Image-2.1", False),  # true CFG 1.0 by default
        ("lumina-2", "Alpha-VLLM/Lumina-Image-2.0", True),
        ("hidream-i1", "HiDream-ai/HiDream-I1-Full", True),
    ],
)
def test_image_default_request(family, repo, guided):
    pipe = _Pipe()
    fam = _image_family(family)
    image._note_multigpu_default_request(pipe, fam, repo, fam.name)
    noted = mg.default_request(pipe)
    assert noted is not None and noted.guided is guided, noted
    assert noted.steps and noted.steps > 0


def test_klein_base_runs_cfg_with_its_own_negative_prompt():
    pipe = _Pipe()
    fam = _image_family("flux.2-klein")
    image._note_multigpu_default_request(
        pipe, fam, "black-forest-labs/FLUX.2-klein-base-4B", fam.name
    )
    noted = mg.default_request(pipe)
    assert noted.guided is True and noted.steps == 20


def test_distilled_klein_checkpoint_is_unguided():
    pipe = _Pipe()
    pipe.config = types.SimpleNamespace(is_distilled = True)
    fam = _image_family("flux.2-klein")
    image._note_multigpu_default_request(
        pipe, fam, "black-forest-labs/FLUX.2-klein-base-4B", fam.name
    )
    assert mg.default_request(pipe).guided is False


@pytest.mark.parametrize(
    "family, guided",
    [
        ("wan2.2-ti2v-5b", True),
        ("wan2.1-t2v-1.3b", True),
        ("hunyuanvideo-1.5", True),
        ("ltx-2", True),
        ("ltx-2.5", False),
        ("minimax-h3", False),
    ],
)
def test_video_default_request(family, guided):
    fam = next(f for f in vf._FAMILIES if f.name == family)
    pipe = types.SimpleNamespace(
        vae_scale_factor_spatial = 16,
        vae_scale_factor_temporal = 4,
        transformer = types.SimpleNamespace(config = types.SimpleNamespace(patch_size = (1, 2, 2))),
    )

    class Holder:
        pass

    holder = Holder()
    holder.__dict__.update(vars(pipe))
    video._note_video_default_request(fam, holder)
    noted = mg.default_request(holder)
    assert noted.guided is guided and noted.steps == fam.default_steps
    w, h = fam.resolution_presets[0]
    assert noted.tokens == mg.video_default_tokens(holder, w, h, fam.default_num_frames)

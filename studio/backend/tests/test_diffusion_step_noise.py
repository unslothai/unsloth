# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""seeded_scheduler_step: a scheduler that draws noise in ``step`` follows the request's seed (HiDream-I1 Fast/Dev)."""

import pytest

torch = pytest.importorskip("torch")

from core.inference.diffusion_step_noise import seeded_scheduler_step  # noqa: E402


class _NoisyScheduler:
    """FlowMatchLCMScheduler's shape: every step mixes in fresh noise drawn from ``generator``."""

    def step(
        self,
        model_output,
        timestep,
        sample,
        generator = None,
        return_dict = True,
    ):
        if isinstance(generator, list):
            noise = torch.cat([torch.randn((1, *sample.shape[1:]), generator = g) for g in generator])
        else:
            noise = torch.randn(sample.shape, generator = generator)
        out = 0.5 * sample + 0.5 * noise
        return (out,) if not return_dict else type("Out", (), {"prev_sample": out})()


class _HiDreamLikePipe:
    """Seeds the initial latents, then steps WITHOUT the generator, as diffusers HiDreamImagePipeline does."""

    def __init__(self, passes_generator = False):
        self.scheduler = _NoisyScheduler()
        self.passes_generator = passes_generator

    def __call__(self, generator):
        gens = generator if isinstance(generator, list) else [generator]
        latents = torch.cat([torch.randn((1, 4), generator = g) for g in gens])
        for t in range(3):
            extra = {"generator": generator} if self.passes_generator else {}
            latents = self.scheduler.step(None, t, latents, return_dict = False, **extra)[0]
        return latents


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _render(
    pipe,
    seeds,
    patched = True,
):
    gens = [_gen(s) for s in seeds] if len(seeds) > 1 else _gen(seeds[0])
    if not patched:
        return pipe(gens)
    with seeded_scheduler_step(pipe, gens):
        return pipe(gens)


def test_unpatched_step_noise_ignores_the_seed():
    """The defect: the same seed renders differently, because step draws from the global RNG."""
    pipe = _HiDreamLikePipe()
    torch.manual_seed(1)
    a = _render(pipe, [7], patched = False)
    torch.manual_seed(2)
    b = _render(pipe, [7], patched = False)
    assert not torch.equal(a, b)


def test_same_seed_renders_the_same_with_the_generator_handed_to_step():
    pipe = _HiDreamLikePipe()
    torch.manual_seed(1)
    a = _render(pipe, [7])
    torch.manual_seed(2)
    b = _render(pipe, [7])
    assert torch.equal(a, b)


def test_each_batch_row_matches_its_own_seeds_solo_render():
    pipe = _HiDreamLikePipe()
    batch = _render(pipe, [3, 9, 11])
    for row, seed in enumerate([3, 9, 11]):
        assert torch.equal(batch[row : row + 1], _render(pipe, [seed]))


def test_a_generator_the_pipeline_passes_is_kept():
    pipe = _HiDreamLikePipe(passes_generator = True)
    seen = []
    inner = pipe.scheduler.step

    def spy(*args, **kwargs):
        seen.append(kwargs.get("generator"))
        return inner(*args, **kwargs)

    pipe.scheduler.step = spy
    own = _gen(5)
    with seeded_scheduler_step(pipe, _gen(99)):
        pipe(own)
    assert seen and all(g is own for g in seen)


def test_a_generator_list_that_does_not_match_the_rows_is_not_handed_over():
    pipe = _HiDreamLikePipe()
    seen = []
    inner = pipe.scheduler.step

    def spy(*args, **kwargs):
        seen.append(kwargs.get("generator"))
        return inner(*args, **kwargs)

    pipe.scheduler.step = spy
    with seeded_scheduler_step(pipe, [_gen(1), _gen(2)]):
        pipe.scheduler.step(None, 0, torch.zeros(4, 4), return_dict = False)
    assert seen == [None]


def test_step_is_restored_after_the_call_and_after_an_error():
    pipe = _HiDreamLikePipe()
    original = pipe.scheduler.step
    with seeded_scheduler_step(pipe, _gen(1)):
        assert pipe.scheduler.step is not original
    assert "step" not in vars(pipe.scheduler)
    assert pipe.scheduler.step == original
    with pytest.raises(RuntimeError):
        with seeded_scheduler_step(pipe, _gen(1)):
            raise RuntimeError("denoise failed")
    assert "step" not in vars(pipe.scheduler)


def test_schedulers_without_a_generator_parameter_are_left_alone():
    class _Plain:
        def step(
            self,
            model_output,
            timestep,
            sample,
            return_dict = True,
        ):
            return (sample,)

    pipe = type("P", (), {})()
    pipe.scheduler = _Plain()
    with seeded_scheduler_step(pipe, _gen(1)):
        assert "step" not in vars(pipe.scheduler)
    with seeded_scheduler_step(type("NoScheduler", (), {})(), _gen(1)):
        pass

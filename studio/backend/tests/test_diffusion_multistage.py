# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Family-agnostic multi-call render helpers: one encode per request, and the denoiser held across calls under model
offload only when the work between the calls fits beside it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_multistage as MS


class _Pipe:
    def __init__(self):
        self.encodes = 0
        self.freed = 0
        self.resets = 0
        self._all_hooks = ["h"]
        self.components = {"transformer": SimpleNamespace(_reset_stateful_cache = self._reset)}

    def _reset(self):
        self.resets += 1

    def encode_prompt(self, **kw):
        self.encodes += 1
        return torch.full((1, 2), float(self.encodes)), None

    def maybe_free_model_hooks(self):
        self.freed += 1


def test_encode_once_shares_the_first_encode_and_isolates_in_place_edits():
    pipe = _Pipe()
    with MS.encode_once(pipe):
        a = pipe.encode_prompt(prompt = "p", device = "cpu")
        a[0].add_(5)
        b = pipe.encode_prompt(prompt = "p", device = "cpu")
        c = pipe.encode_prompt(prompt = "p", device = "cpu")
        other = pipe.encode_prompt(prompt = "q", device = "cpu")
        tensor_arg = pipe.encode_prompt(prompt = "p", prompt_embeds = torch.zeros(1))
    assert pipe.encodes == 3 and torch.equal(b[0], torch.ones(1, 2)) and c is b
    assert torch.equal(other[0], torch.full((1, 2), 2.0)) and torch.equal(tensor_arg[0], torch.full((1, 2), 3.0))
    assert "encode_prompt" not in pipe.__dict__
    # an instance attribute (a remote encoder, a cache wrapper) comes back as it was
    own = lambda **kw: (torch.zeros(1), None)  # noqa: E731
    pipe.encode_prompt = own
    with MS.encode_once(pipe):
        pipe.encode_prompt(prompt = "p")
    assert pipe.__dict__["encode_prompt"] is own


@pytest.mark.parametrize("fits", [True, False])
def test_hold_on_the_pipe_itself_skips_only_the_offload(monkeypatch, fits):
    monkeypatch.setattr(MS, "companions_fit_beside_denoiser", lambda pipe, companions: fits)
    pipe = _Pipe()
    with MS.hold_denoiser_between_calls(pipe, pipe, ()):
        pipe.maybe_free_model_hooks()  # what a diffusers call does at its end
    assert pipe.freed == (0 if fits else 1) and pipe.resets == (1 if fits else 0)
    assert "maybe_free_model_hooks" not in pipe.__dict__


@pytest.mark.parametrize("fits", [True, False])
def test_hold_with_a_derived_runner_frees_the_owner_only_when_it_does_not_fit(monkeypatch, fits):
    monkeypatch.setattr(MS, "companions_fit_beside_denoiser", lambda pipe, companions: fits)
    pipe = _Pipe()
    with MS.hold_denoiser_between_calls(pipe, object(), ()):
        pass
    assert pipe.freed == (0 if fits else 1)


def test_no_offload_hooks_means_no_check(monkeypatch):
    monkeypatch.setattr(MS, "companions_fit_beside_denoiser", lambda *a: pytest.fail("checked"))
    pipe = _Pipe()
    pipe._all_hooks = []
    with MS.hold_denoiser_between_calls(pipe, pipe, ()):
        pipe.maybe_free_model_hooks()
    assert pipe.freed == 1


def test_fit_check_declines_off_cuda_and_counts_module_bytes():
    pipe = SimpleNamespace(transformer = torch.nn.Linear(2, 2))
    assert MS.companions_fit_beside_denoiser(pipe, (torch.nn.Linear(4, 4),)) is False
    assert MS.module_bytes(torch.nn.Linear(4, 4)) == (16 + 4) * 4 and MS.module_bytes(None) == 0

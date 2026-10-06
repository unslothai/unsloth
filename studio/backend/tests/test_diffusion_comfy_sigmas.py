# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-family sigma schedules follow ComfyUI's at every resolution: Qwen-Image-2.1 and FLUX.1
dev / Krea-dev / Kontext sample at ComfyUI's fixed ModelSamplingFlux mu instead of the shipped
token-count mu (+ terminal stretch on 2.1). UNSLOTH_DIFFUSION_COMFY_SIGMAS=0 keeps the shipped one."""

from __future__ import annotations

import math
import types

import pytest

from core.inference.diffusion_families import comfy_flow_shift_for, detect_family
from core.inference.diffusion_flow_shift import (
    COMFY_SIGMAS_ENV,
    apply_comfy_flow_shift,
    flow_shift_overrides,
    flux_mu_shift,
)

# (width, height) the schedules are pinned at: off-square and both sides of 1024.
_SIZES = ((512, 512), (1024, 1024), (1536, 1536), (2048, 2048), (1664, 928))


class _Config(dict):
    pass


class _FakeScheduler:
    def __init__(self, **config):
        self.config = _Config(config)

    @classmethod
    def from_config(cls, config, **overrides):
        return cls(**{**config, **overrides})


# Shipped scheduler configs (the fields that matter).
_Q21 = dict(
    shift = 1.0,
    use_dynamic_shifting = True,
    base_shift = 0.5,
    max_shift = 0.9,
    base_image_seq_len = 256,
    max_image_seq_len = 8192,
    shift_terminal = 0.02,
    time_shift_type = "exponential",
)
_FLUX1_DEV = dict(
    shift = 3.0,
    use_dynamic_shifting = True,
    base_shift = 0.5,
    max_shift = 1.15,
    base_image_seq_len = 256,
    max_image_seq_len = 4096,
    shift_terminal = None,
    time_shift_type = "exponential",
)
_FLUX1_SCHNELL = dict(_FLUX1_DEV, shift = 1.0, use_dynamic_shifting = False)


def _comfy_flux_sigmas(mu, steps):
    """ComfyUI ModelSamplingFlux + 'simple' scheduler at a fixed mu (no image-size input)."""
    ts = [1.0 - i / steps for i in range(steps)]
    return [math.exp(mu) / (math.exp(mu) + (1 / t - 1)) for t in ts] + [0.0]


def _diffusers_sigmas(config, width, height, steps):
    """FlowMatchEulerDiscreteScheduler.set_timesteps for the Flux / Qwen pipelines' default sigmas:
    linspace(1, 1/N, N), token-count mu when dynamic, terminal stretch, then a trailing 0."""
    sig = [1.0 - i * (1.0 - 1.0 / steps) / (steps - 1) for i in range(steps)]
    if config.get("use_dynamic_shifting"):
        seq = (width // 16) * (height // 16)
        m = (config["max_shift"] - config["base_shift"]) / (
            config["max_image_seq_len"] - config["base_image_seq_len"]
        )
        mu = seq * m + config["base_shift"] - m * config["base_image_seq_len"]
        sig = [math.exp(mu) / (math.exp(mu) + (1 / s - 1)) for s in sig]
    else:
        k = config["shift"]
        sig = [k * s / (1 + (k - 1) * s) for s in sig]
    term = config.get("shift_terminal")
    if term:
        scale = (1 - sig[-1]) / (1 - term)
        sig = [1 - (1 - s) / scale for s in sig]
    return sig + [0.0]


def _studio_sigmas(shipped, shift, width, height, steps):
    pipe = types.SimpleNamespace(scheduler = _FakeScheduler(**shipped))
    apply_comfy_flow_shift(pipe, shift)
    return _diffusers_sigmas(pipe.scheduler.config, width, height, steps)


def _close(
    a,
    b,
    tol = 1e-9,
):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def test_flux_mu_shift_equals_comfy_flux_time_shift():
    # e^mu / (e^mu + 1/t - 1) == s*t / (1 + (s - 1)*t) with s = e^mu.
    for mu in (0.69, 1.15):
        s = flux_mu_shift(mu)
        for t in (0.05, 0.3, 0.5, 0.77, 1.0):
            assert s * t / (1 + (s - 1) * t) == pytest.approx(
                math.exp(mu) / (math.exp(mu) + (1 / t - 1)), abs = 1e-12
            )


def test_qwen_image_21_uses_comfy_fixed_mu_at_every_size(monkeypatch):
    monkeypatch.delenv(COMFY_SIGMAS_ENV, raising = False)
    fam = detect_family("Qwen/Qwen-Image-2.1")
    assert fam.name == "qwen-image-2.1"
    shift = comfy_flow_shift_for(fam, None, "Qwen/Qwen-Image-2.1", "Qwen/Qwen-Image-2.1")
    assert shift == pytest.approx(math.exp(0.69))
    comfy = _comfy_flux_sigmas(0.69, 25)
    for w, h in _SIZES:
        assert _close(_studio_sigmas(_Q21, shift, w, h, 25), comfy), (w, h)
    # No terminal stretch: the last nonzero sigma is ComfyUI's 0.0767, not 0.02.
    assert _studio_sigmas(_Q21, shift, 2048, 2048, 25)[-2] == pytest.approx(0.0767, abs = 1e-4)
    # The shipped schedule moves with size (mu 1.31 at 2048) and stretches to 0.02.
    shipped = _diffusers_sigmas(_Q21, 2048, 2048, 25)
    assert shipped[-2] == pytest.approx(0.02)
    assert max(abs(a - b) for a, b in zip(shipped, comfy)) > 0.1


def test_qwen_image_21_gguf_and_prequant_ids_resolve_the_same_shift():
    fam = detect_family("Qwen/Qwen-Image-2.1")
    for ids in (
        ("qwen-image-2.1-Q4_K_M.gguf", "unsloth/Qwen-Image-2.1-GGUF", "Qwen/Qwen-Image-2.1"),
        (None, "unsloth/Qwen-Image-2.1-FP8", "Qwen/Qwen-Image-2.1"),
    ):
        assert comfy_flow_shift_for(fam, *ids) == pytest.approx(math.exp(0.69))


@pytest.mark.parametrize(
    "ids",
    [
        (None, "black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-dev"),
        (None, "unsloth/FLUX.1-dev", "unsloth/FLUX.1-dev"),
        (None, "black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-Krea-dev"),
        (None, "unsloth/FLUX.1-Krea-dev-FP8", "black-forest-labs/FLUX.1-Krea-dev"),
        # A dev GGUF whose base fell back to the family's schnell repo: the file name decides.
        ("flux1-dev-Q4_K_M.gguf", "unsloth/FLUX.1-dev-GGUF", "black-forest-labs/FLUX.1-schnell"),
        (
            "flux1-krea-dev-Q8_0.gguf",
            "unsloth/FLUX.1-Krea-dev-GGUF",
            "black-forest-labs/FLUX.1-schnell",
        ),
        # The family's flux-1 alias, also as flux_1 (normalised to -).
        ("flux_1_dev-Q4_K_M.gguf", "someone/flux-1-models", "black-forest-labs/FLUX.1-schnell"),
        ("flux-1-krea-dev-Q8_0.gguf", "someone/flux-1-models", "black-forest-labs/FLUX.1-schnell"),
    ],
)
def test_flux1_dev_variants_use_comfy_fixed_mu(ids, monkeypatch):
    monkeypatch.delenv(COMFY_SIGMAS_ENV, raising = False)
    fam = detect_family(ids[1])
    assert fam.name == "flux.1"
    shift = comfy_flow_shift_for(fam, *ids)
    assert shift == pytest.approx(math.exp(1.15))
    comfy = _comfy_flux_sigmas(1.15, 20)
    for w, h in _SIZES:
        assert _close(_studio_sigmas(_FLUX1_DEV, shift, w, h, 20), comfy), (w, h)


def test_flux1_dev_1024_is_unchanged():
    # The shipped token-count mu is exactly 1.15 at 1024x1024 (4096 tokens): same schedule as before.
    fam = detect_family("black-forest-labs/FLUX.1-dev")
    shift = comfy_flow_shift_for(fam, None, "black-forest-labs/FLUX.1-dev")
    assert _close(
        _studio_sigmas(_FLUX1_DEV, shift, 1024, 1024, 20),
        _diffusers_sigmas(_FLUX1_DEV, 1024, 1024, 20),
        tol = 1e-12,
    )
    # ...and differs off 1024, where diffusers' mu follows the token count (3.23 at 2048).
    assert not _close(
        _studio_sigmas(_FLUX1_DEV, shift, 2048, 2048, 20),
        _diffusers_sigmas(_FLUX1_DEV, 2048, 2048, 20),
        tol = 1e-3,
    )


def test_local_path_containing_dev_keeps_the_shipped_schedule():
    fam = detect_family("black-forest-labs/FLUX.1-schnell")
    for path in (
        "/home/dev/models/flux",
        "/home/devon/flux-local",
        "D:\\dev\\krea\\flux",
        "C:\\Users\\krea-dev\\models\\flux",
    ):
        assert (
            comfy_flow_shift_for(fam, None, path, path, "black-forest-labs/FLUX.1-schnell") is None
        ), path


def test_local_path_containing_schnell_still_resolves_its_dev_base():
    fam = detect_family("black-forest-labs/FLUX.1-dev")
    path = "/models/schnell/flux.1-local"
    shift = comfy_flow_shift_for(fam, None, path, path, "black-forest-labs/FLUX.1-dev")
    assert shift == pytest.approx(math.exp(1.15))


@pytest.mark.parametrize(
    "ids",
    [
        (None, "black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-schnell"),
        (
            "flux1-schnell-Q4_K_M.gguf",
            "unsloth/FLUX.1-schnell-GGUF",
            "black-forest-labs/FLUX.1-schnell",
        ),
        (None, "unsloth/FLUX.1-schnell-NVFP4", "black-forest-labs/FLUX.1-schnell"),
    ],
)
def test_flux1_schnell_keeps_its_shipped_static_schedule(ids):
    fam = detect_family(ids[1])
    assert comfy_flow_shift_for(fam, *ids) is None
    pipe = types.SimpleNamespace(scheduler = _FakeScheduler(**_FLUX1_SCHNELL))
    assert apply_comfy_flow_shift(pipe, None) is False
    # ComfyUI FluxSchnell: ModelSamplingDiscreteFlow shift 1.0 = linear, same as shipped.
    assert _close(_diffusers_sigmas(_FLUX1_SCHNELL, 2048, 2048, 4), [1.0, 0.75, 0.5, 0.25, 0.0])


def test_flux1_kontext_uses_comfy_fixed_mu():
    fam = detect_family("black-forest-labs/FLUX.1-Kontext-dev")
    assert fam.name == "flux.1-kontext"
    shift = comfy_flow_shift_for(fam, None, "black-forest-labs/FLUX.1-Kontext-dev")
    assert shift == pytest.approx(math.exp(1.15))


def test_families_already_matching_comfy_are_unchanged():
    assert comfy_flow_shift_for(detect_family("Qwen/Qwen-Image"), "Qwen/Qwen-Image") == 3.1
    fam = detect_family("Qwen/Qwen-Image-Edit-2511")
    assert comfy_flow_shift_for(fam, None, "Qwen/Qwen-Image-Edit-2511") == 3.1
    assert comfy_flow_shift_for(fam, None, "Qwen/Qwen-Image-Edit-2509") == 3.0
    assert (
        comfy_flow_shift_for(detect_family("Qwen/Qwen-Image-Layered"), "Qwen/Qwen-Image-Layered")
        == 1.0
    )
    assert (
        comfy_flow_shift_for(detect_family("Tongyi-MAI/Z-Image-Turbo"), "Tongyi-MAI/Z-Image-Turbo")
        == 3.0
    )
    # FLUX.2 dev / klein: ComfyUI's Flux2Scheduler is the same token-count mu as diffusers, so the
    # shipped scheduler stays.
    for repo in ("black-forest-labs/FLUX.2-dev", "black-forest-labs/FLUX.2-klein-4B"):
        assert comfy_flow_shift_for(detect_family(repo), repo) is None


@pytest.mark.parametrize("value", ["0", "false", "off", "no", " 0 "])
def test_kill_switch_keeps_the_shipped_scheduler(value, monkeypatch):
    monkeypatch.setenv(COMFY_SIGMAS_ENV, value)
    for shipped, shift in (
        (_Q21, flux_mu_shift(0.69)),
        (_FLUX1_DEV, flux_mu_shift(1.15)),
        (_Q21, 3.1),
    ):
        sched = _FakeScheduler(**shipped)
        pipe = types.SimpleNamespace(scheduler = sched)
        assert apply_comfy_flow_shift(pipe, shift) is False
        assert pipe.scheduler is sched
        for w, h in _SIZES:
            assert _studio_sigmas(shipped, shift, w, h, 20) == _diffusers_sigmas(shipped, w, h, 20)


@pytest.mark.parametrize("value", [None, "", "1", "true"])
def test_kill_switch_default_and_on_apply(value, monkeypatch):
    if value is None:
        monkeypatch.delenv(COMFY_SIGMAS_ENV, raising = False)
    else:
        monkeypatch.setenv(COMFY_SIGMAS_ENV, value)
    pipe = types.SimpleNamespace(scheduler = _FakeScheduler(**_Q21))
    assert apply_comfy_flow_shift(pipe, flux_mu_shift(0.69)) is True
    assert pipe.scheduler.config["use_dynamic_shifting"] is False
    assert pipe.scheduler.config["shift_terminal"] is None
    assert flow_shift_overrides(_Config(_Q21), 2.0) == {
        "shift": 2.0,
        "use_dynamic_shifting": False,
        "shift_terminal": None,
    }


def test_real_scheduler_matches_comfy_when_diffusers_is_installed(monkeypatch):
    pytest.importorskip("torch")  # no-torch installs ship diffusers' dummy scheduler
    diffusers = pytest.importorskip("diffusers")
    monkeypatch.delenv(COMFY_SIGMAS_ENV, raising = False)
    for shipped, mu, steps in ((_Q21, 0.69, 25), (_FLUX1_DEV, 1.15, 20)):
        pipe = types.SimpleNamespace(
            scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(dict(shipped))
        )
        assert apply_comfy_flow_shift(pipe, flux_mu_shift(mu)) is True
        sig = [1.0 - i * (1.0 - 1.0 / steps) / (steps - 1) for i in range(steps)]
        # The pipelines still pass a token-count mu; a static scheduler ignores it.
        for w, h in _SIZES:
            mu_px = 0.5 + (w // 16) * (h // 16) / 8192
            pipe.scheduler.set_timesteps(sigmas = sig, device = "cpu", mu = mu_px)
            got = [float(x) for x in pipe.scheduler.sigmas]
            assert _close(got, _comfy_flux_sigmas(mu, steps), tol = 1e-6), (w, h)

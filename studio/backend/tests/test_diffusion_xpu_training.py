# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for Intel XPU training on the two flow-matching trainers (#9524).

These pin DECISIONS (device, dtype, which cache is cleared, what is recorded), never Intel kernels:
CI has no Intel GPU. Keep the fake ``torch.xpu`` a bare namespace -- a MagicMock manufactures
attributes, so every "this torch lacks that API" case would pass vacuously.
"""

from __future__ import annotations

import random
import types

import pytest
import torch

import core.training.diffusion_train_common as common
from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_h3_trainer as h3
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    effective_mixed_precision,
    native_bf16_supported_xpu,
    resolve_train_device,
)


def _fake_xpu(**attrs):
    return types.SimpleNamespace(**attrs)


@pytest.fixture
def host(monkeypatch):
    def _set(*, cuda: bool, xpu = None):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        if xpu is None:
            monkeypatch.delattr(torch, "xpu", raising = False)
        else:
            monkeypatch.setattr(torch, "xpu", xpu, raising = False)

    return _set


def test_cuda_still_wins_on_a_box_that_has_both(host):
    host(cuda = True, xpu = _fake_xpu(is_available = lambda: True))
    assert resolve_train_device() == "cuda"


def test_an_xpu_box_resolves_to_xpu_not_cpu(host):
    """#9524: the bnb quantizer already placed the transformer on xpu:0, so "cpu" here split them."""
    host(cuda = False, xpu = _fake_xpu(is_available = lambda: True))
    assert resolve_train_device() == "xpu"


@pytest.mark.parametrize(
    "xpu",
    [
        None,  # a torch build with no xpu module at all
        _fake_xpu(),  # present, but predates is_available()
        _fake_xpu(is_available = True),  # present, non-callable
        _fake_xpu(is_available = lambda: False),  # present, no device
    ],
    ids = ["absent", "no-probe", "non-callable", "unavailable"],
)
def test_a_host_without_a_usable_xpu_stays_on_cpu(host, xpu):
    host(cuda = False, xpu = xpu)
    assert resolve_train_device() == "cpu"


def test_an_uninitialised_xpu_driver_falls_through_instead_of_killing_the_run(host):
    """A raising probe must degrade to CPU, as dit_accelerator_missing_reason does: unguarded, an
    uninitialised driver crashed the run."""

    def _boom():
        raise RuntimeError("driver not initialised")

    host(cuda = False, xpu = _fake_xpu(is_available = _boom))
    assert resolve_train_device() == "cpu"


def test_an_emulation_only_xpu_is_refused(host):
    """is_bf16_supported() defaults including_emulation=True and short-circuits, so the bare call
    answers True for every available XPU: not a capability check."""
    host(
        cuda = False,
        xpu = _fake_xpu(
            is_available = lambda: True,
            is_bf16_supported = lambda including_emulation = True: bool(including_emulation),
        ),
    )
    assert torch.xpu.is_bf16_supported() is True  # what the bare call claims
    assert native_bf16_supported_xpu() is False  # what the hardware can actually do


def test_a_native_bf16_xpu_is_accepted(host):
    host(
        cuda = False,
        xpu = _fake_xpu(
            is_available = lambda: True,
            is_bf16_supported = lambda including_emulation = True: True,
        ),
    )
    assert native_bf16_supported_xpu() is True


def test_a_torch_predating_the_emulation_flag_still_answers(host):
    """On a torch predating including_emulation the no-argument answer is the only one there is."""
    host(cuda = False, xpu = _fake_xpu(is_available = lambda: True, is_bf16_supported = lambda: True))
    assert native_bf16_supported_xpu() is True


@pytest.mark.parametrize(
    "probe",
    [None, True],
    ids = ["absent", "non-callable"],
)
def test_an_xpu_without_a_usable_bf16_probe_is_refused(host, probe):
    attrs = {"is_available": lambda: True}
    if probe is not None:
        attrs["is_bf16_supported"] = probe
    host(cuda = False, xpu = _fake_xpu(**attrs))
    assert native_bf16_supported_xpu() is False


def _dit_cfg():
    return DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
        mixed_precision = "bf16",
    )


def _h3_cfg():
    return DiffusionLoraConfig(
        base_model = "MiniMaxAI/MiniMax-H3",
        data_dir = "/tmp/clips",
        output_dir = "/tmp/out",
        mixed_precision = "bf16",
        resolution = 768,
    )


class _Cut(Exception):
    pass


def _decide(monkeypatch, module, entry, cfg):
    """The (device, weight_dtype) the REAL entry point bound, read from its live frame.

    Cut at _assert_trusted_base_model, the statement right after weight_dtype: observes the decision
    itself rather than a copy of the rule, and nothing downstream (download, cache, training) runs.
    """

    def _cut(*_a, **_k):
        raise _Cut()

    monkeypatch.setattr(module, "_assert_trusted_base_model", _cut)
    # manual_seed fans out into torch.xpu.manual_seed_all, which the bare fake lacks; it runs
    # before the decision and cannot influence it.
    monkeypatch.setattr(torch, "manual_seed", lambda *_a, **_k: None)

    try:
        entry(cfg)
    except _Cut as exc:
        frame = None
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_name in ("run_dit_lora_training", "run_h3_lora_training"):
                frame = tb.tb_frame
            tb = tb.tb_next
        return frame.f_locals["device"], frame.f_locals["weight_dtype"]
    raise AssertionError("the run did not reach the cut")


@pytest.mark.parametrize(
    ("module", "entry", "cfg"),
    [(dit, dit.run_dit_lora_training, _dit_cfg()), (h3, h3.run_h3_lora_training, _h3_cfg())],
    ids = ["dit", "h3"],
)
def test_both_flow_trainers_run_an_xpu_box_in_bf16_on_the_xpu(
    monkeypatch, host, module, entry, cfg
):
    host(
        cuda = False,
        xpu = _fake_xpu(
            is_available = lambda: True,
            is_bf16_supported = lambda including_emulation = True: True,
        ),
    )
    assert _decide(monkeypatch, module, entry, cfg) == ("xpu", torch.bfloat16)


@pytest.mark.parametrize(
    ("module", "entry", "cfg"),
    [(dit, dit.run_dit_lora_training, _dit_cfg()), (h3, h3.run_h3_lora_training, _h3_cfg())],
    ids = ["dit", "h3"],
)
def test_both_flow_trainers_refuse_an_emulation_only_xpu(monkeypatch, host, module, entry, cfg):
    host(
        cuda = False,
        xpu = _fake_xpu(
            is_available = lambda: True,
            is_bf16_supported = lambda including_emulation = True: bool(including_emulation),
        ),
    )
    with pytest.raises(ValueError, match = "bfloat16-capable"):
        _decide(monkeypatch, module, entry, cfg)


@pytest.mark.parametrize("module", [dit, h3], ids = ["dit", "h3"])
def test_every_phase_boundary_frees_the_selected_accelerator(module):
    """Both boundaries free the text encoders and VAE right before the multi-GB transformer load,
    so a CUDA-only clear leaves that memory resident for it on an XPU box."""
    import inspect

    src = inspect.getsource(module)
    cuda_frees = src.count("torch.cuda.empty_cache()")
    xpu_frees = src.count("torch.xpu.empty_cache()")
    assert cuda_frees > 0
    assert xpu_frees == cuda_frees


def test_an_emulation_only_xpu_is_refused_before_the_route_evicts_anything(host):
    """The route calls this BEFORE teardown; dit_accelerator_missing_reason accepts any available
    XPU, so a child-only guard evicted the user's models and then failed."""
    from core.training.diffusion_train_common import bf16_unsupported_reason

    host(
        cuda = False,
        xpu = _fake_xpu(
            is_available = lambda: True,
            is_bf16_supported = lambda including_emulation = True: bool(including_emulation),
        ),
    )
    reason = bf16_unsupported_reason("krea-2")
    assert reason and "bf16 natively" in reason
    host(
        cuda = False,
        xpu = _fake_xpu(
            is_available = lambda: True,
            is_bf16_supported = lambda including_emulation = True: True,
        ),
    )
    assert bf16_unsupported_reason("krea-2") is None


def test_an_unprobeable_xpu_is_left_to_the_child_not_refused_up_front(host):
    """The preflight runs before teardown on a host whose XPU may not be interrogable at all, so an
    undeterminable capability must fail OPEN. Collapsing it to "unsupported" refused nf4 on every
    such host (caught by test_training_precision_preflight_error)."""
    from core.training.diffusion_train_common import (
        bf16_unsupported_reason,
        native_bf16_supported_xpu,
        xpu_native_bf16_probe,
    )

    def _unprobeable(including_emulation = True):
        raise RuntimeError("no device properties")

    host(cuda = False, xpu = _fake_xpu(is_available = lambda: True, is_bf16_supported = _unprobeable))
    assert xpu_native_bf16_probe() is None
    assert bf16_unsupported_reason("krea-2") is None   # route proceeds
    assert native_bf16_supported_xpu() is False        # child still refuses, on the device itself


def test_an_xpu_run_captures_and_restores_its_own_noise_generator(host, monkeypatch):
    """The loops draw randn_like on the training device, so an XPU run's per-step noise lives in the
    XPU generator; capturing only torch_cpu resumed it from a fresh seed."""
    from core.training.diffusion_checkpoint import capture_rng_state, restore_rng_state

    seen = {"set": []}
    states = [torch.tensor([7, 7, 7], dtype = torch.uint8)]
    fake = _fake_xpu(
        is_available = lambda: True,
        device_count = lambda: 1,
        get_rng_state_all = lambda: states,
        set_rng_state = lambda st, i: seen["set"].append((i, st.tolist())),
    )
    host(cuda = False, xpu = fake)

    captured = capture_rng_state()
    assert "torch_xpu_0" in captured["tensors"]

    restore_rng_state(captured["json"], captured["tensors"])
    assert seen["set"] == [(0, [7, 7, 7])]


def test_a_checkpoint_cannot_silently_resume_across_accelerator_backends(host, tmp_path):
    """Recording bf16 for XPU too (above) removed the accidental barrier that used to stop a
    CUDA checkpoint resuming on XPU: the precisions now match, the identity carries no backend, and
    the restore finds no torch_xpu_* key, so the device generator stays freshly seeded while the
    resume reports success. Refused at the preflight, before teardown."""
    from pathlib import Path

    from core.training.diffusion_checkpoint import (
        ResumeError,
        _assert_required_state,
        capture_rng_state,
    )

    complete = {
        "kind": "image",
        "sampler": {"pos": 0},
        "files": {k: f"{k}.pt" for k in ("adapter", "optimizer", "scheduler", "rng")},
    }
    written_here = capture_rng_state({"loop": random.Random(1), "variant": random.Random(2)})["json"]
    assert written_here["accelerator"] in ("cuda", "xpu", "cpu")

    # Same host: accepted.
    _assert_required_state(Path("ckpt"), {**complete, "rng": written_here})

    # Written on the other backend: refused, and the message says which.
    other = "xpu" if written_here["accelerator"] != "xpu" else "cuda"
    with pytest.raises(ResumeError, match = "this accelerator"):
        _assert_required_state(
            Path("ckpt"), {**complete, "rng": {**written_here, "accelerator": other}}
        )

    # A bundle predating the field must still resume: unknown is not a mismatch.
    legacy = {k: v for k, v in written_here.items() if k != "accelerator"}
    _assert_required_state(Path("ckpt"), {**complete, "rng": legacy})


def test_an_xpu_flow_run_records_the_bf16_it_actually_trains_in(host):
    """identity_for_config stores this and a resume is refused when it disagrees, so a CUDA-only
    rule would record "no" for a run the loop trained in bf16."""
    cfg = _dit_cfg().normalized()
    host(cuda = False, xpu = _fake_xpu(is_available = lambda: True))
    assert effective_mixed_precision(cfg) == "bf16"
    host(cuda = False, xpu = None)
    assert effective_mixed_precision(cfg) == "no"
    assert common.resolve_train_device() == "cpu"

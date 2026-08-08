"""Regression test for unslothai/unsloth#4631: xformers must not be blanket-disabled
on sm_120 GPUs where its kernel actually runs (a ~57% attention-memory saving over the
SDPA packed-mask fallback). The gate probes the real op instead of guessing by the
compute-capability major version.

Also covers NVIDIA QA P0-1: the probe used to be skipped entirely below sm_120, on the
assumption that xformers always works there. It does not when the wheel was built for a
different torch or CUDA major, and skipping the probe is what let a cu128-built managed
Windows package ship beside a cu130 runtime with its kernels silently dead."""

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.utils import attention_dispatch as ad

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "capability, probe_result, expect_disabled",
    [
        # #4631: on sm_120 the answer must come from the real op, both ways.
        ((12, 0), True, False),  # sm_120 where the kernel runs: keep xformers
        ((12, 0), False, True),  # sm_120 where the kernel can't run: fall back to SDPA
        # Every other capability is now probed too, and a working kernel is still kept:
        # this is the half of #4631 that must not regress into "probe means disable".
        ((7, 5), True, False),  # Turing
        ((8, 0), True, False),  # Ampere
        ((8, 9), True, False),  # Ada
        ((9, 0), True, False),  # Hopper
        ((10, 0), True, False),  # Blackwell B200 (sm_100)
        # P0-1: a mismatched build is dead on these too, and must be caught.
        ((8, 9), False, True),
        ((9, 0), False, True),
        ((10, 0), False, True),
    ],
)
def test_capability_gate(capability, probe_result, expect_disabled):
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return probe_result

    assert ad._xformers_disabled_for_capability(capability, probe = probe) is expect_disabled
    # Exactly once, on every capability: the decision is the kernel's to make, and it is
    # one tiny forward, so there is no reason to run it twice or to skip it.
    assert calls["n"] == 1


@pytest.mark.skipif(
    not (torch.cuda.is_available() and ad.HAS_XFORMERS),
    reason = "needs a CUDA GPU with a working xformers build",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12,
    reason = "on real sm_120+ the probe legitimately returns False when the build ships no "
    "sm_120 kernel, so asserting True there would be a false failure",
)
def test_probe_shapes_are_valid_on_working_gpu():
    # Guards against a malformed probe that raises on every GPU and would silently
    # disable xformers on Blackwell even where it works. On a pre-sm_120 GPU with a
    # functional xformers the real probe must succeed; sm_120+ is skipped above because
    # there a False is a correct answer, not a malformed probe.
    assert ad._xformers_runs_on_device() is True


@pytest.mark.parametrize(
    "supports_bf16, expected_dtype",
    [(True, torch.bfloat16), (False, torch.float16)],
)
def test_probe_dtype_follows_bf16_support(monkeypatch, supports_bf16, expected_dtype):
    # Pre-Ampere GPUs (sm < 80: Turing/Volta, e.g. T4/V100) run xformers fine in
    # float16 but have no bfloat16 attention kernel, so a hardcoded bf16 probe would
    # raise there, get swallowed to False, and misreport a working xformers as broken.
    # The probe must pick its dtype from SUPPORTS_BFLOAT16 (no Turing GPU needed here).
    captured = {}

    def fake_zeros(
        *args,
        dtype = None,
        **kwargs,
    ):
        captured["dtype"] = dtype
        raise RuntimeError("stop after capturing the probe dtype")

    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", supports_bf16)
    monkeypatch.setattr(ad.torch, "zeros", fake_zeros)
    ad._xformers_runs_on_device()  # RuntimeError is swallowed; only the dtype matters
    assert captured["dtype"] is expected_dtype


def test_probe_syncs_and_fails_on_deferred_async_error(monkeypatch):
    # A CUDA kernel launch is async: xformers_attention can return before the GPU
    # reports a failure. The probe must synchronize so a deferred launch/runtime error
    # is caught and disables xformers here, instead of surfacing later on an unrelated
    # CUDA call (unslothai/unsloth#6828 review). No GPU needed: everything is stubbed.
    _bias = type(
        "B",
        (),
        {
            "BlockDiagonalCausalMask": type(
                "M", (), {"from_seqlens": staticmethod(lambda seqlens: None)}
            )
        },
    )
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", True)
    monkeypatch.setattr(ad.torch, "zeros", lambda *a, **k: object())
    monkeypatch.setattr(ad, "xformers", type("X", (), {"attn_bias": _bias}))
    monkeypatch.setattr(ad, "xformers_attention", lambda *a, **k: None)  # "succeeds"

    def deferred_cuda_error():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(ad.torch.cuda, "synchronize", deferred_cuda_error)
    # Without the synchronize the stubbed op returns cleanly and the probe wrongly
    # reports True; the sync surfaces the deferred error so the probe returns False.
    assert ad._xformers_runs_on_device() is False


def test_probe_caches_the_failure_reason(monkeypatch):
    # "xformers is off" with no reason is what made the mismatched Windows build so hard
    # to diagnose. A failed probe must leave something a report can print.
    monkeypatch.setattr(ad, "XFORMERS_PROBE_REASON", None)
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", True)

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA error: no kernel image is available for execution")

    monkeypatch.setattr(ad.torch, "zeros", boom)
    assert ad._xformers_runs_on_device() is False
    assert "no kernel image is available" in ad.XFORMERS_PROBE_REASON
    assert ad.XFORMERS_PROBE_REASON.startswith("RuntimeError: ")


def test_probe_clears_a_stale_reason_on_success(monkeypatch):
    # A later success must not leave the previous failure's reason behind, or a healthy
    # install reports itself broken.
    monkeypatch.setattr(ad, "XFORMERS_PROBE_REASON", "RuntimeError: stale")
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", True)
    monkeypatch.setattr(ad.torch, "zeros", lambda *a, **k: object())
    monkeypatch.setattr(
        ad,
        "xformers",
        type(
            "X",
            (),
            {
                "attn_bias": type(
                    "B",
                    (),
                    {
                        "BlockDiagonalCausalMask": type(
                            "M", (), {"from_seqlens": staticmethod(lambda seqlens: None)}
                        )
                    },
                )
            },
        ),
    )
    monkeypatch.setattr(ad, "xformers_attention", lambda *a, **k: None)
    monkeypatch.setattr(ad.torch.cuda, "synchronize", lambda *a, **k: None)
    assert ad._xformers_runs_on_device() is True
    assert ad.XFORMERS_PROBE_REASON is None


def test_probe_never_raises_even_when_xformers_is_none(monkeypatch):
    # Patch the globals the probe writes through, or its `global` assignment leaks the
    # failure into every later test in the session.
    monkeypatch.setattr(ad, "XFORMERS_PROBE_REASON", None)
    monkeypatch.setattr(ad, "XFORMERS_PROBE_INCONCLUSIVE", False)
    # The probe runs at import time on every CUDA capability now, so anything it touches
    # being missing or broken must degrade to False, never to an ImportError at `import
    # unsloth`.
    monkeypatch.setattr(ad, "xformers", None)
    monkeypatch.setattr(ad, "xformers_attention", None)
    assert ad._xformers_runs_on_device() is False
    assert ad.XFORMERS_PROBE_REASON


@pytest.mark.parametrize(
    "message, inconclusive",
    [
        ("CUDA out of memory. Tried to allocate 2.00 GiB", True),
        ("CUDA error: all CUDA-capable devices are busy or unavailable", True),
        ("CUDA error: no kernel image is available for execution on the device", False),
        ("undefined symbol: _ZN3c105ErrorC1E", False),
    ],
)
def test_a_busy_or_full_gpu_does_not_count_as_a_broken_build(monkeypatch, message, inconclusive):
    # Device 0 being full, or claimed by another rank under EXCLUSIVE_PROCESS, says
    # nothing about the wheel. Turning memory-efficient attention off for the whole
    # process on that basis is a silent 2x memory regression caused by the probe itself.
    monkeypatch.setattr(ad, "XFORMERS_PROBE_REASON", None)
    monkeypatch.setattr(ad, "XFORMERS_PROBE_INCONCLUSIVE", False)
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", True)

    def boom(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(ad.torch, "zeros", boom)
    assert ad._xformers_runs_on_device() is False
    assert ad.XFORMERS_PROBE_INCONCLUSIVE is inconclusive


def test_the_probe_targets_this_rank_s_device(monkeypatch):
    # Under torchrun each rank owns a different GPU, and on a mixed box device 0 is often
    # the small display card. Probing 0 for everyone lets a wheel with no kernel for the
    # weakest GPU disable xformers on the good ones.
    captured = {}

    def fake_zeros(
        *args,
        device = None,
        **kwargs,
    ):
        captured["device"] = device
        raise RuntimeError("stop after capturing the device")

    # torch.cuda.device is stubbed so this runs on a host with any number of GPUs, and so
    # the index the context is entered with can be asserted directly.
    @contextlib.contextmanager
    def fake_device(index):
        captured["context"] = index
        yield

    monkeypatch.setattr(ad, "XFORMERS_PROBE_REASON", None)
    monkeypatch.setattr(ad, "XFORMERS_PROBE_INCONCLUSIVE", False)
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", True)
    monkeypatch.setattr(ad, "_PROBE_DEVICE_INDEX", 3)
    monkeypatch.setattr(ad.torch.cuda, "device", fake_device)
    monkeypatch.setattr(ad.torch, "zeros", fake_zeros)
    ad._xformers_runs_on_device()
    assert captured["device"] == "cuda:3"
    # The attn_bias must land there too. BlockDiagonalCausalMask builds its seqstart
    # tensors on the CURRENT device, so without this context q went to cuda:N and the bias
    # to cuda:0, and xformers rejected the pair -- every rank but 0 lost xformers on a
    # healthy install, which is the exact silent downgrade this gate exists to prevent.
    assert captured["context"] == 3


def test_the_probe_device_is_clamped_to_a_device_that_exists():
    """LOCAL_RANK is a rank, not an index into the visible devices.

    Slurm with --gpus-per-task=1, and anything that narrows CUDA_VISIBLE_DEVICES per rank,
    gives a rank one visible device while still exporting its global rank. accelerate and
    transformers also use -1 for "not distributed". torch raises on an invalid ordinal and
    the capability read is at module scope, so an unclamped index makes `import unsloth`
    itself crash.
    """
    visible = ad.torch.cuda.device_count() if ad.torch.cuda.is_available() else 0
    assert 0 <= ad._PROBE_DEVICE_INDEX < max(visible, 1)
    if visible:
        # Would raise "Invalid device id" for an out-of-range index.
        ad.torch.cuda.get_device_capability(ad._PROBE_DEVICE_INDEX)


@pytest.mark.parametrize("local_rank", ["3", "-1", "07", "abc", ""])
def test_import_survives_a_local_rank_that_names_no_visible_device(local_rank):
    """The regression test the mocked one above cannot be: a real import, real env."""
    if not (ad.torch.cuda.is_available() and ad.torch.cuda.device_count() >= 1):
        pytest.skip("needs at least one CUDA device")
    env = {
        **os.environ,
        "LOCAL_RANK": local_rank,
        # One visible device, so any rank above 0 is out of range.
        "CUDA_VISIBLE_DEVICES": (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import unsloth;"
            "from unsloth.utils import attention_dispatch as a;"
            "print('IDX', a._PROBE_DEVICE_INDEX)",
        ],
        capture_output = True,
        text = True,
        timeout = 600,
        env = env,
        cwd = str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"import unsloth crashed with LOCAL_RANK={local_rank!r}:\n{result.stderr[-2000:]}"
    )
    assert "IDX 0" in result.stdout, result.stdout[-2000:]

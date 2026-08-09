# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    # `capability` is documentation now: the gate reads the kernel, not the number. It used
    # to be read at the call site, which is what put an unguarded CUDA query at module scope.
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return probe_result

    assert ad._xformers_disabled(probe = probe) is expect_disabled
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
    "capability, expected_dtype",
    [((8, 0), torch.bfloat16), ((7, 5), torch.float16)],
)
def test_probe_dtype_follows_the_probed_device(monkeypatch, capability, expected_dtype):
    # Pre-Ampere GPUs (sm < 80: Turing/Volta, e.g. T4/V100) run xformers fine in
    # float16 but have no bfloat16 attention kernel, so a hardcoded bf16 probe would
    # raise there, get swallowed to False, and misreport a working xformers as broken.
    #
    # Read off THE DEVICE BEING PROBED, not the module-level SUPPORTS_BFLOAT16, which
    # describes device 0. On a mixed box where 0 is Ampere-or-newer and LOCAL_RANK selects
    # a Turing card, the global says bf16 and this rank writes off a healthy install.
    captured = {}

    def fake_zeros(
        *args,
        dtype = None,
        **kwargs,
    ):
        captured["dtype"] = dtype
        raise RuntimeError("stop after capturing the probe dtype")

    # The opposite of the device's own answer, so a dtype taken from here fails the test.
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", capability[0] < 8)
    monkeypatch.setattr(ad.torch.cuda, "get_device_capability", lambda index = None: capability)
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
    assert (
        result.returncode == 0
    ), f"import unsloth crashed with LOCAL_RANK={local_rank!r}:\n{result.stderr[-2000:]}"
    assert "IDX 0" in result.stdout, result.stdout[-2000:]


def test_no_unguarded_cuda_query_runs_at_import():
    """A CUDA capability query is not safe to run at module scope.

    The driver refuses it when the device is busy, in exclusive-compute mode, or otherwise
    unavailable, and at module scope that is `import unsloth` raising -- over a diagnostic
    whose worst outcome is "keep xformers on and let the real forward decide". Every such
    read has to sit inside a function that can answer "unknown"."""
    import ast

    tree = ast.parse((_REPO_ROOT / "unsloth" / "utils" / "attention_dispatch.py").read_text())
    offenders = []
    for node in tree.body:
        # Module scope only. A call inside a def runs when something calls it, and every
        # such caller here is already guarded.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Attribute) and child.func.attr in (
                "get_device_capability",
                "get_device_properties",
            ):
                offenders.append(ast.unparse(child))
    assert offenders == [], f"unguarded CUDA query at import: {offenders}"


def test_an_unavailable_device_leaves_the_fp32_answer_unknown(monkeypatch):
    # Same query, same refusal. It must degrade to "cannot tell" rather than propagate.
    monkeypatch.setattr(ad.torch.cuda, "is_available", lambda: True)

    def _refuse(index = None):
        raise RuntimeError("CUDA error: device is currently in use by another process")

    monkeypatch.setattr(ad.torch.cuda, "get_device_capability", _refuse)
    assert ad._probe_device_major() is None


def test_the_model_code_reads_the_probed_verdict_not_the_bare_import():
    """`HAS_XFORMERS = xformers is not None` recomputed in llama.py ignored the probe.

    Mistral answers "xFormers is on" by skipping the 4D sliding-window mask and letting the
    xFormers bias carry the window. With the dispatcher already fallen back to SDPA, that
    mask is the only thing making the window local, so every sequence longer than
    config.sliding_window silently attended to the whole causal history."""
    import ast

    tree = ast.parse((_REPO_ROOT / "unsloth" / "models" / "llama.py").read_text())
    assigned = [
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "HAS_XFORMERS"
    ]
    assert assigned == [], "llama.py recomputes HAS_XFORMERS instead of taking the probed one"
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("attention_dispatch")
        for alias in node.names
    ]
    assert "HAS_XFORMERS" in imported

    from unsloth.models import llama
    from unsloth.utils import attention_dispatch

    assert llama.HAS_XFORMERS is attention_dispatch.HAS_XFORMERS


def test_an_exclusive_mode_refusal_keeps_xformers_on(monkeypatch):
    """ "device is currently in use by another process" is the driver saying the GPU is
    someone else's right now, not that the wheel is broken. Recorded as conclusive, it turned
    a transient into a process-wide 2x memory regression -- caused by the diagnostic."""
    monkeypatch.setattr(
        ad.torch.cuda,
        "get_device_capability",
        lambda index = None: (_ for _ in ()).throw(
            RuntimeError("CUDA error: device is currently in use by another process")
        ),
    )
    assert ad._xformers_runs_on_device() is False
    assert (
        ad.XFORMERS_PROBE_INCONCLUSIVE is True
    ), "an inconclusive failure must leave xformers enabled for the real forward to decide"


def test_the_probed_device_follows_the_caller_selection(monkeypatch):
    """A single-process app that calls torch.cuda.set_device(1) before importing us has said
    where its work goes. Probing 0 anyway can disable xformers over a card nothing will
    touch, and creates a context on it to do so. LOCAL_RANK still wins when it is usable."""
    monkeypatch.setattr(ad.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ad.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(ad.torch.cuda, "current_device", lambda: 1)

    monkeypatch.delenv("LOCAL_RANK", raising = False)
    assert ad._resolve_probe_device_index() == 1

    monkeypatch.setenv("LOCAL_RANK", "3")
    assert ad._resolve_probe_device_index() == 3

    # Out of range, the not-distributed sentinel, and junk all fall back to the selection.
    for value in ("9", "-1", "not a rank", ""):
        monkeypatch.setenv("LOCAL_RANK", value)
        assert ad._resolve_probe_device_index() == 1

    # And a current_device that raises still lands on 0 rather than propagating.
    monkeypatch.setattr(
        ad.torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(RuntimeError("no context")),
    )
    assert ad._resolve_probe_device_index() == 0


def test_the_inconclusive_branch_can_actually_read_the_logging_flag():
    """The inconclusive arm prints behind UNSLOTH_ENABLE_LOGGING at MODULE scope.

    attention_dispatch pulls _utils in with `import *`, and UNSLOTH_ENABLE_LOGGING is not in
    _utils.__all__, so the name only exists here because it is imported explicitly. Without
    that import a busy or out-of-memory GPU -- newly classified as inconclusive -- raises
    NameError during `import unsloth` instead of keeping xformers on.
    """
    from unsloth.models import _utils

    assert "UNSLOTH_ENABLE_LOGGING" not in getattr(
        _utils, "__all__", ()
    ), "if the flag is exported, this explicit import can go -- but not before"
    assert hasattr(
        ad, "UNSLOTH_ENABLE_LOGGING"
    ), "the inconclusive branch reads this name at import time"

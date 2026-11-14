# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``safetensors_pread.py``: on Windows a media load reads transformers' safetensors shards with the ``pread`` backend
instead of torch's copy-on-write file mapping (#9977), and every other platform keeps the byte-identical mmap path.
CPU only."""

from __future__ import annotations

import inspect
import sys
import threading

import pytest
import torch

import core.inference.safetensors_pread as sp

transformers = pytest.importorskip("transformers")
import transformers.modeling_utils as modeling_utils  # noqa: E402

# transformers < 5.6 passes no backend (None), later releases pass "mmap" off Windows.
MMAP = {None, "mmap"}

requires_backend = pytest.mark.skipif(
    not sp._supports_backend(modeling_utils.safe_open),
    reason = "installed safetensors has no backend keyword",
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(sp.PREAD_ENV, raising = False)
    yield
    assert sp._depth == 0 and sp._original is None and not sp._active_rules()


def _record_backends(monkeypatch):
    """Spy under the patch: the backend keyword each transformers safe_open call carried (None = default mmap)."""
    seen = []
    real = modeling_utils.safe_open

    def spy(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return real(*args, **kwargs)

    spy.__text_signature__ = real.__text_signature__
    monkeypatch.setattr(modeling_utils, "safe_open", spy)
    return seen


@pytest.mark.parametrize(
    "platform, env, wanted",
    [
        ("win32", None, True),
        ("linux", None, False),
        ("darwin", None, False),
        ("win32", "0", False),
        ("linux", "1", True),
    ],
)
def test_pread_wanted(monkeypatch, platform, env, wanted):
    monkeypatch.setattr(sys, "platform", platform)
    if env is not None:
        monkeypatch.setenv(sp.PREAD_ENV, env)
    assert sp.pread_wanted() is wanted


def test_off_windows_nothing_is_patched(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    before = modeling_utils.safe_open
    with sp.pread_safetensors() as engaged:
        assert engaged is False
        assert modeling_utils.safe_open is before


def test_old_safetensors_without_backend_is_left_alone(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    def old_safe_open(
        filename,
        framework,
        device = "cpu",
    ):
        raise AssertionError("not called")

    monkeypatch.setattr(modeling_utils, "safe_open", old_safe_open)
    with sp.pread_safetensors() as engaged:
        assert engaged is False
        assert modeling_utils.safe_open is old_safe_open


def _save_llama(path, dtype):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).to(dtype).save_pretrained(path, max_shard_size = "20KB")
    assert len(list(path.glob("*.safetensors"))) > 1
    return LlamaForCausalLM


@requires_backend
def test_windows_fp32_load_reads_with_pread_and_matches_mmap(monkeypatch, tmp_path):
    """The reported path: an fp32 checkpoint loaded at bf16 through transformers' from_pretrained."""
    model_cls = _save_llama(tmp_path, torch.float32)
    seen = _record_backends(monkeypatch)
    dense = model_cls.from_pretrained(tmp_path, dtype = torch.bfloat16)
    assert seen and set(seen) <= MMAP

    monkeypatch.setenv(sp.PREAD_ENV, "1")
    seen.clear()
    spy = modeling_utils.safe_open
    with sp.pread_safetensors() as engaged:
        assert engaged is True
        assert modeling_utils.safe_open.__wrapped__ is spy
        loaded = model_cls.from_pretrained(tmp_path, dtype = torch.bfloat16)
    assert seen and set(seen) == {"pread"}
    assert modeling_utils.safe_open is spy

    expected = dense.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor, expected[name]), name


@requires_backend
def test_windows_bf16_shards_keep_mmap(monkeypatch, tmp_path):
    """A bf16 shard's mapping is the loaded weight, so it stays on the default backend."""
    model_cls = _save_llama(tmp_path, torch.bfloat16)
    monkeypatch.setenv(sp.PREAD_ENV, "1")
    seen = _record_backends(monkeypatch)
    with sp.pread_safetensors() as engaged:
        assert engaged is True
        model_cls.from_pretrained(tmp_path, dtype = torch.bfloat16)
    assert seen and set(seen) <= MMAP


@requires_backend
@pytest.mark.parametrize(
    "dtype, passed, expected",
    [
        (torch.float32, None, "pread"),
        (torch.float32, "mmap", "pread"),
        (torch.float32, "pread", "pread"),
        (torch.bfloat16, None, None),
        (torch.bfloat16, "mmap", "mmap"),
    ],
)
def test_backend_choice(monkeypatch, tmp_path, dtype, passed, expected):
    from safetensors.torch import save_file

    path = tmp_path / "shard.safetensors"
    save_file({"w": torch.zeros(64, 64, dtype = dtype)}, str(path))
    monkeypatch.setattr(sys, "platform", "win32")
    seen = _record_backends(monkeypatch)
    kwargs = {} if passed is None else {"backend": passed}
    with sp.pread_safetensors():
        with modeling_utils.safe_open(str(path), framework = "pt", device = "cpu", **kwargs) as handle:
            assert torch.equal(handle.get_slice("w")[...], torch.zeros(64, 64, dtype = dtype))
    assert seen == [expected]


@requires_backend
def test_nested_and_concurrent_loads_restore_once(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    original = modeling_utils.safe_open
    inside = threading.Event()
    release = threading.Event()

    def other_load():
        with sp.pread_safetensors() as engaged:
            assert engaged
            inside.set()
            release.wait(5)

    worker = threading.Thread(target = other_load)
    with sp.pread_safetensors() as outer:
        assert outer
        patched = modeling_utils.safe_open
        worker.start()
        assert inside.wait(5)
        with sp.pread_safetensors() as inner:
            assert inner and modeling_utils.safe_open is patched
        assert modeling_utils.safe_open is patched
    assert modeling_utils.safe_open is patched
    release.set()
    worker.join(5)
    assert modeling_utils.safe_open is original


def test_video_load_runs_inside_the_scope():
    from core.inference.video import VideoBackend

    source = inspect.getsource(VideoBackend._run_load)
    scope = source.index("with pread_safetensors():")
    assert scope < source.index("self.load_pipeline(**kwargs)", scope)


@requires_backend
@pytest.mark.parametrize(
    "stored, convert_to, keep, expected",
    [
        (torch.bfloat16, torch.float16, (), "pread"),
        (torch.bfloat16, torch.float32, (), "pread"),
        (torch.float16, torch.float16, (), None),
        (torch.bfloat16, torch.bfloat16, (), None),
        # the small-host route loads this component at its stored dtype: its mapping is the weight
        (torch.bfloat16, torch.float16, ("text_encoder",), None),
        (torch.int8, torch.float16, (), None),
        (torch.bfloat16, None, (), None),
        # fp32 loaded at fp32 (CPU fallback) is the loaded weight; fp32 converted down reads through pread
        (torch.float32, torch.float32, (), None),
        (torch.float32, torch.bfloat16, (), "pread"),
        # the video scope (no target dtype) keeps its fp32-only rule
        (torch.float32, None, (), "pread"),
    ],
)
def test_converted_shard_backend_choice(monkeypatch, tmp_path, stored, convert_to, keep, expected):
    from safetensors.torch import save_file

    comp = tmp_path / "text_encoder"
    comp.mkdir()
    path = comp / "model.safetensors"
    save_file({"w": torch.zeros(64, 64, dtype = stored)}, str(path))
    monkeypatch.setattr(sys, "platform", "win32")
    seen = _record_backends(monkeypatch)
    with sp.pread_safetensors(convert_to = convert_to, keep = keep):
        with modeling_utils.safe_open(str(path), framework = "pt", device = "cpu") as handle:
            handle.get_slice("w")[...]
    assert seen == [expected]
    assert not sp._active_rules()


@requires_backend
def test_converting_bf16_load_reads_with_pread_and_matches_mmap(monkeypatch, tmp_path):
    """#14513: a bf16 encoder loaded at fp16 (fp16-only GPU) reads through pread, weights unchanged."""
    model_cls = _save_llama(tmp_path, torch.bfloat16)
    seen = _record_backends(monkeypatch)
    dense = model_cls.from_pretrained(tmp_path, dtype = torch.float16)
    assert seen and set(seen) <= MMAP

    monkeypatch.setenv(sp.PREAD_ENV, "1")
    seen.clear()
    with sp.pread_safetensors(convert_to = torch.float16) as engaged:
        assert engaged is True
        loaded = model_cls.from_pretrained(tmp_path, dtype = torch.float16)
    assert seen and set(seen) == {"pread"}

    expected = dense.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == torch.float16
        assert torch.equal(tensor, expected[name]), name


_COMMIT_ARM = r"""
import ctypes, sys, torch
sys.path.insert(0, sys.argv[1])
import core.inference.safetensors_pread as sp
from transformers import LlamaForCausalLM

class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
        "PeakPagefileUsage")]

k = ctypes.WinDLL("kernel32", use_last_error = True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]

def counters():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    assert k.K32GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb), ctypes.get_last_error()
    return c

base = counters().PagefileUsage
if sys.argv[3] == "pread":
    with sp.pread_safetensors(convert_to = torch.float16):
        LlamaForCausalLM.from_pretrained(sys.argv[2], dtype = torch.float16)
else:
    LlamaForCausalLM.from_pretrained(sys.argv[2], dtype = torch.float16)
print("PEAK_MIB", (counters().PeakPagefileUsage - base) >> 20)
"""


@pytest.mark.skipif(sys.platform != "win32", reason = "process commit is a Windows counter")
@requires_backend
def test_windows_converting_load_commits_about_one_copy(tmp_path):
    """Windows bf16 load as fp16: mmap commits stored bytes plus the fp16 copy; pread does not."""
    import subprocess
    import warnings
    from pathlib import Path

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 32000,
        hidden_size = 2048,
        intermediate_size = 5632,
        num_hidden_layers = 12,
        num_attention_heads = 16,
        num_key_value_heads = 16,
    )
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model = model.to_empty(device = "cpu").to(torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    model.save_pretrained(tmp_path, max_shard_size = "400MB")
    stored_mib = sum(f.stat().st_size for f in tmp_path.glob("*.safetensors")) >> 20
    del model

    backend = str(Path(sp.__file__).resolve().parents[2])
    peaks = {}
    for arm in ("mmap", "pread"):
        out = subprocess.run(
            [sys.executable, "-c", _COMMIT_ARM, backend, str(tmp_path), arm],
            capture_output = True,
            text = True,
            timeout = 900,
        )
        assert out.returncode == 0, out.stderr[-3000:]
        peaks[arm] = int(
            next(l for l in out.stdout.splitlines() if l.startswith("PEAK_MIB")).split()[1]
        )
    warnings.warn(f"windows commit for a {stored_mib} MiB bf16 -> fp16 load: {peaks}")
    # windows-latest, 1426 MiB: mmap 3477, pread 2077 (both carry ~0.6 GB of load-time allocations besides the
    # fp16 copy); pread drops the copy-on-write charge for the stored bytes
    assert peaks["mmap"] - peaks["pread"] >= stored_mib * 0.85, peaks


@requires_backend
@pytest.mark.parametrize(
    "stored, convert_to, keep, sent",
    [
        (torch.bfloat16, torch.float16, (), None),  # converted: this scope asks for pread itself
        (
            torch.bfloat16,
            torch.bfloat16,
            (),
            True,
        ),  # same dtype: mmap, unsloth's pread patch told to keep it
        (
            torch.bfloat16,
            torch.float16,
            ("text_encoder",),
            True,
        ),  # small-host route keeps the encoder mapped
    ],
)
def test_unsloth_windows_pread_patch_below_keeps_this_scopes_mmap_choices(
    monkeypatch, tmp_path, stored, convert_to, keep, sent
):
    """unsloth (transformers 5.0 - 5.16 on Windows) wraps safe_open to default every CPU open to pread; inside a media
    load this scope's mmap choices win via ``_unsloth_keep_mmap``."""
    import functools

    from safetensors.torch import save_file

    comp = tmp_path / "text_encoder"
    comp.mkdir()
    path = comp / "model.safetensors"
    save_file({"w": torch.zeros(64, 64, dtype = stored)}, str(path))
    received = []
    real = modeling_utils.safe_open

    @functools.wraps(real)
    def core_patch(*args, **kwargs):
        received.append(kwargs.pop("_unsloth_keep_mmap", None))
        return real(*args, **kwargs)

    core_patch._unsloth_windows_pread = True
    monkeypatch.setattr(modeling_utils, "safe_open", core_patch)
    monkeypatch.setattr(sys, "platform", "win32")
    with sp.pread_safetensors(convert_to = convert_to, keep = keep):
        with modeling_utils.safe_open(str(path), framework = "pt", device = "cpu") as handle:
            handle.get_slice("w")[...]
    assert received == [sent]


@requires_backend
@pytest.mark.parametrize("passed, expected", [("pread", "mmap"), (None, None), ("mmap", "mmap")])
def test_kept_shard_overrides_a_callers_pread(monkeypatch, tmp_path, passed, expected):
    """transformers >= 5.17 passes backend="pread" on Windows itself; a shard the small-host route keeps stays mapped."""
    from safetensors.torch import save_file

    comp = tmp_path / "text_encoder"
    comp.mkdir()
    path = comp / "model.safetensors"
    save_file({"w": torch.zeros(64, 64, dtype = torch.bfloat16)}, str(path))
    monkeypatch.setattr(sys, "platform", "win32")
    seen = _record_backends(monkeypatch)
    kwargs = {} if passed is None else {"backend": passed}
    with sp.pread_safetensors(convert_to = torch.float16, keep = ("text_encoder",)):
        with modeling_utils.safe_open(str(path), framework = "pt", device = "cpu", **kwargs) as handle:
            handle.get_slice("w")[...]
    assert seen == [expected]


@requires_backend
def test_concurrent_loads_keep_their_own_rules(monkeypatch, tmp_path):
    """An image load's keep / convert rules never reach a video load running in another thread (multi-model policy)."""
    from safetensors.torch import save_file

    image, video = tmp_path / "image" / "text_encoder", tmp_path / "video" / "text_encoder"
    for d, dtype in ((image, torch.bfloat16), (video, torch.float32)):
        d.mkdir(parents = True)
        save_file({"w": torch.zeros(64, 64, dtype = dtype)}, str(d / "model.safetensors"))
    monkeypatch.setattr(sys, "platform", "win32")
    seen = {}
    real = modeling_utils.safe_open

    def spy(filename, *args, **kwargs):
        seen[str(filename)] = kwargs.get("backend")
        return real(filename, *args, **kwargs)

    spy.__text_signature__ = real.__text_signature__
    monkeypatch.setattr(modeling_utils, "safe_open", spy)
    opened, release = threading.Event(), threading.Event()

    def video_load():
        with sp.pread_safetensors():
            with modeling_utils.safe_open(
                str(video / "model.safetensors"), framework = "pt", device = "cpu"
            ) as h:
                h.get_slice("w")[...]
            opened.set()
            release.wait(5)

    with sp.pread_safetensors(convert_to = torch.float16, keep = ("text_encoder",)):
        worker = threading.Thread(target = video_load)
        worker.start()
        assert opened.wait(5)
        with modeling_utils.safe_open(
            str(image / "model.safetensors"), framework = "pt", device = "cpu"
        ) as h:
            h.get_slice("w")[...]
        release.set()
        worker.join(5)
    assert (
        seen[str(video / "model.safetensors")] == "pread"
    )  # its fp32 encoder, despite the image's keep entry
    assert seen[str(image / "model.safetensors")] is None  # the image's routed encoder stays mapped


@requires_backend
def test_a_thread_outside_any_scope_keeps_unsloths_default(monkeypatch, tmp_path):
    """While a video scope holds the process-wide wrapper, an eager image-side load in another thread (HiDream's Llama)
    gets no keep flag, so unsloth's Windows pread patch still applies there."""
    import functools

    from safetensors.torch import save_file

    path = tmp_path / "text_encoder_4" / "model.safetensors"
    path.parent.mkdir()
    save_file({"w": torch.zeros(64, 64, dtype = torch.bfloat16)}, str(path))
    received = []
    real = modeling_utils.safe_open

    @functools.wraps(real)
    def core_patch(*args, **kwargs):
        received.append(kwargs.pop("_unsloth_keep_mmap", None))
        return real(*args, **kwargs)

    core_patch._unsloth_windows_pread = True
    monkeypatch.setattr(modeling_utils, "safe_open", core_patch)
    monkeypatch.setattr(sys, "platform", "win32")
    entered, release = threading.Event(), threading.Event()

    def video_load():
        with sp.pread_safetensors():
            entered.set()
            release.wait(5)

    worker = threading.Thread(target = video_load)
    worker.start()
    assert entered.wait(5)
    try:
        with modeling_utils.safe_open(str(path), framework = "pt", device = "cpu") as handle:
            handle.get_slice("w")[...]
    finally:
        release.set()
        worker.join(5)
    assert received == [None]

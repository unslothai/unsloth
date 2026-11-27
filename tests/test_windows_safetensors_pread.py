# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Windows safetensors ``pread`` backport (transformers 5.0 - 5.16): gating, the backend each shard open gets, weight
identity, the keep-mmap escape, and on Windows the real process commit of a converting load."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import transformers.modeling_utils as modeling_utils  # noqa: E402

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "unsloth" / "models" / "_windows_safetensors.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("windows_safetensors_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ws = _load_module()
requires_backend = pytest.mark.skipif(
    not ws._supports_backend(), reason = "safetensors < 0.8 has no backend keyword"
)


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    import transformers.modeling_layers as modeling_layers

    original = modeling_utils.safe_open
    original_layers = getattr(modeling_layers, "safe_open", None)
    monkeypatch.delenv(ws.PREAD_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_UMA", raising = False)
    monkeypatch.setattr(ws, "_UMA", None)
    yield
    modeling_utils.safe_open = original
    if original_layers is not None:
        modeling_layers.safe_open = original_layers


def _spy(monkeypatch):
    seen = []
    real = modeling_utils.safe_open

    def spy(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return real(*args, **kwargs)

    monkeypatch.setattr(modeling_utils, "safe_open", spy)
    return seen


@pytest.mark.parametrize(
    "platform, machine, env, version, installed",
    [
        ("win32", "AMD64", None, "5.5.0", True),
        ("win32", "AMD64", None, "5.16.0", True),
        ("win32", "AMD64", None, "5.17.0", False),
        ("win32", "AMD64", None, "4.57.6", False),
        ("win32", "AMD64", "0", "5.5.0", False),
        ("linux", "x86_64", None, "5.5.0", False),
        ("darwin", "x86_64", None, "5.5.0", False),
        ("linux", "x86_64", "1", "5.17.0", True),
        # ARM hosts may be unified memory (GB10, Jetson): wrapped, decided per open
        ("linux", "aarch64", None, "4.57.6", True),
        ("linux", "aarch64", None, "5.17.0", True),
        ("win32", "ARM64", None, "5.17.0", True),
        ("linux", "aarch64", "0", "5.5.0", False),
        # Apple Silicon has no CUDA: nothing to decide per open
        ("darwin", "arm64", None, "5.5.0", False),
    ],
)
@requires_backend
def test_gating(monkeypatch, platform, machine, env, version, installed):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(ws.platform, "machine", lambda: machine)
    monkeypatch.setattr(transformers, "__version__", version)
    if env is not None:
        monkeypatch.setenv(ws.PREAD_ENV, env)
    before = modeling_utils.safe_open
    assert ws.patch_windows_safetensors_pread() is installed
    assert (modeling_utils.safe_open is not before) is installed


@requires_backend
def test_idempotent_and_found_under_another_wrapper(monkeypatch):
    import functools

    monkeypatch.setenv(ws.PREAD_ENV, "1")
    assert ws.patch_windows_safetensors_pread()
    patched = modeling_utils.safe_open
    assert ws.patch_windows_safetensors_pread() and modeling_utils.safe_open is patched

    @functools.wraps(patched)
    def outer(*args, **kwargs):  # e.g. the UMA clone wrapper installed on top
        return patched(*args, **kwargs)

    modeling_utils.safe_open = outer
    assert ws.patch_windows_safetensors_pread() and modeling_utils.safe_open is outer


@requires_backend
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"device": "cpu"}, "pread"),
        ({}, "pread"),
        ({"device": "cpu", "backend": "mmap"}, "pread"),
        ({"device": "cpu", "backend": "pread"}, "pread"),
        ({"device": "cpu", ws.KEEP_MMAP_KWARG: True}, None),
        ({"device": "cuda:0"}, None),
    ],
)
def test_backend_per_open(monkeypatch, kwargs, expected):
    received = []

    def real(
        filename,
        framework = None,
        **kw,
    ):
        received.append(kw)

    monkeypatch.setattr(modeling_utils, "safe_open", real)
    monkeypatch.setenv(ws.PREAD_ENV, "1")
    assert ws.patch_windows_safetensors_pread()
    modeling_utils.safe_open("shard.safetensors", framework = "pt", **kwargs)
    assert received[0].get("backend") == expected
    assert ws.KEEP_MMAP_KWARG not in received[0]


def _save_llama(path, dtype):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).to(dtype).save_pretrained(path, max_shard_size = "20KB")
    assert len(list(path.glob("*.safetensors"))) > 1
    return LlamaForCausalLM


@requires_backend
@pytest.mark.parametrize("load_dtype", [torch.bfloat16, torch.float16])
def test_from_pretrained_reads_with_pread_and_weights_match(monkeypatch, tmp_path, load_dtype):
    model_cls = _save_llama(tmp_path, torch.bfloat16)
    expected = model_cls.from_pretrained(tmp_path, dtype = load_dtype).state_dict()
    seen = _spy(monkeypatch)
    monkeypatch.setenv(ws.PREAD_ENV, "1")
    assert ws.patch_windows_safetensors_pread()
    loaded = model_cls.from_pretrained(tmp_path, dtype = load_dtype)
    assert seen and set(seen) == {"pread"}
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == load_dtype
        assert torch.equal(tensor, expected[name]), name


_COMMIT_ARM = r"""
import ctypes, importlib.util, os, sys, torch
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

spec = importlib.util.spec_from_file_location("ws", sys.argv[1]); ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)
os.environ[ws.PREAD_ENV] = "1" if sys.argv[3] == "pread" else "0"
print("PATCHED", ws.patch_windows_safetensors_pread())
base = counters().PagefileUsage
LlamaForCausalLM.from_pretrained(sys.argv[2], dtype = torch.float16)
print("PEAK_MIB", (counters().PeakPagefileUsage - base) >> 20)
"""


@pytest.mark.skipif(sys.platform != "win32", reason = "process commit is a Windows counter")
@pytest.mark.skipif(
    not ws._transformers_needs_it(transformers.__version__),
    reason = "transformers >= 5.17 already reads with pread on Windows",
)
@requires_backend
def test_windows_converting_load_commits_about_one_copy(tmp_path):
    """bf16 -> fp16 load on Windows: mmap also commits the stored bytes, pread does not."""
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

    peaks = {}
    for arm in ("mmap", "pread"):
        out = subprocess.run(
            [sys.executable, "-c", _COMMIT_ARM, str(_MODULE_PATH), str(tmp_path), arm],
            capture_output = True,
            text = True,
            timeout = 900,
        )
        assert out.returncode == 0, out.stderr[-3000:]
        assert f"PATCHED {arm == 'pread'}" in out.stdout, out.stdout
        peaks[arm] = int(
            next(l for l in out.stdout.splitlines() if l.startswith("PEAK_MIB")).split()[1]
        )
    warnings.warn(f"windows commit for a {stored_mib} MiB bf16 -> fp16 load: {peaks}")
    assert peaks["mmap"] - peaks["pread"] >= stored_mib * 0.85, peaks


@requires_backend
def test_the_mtp_loaders_own_reference_is_wrapped(monkeypatch):
    # transformers 5.16's MtpModel loader imports safe_open into modeling_layers and opens its files through it
    import transformers.modeling_layers as modeling_layers

    received = []

    def real(
        filename,
        framework = None,
        **kw,
    ):
        received.append(kw)

    monkeypatch.setattr(modeling_layers, "safe_open", real, raising = False)
    monkeypatch.setenv(ws.PREAD_ENV, "1")
    assert ws.patch_windows_safetensors_pread()
    assert getattr(modeling_layers.safe_open, "_unsloth_windows_pread", False)
    modeling_layers.safe_open("mtp.safetensors", framework = "pt", device = "cpu")
    assert received[0].get("backend") == "pread"
    assert ws.patch_windows_safetensors_pread() and modeling_layers.safe_open.__wrapped__ is real


def _recording_real(monkeypatch):
    received = []

    def real(
        filename,
        framework = None,
        **kw,
    ):
        received.append(kw)

    monkeypatch.setattr(modeling_utils, "safe_open", real)
    return received


@requires_backend
@pytest.mark.parametrize("uma, expected", [("1", "pread"), ("0", None)])
def test_unified_memory_is_decided_per_open(monkeypatch, uma, expected):
    received = _recording_real(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ws.platform, "machine", lambda: "aarch64")
    assert ws.patch_windows_safetensors_pread()
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", uma)
    modeling_utils.safe_open("shard.safetensors", framework = "pt", device = "cpu")
    modeling_utils.safe_open("shard.safetensors", framework = "pt", device = "cpu", backend = "mmap")
    modeling_utils.safe_open(
        "shard.safetensors", framework = "pt", device = "cpu", **{ws.KEEP_MMAP_KWARG: True}
    )
    assert [r.get("backend") for r in received] == [
        expected,
        expected if uma == "1" else "mmap",
        None,
    ]


def test_unified_memory_check_never_initialises_cuda(monkeypatch):
    if torch.cuda.is_initialized():
        pytest.skip("CUDA already initialised in this process")
    assert ws._unified_memory_now() is False
    assert not torch.cuda.is_initialized()
    assert ws._UMA is None  # not cached: re-checked once a context exists


@pytest.mark.parametrize(
    "platform, env, uma, supported, device, expected",
    [
        ("linux", None, "0", True, "cpu", {}),
        ("linux", None, "1", True, "cpu", {"backend": "pread"}),
        ("linux", None, "1", False, "cpu", {}),
        ("linux", None, "1", True, "cuda:0", {}),
        ("linux", "0", "1", True, "cpu", {}),
        ("linux", "1", "0", True, "cpu", {"backend": "pread"}),
        # discrete Windows keeps mmap for Unsloth's own loaders (pread is slower there)
        ("win32", None, "0", True, "cpu", {}),
        ("win32", None, "1", True, "cpu", {"backend": "pread"}),
        ("win32", "0", "1", True, "cpu", {}),
    ],
)
def test_pread_backend_kwargs(monkeypatch, platform, env, uma, supported, device, expected):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", uma)
    if env is not None:
        monkeypatch.setenv(ws.PREAD_ENV, env)
    monkeypatch.setattr(ws, "_supports_backend", lambda: supported)
    assert ws.pread_backend_kwargs(device) == expected


# safetensors < 0.8.0's extension class, as far as detection can see
_OldSafeOpen = types.SimpleNamespace(__text_signature__ = "(filename, framework, device=...)")


def test_safetensors_without_backend_keeps_mmap(monkeypatch):
    """safetensors < 0.8.0: no backend keyword, so nothing is wrapped and the helper adds nothing."""
    import safetensors._safetensors_rust as rust

    monkeypatch.setattr(rust, "safe_open", _OldSafeOpen)
    ws._supports_backend.cache_clear()
    try:
        assert ws._supports_backend() is False
        monkeypatch.setenv(ws.PREAD_ENV, "1")
        assert ws.pread_backend_kwargs() == {}
        before = modeling_utils.safe_open
        assert ws.patch_windows_safetensors_pread() is False
        assert modeling_utils.safe_open is before
    finally:
        ws._supports_backend.cache_clear()


def test_backend_detection_sees_through_a_wrapped_safe_open(monkeypatch, tmp_path):
    """Another library wrapping safetensors.safe_open as (*args, **kwargs) must not hide the keyword."""
    import safetensors
    from safetensors.torch import save_file

    real = safetensors.safe_open
    monkeypatch.setattr(safetensors, "safe_open", lambda *a, **k: real(*a, **k))
    ws._supports_backend.cache_clear()
    path = str(tmp_path / "x.safetensors")
    save_file({"w": torch.arange(4, dtype = torch.float32)}, path)
    try:
        real(path, framework = "pt", device = "cpu", backend = "pread")
        accepts = True
    except TypeError:
        accepts = False
    try:
        assert ws._supports_backend() is accepts
    finally:
        ws._supports_backend.cache_clear()


@requires_backend
def test_env_off_after_install_stops_pread(monkeypatch):
    received = _recording_real(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ws.platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", "1")
    assert ws.patch_windows_safetensors_pread()
    monkeypatch.setenv(ws.PREAD_ENV, "0")
    modeling_utils.safe_open("shard.safetensors", framework = "pt", device = "cpu")
    assert received[0].get("backend") is None


@requires_backend
def test_unified_memory_from_pretrained_reads_with_pread_and_weights_match(monkeypatch, tmp_path):
    model_cls = _save_llama(tmp_path, torch.bfloat16)
    expected = model_cls.from_pretrained(tmp_path, dtype = torch.bfloat16).state_dict()
    seen = _spy(monkeypatch)
    monkeypatch.setattr(
        sys, "platform", "linux"
    )  # an ARM Linux box (GB10), also on macOS / Windows runners
    monkeypatch.setattr(ws.platform, "machine", lambda: "aarch64")
    assert ws.patch_windows_safetensors_pread()
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", "1")
    loaded = model_cls.from_pretrained(tmp_path, dtype = torch.bfloat16)
    assert seen and set(seen) == {"pread"}
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor.view(torch.uint8), expected[name].view(torch.uint8)), name

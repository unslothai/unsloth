# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for attention-backend selection. No torch/diffusers needed:
``_is_cuda_nvidia`` is monkeypatched for the policy tests, and the apply path uses a fake
transformer that records / raises on ``set_attention_backend``.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import (
    ATTN_AUTO,
    apply_attention_backend,
    normalize_attention_backend,
    select_attention_backend,
)


def _target(device = "cuda"):
    return types.SimpleNamespace(device = device)


# ── normalize ────────────────────────────────────────────────────────────────────
def test_normalize_defaults_and_aliases():
    assert normalize_attention_backend(None) == ATTN_AUTO
    assert normalize_attention_backend("") == ATTN_AUTO
    assert normalize_attention_backend("auto") == ATTN_AUTO
    assert normalize_attention_backend("CuDNN") == "cudnn"
    assert normalize_attention_backend("FLASH3") == "flash3"
    assert normalize_attention_backend("sdpa") == "sdpa"


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_attention_backend("bogus")
    # dashes are no longer silently rewritten to underscores -> a dashed alias is rejected.
    with pytest.raises(ValueError):
        normalize_attention_backend("flash-3")


def test_sdpa_alias_maps_to_native():
    # sdpa is an alias for native -> nothing to set on the dispatcher.
    assert select_attention_backend(_target(), "sdpa", speed_active = True) is None


# ── select policy ─────────────────────────────────────────────────────────────────
def test_auto_upgrades_to_cudnn_on_nvidia_when_speed_active(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 0))  # Ampere+: cuDNN ok
    assert select_attention_backend(_target(), "auto", speed_active = True) == "_native_cudnn"


def test_auto_does_not_pin_cudnn_below_sm80(monkeypatch):
    # cuDNN fused SDPA fails at run time on pre-SM80 (T4 SM75 / V100 SM70); auto must stay
    # on the native default there rather than pin a backend that crashes on first generation.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (7, 5))  # Turing T4
    assert select_attention_backend(_target(), "auto", speed_active = True) is None


def test_auto_stays_native_when_speed_off(monkeypatch):
    # off must stay bit-identical -> no backend change even on NVIDIA.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    assert select_attention_backend(_target(), "auto", speed_active = False) is None


def test_auto_stays_native_off_nvidia(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(device = "mps"), "auto", speed_active = True) is None


def test_explicit_backend_honored_regardless_of_speed(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    # Pin a high capability so the arch-gated flash4 isn't dropped by the runtime check.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    assert select_attention_backend(_target(), "sage", speed_active = False) == "sage"
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"
    assert select_attention_backend(_target(), "cudnn", speed_active = False) == "_native_cudnn"


def test_explicit_backend_dropped_off_nvidia_cuda(monkeypatch):
    # Explicit cuDNN/flash/sage on ROCm / MPS / CPU passes diffusers' set-time check
    # and crashes at the first generation, so selection drops to the native default.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    for alias in ("sage", "flash", "flash4", "cudnn"):
        assert select_attention_backend(_target(device = "mps"), alias, speed_active = True) is None


def test_explicit_native_returns_none():
    # native is the default -> nothing to set.
    assert select_attention_backend(_target(), "native", speed_active = True) is None


# ── arch gating (flash3/flash4 need a specific CUDA capability) ─────────────────────
def test_flash3_dropped_below_hopper(monkeypatch):
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 9))  # Ada / consumer
    assert select_attention_backend(_target(), "flash3", speed_active = False) is None


def test_flash4_dropped_below_blackwell(monkeypatch):
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))  # Hopper, but FA4 needs SM100
    assert select_attention_backend(_target(), "flash4", speed_active = False) is None
    # flash3 still allowed on Hopper.
    assert select_attention_backend(_target(), "flash3", speed_active = False) == "_flash_3_hub"


def test_arch_gate_does_not_block_when_capability_unknown(monkeypatch):
    # Unknown capability (e.g. no CUDA) must not block -> diffusers' set-time check still guards.
    monkeypatch.setattr(att, "_cuda_capability", lambda: None)
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"


def test_flash3_dropped_on_blackwell(monkeypatch):
    # FlashAttention 3 is a Hopper-SM90 rewrite with no Blackwell kernel: an explicit
    # flash3 on a B200 (SM100) must drop to native rather than set fine then crash.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    assert select_attention_backend(_target(), "flash3", speed_active = False) is None
    # FA4 is still honored on Blackwell.
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"
    # flash3 is allowed exactly on Hopper SM90.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    assert select_attention_backend(_target(), "flash3", speed_active = False) == "_flash_3_hub"


def test_explicit_cudnn_dropped_below_sm80(monkeypatch):
    # An explicit cuDNN request on pre-Ampere (T4 SM75 / V100 SM70) must drop to native,
    # not set fine and crash at first generation -- the same gate the auto path applies.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (7, 5))
    assert select_attention_backend(_target(), "cudnn", speed_active = False) is None
    # Ampere+ still honors it.
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 0))
    assert select_attention_backend(_target(), "cudnn", speed_active = False) == "_native_cudnn"


# ── apply ─────────────────────────────────────────────────────────────────────────
class _FakeTransformer:
    def __init__(self, *, fail = False):
        self.fail = fail
        self.set_to = None

    def set_attention_backend(self, name):
        if self.fail:
            raise RuntimeError(f"{name} kernel unavailable")
        self.set_to = name


def _pipe(transformer):
    return types.SimpleNamespace(transformer = transformer)


def test_apply_none_leaves_native_when_global_already_native(monkeypatch):
    # Global already native -> no redundant set call, returns None.
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer()
    assert apply_attention_backend(_pipe(t), None) is None
    assert t.set_to is None


def test_apply_none_restores_native_when_global_polluted(monkeypatch):
    # A previous load pinned cuDNN process-wide; a native load must reset it so it can't
    # silently inherit cuDNN (the bit-identical/off guarantee).
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "_native_cudnn")
    t = _FakeTransformer()
    assert apply_attention_backend(_pipe(t), None) is None
    assert t.set_to == "native"


def test_apply_sets_backend():
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"


def test_apply_sets_backend_on_both_dits():
    # A dual-DiT family (Ideogram) runs transformer + unconditional_transformer each step, so the
    # backend must be set on BOTH; otherwise the second DiT keeps the native default while status
    # reports the requested kernel as engaged.
    t1, t2 = _FakeTransformer(), _FakeTransformer()
    pipe = types.SimpleNamespace(transformer = t1, unconditional_transformer = t2)
    engaged = apply_attention_backend(pipe, "_native_cudnn")
    assert engaged == "_native_cudnn"
    assert t1.set_to == "_native_cudnn" and t2.set_to == "_native_cudnn"


def test_apply_falls_back_on_unavailable_kernel(monkeypatch):
    # an unavailable kernel must not fail the load -> returns None (diffusers default).
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer(fail = True)
    assert apply_attention_backend(_pipe(t), "sage") is None


def test_apply_failed_kernel_restores_native_when_polluted(monkeypatch):
    # Requested kernel fails AND the global is polluted: restore native before returning.
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "_native_cudnn")

    class _FailOnceTransformer:
        def __init__(self):
            self.calls = []

        def set_attention_backend(self, name):
            self.calls.append(name)
            if name != "native":
                raise RuntimeError(f"{name} kernel unavailable")

    t = _FailOnceTransformer()
    assert apply_attention_backend(_pipe(t), "sage") is None
    assert t.calls == ["sage", "native"]


def test_apply_handles_missing_method():
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert apply_attention_backend(pipe, "_native_cudnn") is None


def test_apply_resets_global_registry_after_success(monkeypatch):
    # After a successful per-transformer set, the process-wide registry must be reset to
    # native so a later component (unconfigured processors) can't inherit this kernel --
    # while the transformer's own backend stays the engaged one.
    called = {"reset": False}
    monkeypatch.setattr(
        att, "_reset_global_backend_to_native", lambda logger: called.__setitem__("reset", True)
    )
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"
    assert called["reset"] is True


def test_active_attention_backend_reads_tuple_return():
    # get_active_backend() returns a (AttentionBackendName, fn) tuple; the helper must read
    # the name's .value, not stringify the tuple (which never compares equal to a name).
    pytest.importorskip("diffusers")
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    _AttentionBackendRegistry.set_active_backend(AttentionBackendName.NATIVE)
    assert att._active_attention_backend() == "native"


# ── on-demand wheel-only install of optional kernels ─────────────────────────────
@pytest.fixture(autouse = True)
def _no_real_installs(monkeypatch):
    # Unit tests must never shell out to pip: the apply path probes installable
    # backends (sage/flash*), so hard-disable the gate; install tests re-enable it
    # with a stubbed subprocess.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "0")
    # The install once-per-process memo is module state; clear it so each test starts
    # with a fresh "not yet attempted" set (otherwise an earlier test's attempt would
    # make a later install a no-op).
    att._INSTALL_ATTEMPTED.clear()


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return types.SimpleNamespace(returncode = 0)


def _stub_subprocess(monkeypatch, run):
    import subprocess
    monkeypatch.setattr(subprocess, "run", run)


def test_install_skipped_when_gate_disabled(monkeypatch):
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert run.calls == []


def test_install_skipped_when_module_present(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "sageattention" else None
    )
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert run.calls == []


def test_install_runs_wheel_only_for_missing_kernel(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("sage")
    assert len(run.calls) == 1
    cmd = run.calls[0]
    assert "--only-binary" in cmd and ":all:" in cmd and "sageattention" in cmd


def test_install_uses_no_deps_to_protect_core_deps(monkeypatch):
    # A kernel add-on (xformers/flash-attn) pins an exact torch, so a normal install would
    # upgrade/replace the running torch/triton. --no-deps installs only the kernel wheel;
    # an ABI-incompatible one fails to import and falls back to native rather than clobbering
    # the environment's core deps.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("xformers")
    assert len(run.calls) == 1
    assert "--no-deps" in run.calls[0]


def test_failed_install_not_retried_in_same_process(monkeypatch):
    # The loader pre-installs the kernel OUTSIDE its locks and then re-resolves the same
    # backend under _generate_lock; if the pre-install failed (no wheel / offline) the
    # in-lock apply path must NOT re-run pip (a second up-to-600s install holding the load
    # lock blocks unload/cancel). The once-per-process memo makes the retry a no-op.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)  # stays missing

    calls: list[list[str]] = []

    def _boom(cmd, **kwargs):
        calls.append(list(cmd))
        raise sp.CalledProcessError(returncode = 1, cmd = cmd)

    _stub_subprocess(monkeypatch, _boom)
    att._ensure_attention_backend_installed("sage")  # pre-install attempt (outside lock)
    att._ensure_attention_backend_installed("sage")  # in-lock retry -> must be skipped
    assert len(calls) == 1


def test_install_invalidates_import_caches_on_success(monkeypatch):
    # A wheel written to site-packages after the finder cached that directory can be
    # missed by the very next import, so a successful install must invalidate the caches
    # (otherwise set_attention_backend imports the missing package and falls back).
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    _stub_subprocess(monkeypatch, _Recorder())
    invalidated = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: invalidated.append(True))
    att._ensure_attention_backend_installed("sage")
    assert invalidated == [True]


def test_install_failure_skips_cache_invalidation(monkeypatch):
    # A failed install left nothing to import, so the finder caches must be left alone.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def _boom(cmd, **kwargs):
        raise sp.CalledProcessError(returncode = 1, cmd = cmd)

    _stub_subprocess(monkeypatch, _boom)
    invalidated = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: invalidated.append(True))
    att._ensure_attention_backend_installed("sage")
    assert invalidated == []


def test_install_never_attempted_for_builtin_backends(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    run = _Recorder()
    _stub_subprocess(monkeypatch, run)
    att._ensure_attention_backend_installed("_native_cudnn")
    att._ensure_attention_backend_installed("native")
    assert run.calls == []


def test_install_failure_logs_pip_stderr(monkeypatch):
    # A CalledProcessError's str() hides the pip reason; the warning must surface the
    # captured stderr (decoding bytes) so a fallback to native is diagnosable.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def _boom(cmd, **kwargs):
        raise sp.CalledProcessError(
            returncode = 1, cmd = cmd, stderr = b"ERROR: No matching distribution found"
        )

    _stub_subprocess(monkeypatch, _boom)

    warnings: list[str] = []

    class _Logger:
        def info(self, *a, **k):
            pass

        def warning(self, msg, *args):
            warnings.append(msg % args if args else msg)

    att._ensure_attention_backend_installed("sage", _Logger())
    assert warnings and "No matching distribution found" in warnings[-1]


def test_install_failure_falls_back_to_native(monkeypatch):
    # pip failing (no wheel for this platform) must not break the load: the apply
    # path proceeds, set_attention_backend raises on the missing package, and the
    # dispatcher is restored to native -- same contract as before the hook.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "auto")
    import importlib.util
    import subprocess as sp

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def _boom(cmd, **kwargs):
        raise sp.CalledProcessError(returncode = 1, cmd = cmd)

    _stub_subprocess(monkeypatch, _boom)
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    t = _FakeTransformer(fail = True)
    assert apply_attention_backend(_pipe(t), "sage") is None


# ── HunyuanVideo-1.5 padded-text attention trim ─────────────────────────────────────
# _trim_stream / _hunyuan_trim_pre_hook use real torch tensor ops, so these run on CPU torch.
import torch  # noqa: E402


def test_trim_stream_drops_trailing_padding():
    # right-padded (valid prefix): drop the globally-invalid tail, keep valid, flag all_valid.
    states = torch.arange(6.0).reshape(1, 6, 1)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (1, 3, 1)
    assert torch.equal(out_s[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert out_m.shape == (1, 3) and all_valid is True


def test_trim_stream_layout_agnostic_drops_only_global_padding():
    # left-padded (valid suffix): any(dim=0) keeps positions valid for at least one element,
    # so the leading globally-invalid columns are dropped regardless of padding side.
    states = torch.arange(4.0).reshape(1, 4, 1)
    mask = torch.tensor([[0, 0, 1, 1]])
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert torch.equal(out_s[0, :, 0], torch.tensor([2.0, 3.0])) and all_valid is True


def test_trim_stream_full_mask_is_noop():
    states = torch.ones(1, 4, 2)
    mask = torch.ones(1, 4, dtype=torch.long)
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (1, 4, 2) and all_valid is True


def test_trim_stream_none_mask_passthrough():
    states = torch.ones(1, 4, 2)
    out_s, out_m, all_valid = att._trim_stream(states, None)
    assert out_s is states and out_m is None and all_valid is True


def test_trim_stream_mixed_batch_not_all_valid():
    # batch>1 with different valid sets: the union is kept, but a column valid for only one
    # element remains partially padded -> all_valid False -> caller keeps the dense mask.
    states = torch.ones(2, 4, 1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])  # elem1 has 2 valid, elem2 has 3
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (2, 3, 1)  # dropped the last col (invalid for both)
    assert all_valid is False


def _fake_dit(n_blocks=2):
    blocks = [types.SimpleNamespace(attn=types.SimpleNamespace()) for _ in range(n_blocks)]
    return types.SimpleNamespace(transformer_blocks=blocks)


def test_trim_pre_hook_empties_t2v_image_and_trims_and_flags():
    dit = _fake_dit()
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),  # all-zero -> t2v -> emptied
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 0, 0]]),
        "encoder_hidden_states_2": torch.arange(3.0).reshape(1, 3, 1),
        "encoder_attention_mask_2": torch.tensor([[1, 0, 0]]),
    }
    args, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["image_embeds"].shape == (1, 0, 3)  # image tokens dropped
    assert out["encoder_hidden_states"].shape == (1, 2, 1)  # mllm trimmed to 2 valid
    assert out["encoder_hidden_states_2"].shape == (1, 1, 1)  # byt5 trimmed to 1 valid
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_trim_stream_all_invalid_yields_empty_but_valid():
    # A fully-padded secondary stream (e.g. unused byt5 in t2v) trims to 0 length and reports
    # all_valid True (vacuous) so it does NOT drop the fast path -- it just contributes no tokens.
    states = torch.ones(1, 5, 2)
    mask = torch.zeros(1, 5, dtype=torch.long)
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (1, 0, 2) and all_valid is True


def test_trim_pre_hook_byt5_all_invalid_keeps_fast_path():
    # The real t2v case: byt5 is entirely padding (valid=0). It must be emptied WITHOUT dropping
    # the null-mask fast path, since mllm still carries the prompt.
    dit = _fake_dit()
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "encoder_hidden_states_2": torch.ones(1, 6, 1),
        "encoder_attention_mask_2": torch.zeros(1, 6, dtype=torch.long),  # all padding
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["encoder_hidden_states"].shape == (1, 3, 1)
    assert out["encoder_hidden_states_2"].shape == (1, 0, 1)  # byt5 emptied
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_trim_pre_hook_empty_primary_reverts_and_disables():
    # Pathological empty prompt: mllm has 0 valid tokens. The TokenRefiner must not get a
    # 0-length sequence -> revert all inputs to original and take the stock dense-mask path.
    dit = _fake_dit()
    mllm = torch.ones(1, 4, 1)
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),
        "encoder_hidden_states": mllm,
        "encoder_attention_mask": torch.zeros(1, 4, dtype=torch.long),  # 0 valid
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["encoder_hidden_states"] is mllm  # reverted (not emptied)
    assert out["image_embeds"].shape == (1, 5, 3)  # image revert too
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_keeps_i2v_image():
    dit = _fake_dit()
    img = torch.ones(1, 5, 3)  # nonzero -> i2v -> kept
    kwargs = {
        "image_embeds": img,
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["image_embeds"] is img  # not emptied
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_trim_pre_hook_mixed_batch_flags_false():
    dit = _fake_dit()
    kwargs = {
        "image_embeds": torch.zeros(2, 2, 3),
        "encoder_hidden_states": torch.ones(2, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_never_raises_sets_flag_false():
    # A malformed mask (not a tensor) must not break the forward: flag False, no exception.
    dit = _fake_dit()
    kwargs = {"encoder_hidden_states": torch.ones(1, 2, 1), "encoder_attention_mask": "oops"}
    args, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_absent_stream_not_written_back():
    # If encoder_hidden_states is absent from kwargs (a caller passing it positionally), the hook
    # must NOT write it back as None (that would collide: "got multiple values for argument") and
    # must drop the fast path (flag False) rather than null a mask it never verified.
    dit = _fake_dit()
    kwargs = {"image_embeds": torch.zeros(1, 4, 3)}  # no encoder_hidden_states key
    _, out = att._hunyuan_trim_pre_hook(dit, (torch.ones(1, 5, 1),), kwargs)
    assert "encoder_hidden_states" not in out
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_install_trim_noop_for_non_hunyuan_family():
    fam = types.SimpleNamespace(transformer_class="WanTransformer3DModel")
    pipe = types.SimpleNamespace(transformer=types.SimpleNamespace())
    assert att.install_hunyuan_attention_trim(pipe, fam) is False


def test_install_trim_noop_when_transformer_class_mismatch():
    # Family claims Hunyuan but the loaded module isn't -> no processors touched, no diffusers
    # import; returns False rather than swapping an unknown attention processor.
    fam = types.SimpleNamespace(transformer_class="HunyuanVideo15Transformer3DModel")
    pipe = types.SimpleNamespace(transformer=types.SimpleNamespace())  # class name mismatch
    assert att.install_hunyuan_attention_trim(pipe, fam) is False

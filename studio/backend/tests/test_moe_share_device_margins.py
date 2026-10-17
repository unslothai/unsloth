# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MoE resident share prices the margin and projector the child really applies.

Fixture: 20 GiB model, 15 GiB routed experts, one GPU with 20,000 / 32,000 MiB free,
4096 context, 256 MiB KV, 256 GiB host RAM (``_cache_launch``). The share decides
``--moe-cache-mib auto`` (band 0.35..0.75); an unknown share means no automatic cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_llama_cpp_placement as placement  # noqa: E402
from test_llama_cpp_placement import (  # noqa: E402,F401
    _GIB,
    _cache_launch,
    _discrete_linux_host,
    _has_moe_cache,
    _moe_cache_host,
    _ubatch_values,
    _write_gguf,
    llama_cpp_module,
)


def _spy_shares(monkeypatch):
    seen = []
    original = llama_cpp_module._moe_resident_expert_share

    def spy(shortfall, expert_bytes):
        share = original(shortfall, expert_bytes)
        seen.append(share)
        return share

    monkeypatch.setattr(llama_cpp_module, "_moe_resident_expert_share", spy)
    return seen


def _set_env(monkeypatch, env):
    for name in ("LLAMA_ARG_FIT_TARGET", "MTMD_BACKEND_DEVICE", "LLAMA_ARG_MMPROJ_OFFLOAD"):
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


# --- per-device --fit-target ------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["--fit-target", "0,6000"], {}),
        (["-fitt=0/6000"], {}),
        ([], {"LLAMA_ARG_FIT_TARGET": "0,6000"}),
        # Last wins for device 0: a later list rewrites entry 0.
        (["--fit-target", "6000", "--fit-target", "0,6000"], {}),
    ],
    ids = ["argv", "short_slash", "env", "single_then_list"],
)
def test_a_per_device_fit_target_prices_the_selected_cards_entry(
    tmp_path, _moe_cache_host, monkeypatch, extra_args, env
):
    """One device in the child: llama.cpp reads margins_s[0] only (common/fit.cpp), so
    device 0's 0 MiB margin applies, not the 6000 MiB of a device that does not exist.
    Priced at 0 the share is ~0.598 (inside the band); at max() it was ~0.207."""
    _set_env(monkeypatch, env)
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _cache_launch(tmp_path, extra_args = extra_args)

    assert seen and seen[-1] == pytest.approx(0.597917, abs = 1e-3), seen
    assert _has_moe_cache(cmd), cmd


def test_the_selected_cards_own_entry_is_honoured(tmp_path, _moe_cache_host, monkeypatch):
    # Control: entry 0 is 6000 here, so the card keeps 6000 MiB back and the share is low.
    _set_env(monkeypatch, {})
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _cache_launch(tmp_path, extra_args = ["--fit-target", "6000,0"])

    assert seen and seen[-1] == pytest.approx(0.207292, abs = 1e-3), seen
    assert not _has_moe_cache(cmd), cmd


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--fit-target", "0,6000", "--device", "CUDA0,CUDA1"],
        ["--fit-target", "0,6000", "--rpc", "127.0.0.1:50052"],
    ],
    ids = ["two_devices_named", "rpc_device_first"],
)
def test_a_per_device_fit_target_without_a_provable_index_is_unpriced(
    tmp_path, _moe_cache_host, monkeypatch, extra_args
):
    """A --device list of two, or an RPC server (llama.cpp puts RPC devices first), means
    entry 0 may not be this card: the share abstains instead of guessing."""
    _set_env(monkeypatch, {})
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _cache_launch(tmp_path, extra_args = extra_args)

    assert all(share is None for share in seen), seen
    assert not _has_moe_cache(cmd), cmd


# --- projector placement ------------------------------------------------------------------


def _projector_launch(
    tmp_path,
    monkeypatch,
    extra_args,
    projector_gib = 6,
):
    """_cache_launch with a sized vision projector, as the #13600 review measured it."""
    real_launch = placement._launch

    def launch(backend, gguf, **kwargs):
        projector = _write_gguf(gguf.parent / "mmproj.gguf", architecture = "clip")
        backend._resolve_launch_mmproj_path = lambda **_kw: str(projector)
        backend._mmproj_vram_bytes = lambda path: int(projector_gib * _GIB) if path else 0
        return real_launch(backend, gguf, **kwargs)

    monkeypatch.setattr(placement, "_launch", launch)
    return _cache_launch(tmp_path, is_vision = True, extra_args = list(extra_args))


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["--mmproj-offload", "--mmproj-device", "none"], {}),
        (["--mmproj-offload", "-mmdev=none"], {}),
        # arg.cpp applies env before argv, and -mmdev's env twin after --mmproj-offload's.
        ([], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1", "MTMD_BACKEND_DEVICE": "none"}),
        (["--mmproj-device", "none"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "1"}),
        (["--no-mmproj-offload"], {}),
    ],
    ids = [
        "argv_none_last",
        "short_none_last",
        "env_pair",
        "argv_none_over_env_offload",
        "no_offload",
    ],
)
def test_a_projector_that_resolves_to_host_ram_is_not_charged_to_the_card(
    tmp_path, _moe_cache_host, monkeypatch, extra_args, env
):
    """The last placement token wins (env first, then argv in order): device none clears
    mmproj_use_gpu (common/arg.cpp), so the 6 GiB projector and its allowance leave the
    card and the share is ~0.531 (inside the band) instead of 0.0."""
    _set_env(monkeypatch, env)
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _projector_launch(tmp_path, monkeypatch, extra_args)

    assert "--mmproj" in cmd, cmd
    assert seen and seen[-1] == pytest.approx(0.53125, abs = 1e-3), seen
    assert _has_moe_cache(cmd), cmd


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["--mmproj-device", "none", "--mmproj-offload"], {}),
        (["--mmproj-offload"], {"MTMD_BACKEND_DEVICE": "none"}),
    ],
    ids = ["offload_after_none", "argv_offload_over_env_none"],
)
def test_a_projector_turned_back_on_is_charged_to_the_card(
    tmp_path, _moe_cache_host, monkeypatch, extra_args, env
):
    """Controls: a later --mmproj-offload puts the projector back on the model's device
    (mmproj_device = devices.front(), common/arg.cpp), so it stays charged there."""
    _set_env(monkeypatch, env)
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _projector_launch(tmp_path, monkeypatch, extra_args)

    assert "--mmproj" in cmd, cmd
    assert seen and seen[-1] == pytest.approx(0.0, abs = 1e-3), seen
    assert not _has_moe_cache(cmd), cmd


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["--no-mmproj-offload", "--mmproj-device", "CUDA0"], {}),
        (["--mmproj-device", "CUDA0"], {"LLAMA_ARG_MMPROJ_OFFLOAD": "0"}),
        # A CPU name is a child that never starts (parse_device_list rejects CPU devices).
        (["--no-mmproj-offload", "--mmproj-device", "CPU"], {}),
    ],
    ids = [
        "argv_named_after_no_offload",
        "argv_named_over_env_no_offload",
        "argv_cpu_after_no_offload",
    ],
)
def test_a_named_projector_device_after_offload_off_is_unpriced(
    tmp_path, _moe_cache_host, monkeypatch, extra_args, env
):
    """A named --mmproj-device turns GPU offload back on (mmproj_use_gpu = true, common/arg.cpp)
    after an earlier offload-off left the budgeted projector at 0 bytes. The named-device
    abstention ran only when that size was positive, so the share read a full residency and
    the -ub 2048 raise was withheld; unknown keeps the raise and adds no cache."""
    _set_env(monkeypatch, env)
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _projector_launch(tmp_path, monkeypatch, extra_args)

    assert "--mmproj" in cmd, cmd
    assert not seen, seen
    assert cmd[cmd.index("--ubatch-size") + 1] == "2048", cmd
    assert not _has_moe_cache(cmd), cmd


# --- the cache's RAM admission prices the same margin ---------------------------------------


@pytest.mark.parametrize(
    "launch, extra_args, env",
    [
        # 20 GiB model, 15 GiB experts, 20,000 MiB free, 26 GiB RAM: entry 0 is 0.
        (dict(ram_gib = 26), ["--fit-target", "0"], {}),
        (dict(ram_gib = 26), ["--fit-target", "0,16384"], {}),
        (dict(ram_gib = 26), [], {"LLAMA_ARG_FIT_TARGET": "0,16384"}),
        # 8 GiB model, 4 GiB experts, 13,000 MiB free, 14 GiB RAM: entry 0 is 1024.
        (
            dict(ram_gib = 14, size_gib = 8, expert_gib = 4, memory = [(0, 13_000, 16_000)]),
            ["--fit-target", "1024"],
            {},
        ),
        (
            dict(ram_gib = 14, size_gib = 8, expert_gib = 4, memory = [(0, 13_000, 16_000)]),
            ["--fit-target", "1024,4096"],
            {},
        ),
        (
            dict(ram_gib = 14, size_gib = 8, expert_gib = 4, memory = [(0, 13_000, 16_000)]),
            [],
            {"LLAMA_ARG_FIT_TARGET": "1024,4096"},
        ),
    ],
    ids = [
        "single_0",
        "list_0_16384",
        "env_list_0_16384",
        "single_1024",
        "list_1024_4096",
        "env_list_1024_4096",
    ],
)
def test_the_cache_admission_charges_the_selected_cards_entry(
    tmp_path, _moe_cache_host, monkeypatch, launch, extra_args, env
):
    """One device in the child: the fitter keeps only entry 0 free (common/fit.cpp), the
    margin the share already priced. The RAM admission charged the largest entry, the
    16384 / 4096 MiB of a device that does not exist, and rejected a cache that the
    equivalent single value admits."""
    _set_env(monkeypatch, env)
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _cache_launch(tmp_path, extra_args = extra_args, **launch)

    assert seen and seen[-1] is not None, seen
    assert _has_moe_cache(cmd), cmd


# --- pass-through adapters ----------------------------------------------------------------


def _adapter(tmp_path, gib):
    (tmp_path / "launch").mkdir(exist_ok = True)
    path = tmp_path / f"adapter_{gib}.gguf"
    with open(path, "wb") as fh:
        fh.truncate(int(gib * _GIB))
    return str(path)


@pytest.mark.parametrize(
    "flag, gib",
    [("--lora", 4), ("--control-vector", 1), ("--lora", None)],
    ids = ["lora_4gib", "control_vector_1gib", "unreadable_lora"],
)
def test_a_pass_through_adapter_leaves_the_share_unpriced(
    tmp_path, _moe_cache_host, monkeypatch, flag, gib
):
    """common/fit.cpp sizes the split before the adapters load (common_init_from_params),
    and their tensors then land on each base layer's buffer (llama-adapter.cpp), so which
    expert slices they displace is unknown: the share abstains, as the load-mode fit and
    the context ceiling do. An unreadable file is unsized. Unknown keeps the -ub raise and
    adds no cache; priced without the adapter the share read ~0.531, inside the band."""
    _set_env(monkeypatch, {})
    path = _adapter(tmp_path, gib or 0)
    if gib is None:
        path += ".missing"
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _cache_launch(tmp_path / "launch", extra_args = [flag, path])

    assert all(share is None for share in seen), seen
    assert _ubatch_values(cmd) == ["2048"], cmd
    assert not _has_moe_cache(cmd), cmd


def test_an_empty_adapter_keeps_the_share(tmp_path, _moe_cache_host, monkeypatch):
    # Control: a 0-byte adapter adds nothing, so the share is the adapter-free one.
    _set_env(monkeypatch, {})
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _cache_launch(tmp_path / "launch", extra_args = ["--lora", _adapter(tmp_path, 0)])

    assert seen and seen[-1] == pytest.approx(0.53125, abs = 1e-3), seen
    assert _has_moe_cache(cmd), cmd


# --- an explicit --fit-target the child cannot read -------------------------------------------


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["--fit-target", "foo"], {}),
        (["--fit-target", "foo"], {"LLAMA_ARG_FIT_TARGET": "1024"}),
        (["-fitt=0,foo"], {}),
        (["--fit-target", "-512"], {}),
        ([], {"LLAMA_ARG_FIT_TARGET": "foo"}),
        ([], {"LLAMA_ARG_FIT_TARGET": "-512"}),
    ],
    ids = ["argv", "argv_over_env", "argv_list", "argv_negative", "env", "env_negative"],
)
def test_an_unreadable_fit_target_leaves_the_share_unpriced(
    tmp_path, _moe_cache_host, monkeypatch, extra_args, env
):
    """std::stoull throws on "foo" (common/arg.cpp) and wraps a negative to a huge margin,
    so the margin the child keeps is unknown. Falling back to Studio's value or the env
    priced a margin the child never applies; the share now abstains."""
    _set_env(monkeypatch, env)
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _cache_launch(tmp_path, extra_args = extra_args)

    assert all(share is None for share in seen), seen
    assert not _has_moe_cache(cmd), cmd


@pytest.mark.parametrize(
    "extra_args, env",
    [(["--fit-target", "1024"], {}), ([], {"LLAMA_ARG_FIT_TARGET": "1024"}), ([], {})],
    ids = ["argv", "env", "unset"],
)
def test_a_readable_fit_target_keeps_the_share(
    tmp_path, _moe_cache_host, monkeypatch, extra_args, env
):
    # Control: llama.cpp's own 1024 MiB default, however it is spelled.
    _set_env(monkeypatch, env)
    seen = _spy_shares(monkeypatch)
    _backend, cmd = _cache_launch(tmp_path, extra_args = extra_args)

    assert seen and seen[-1] == pytest.approx(0.53125, abs = 1e-3), seen
    assert _has_moe_cache(cmd), cmd


# --- the no-spill log ------------------------------------------------------------------------


def test_a_launch_with_nothing_spilled_says_so(tmp_path, _moe_cache_host, monkeypatch):
    """Share 1.0 skips the raise with its own reason, not a "100% -> 100%" movement."""
    lines = []
    monkeypatch.setattr(
        llama_cpp_module.logger,
        "info",
        lambda msg, *a, **kw: lines.append(msg % a if a else msg),
    )
    monkeypatch.setattr(llama_cpp_module, "_moe_resident_expert_share", lambda *_a: 1.0)
    _backend, cmd = _cache_launch(tmp_path, memory = [(0, 14_000, 32_000)])

    assert "2048" not in _ubatch_values(cmd), cmd
    assert any("keeps every routed expert on the GPU" in line for line in lines), lines
    assert not any("resident share" in line for line in lines), lines


# --- a zero-layer drafter whose tensors are overridden onto the card -----------------------


def test_a_zero_layer_drafter_with_gpu_tensor_overrides_is_unpriced(
    tmp_path, _moe_cache_host, monkeypatch
):
    """-otd places each matching drafter tensor on its named buffer whatever -ngld says, so
    -ngld 0 -otd .*=CUDA0 keeps drafter weights on the card. Classified as a CPU drafter it
    left the budget and skipped the draft tensor-placement guard; the share now abstains."""
    _set_env(monkeypatch, {})
    _cmd, cpu = placement._drafter_cache_launch(
        tmp_path / "cpu", monkeypatch, ["-ngld", "0", "-otd", ".*=CPU"]
    )
    assert cpu and cpu[-1] is not None, cpu

    cmd, seen = placement._drafter_cache_launch(
        tmp_path / "gpu", monkeypatch, ["-ngld", "0", "-otd", ".*=CUDA0"]
    )
    assert all(share is None for share in seen), seen
    assert not _has_moe_cache(cmd), cmd

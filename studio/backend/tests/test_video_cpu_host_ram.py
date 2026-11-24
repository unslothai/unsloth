# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only torch: a video load whose float32 host floor cannot fit available RAM is refused before any download;
GPU / MPS targets never reach the sizing."""

import pytest
import torch

import core.inference.diffusion_device as dd
import core.inference.diffusion_memory as dm
import core.inference.video as v
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

try:
    import core.inference.video_cpu_host as vch
except (
    ImportError
):  # the base arm (no CPU host check): the behaviour tests below still run against it
    vch = None

FAM = {f.name: f for f in VIDEO_FAMILIES}
LTX25, LTX2, H3 = FAM["ltx-2.5"], FAM["ltx-2"], FAM["minimax-h3"]
WAN5B, A14B, WAN13 = FAM["wan2.2-ti2v-5b"], FAM["wan2.2-t2v-a14b"], FAM["wan2.1-t2v-1.3b"]
HV15 = FAM["hunyuanvideo-1.5"]
GIB_MIB = 1024
GB_PER_GIB = 1024**3 / 1e9

# Colab CPU-HM, LTX-2.5 distilled UD-Q4_K_XL GGUF, 768x512x33: process-tree RssAnon peaks (decimal GB).
LTX25_MEASURED_ANON_GB = 28.7
LTX25_MEASURED_ANON_TIGHT_GB = 29.2
LTX25_RENDERED_AVAILABLE_GB = 32.2
# Wan2.1-T2V-1.3B pipeline on CPU, 832x480x5: 24.5 GiB peak server RSS (PR #14247).
WAN13_MEASURED_RSS_GB = 24.5 * GB_PER_GIB


def _target(
    device,
    dtype = torch.float32,
    backend = None,
):
    return DiffusionDeviceTarget(
        device = device,
        dtype = dtype,
        backend = backend or device,
        vendor = "nvidia" if device == "cuda" else None,
        supports_model_cpu_offload = device == "cuda",
        supports_default_torch_compile = device == "cuda",
        supports_pinned_transfer = device == "cuda",
    )


@pytest.fixture
def host(monkeypatch):
    """``host(device, total_gb, available_gb, held_gb=0)``: the diffusion target and the host RAM readings."""
    for name in (
        "UNSLOTH_DIFFUSION_SMALL_HOST",
        "UNSLOTH_DIFFUSION_HOST_RAM_CHECK",
        "UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB",
        "UNSLOTH_LTX25_TE_INT8",
        "UNSLOTH_DIFFUSION_CPU_GGUF_DIFFUSERS",
    ):
        monkeypatch.delenv(name, raising = False)

    def _set(
        device,
        total_gb,
        available_gb,
        held_gb = 0.0,
        *,
        probe_free_cpu = None,
    ):
        target = _target(device)
        is_cpu = device == "cpu" if probe_free_cpu is None else probe_free_cpu
        monkeypatch.setattr(dd, "diffusion_target_is_cpu", lambda: is_cpu)
        monkeypatch.setattr(dd, "resolve_diffusion_device_target", lambda *a, **k: target)
        monkeypatch.setattr(v, "resolve_diffusion_device_target", lambda *a, **k: target)
        monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: int(total_gb * 1e9 / 2**20))
        monkeypatch.setattr(
            dm, "_available_system_memory_mib", lambda: int(available_gb * 1e9 / 2**20)
        )
        if vch is not None:
            monkeypatch.setattr(vch, "_process_tree_held_mib", lambda: int(held_gb * 1e9 / 2**20))

    return _set


def _load(fam, kind, **kw):
    v.assert_video_precision_available(fam, model_kind = kind, **kw)


def _refusal(fam, kind, **kw):
    with pytest.raises(RuntimeError) as err:
        _load(fam, kind, **kw)
    msg = str(err.value)
    assert msg.startswith("Not enough system RAM to load ")
    assert "no GPU PyTorch can use" in msg and "UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0" in msg
    return msg


# ---- refusals (each passes the load on origin/main) ------------------------------------------------------------


def test_ltx25_gguf_on_a_24_gib_cgroup_is_refused_before_download(host):
    cap = 24 * GB_PER_GIB
    host("cpu", cap, cap)
    msg = _refusal(LTX25, "gguf")
    assert "needs about 28.4 GB of system RAM" in msg
    assert f"about {cap:.1f} GB is available" in msg
    assert "denoiser read from its GGUF file" in msg and "text encoder 8.5" in msg
    # The GGUF denoiser is mapped, so a lower quant cannot help: say so rather than suggest one.
    assert "a lower GGUF quant does not fit either" in msg


def test_real_cgroup_limit_reading_drives_the_refusal(host, monkeypatch):
    """Through the real host-RAM readers: a 24 GiB cgroup (1 GiB used, so 23 GiB left) on a huge host refuses."""
    host("cpu", 1, 1)
    monkeypatch.undo()
    target = _target("cpu")
    monkeypatch.setattr(dd, "diffusion_target_is_cpu", lambda: True)
    monkeypatch.setattr(dd, "resolve_diffusion_device_target", lambda *a, **k: target)
    if vch is not None:
        monkeypatch.setattr(vch, "_process_tree_held_mib", lambda: 0)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HOST_RAM_CHECK", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB", raising = False)
    limit = 24 << 30
    monkeypatch.setattr(
        LlamaCppBackend,
        "_cgroup_memory_budgets",
        staticmethod(lambda: [(limit - (1 << 30), limit)]),
    )
    monkeypatch.setattr("utils.host_memory.system_available_mib", lambda: 1_000_000)
    monkeypatch.setattr("utils.host_memory.system_total_mib", lambda: 2_000_000)
    assert "about 24.7 GB is available" in _refusal(LTX25, "gguf")


def test_host_ram_limit_test_hook_caps_the_reading(host, monkeypatch):
    host("cpu", 1000, 1000)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB", str(24 * GIB_MIB))
    _refusal(LTX25, "gguf")


def test_ltx25_dense_encoder_suggests_the_int8_encoder(host):
    host("cpu", 50 * GB_PER_GIB, 50 * GB_PER_GIB)
    msg = _refusal(LTX25, "gguf", text_encoder_quant = "none")
    assert "text encoder 47.8" in msg
    assert "leave the text encoder precision on Auto for the int8 text encoder" in msg
    assert "about 28.4 GB." in msg


def test_wan5b_dense_pipeline_suggests_its_gguf(host):
    cap = 24 * GB_PER_GIB
    host("cpu", cap, cap)
    msg = _refusal(WAN5B, "pipeline")
    # DiT 10.0 bf16 -> 20.0 float32, hosted fp8 umT5 5.7, float32 VAE 2.8, working set 1.5
    assert "needs about 30.0 GB" in msg and "denoiser 20.0" in msg
    assert "load a GGUF of it from 'unsloth/Wan2.2-TI2V-5B-GGUF'" in msg
    assert "about 10.0 GB." in msg


@pytest.mark.parametrize(
    "fam, kind, host_gib",
    [
        (A14B, "pipeline", 64),
        (HV15, "pipeline", 32),
        (LTX2, "gguf", 16),
        (WAN5B, "gguf", 8),
        (H3, "gguf", 4),
    ],
)
def test_other_families_refuse_below_their_floor(host, fam, kind, host_gib):
    cap = host_gib * GB_PER_GIB
    host("cpu", cap, cap)
    _refusal(fam, kind)


def test_available_not_total_is_the_ceiling(host):
    """50 GiB host with 20 GB held by another process: the 26 GB left is what the load gets."""
    host("cpu", 50 * GB_PER_GIB, 26.0)
    _refusal(LTX25, "gguf")


# ---- calibration and loads that fit ----------------------------------------------------------------------------


def test_ltx25_price_sits_under_the_measured_peak_and_above_a_24_gib_cap():
    need = vch.cpu_video_host_need(LTX25, model_kind = "gguf")
    assert need.denoiser_gb == 0.0 and need.denoiser_mapped
    assert need.total_gb == pytest.approx(28.4, abs = 0.05)
    assert 24 * GB_PER_GIB < need.total_gb <= LTX25_MEASURED_ANON_GB <= LTX25_MEASURED_ANON_TIGHT_GB


def test_wan13_price_never_exceeds_its_measured_peak():
    assert vch.cpu_video_host_need(WAN13, model_kind = "pipeline").total_gb <= WAN13_MEASURED_RSS_GB


@pytest.mark.parametrize(
    "total_gb, available_gb, held_gb",
    [
        # Colab CPU-HM with a 20.9 GB ballast: 32.2 GB available, the load rendered.
        (50 * GB_PER_GIB, LTX25_RENDERED_AVAILABLE_GB, 0.0),
        # Uncapped 50 GiB host.
        (50 * GB_PER_GIB, 48.0, 0.0),
        # Replacing a resident model this process tree holds (credited back).
        (50 * GB_PER_GIB, 10.0, 25.0),
    ],
)
def test_ltx25_loads_that_fit_are_not_refused(host, total_gb, available_gb, held_gb):
    host("cpu", total_gb, available_gb, held_gb)
    _load(LTX25, "gguf")
    _load(LTX25, "gguf", text_encoder_quant = "int8")


def test_wan13_on_a_32_gib_host_is_not_refused(host):
    host("cpu", 32 * GB_PER_GIB, 28.0)
    _load(WAN13, "pipeline")


def test_kill_switch_lets_the_load_try(host, monkeypatch):
    host("cpu", 8, 8)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HOST_RAM_CHECK", "0")
    _load(LTX25, "gguf")


def test_unreadable_ram_refuses_nothing(host, monkeypatch):
    host("cpu", 8, 8)
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: None)
    _load(LTX25, "gguf")


def test_unsized_family_refuses_nothing(host):
    host("cpu", 1, 1)
    fam = type("F", (), {"name": "custom", "bf16_components_gb": None})()
    _load(fam, "pipeline")


# ---- accelerator controls: the sizing never runs ---------------------------------------------------------------


@pytest.fixture
def sizing_must_not_run(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("CPU host sizing ran on an accelerator target")

    if vch is not None:
        monkeypatch.setattr(vch, "cpu_video_host_shortfall", _boom)
        monkeypatch.setattr(vch, "cpu_video_host_capacity_mib", _boom)


@pytest.mark.parametrize("device, probe_free_cpu", [("cuda", None), ("mps", None), ("cuda", True)])
@pytest.mark.parametrize(
    "fam, kind", [(LTX25, "gguf"), (WAN5B, "pipeline"), (A14B, "pipeline"), (H3, "gguf")]
)
def test_accelerator_targets_are_untouched(
    host, sizing_must_not_run, device, probe_free_cpu, fam, kind
):
    host(device, 4, 4, probe_free_cpu = probe_free_cpu)
    _load(fam, kind)


def test_accelerator_host_skips_the_target_probe(host, monkeypatch):
    host("cuda", 4, 4)

    def _no_probe(*a, **k):
        raise AssertionError("dtype probes ran on an accelerator host")

    monkeypatch.setattr(dd, "resolve_diffusion_device_target", _no_probe)
    monkeypatch.setattr(v, "resolve_diffusion_device_target", _no_probe)
    _load(LTX25, "gguf")


def test_cpu_check_follows_the_callers_target(host, sizing_must_not_run, monkeypatch):
    """The probe-free reading says CPU but the loader's own resolver says CUDA (as the fp16 host tests mock it): the
    loader's answer wins and nothing is sized."""
    host("cuda", 4, 4, probe_free_cpu = True)
    cpu = _target("cpu")
    monkeypatch.setattr(dd, "resolve_diffusion_device_target", lambda *a, **k: cpu)
    _load(WAN5B, "pipeline")

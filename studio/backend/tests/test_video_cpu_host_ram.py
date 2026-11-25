# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only torch: a video load whose float32 host floor cannot fit available RAM is refused before any download;
GPU / MPS targets never reach the sizing."""

import os

import pytest
import torch

import core.inference.diffusion as diffusion_module
import core.inference.diffusion_device as dd
import core.inference.diffusion_gguf_mmap as gguf_mmap
import core.inference.diffusion_memory as dm
import core.inference.diffusion_te_prequant as tp
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

_REAL_CAPACITY = dm._host_ram_capacity_mib
_REAL_AVAILABLE = dm._available_system_memory_mib
_REAL_MMAP = gguf_mmap.gguf_mmap_enabled
_REAL_RELEASE = getattr(v, "_resident_video_cpu_release_gb", None)

# Declared on-disk sizes of the tiny test checkpoints: NTFS allocates sparse files in full, so the files hold only
# their headers and the size the guard reads comes from here.
_DECLARED_SIZES = {}


@pytest.fixture(autouse = True)
def _declared_checkpoint_sizes(monkeypatch):
    _DECLARED_SIZES.clear()
    if vch is None:
        return
    real = vch._checkpoint_size

    def sized(repo, filename):
        size, path = real(repo, filename)
        if path is not None and os.path.basename(path) in _DECLARED_SIZES:
            return _DECLARED_SIZES[os.path.basename(path)], path
        return size, path

    monkeypatch.setattr(vch, "_checkpoint_size", sized)


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
# Colab CPU-HM, Wan2.1-T2V-1.3B pipeline at 832x480x33, 2 steps: process-tree RssAnon / Pss_Anon peak (decimal GB).
WAN13_MEASURED_ANON_GB = 9.45


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
    """``host(device, total_gb, available_gb, held_gb=0)``: the diffusion target and the host RAM readings.
    ``held_gb``: the resident video model this load replaces (its estimate and the process tree's PSS). Hermetic:
    the Hub is never asked, the GGUF mmap hook reads as on Linux unless a test turns it off."""
    for name in (
        "UNSLOTH_DIFFUSION_SMALL_HOST",
        "UNSLOTH_DIFFUSION_HOST_RAM_CHECK",
        "UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB",
        "UNSLOTH_LTX25_TE_INT8",
        "UNSLOTH_DIFFUSION_CPU_GGUF_DIFFUSERS",
        "UNSLOTH_DIFFUSION_FP8_STORAGE",
        "UNSLOTH_DIFFUSION_GGUF_MMAP",
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
            # raising = False: the previous head of this branch has neither, and its arm still runs these tests
            monkeypatch.setattr(
                vch, "_process_tree_pss_mib", lambda: int(held_gb * 1e9 / 2**20), raising = False
            )
            monkeypatch.setattr(
                v, "_resident_video_cpu_release_gb", lambda: float(held_gb), raising = False
            )
            monkeypatch.setattr(
                vch, "_process_tree_held_mib", lambda: int(held_gb * 1e9 / 2**20), raising = False
            )
            monkeypatch.setattr(gguf_mmap, "gguf_mmap_enabled", lambda: True)
        # LTX-2.5's automatic int8 encoder is confirmed on the Hub; never ask it here.
        monkeypatch.setattr(tp, "te_prequant_sources_confirmed", lambda *a, **k: True)
        monkeypatch.setattr(diffusion_module, "_hub_file_size", lambda *a, **k: 0)

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
    # Undo only the two readings the fake cgroup must now drive; system total / available and the credit are fixed so
    # the number is the same on every OS and runner.
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", _REAL_CAPACITY)
    monkeypatch.setattr(dm, "_available_system_memory_mib", _REAL_AVAILABLE)
    monkeypatch.setattr(dm, "_system_memory_mib", lambda: (2_000_000, 1_000_000))
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


@pytest.mark.parametrize("request_kw", [{"text_encoder_quant": "none"}, {"speed_mode": "off"}])
def test_ltx25_dense_encoder_suggests_the_int8_encoder(host, request_kw):
    """Encoder Off, and Speed Off with the encoder unset, both load the dense Gemma 4 in float32 (67.7 GB)."""
    host("cpu", 50 * GB_PER_GIB, 50 * GB_PER_GIB)
    msg = _refusal(LTX25, "gguf", **request_kw)
    assert "needs about 67.7 GB" in msg and "text encoder 47.8" in msg
    assert "use the int8 text encoder" in msg
    assert "about 28.4 GB." in msg


def test_ltx25_speed_off_at_32_gb_is_refused(host):
    host("cpu", 50 * GB_PER_GIB, LTX25_RENDERED_AVAILABLE_GB)
    _refusal(LTX25, "gguf", speed_mode = "off")


def test_ltx25_unconfirmed_hosted_encoder_prices_the_dense_one(host, monkeypatch):
    host("cpu", 50 * GB_PER_GIB, LTX25_RENDERED_AVAILABLE_GB)
    monkeypatch.setattr(tp, "te_prequant_sources_confirmed", lambda *a, **k: False)
    assert "text encoder 47.8" in _refusal(LTX25, "gguf")


def test_wan5b_auto_encoder_is_priced_dense_on_cpu(host):
    """Auto fp8 is dropped on the CPU target and the released umT5 loads in float32: 32 GB is not enough."""
    host("cpu", 50 * GB_PER_GIB, 32.0)
    msg = _refusal(WAN5B, "pipeline")
    # DiT 10.0 bf16 -> 20.0 float32, umT5 11.4 -> 22.8, float32 VAE 2.8, working set 1.5
    assert "needs about 47.1 GB" in msg and "denoiser 20.0" in msg and "text encoder 22.8" in msg
    assert "load a GGUF of it from 'unsloth/Wan2.2-TI2V-5B-GGUF'" in msg
    assert "about 27.1 GB." in msg


def test_gguf_copy_is_priced_when_the_mmap_hook_is_off(host, monkeypatch, tmp_path):
    """Windows, or UNSLOTH_DIFFUSION_GGUF_MMAP=0: diffusers copies the whole GGUF into memory."""
    host("cpu", 50 * GB_PER_GIB, LTX25_RENDERED_AVAILABLE_GB)
    monkeypatch.setattr(gguf_mmap, "gguf_mmap_enabled", lambda: False)
    name = "ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf"
    (tmp_path / name).write_bytes(b"GGUF")
    _DECLARED_SIZES[name] = 14_000_234_880
    msg = _refusal(LTX25, "gguf", checkpoint_repo = str(tmp_path), checkpoint_filename = name)
    assert "needs about 42.4 GB" in msg and "GGUF denoiser copied into memory 14.0" in msg
    # The copy is what a lower quant shrinks, so the advice names the size that fits.
    assert "a GGUF quant of at most about 3.8 GB fits" in msg


def test_gguf_copy_env_switch_reaches_the_price(host, monkeypatch):
    host("cpu", 50 * GB_PER_GIB, LTX25_RENDERED_AVAILABLE_GB)
    monkeypatch.setattr(gguf_mmap, "gguf_mmap_enabled", _REAL_MMAP)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GGUF_MMAP", "0")
    # Unsized (no file, Hub not asked): the smallest curated quant's share of bf16, 42.0 x 0.2.
    msg = _refusal(
        LTX25, "gguf", checkpoint_repo = "unsloth/LTX-2.5-GGUF", checkpoint_filename = "x.gguf"
    )
    assert "GGUF denoiser copied into memory 8.4" in msg


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


def test_memory_the_load_keeps_is_not_credited(host, monkeypatch):
    """16 GB available and 16 GB of chat in the process tree: the CPU branch leaves chat resident, so no credit."""
    host("cpu", 64 * GB_PER_GIB, 16.0)
    monkeypatch.setattr(vch, "_process_tree_pss_mib", lambda: int(16e9 / 2**20), raising = False)
    monkeypatch.setattr(vch, "_process_tree_held_mib", lambda: int(16e9 / 2**20), raising = False)
    _refusal(LTX25, "gguf")


def test_resident_video_credit_is_capped_by_measured_pss(host, monkeypatch):
    """A resident video model priced at 25 GB but holding only 4 GB of PSS credits 4 GB, not 25."""
    host("cpu", 64 * GB_PER_GIB, 16.0, 25.0)
    monkeypatch.setattr(vch, "_process_tree_pss_mib", lambda: int(4e9 / 2**20), raising = False)
    _refusal(LTX25, "gguf")


# ---- calibration and loads that fit ----------------------------------------------------------------------------


def test_ltx25_price_sits_under_the_measured_peak_and_above_a_24_gib_cap():
    need = vch.cpu_video_host_need(LTX25, model_kind = "gguf", te_scheme = "int8", te_hosted = True)
    assert need.denoiser_gb == 0.0 and need.denoiser_mapped
    assert need.total_gb == pytest.approx(28.4, abs = 0.05)
    assert 24 * GB_PER_GIB < need.total_gb <= LTX25_MEASURED_ANON_GB <= LTX25_MEASURED_ANON_TIGHT_GB


def test_cpu_prices_the_dense_float32_encoder_the_loader_keeps():
    """Wan2.2 5B stores umT5 in bf16: the float32 CPU target converts it, 22.8 GB of anonymous memory."""
    need = vch.cpu_video_host_need(WAN5B, model_kind = "pipeline")
    assert need.text_encoder_gb == pytest.approx(22.8, abs = 0.05) and need.text_encoder_dense


def test_wan13_price_matches_its_cpu_measurement():
    """Wan2.1 1.3B stores umT5 and the DiT in float32; measured on Colab CPU-HM at 832x480x33: 9.45 GB anonymous peak
    (umT5 read through its file mapping). Priced just under it."""
    need = vch.cpu_video_host_need(WAN13, model_kind = "pipeline")
    assert need.text_encoder_gb == 0.0
    assert WAN13_MEASURED_ANON_GB - 0.5 <= need.total_gb <= WAN13_MEASURED_ANON_GB


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


@pytest.mark.parametrize("available_gb", [26.0, 12.0])
def test_wan13_on_a_32_gib_host_is_not_refused(host, available_gb):
    host("cpu", 32 * GB_PER_GIB, available_gb)
    _load(WAN13, "pipeline")


def test_wan13_below_its_measured_floor_is_refused(host):
    host("cpu", 32 * GB_PER_GIB, 8.0)
    assert "text encoder read from its float32 file" in _refusal(WAN13, "pipeline")


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


# ---- single-file checkpoints priced as the loader keeps them -------------------------------------------------


def _sparse_safetensors(path, dtype, nbytes):
    """A header-only safetensors file declaring one ``dtype`` tensor of ``nbytes``; its size is declared, not written."""
    import json
    import struct

    header = json.dumps(
        {
            "blocks.0.attn.to_q.weight": {
                "dtype": dtype,
                "shape": [nbytes],
                "data_offsets": [0, nbytes],
            }
        }
    ).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header)
    _DECLARED_SIZES[path.name] = 8 + len(header) + nbytes
    return path.name


def test_fp8_single_file_is_priced_at_its_packed_size(host, tmp_path):
    """An fp8 checkpoint with transformer_quant unset: the loader keeps it packed when widening would not fit
    (fp8_storage_wanted), so it is priced at its stored 5.0 GB, not as a 20 GB float32 DiT."""
    host("cpu", 64 * GB_PER_GIB, 34.0)
    name = _sparse_safetensors(
        tmp_path / "wan2.2_ti2v_5B_scaled.safetensors", "F8_E4M3", 5_000_000_000
    )
    _load(WAN5B, "single_file", checkpoint_repo = str(tmp_path), checkpoint_filename = name)
    need = vch.cpu_video_host_need(
        WAN5B,
        model_kind = "single_file",
        single_file_packed_gb = vch.single_file_packed_gb(WAN5B, str(tmp_path), name),
    )
    assert need.denoiser_gb == pytest.approx(5.0, abs = 0.01) and need.total_gb == pytest.approx(
        32.1, abs = 0.05
    )


@pytest.mark.parametrize("pin", ["none", "off"])
def test_fp8_single_file_with_the_dense_dit_pinned_is_priced_float32(host, tmp_path, pin):
    """transformer_quant "none" / "off" sets comfy_keep False, so the loader dequantizes the fp8 file: no packed
    discount (47.1 GB for Wan 5B, as a dense float32 DiT), and a 34 GB host refuses it before any download."""
    host("cpu", 64 * GB_PER_GIB, 34.0)
    name = _sparse_safetensors(
        tmp_path / "wan2.2_ti2v_5B_scaled.safetensors", "F8_E4M3", 5_000_000_000
    )
    packed = vch.single_file_packed_gb(WAN5B, str(tmp_path), name)
    kept = vch.cpu_video_host_need(WAN5B, model_kind = "single_file", single_file_packed_gb = packed)
    dense = vch.cpu_video_host_need(
        WAN5B, model_kind = "single_file", transformer_quant = pin, single_file_packed_gb = packed
    )
    assert kept.denoiser_gb == pytest.approx(5.0, abs = 0.01)
    assert dense.denoiser_gb == pytest.approx(20.0, abs = 0.01) and dense.total_gb == pytest.approx(
        47.1, abs = 0.05
    )


@pytest.mark.parametrize("pin", [None, "auto"])
def test_fp8_single_file_under_speed_off_is_refused_as_float32(host, tmp_path, pin):
    """Speed "off" with transformer_quant unset / auto: the loader resolves it to "off" (video_transformer_quant_request),
    comfy_keep is False and the fp8 file dequantizes, so the route prices 47.1 GB and a 34 GB host refuses; the same
    request at the default speed keeps the packed 32.1 GB price and loads."""
    host("cpu", 64 * GB_PER_GIB, 34.0)
    name = _sparse_safetensors(
        tmp_path / "wan2.2_ti2v_5B_scaled.safetensors", "F8_E4M3", 5_000_000_000
    )
    msg = _refusal(
        WAN5B,
        "single_file",
        checkpoint_repo = str(tmp_path),
        checkpoint_filename = name,
        transformer_quant = pin,
        speed_mode = "off",
    )
    assert "47.1" in msg
    _load(
        WAN5B,
        "single_file",
        checkpoint_repo = str(tmp_path),
        checkpoint_filename = name,
        transformer_quant = pin,
    )


def test_unsized_dual_expert_gguf_copy_is_not_doubled(monkeypatch):
    """Unreadable GGUF size: the table's denoiser term already holds both A14B experts, so the fallback is that share
    once (57.2 GB x the smallest quant fraction), while a measured single-expert file still counts twice."""
    monkeypatch.setattr(vch, "_checkpoint_size", lambda repo, filename: (None, None))
    table = vch._sizes(A14B)[0]
    assert vch.gguf_copy_gb(A14B, "x/y", "a.gguf", mapped = False) == pytest.approx(
        table * vch.GGUF_MIN_BF16_FRACTION
    )
    monkeypatch.setattr(vch, "_checkpoint_size", lambda repo, filename: (7_000_000_000, None))
    assert vch.gguf_copy_gb(A14B, "x/y", "a.gguf", mapped = False) == pytest.approx(14.0)
    assert vch.gguf_copy_gb(WAN5B, "x/y", "a.gguf", mapped = False) == pytest.approx(7.0)


def test_fp8_single_file_by_name_without_the_file(host):
    host("cpu", 64 * GB_PER_GIB, 34.0)
    # Not local, Hub size unread: half the bf16 table (10.0 / 2).
    _load(
        WAN5B,
        "single_file",
        checkpoint_repo = "someone/wan",
        checkpoint_filename = "wan2.2_ti2v_5B_fp8_scaled.safetensors",
    )


def test_dense_single_file_and_disabled_fp8_storage_stay_float32(host, monkeypatch, tmp_path):
    host("cpu", 64 * GB_PER_GIB, 34.0)
    bf16 = _sparse_safetensors(tmp_path / "wan2.2_ti2v_5B_bf16.safetensors", "BF16", 10_000_000_000)
    assert "denoiser 20.0" in _refusal(
        WAN5B, "single_file", checkpoint_repo = str(tmp_path), checkpoint_filename = bf16
    )
    fp8 = _sparse_safetensors(
        tmp_path / "wan2.2_ti2v_5B_scaled.safetensors", "F8_E4M3", 5_000_000_000
    )
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FP8_STORAGE", "0")
    assert "denoiser 20.0" in _refusal(
        WAN5B, "single_file", checkpoint_repo = str(tmp_path), checkpoint_filename = fp8
    )


def test_fp8_dual_expert_single_file_prices_both_experts(host, tmp_path):
    """A Wan2.2-A14B fp8 pick loads its partner expert too: both ~14.3 GB files count (53.4 GB, not 39.1)."""
    host("cpu", 64 * GB_PER_GIB, 45.0)
    high = _sparse_safetensors(
        tmp_path / "wan2.2_t2v_high_noise_14B_scaled.safetensors", "F8_E4M3", 14_300_000_000
    )
    _sparse_safetensors(
        tmp_path / "wan2.2_t2v_low_noise_14B_scaled.safetensors", "F8_E4M3", 14_300_000_000
    )
    msg = _refusal(A14B, "single_file", checkpoint_repo = str(tmp_path), checkpoint_filename = high)
    assert "needs about 53.4 GB" in msg and "denoiser 28.6" in msg


def test_fp8_dual_expert_partner_unread_counts_the_pick_twice(host, tmp_path):
    host("cpu", 64 * GB_PER_GIB, 45.0)
    low = _sparse_safetensors(
        tmp_path / "wan2.2_t2v_low_noise_14B_scaled.safetensors", "F8_E4M3", 14_300_000_000
    )
    assert "denoiser 28.6" in _refusal(
        A14B, "single_file", checkpoint_repo = str(tmp_path), checkpoint_filename = low
    )


def test_resident_copied_gguf_is_credited_at_its_copy(host, monkeypatch, tmp_path):
    """mmap hook off (Windows, UNSLOTH_DIFFUSION_GGUF_MMAP=0): a resident LTX-2.5 GGUF holds its 14 GB copy, so
    replacing it releases about 40.9 GB, not the 26.9 GB a mapped GGUF would."""
    import types

    host("cpu", 64 * GB_PER_GIB, 8.0)
    monkeypatch.setattr(v, "_resident_video_cpu_release_gb", _REAL_RELEASE)
    monkeypatch.setattr(vch, "_process_tree_pss_mib", lambda: int(45e9 / 2**20), raising = False)
    monkeypatch.setattr(gguf_mmap, "gguf_mmap_enabled", lambda: False)
    name = "ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf"
    (tmp_path / name).write_bytes(b"GGUF")
    _DECLARED_SIZES[name] = 14_000_234_880
    state = types.SimpleNamespace(
        device = "cpu",
        family = LTX25,
        kind = "gguf",
        repo_id = str(tmp_path),
        gguf_filename = name,
        text_encoder_quant = "int8",
        transformer_quant = None,
    )
    monkeypatch.setattr(v, "_backend", types.SimpleNamespace(_state = state))
    assert v._resident_video_cpu_release_gb() == pytest.approx(40.9, abs = 0.05)
    # The same pick again (42.4 GB) fits in 8 GB free plus the 40.9 GB it releases.
    _load(LTX25, "gguf", checkpoint_repo = str(tmp_path), checkpoint_filename = name)

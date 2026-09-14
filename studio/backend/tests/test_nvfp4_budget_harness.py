# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pure parts of the NVFP4 time-budget harness that decide what a measurement MEANS."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _script(name: str):
    """One harness script as a module. Imported by path: ``scripts/`` is not a package."""
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_nvfp4_gemm_is_not_charged_to_attention():
    # "flash" is a substring of "flashinfer": testing attention first would file the NVFP4 GEMM under attention.
    profile = _script("nvfp4_budget_profile")
    fp4 = "flashinfer::DeviceGemmFp4_128x128"
    assert profile.classify(fp4, "phase:denoise") == "fp4_gemm"
    assert profile.classify(fp4, None) == "fp4_gemm"
    assert profile.classify("nvfp4_quantize_with_block_size", "phase:denoise") == "fp4_quantize"
    assert (
        profile.classify(
            "cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_36", "phase:denoise"
        )
        == "attention_cudnn"
    )
    assert (
        profile.classify(
            "void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128", "phase:denoise"
        )
        == "attention_flash"
    )
    assert (
        profile.classify("fmha_cutlassF_bf16_aligned_64x128_rf_sm80", "phase:denoise")
        == "attention_mem_efficient"
    )


def test_an_inductor_kernel_is_inductor_time_whatever_it_was_fused_from():
    profile = _script("nvfp4_budget_profile")
    fused = "triton_poi_fused__scaled_dot_product_cudnn_attention_add_7"
    assert profile.classify(fused, "phase:denoise") == "inductor_triton"


def test_the_phase_window_outranks_the_kernel_name():
    # Window overrides keep a text-encoder GEMM out of the denoise fp8 bucket.
    profile = _script("nvfp4_budget_profile")
    gemm = "nvjet_tst_128x128_64x4_1x1_v_bz_coopA_NTn"
    assert profile.classify(gemm, "phase:te") == "text_encoder"
    assert profile.classify(gemm, "phase:vae") == "vae_decode"
    assert profile.classify(gemm, "phase:denoise") == "fp8_scaled_mm"
    assert profile.classify(gemm, None) == "gemm_other"


def test_gpu_busy_unions_overlapping_intervals_instead_of_summing_them(tmp_path):
    # Two 10 ms kernels overlapping by 5 ms are 15 ms of busy GPU, not 20; a sum would exceed the wall clock.
    profile = _script("nvfp4_budget_profile")
    trace = tmp_path / "trace.json"
    trace.write_text(
        '{"traceEvents": ['
        '{"ph": "X", "cat": "kernel", "ts": 0.0, "dur": 10000.0, "name": "a"},'
        '{"ph": "X", "cat": "kernel", "ts": 5000.0, "dur": 10000.0, "name": "b"},'
        '{"ph": "X", "cat": "cpu_op", "ts": 0.0, "dur": 99000.0, "name": "host"}'
        "]}"
    )
    busy = profile._union_busy_us(trace)
    assert busy["busy_us"] == pytest.approx(15000.0)
    assert busy["sum_dur_us"] == pytest.approx(20000.0)
    assert busy["n_intervals"] == 2
    assert profile._union_busy_us(trace, window = (10**9, 2 * 10**9))["busy_us"] == 0.0


def test_paired_times_reports_the_direction_it_claims():
    # speedup > 1 means the SECOND argument is faster; backwards would invert every A/B verdict.
    profile = _script("nvfp4_budget_profile")
    slow = [1.0, 1.1, 1.2]
    fast = [0.5, 0.55, 0.6]
    row = profile.paired_times(slow, fast)
    assert row["speedup_p50"] == pytest.approx(2.0)
    assert row["wins"] == 3
    assert row["n"] == 3
    assert profile.paired_times(slow, slow)["speedup_p50"] == pytest.approx(1.0)


def test_the_summariser_reads_a_results_directory_and_recomputes_nothing(tmp_path):
    summarise = _script("nvfp4_budget_summarise")
    (tmp_path / "cell_a.json").write_text(
        '{"tag": "cell_a", "arm": "nvfp4", "graphs": "on", "model": "m", "steps": 4,'
        ' "resolution": "1024", "p50_s": 0.5, "min_s": 0.49, "gpu_busy_union_s": 0.4,'
        ' "host_idle_s": 0.1, "gpu_busy_fraction_of_wall": 0.8, "clean": true,'
        ' "profiler_overhead_ratio": 1.1, "phase_sync_overhead_s": 0.01,'
        ' "contention": {"pre": {"verdict": "clean"}, "post": {"verdict": "clean"}},'
        ' "buckets": {"fp4_gemm": {"calls_per_render": 8, "ms_per_render": 12.5,'
        ' "pct_busy": 3.1}}, "attention_by_kernel": [], "memcpy_d2d":'
        ' {"calls_per_render": 2, "calls_per_step": 0.5, "ms_per_render": 0.1}}\n'
    )
    order = tmp_path / "order.txt"
    order.write_text("# the pass, in order\ncell_a\ncell_that_never_ran\n")
    notes = tmp_path / "notes.md"
    notes.write_text("## Caveats\n\nThe card was shared.\n")
    out = tmp_path / "report" / "budget.md"
    assert (
        summarise.main(
            [
                "--results-dir",
                str(tmp_path),
                "--out",
                str(out),
                "--order",
                str(order),
                "--notes",
                str(notes),
                "--title",
                "A pass",
            ]
        )
        == 0
    )
    text = out.read_text()
    assert text.startswith("# A pass")
    assert "| cell_a | nvfp4 | on | 0.5000 | 0.4000 | 0.1000 | 80.0% | True |" in text
    assert "| fp4_gemm | 8 | 12.50 | 3.1% |" in text
    assert "The card was shared." in text
    assert "cell_that_never_ran" not in text


def test_every_harness_script_parses_its_arguments_without_a_gpu():
    # torch, diffusers and the Studio backend are imported inside main(), so --help works without them.
    for name in (
        "nvfp4_budget_profile",
        "nvfp4_budget_attention_ab",
        "nvfp4_budget_summarise",
        "nvfp4_budget_vae_numerics",
    ):
        module = _script(name)
        with pytest.raises(SystemExit) as exc:
            module.main(["--help"])
        assert exc.value.code == 0


class _Processor:
    def __init__(self, backend):
        self._attention_backend = backend


class _Attn:
    def __init__(self, backend):
        self.processor = _Processor(backend)


class _Denoiser:
    """The two levels ``observed_backends`` walks: a module tree whose attention submodules hold a processor."""

    def __init__(self, *backends):
        self._subs = [_Attn(b) for b in backends]

    def modules(self):
        return [self, *self._subs]


def test_a_refused_attention_switch_is_dropped_before_the_arm_is_timed():
    # set_attention_backend leaves the PREVIOUS backend installed when it refuses, so a render that merely succeeds
    # would publish the old kernel's time under the new kernel's label.
    ab = _script("nvfp4_budget_attention_ab")
    ok = {"requested": "_native_flash", "observed": ["_native_flash"], "errors": []}
    assert ab.switch_failure(ok) is None
    refused = {
        "requested": "_native_flash",
        "observed": ["_native_cudnn"],
        "errors": ["FluxTransformer2DModel: ValueError: `backend=` must be one of"],
    }
    assert "refused" in ab.switch_failure(refused)
    silent = {"requested": "_native_flash", "observed": ["_native_cudnn"], "errors": []}
    assert "not _native_flash" in ab.switch_failure(silent)
    assert ab.switch_failure({"requested": "_native_flash", "observed": [], "errors": []}) is None


def test_the_observed_backend_is_read_off_the_processors_not_the_denoiser():
    # diffusers writes processor._attention_backend; the denoiser module carries no such attribute.
    ab = _script("nvfp4_budget_attention_ab")
    backend = types.SimpleNamespace(value = "_native_cudnn")
    assert ab.observed_backends([_Denoiser(backend, backend)]) == ["_native_cudnn"]
    assert ab.observed_backends([_Denoiser(None)]) == []
    mixed = _Denoiser(backend, types.SimpleNamespace(value = "_native_flash"))
    assert ab.observed_backends([mixed]) == ["_native_cudnn", "_native_flash"]


def test_compile_off_loads_the_eager_tier_rather_than_relabelling_a_compiled_run():
    profile = _script("nvfp4_budget_profile")
    assert profile.resolve_load_speed_mode("off", "default") == "eager"
    assert profile.resolve_load_speed_mode("off", "max") == "eager"
    assert profile.resolve_load_speed_mode("regional", "default") == "default"
    assert profile.resolve_load_speed_mode("whole", "max") == "max"

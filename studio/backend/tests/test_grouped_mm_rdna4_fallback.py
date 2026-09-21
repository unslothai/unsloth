# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Numerics + gating for the RDNA4 _grouped_mm CPU fallback (PRs #7276 / #7292).

RDNA4 (gfx1200/gfx1201) ships a null HIP `_grouped_mm` kernel on ROCm <= 7.12
(fixed in 7.13, ROCm/TheRock #5284). Training MoE models there crashes with
0xC0000005 on Windows and a plain segfault on Linux, so worker.py registers a
Python mm/bmm fallback on the CUDA dispatch key.

The fallback is silent, GPU-gated, and reimplements a matmul: if it is wrong, an
RX 9070 user does not crash, they train on quietly wrong gradients. Until now the
only coverage was `assert '_gm_lib.impl("_grouped_mm"' in source` -- the math was
never executed once, in any suite.

worker.py cannot be imported here (module-level structlog/backend imports), so
`_install_grouped_mm_cpu_fallback` is lifted out with ast and driven with a fake
`torch_mod` that forwards to real CPU torch. That also pins the op surface: the
fallback may only use the ops the fake exposes, and the registration is captured
instead of hitting a real CUDA dispatch key that CI runners do not have.

The two gates around it are exec'd straight out of the source so this file tests
the shipped expressions rather than a copy of them.
"""

import ast
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_WORKER_PATH = Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py"
_WORKER_SOURCE = _WORKER_PATH.read_text(encoding = "utf-8")


def _load_installer():
    """exec just _install_grouped_mm_cpu_fallback out of worker.py."""
    tree = ast.parse(_WORKER_SOURCE)
    fn = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_install_grouped_mm_cpu_fallback"
    ]
    assert fn, "_install_grouped_mm_cpu_fallback not found in core/training/worker.py"
    ns: dict = {}
    exec(compile(ast.Module(body = fn, type_ignores = []), str(_WORKER_PATH), "exec"), ns)
    return ns["_install_grouped_mm_cpu_fallback"]


_install_grouped_mm_cpu_fallback = _load_installer()


class _RecordingLibrary:
    """Stands in for torch.library.Library: captures the registration instead of
    binding it to a CUDA dispatch key no CI runner has."""

    def __init__(self, namespace, kind):
        self.namespace = namespace
        self.kind = kind
        self.registrations = []

    def impl(self, name, fn, dispatch_key):
        self.registrations.append((name, fn, dispatch_key))


class _RecordingLogger:
    def __init__(self):
        self.info_calls = []
        self.warning_calls = []

    def info(self, *args, **kwargs):
        self.info_calls.append(args)

    def warning(self, *args, **kwargs):
        self.warning_calls.append(args)


def _fake_torch():
    """Real CPU torch behind the exact op surface the fallback is allowed to use.

    Anything else the fallback reaches for raises AttributeError here, which is
    the point: a new dependency has to be a deliberate edit, not a silent one."""
    return SimpleNamespace(
        library = SimpleNamespace(Library = _RecordingLibrary),
        mm = torch.mm,
        bmm = torch.bmm,
        matmul = torch.matmul,
        cat = torch.cat,
        zeros = torch.zeros,
    )


@pytest.fixture
def fallback():
    """The registered _grouped_mm implementation, plus the Library it landed on."""
    torch_mod = _fake_torch()
    logger = _RecordingLogger()
    lib = _install_grouped_mm_cpu_fallback(torch_mod, logger, "test")
    assert lib.registrations, "the fallback registered nothing"
    name, fn, key = lib.registrations[0]
    return SimpleNamespace(fn = fn, lib = lib, logger = logger, name = name, key = key)


class TestRegistration:
    """Where the override lands. Getting the namespace or dispatch key wrong is a
    silent no-op: training still crashes on the null HIP kernel."""

    def test_overrides_aten_grouped_mm_on_the_cuda_key(self, fallback):
        assert fallback.lib.namespace == "aten"
        assert fallback.lib.kind == "IMPL"
        assert fallback.name == "_grouped_mm"
        # ROCm dispatches through the CUDA key; "HIP"/"PrivateUse1" would not bind.
        assert fallback.key == "CUDA"

    def test_registers_exactly_once(self, fallback):
        assert len(fallback.lib.registrations) == 1

    def test_returns_the_library_so_the_caller_can_keep_it_alive(self, fallback):
        """A dropped Library is garbage collected and the override silently
        unregisters mid-run; worker.py parks it in a module global."""
        assert isinstance(fallback.lib, _RecordingLibrary)
        assert "_WINDOWS_ROCM_GROUPED_MM_LIB = _install_grouped_mm_cpu_fallback(" in _WORKER_SOURCE

    def test_logs_the_patch_with_its_label(self, fallback):
        assert fallback.logger.info_calls, "the patch must be visible in the run log"
        assert "test" in fallback.logger.info_calls[0]


class TestUngroupedNumerics:
    """offs=None: plain matmul, one path per rank combination. The 3-D case is
    the regression #7292 fixed -- an unconditional mm() broke MoE experts."""

    def test_2d_by_2d_matches_mm(self, fallback):
        a = torch.randn(6, 4)
        b = torch.randn(4, 5)
        torch.testing.assert_close(fallback.fn(a, b), torch.mm(a, b))

    def test_3d_by_3d_matches_bmm(self, fallback):
        a = torch.randn(3, 6, 4)
        b = torch.randn(3, 4, 5)
        torch.testing.assert_close(fallback.fn(a, b), torch.bmm(a, b))

    def test_3d_by_2d_matches_matmul(self, fallback):
        a = torch.randn(3, 6, 4)
        b = torch.randn(4, 5)
        torch.testing.assert_close(fallback.fn(a, b), torch.matmul(a, b))

    def test_2d_by_3d_matches_matmul(self, fallback):
        a = torch.randn(6, 4)
        b = torch.randn(3, 4, 5)
        torch.testing.assert_close(fallback.fn(a, b), torch.matmul(a, b))

    def test_non_contiguous_inputs_are_handled(self, fallback):
        """Transposed views reach _grouped_mm constantly; every path calls
        .contiguous() and this catches it if one stops."""
        a = torch.randn(4, 6).t()
        b = torch.randn(5, 4).t()
        torch.testing.assert_close(fallback.fn(a, b), torch.mm(a, b))


class TestGroupedNumerics:
    """offs=[end-row of each group], the MoE token-routing layout."""

    def test_matches_per_group_mm_with_3d_weights(self, fallback):
        a = torch.randn(7, 4)
        b = torch.randn(3, 4, 5)
        offs = torch.tensor([2, 5, 7])
        expected = torch.cat([a[0:2] @ b[0], a[2:5] @ b[1], a[5:7] @ b[2]], dim = 0)
        torch.testing.assert_close(fallback.fn(a, b, offs), expected)

    def test_shared_2d_weight_is_reused_for_every_group(self, fallback):
        a = torch.randn(7, 4)
        b = torch.randn(4, 5)
        offs = torch.tensor([2, 5, 7])
        torch.testing.assert_close(fallback.fn(a, b, offs), a @ b)

    def test_empty_group_produces_no_rows(self, fallback):
        """An expert that routed zero tokens (offs[i] == offs[i-1]) must
        contribute nothing, not a stray row."""
        a = torch.randn(5, 4)
        b = torch.randn(3, 4, 5)
        offs = torch.tensor([2, 2, 5])
        expected = torch.cat([a[0:2] @ b[0], a[2:5] @ b[2]], dim = 0)
        got = fallback.fn(a, b, offs)
        assert got.shape == (5, 5)
        torch.testing.assert_close(got, expected)

    def test_rows_past_the_last_offset_are_not_dropped(self, fallback):
        """Trailing tokens beyond offs[-1] go through the last expert; dropping
        them would silently shrink the output instead of raising."""
        a = torch.randn(7, 4)
        b = torch.randn(3, 4, 5)
        offs = torch.tensor([2, 5])
        expected = torch.cat([a[0:2] @ b[0], a[2:5] @ b[1], a[5:7] @ b[-1]], dim = 0)
        got = fallback.fn(a, b, offs)
        assert got.shape[0] == a.shape[0]
        torch.testing.assert_close(got, expected)

    def test_zero_rows_returns_an_empty_result_not_an_error(self, fallback):
        a = torch.randn(0, 4)
        b = torch.randn(3, 4, 5)
        offs = torch.tensor([], dtype = torch.int64)
        got = fallback.fn(a, b, offs)
        assert got.shape == (0, 5)
        assert got.dtype == a.dtype

    def test_offsets_may_arrive_as_a_device_tensor_of_any_int_dtype(self, fallback):
        a = torch.randn(4, 4)
        b = torch.randn(2, 4, 5)
        expected = torch.cat([a[0:2] @ b[0], a[2:4] @ b[1]], dim = 0)
        for dtype in (torch.int32, torch.int64):
            torch.testing.assert_close(
                fallback.fn(a, b, torch.tensor([2, 4], dtype = dtype)), expected
            )


class TestBiasAndDtype:
    def test_bias_is_added(self, fallback):
        a = torch.randn(6, 4)
        b = torch.randn(4, 5)
        bias = torch.randn(5)
        torch.testing.assert_close(fallback.fn(a, b, None, bias), torch.mm(a, b) + bias)

    def test_bias_is_added_on_the_grouped_path_too(self, fallback):
        a = torch.randn(4, 4)
        b = torch.randn(2, 4, 5)
        bias = torch.randn(5)
        offs = torch.tensor([2, 4])
        expected = torch.cat([a[0:2] @ b[0], a[2:4] @ b[1]], dim = 0) + bias
        torch.testing.assert_close(fallback.fn(a, b, offs, bias), expected)

    def test_out_dtype_is_honoured(self, fallback):
        a = torch.randn(6, 4)
        b = torch.randn(4, 5)
        got = fallback.fn(a, b, None, None, torch.float64)
        assert got.dtype == torch.float64
        torch.testing.assert_close(got, torch.mm(a, b).to(torch.float64))

    def test_promotion_from_bias_is_cast_back_to_the_input_dtype(self, fallback):
        """Without the restore, a promoted result changes the autograd dtype
        downstream of every MoE layer."""
        a = torch.randn(6, 4, dtype = torch.float32)
        b = torch.randn(4, 5, dtype = torch.float32)
        bias = torch.randn(5, dtype = torch.float64)
        got = fallback.fn(a, b, None, bias)
        assert got.dtype == torch.float32

    def test_out_dtype_wins_over_the_input_dtype_restore(self, fallback):
        a = torch.randn(6, 4, dtype = torch.float32)
        b = torch.randn(4, 5, dtype = torch.float32)
        bias = torch.randn(5, dtype = torch.float64)
        got = fallback.fn(a, b, None, bias, torch.float64)
        assert got.dtype == torch.float64

    def test_bf16_inputs_stay_bf16(self, fallback):
        """The dtype training actually runs in."""
        a = torch.randn(6, 4).to(torch.bfloat16)
        b = torch.randn(4, 5).to(torch.bfloat16)
        got = fallback.fn(a, b)
        assert got.dtype == torch.bfloat16
        torch.testing.assert_close(got.float(), (a.float() @ b.float()), rtol = 2e-2, atol = 2e-2)


def _exec_source_snippet(anchor: str, last_line: str, **variables):
    """Run a slice of worker.py verbatim, so the gate under test is the shipped
    one and not a copy that can drift."""
    start = _WORKER_SOURCE.find(anchor)
    assert start != -1, f"gate snippet not found in worker.py: {anchor!r}"
    start = _WORKER_SOURCE.rfind("\n", 0, start) + 1  # keep the indent for dedent()
    end = _WORKER_SOURCE.find(last_line, start)
    assert end != -1, f"end of gate snippet not found: {last_line!r}"
    snippet = textwrap.dedent(_WORKER_SOURCE[start : end + len(last_line)])
    ns = {"re": re, **variables}
    exec(compile(snippet, str(_WORKER_PATH), "exec"), ns)
    return ns


class TestLinuxHipVersionGate:
    """PR #7292's Linux gate. Too low a floor keeps the slow Python fallback on
    fixed ROCm 7.13+; too high reintroduces the segfault on 7.12."""

    _ANCHOR = '_m = re.match(r"(\\d+)\\.(\\d+)", _hip_str)'
    _LAST = '_hip_lt_713 = "rocmsdk" not in _ver'

    def _decide(self, hip_str, version):
        ns = _exec_source_snippet(self._ANCHOR, self._LAST, _hip_str = hip_str, _ver = version.lower())
        return ns["_hip_lt_713"]

    @pytest.mark.parametrize(
        "hip_str,version,affected",
        [
            ("7.12.0", "2.10.0+rocm7.12.0", True),  # the broken kernel
            ("7.6.0", "2.9.0+rocm7.6.0", True),
            ("6.4.0", "2.8.0+rocm6.4.0", True),
            ("7.13.0", "2.11.0+rocm7.13.0", False),  # AMD's fix
            ("7.14.0", "2.11.0+rocm7.14.0", False),
            ("8.0.0", "2.12.0+rocm8.0.0", False),
        ],
    )
    def test_torch_version_hip_decides_when_present(self, hip_str, version, affected):
        assert self._decide(hip_str, version) is affected

    @pytest.mark.parametrize(
        "version,affected",
        [
            ("2.10.0+rocm7.12.0", True),
            ("2.11.0+rocm7.13.0", False),
            ("2.11.0+rocm7.14.0", False),
        ],
    )
    def test_falls_back_to_the_rocm_tag_in_torch_version(self, version, affected):
        """AMD SDK / Radeon wheels leave torch.version.hip unset."""
        assert self._decide("", version) is affected

    def test_unknown_version_is_assumed_affected(self):
        """Fallback is slow but correct; a missed guard is a crash."""
        assert self._decide("", "2.9.0+unknown") is True

    def test_rocmsdk_wheels_without_a_version_are_assumed_fixed(self):
        """rocmsdk wheels post-date the gfx120X fix."""
        assert self._decide("", "2.10.0+rocmsdk20260107") is False


class TestLinuxRdna4NameMatch:
    """The name regex is the fallback when a wheel omits gcnArchName."""

    def _pattern(self):
        """Read whatever pattern worker.py currently uses, not a copy of the one
        it used when this test was written. Anchoring on the literal pattern text
        would make a *widened* regex -- the dangerous edit, since it silently
        forces the slow Python fallback onto RDNA3 users -- fail as "moved"
        instead of being checked against the cases below."""
        m = re.search(r"re\.search\(r\"([^\"]+)\",\s*_lin_name\)", _WORKER_SOURCE)
        assert m, "could not locate the RDNA4 device-name regex in worker.py"
        return m.group(1)

    def test_name_is_lowercased_before_matching(self):
        """The pattern is all-lowercase, so it only works against a lowercased
        name. Device names arrive mixed case ("AMD Radeon RX 9070 XT")."""
        assert self._pattern() == self._pattern().lower(), "pattern is not all-lowercase"
        assert re.search(
            r"_lin_name\s*=\s*\(getattr\(_props,\s*\"name\",\s*\"\"\)\s*or\s*\"\"\)\.lower\(\)",
            _WORKER_SOURCE,
        ), "worker.py must lowercase the device name before matching the RDNA4 pattern"

    def test_name_match_is_only_a_fallback_when_arch_is_unknown(self):
        """gcnArchName is authoritative when present. Letting the name regex fire
        alongside a known arch would misclassify any card whose marketing name
        happens to look RDNA4."""
        assert re.search(
            r"not _lin_arch and re\.search\(r\"[^\"]+\",\s*_lin_name\)", _WORKER_SOURCE
        ), "the RDNA4 name regex must be guarded by `not _lin_arch`"

    @pytest.mark.parametrize(
        "name,is_rdna4",
        [
            ("AMD Radeon RX 9070 XT", True),
            ("AMD Radeon RX 9060 XT", True),
            ("Radeon RX9070", True),
            ("AMD Radeon AI PRO R9700", True),
            ("AMD Radeon RX 7900 XTX", False),  # RDNA3, kernel is fine
            ("AMD Radeon 8060S Graphics", False),  # Strix Halo
            ("AMD Radeon RX 6800 XT", False),
            ("NVIDIA GeForce RTX 4090", False),
        ],
    )
    def test_matches_only_rdna4_cards(self, name, is_rdna4):
        assert bool(re.search(self._pattern(), name.lower())) is is_rdna4


class TestLinuxGateStructure:
    """The block is a few hundred lines into run_training_process and can only be
    checked structurally; these pin the parts a refactor would quietly drop."""

    def _linux_block(self):
        start = _WORKER_SOURCE.find("1f-linux")
        assert start != -1, "the Linux ROCm gfx120X guard (#7292) is gone from worker.py"
        end = _WORKER_SOURCE.find("1g.", start)
        assert end != -1
        return _WORKER_SOURCE[start:end]

    def test_gated_on_linux_and_rocm(self):
        block = self._linux_block()
        assert 'sys.platform.startswith("linux")' in block
        assert "_hw.IS_ROCM" in block, "guard must not run on NVIDIA/CPU hosts"

    def test_requires_both_rdna4_and_an_affected_hip(self):
        block = self._linux_block()
        assert "if _rdna4 and _hip_lt_713:" in block

    def test_scans_every_visible_device(self):
        """device_map="balanced" can place layers on a later card, so checking
        device 0 alone misses the RDNA4 GPU."""
        block = self._linux_block()
        assert "for _i in range(_torch_lin.cuda.device_count()):" in block

    def test_matches_both_rdna4_arch_ids(self):
        block = self._linux_block()
        assert '("gfx1200", "gfx1201")' in block

    def test_failure_to_patch_is_non_fatal(self):
        """A broken patch attempt must not take down the whole training run."""
        block = self._linux_block()
        assert "except Exception" in block
        assert "logger.warning" in block

    def test_windows_and_linux_share_one_implementation(self):
        """Two copies of this fallback would drift; #7292 deliberately hoisted it."""
        assert _WORKER_SOURCE.count("def _install_grouped_mm_cpu_fallback(") == 1
        assert _WORKER_SOURCE.count("_install_grouped_mm_cpu_fallback(") >= 3  # def + win32 + linux


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

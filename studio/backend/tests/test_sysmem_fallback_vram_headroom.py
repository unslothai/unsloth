"""Windows sysmem-fallback headroom in the VRAM budget (unslothai/unsloth#11349).

On Windows with a discrete NVIDIA GPU, a cudaMalloc that does not fit in VRAM is served
out of host RAM by WDDM instead of failing, so an over-optimistic fit estimate produces a
load that starts, answers /health, exits 0 and decodes 5-10x slow. Linux refuses the same
allocation outright ("cudaMalloc failed: out of memory"), which is what makes a thin
margin self-correcting there and silent on Windows.

These tests pin the two halves of the fix:
  * the raised floor applies ONLY on Windows + a CUDA build, and
  * every other host keeps byte-for-byte the budget it had.

The second half is the one worth guarding: the regression risk of this change is entirely
"did it leak onto a platform it was not meant for".
"""

import sys

import pytest

from core.inference.llama_cpp import (
    _CTX_FIT_VRAM_FRACTION,
    _LLAMA_FIT_TARGET_DEFAULT_MIB,
    _VRAM_FLOOR_RESERVE_MIB,
    _WINDOWS_SYSMEM_FALLBACK_MAX_FRACTION,
    _WINDOWS_SYSMEM_FALLBACK_RESERVE_MIB,
    LlamaCppBackend,
    _vram_reserve_floor_mib,
    _vram_usable_mib,
)

GIB = 1024.0
CARD_SIZES_MIB = [4 * GIB, 6 * GIB, 8 * GIB, 12 * GIB, 16 * GIB, 24 * GIB, 48 * GIB]
FRACTIONS = [0.80, 0.90, 0.95, _CTX_FIT_VRAM_FRACTION, 0.971, 0.98, 0.99, 1.0]


class TestFlagDefaultsToTodaysBehaviour:
    """The flag must be opt-in, so an untaught caller cannot change platform behaviour."""

    @pytest.mark.parametrize("total", CARD_SIZES_MIB)
    @pytest.mark.parametrize("frac", FRACTIONS)
    def test_default_matches_explicit_false(self, total, frac):
        free = total * 0.94
        assert _vram_usable_mib(free, total, frac) == _vram_usable_mib(
            free, total, frac, sysmem_fallback = False
        )

    @pytest.mark.parametrize("total", CARD_SIZES_MIB)
    def test_floor_default_matches_explicit_false(self, total):
        assert _vram_reserve_floor_mib(total) == _vram_reserve_floor_mib(
            total, sysmem_fallback = False
        )

    def test_unknown_total_branch_also_defaults_off(self):
        # A MIG/vGPU or two-column probe reports free with no total.
        assert _vram_usable_mib(9000, 0, 0.97) == _vram_usable_mib(
            9000, 0, 0.97, sysmem_fallback = False
        )


class TestRaisedFloor:
    @pytest.mark.parametrize("total", CARD_SIZES_MIB)
    def test_never_reserves_less_than_before(self, total):
        assert _vram_reserve_floor_mib(total, sysmem_fallback = True) >= _vram_reserve_floor_mib(
            total
        )

    @pytest.mark.parametrize("total", CARD_SIZES_MIB)
    def test_never_exceeds_an_eighth_of_the_card(self, total):
        floor = _vram_reserve_floor_mib(total, sysmem_fallback = True)
        assert floor <= _WINDOWS_SYSMEM_FALLBACK_MAX_FRACTION * total + 1e-9

    @pytest.mark.parametrize("total", [8 * GIB, 12 * GIB, 16 * GIB, 24 * GIB, 48 * GIB])
    def test_adopts_llama_cpp_own_fit_target_above_8gib(self, total):
        # Not a new magic number: llama.cpp's own per-device --fit-target default.
        assert _vram_reserve_floor_mib(total, sysmem_fallback = True) == pytest.approx(
            _LLAMA_FIT_TARGET_DEFAULT_MIB
        )

    def test_small_cards_stay_proportionate(self):
        # A flat 1024 would be a quarter of a 4 GiB card; the cap holds it to an eighth.
        assert _vram_reserve_floor_mib(4 * GIB, sysmem_fallback = True) == pytest.approx(512.0)

    def test_reserve_constant_is_llama_cpp_fit_target(self):
        assert _WINDOWS_SYSMEM_FALLBACK_RESERVE_MIB == _LLAMA_FIT_TARGET_DEFAULT_MIB
        assert _WINDOWS_SYSMEM_FALLBACK_RESERVE_MIB > _VRAM_FLOOR_RESERVE_MIB


class TestMonotonicity:
    """Raising the VRAM-budget slider must never hand back LESS context.

    This is the property `_vram_reserve_floor_mib`'s docstring exists to protect, and the
    reason the pre-existing floor is capped rather than flat. A raised floor could plausibly
    reintroduce the bug, so it is checked on both platforms at every card size.
    """

    @pytest.mark.parametrize("sysmem_fallback", [False, True])
    @pytest.mark.parametrize("total", CARD_SIZES_MIB)
    def test_budget_never_decreases_as_fraction_rises(self, total, sysmem_fallback):
        budgets = [
            _vram_usable_mib(total, total, f, sysmem_fallback = sysmem_fallback)
            for f in FRACTIONS
        ]
        for lower, higher in zip(budgets, budgets[1:]):
            assert higher >= lower - 1e-9

    def test_docstring_8gib_example_unchanged_on_linux(self):
        # The worked example in the docstring: 7946 MiB at both 0.97 and 0.971.
        t = 8 * GIB
        assert _vram_usable_mib(t, t, 0.97) == pytest.approx(7946.2, abs = 0.5)
        assert _vram_usable_mib(t, t, 0.971) == pytest.approx(7946.2, abs = 0.5)

    def test_docstring_8gib_example_is_flat_under_fallback(self):
        t = 8 * GIB
        at_default = _vram_usable_mib(t, t, 0.97, sysmem_fallback = True)
        past_default = _vram_usable_mib(t, t, 0.971, sysmem_fallback = True)
        assert at_default == pytest.approx(7168.0)
        assert past_default == pytest.approx(at_default)

    def test_a_lowered_slider_can_still_reserve_more_than_the_floor(self):
        # The floor only ever overrides upward. A user asking for a bigger margin
        # than the floor must still get it.
        t = 8 * GIB
        assert _vram_usable_mib(t, t, 0.80, sysmem_fallback = True) == pytest.approx(
            t - 0.20 * t
        )


class TestPooledIsNotChargedTwice:
    @pytest.mark.parametrize("frac", [0.90, _CTX_FIT_VRAM_FRACTION, 1.0])
    def test_pooled_ignores_the_flag(self, frac):
        # Pooled MiB are already absolute: each card paid its own reserve, raised floor
        # included. Charging the floor again here would take it twice.
        assert _vram_usable_mib(9000, 0, frac, pooled = True, sysmem_fallback = True) == (
            _vram_usable_mib(9000, 0, frac, pooled = True)
        )


class TestRiskClassifier:
    @pytest.fixture(autouse = True)
    def _clear_cache(self):
        # The classifier memoises per binary; a value cached by one case would
        # otherwise answer for the next and the platform gates would look correct
        # while never actually running.
        LlamaCppBackend._SYSMEM_FALLBACK_RISK.clear()
        yield
        LlamaCppBackend._SYSMEM_FALLBACK_RISK.clear()

    def test_false_off_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        # Would be True on Windows with this backend set; the platform gate must win,
        # and must short-circuit before the backend is even consulted.
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_ggml_backends",
            staticmethod(lambda binary = None: frozenset({"cuda", "base"})),
        )
        assert LlamaCppBackend._sysmem_fallback_risk() is False

    def test_false_on_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert LlamaCppBackend._sysmem_fallback_risk() is False

    def test_true_on_windows_cuda(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_ggml_backends",
            staticmethod(lambda binary = None: frozenset({"cuda", "base", "cpu"})),
        )
        assert LlamaCppBackend._sysmem_fallback_risk() is True

    @pytest.mark.parametrize(
        "backends",
        [
            frozenset({"hip", "base"}),      # ROCm: different vendor, different mechanism
            frozenset({"vulkan", "base"}),   # Vulkan does not allocate through CUDA
            frozenset({"cpu", "base"}),      # no GPU backend at all
            frozenset(),                     # unreadable lib dir
        ],
    )
    def test_false_on_windows_without_cuda(self, monkeypatch, backends):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_ggml_backends",
            staticmethod(lambda binary = None: backends),
        )
        assert LlamaCppBackend._sysmem_fallback_risk() is False

    def test_multi_backend_build_with_cuda_counts(self, monkeypatch):
        # If a CUDA lib is present it is what a discrete NVIDIA card will be driven by.
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_ggml_backends",
            staticmethod(lambda binary = None: frozenset({"cuda", "hip", "vulkan"})),
        )
        assert LlamaCppBackend._sysmem_fallback_risk() is True

    def test_classification_failure_keeps_todays_budget(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")

        def _boom(binary = None):
            raise OSError("lib dir vanished mid-load")

        monkeypatch.setattr(
            LlamaCppBackend, "_installed_ggml_backends", staticmethod(_boom)
        )
        # An advisory must never be what fails a model load.
        assert LlamaCppBackend._sysmem_fallback_risk() is False

    def test_read_error_is_not_cached(self, monkeypatch):
        # A transient failure must not pin the safe-but-wrong answer for the process.
        monkeypatch.setattr(sys, "platform", "win32")
        calls = {"n": 0}

        def _flaky(binary = None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient")
            return frozenset({"cuda"})

        monkeypatch.setattr(
            LlamaCppBackend, "_installed_ggml_backends", staticmethod(_flaky)
        )
        assert LlamaCppBackend._sysmem_fallback_risk() is False
        assert LlamaCppBackend._sysmem_fallback_risk() is True

    def test_cache_is_keyed_per_binary(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        table = {
            "C:\\cuda-build\\llama-server.exe": frozenset({"cuda"}),
            "C:\\vulkan-build\\llama-server.exe": frozenset({"vulkan"}),
        }
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_ggml_backends",
            staticmethod(lambda binary = None: table[binary]),
        )
        assert LlamaCppBackend._sysmem_fallback_risk("C:\\cuda-build\\llama-server.exe") is True
        assert (
            LlamaCppBackend._sysmem_fallback_risk("C:\\vulkan-build\\llama-server.exe") is False
        )

    def test_second_call_does_not_relist_the_directory(self, monkeypatch):
        # The planner asks once per candidate GPU subset; that must not be a syscall each.
        monkeypatch.setattr(sys, "platform", "win32")
        calls = {"n": 0}

        def _counted(binary = None):
            calls["n"] += 1
            return frozenset({"cuda"})

        monkeypatch.setattr(
            LlamaCppBackend, "_installed_ggml_backends", staticmethod(_counted)
        )
        for _ in range(5):
            assert LlamaCppBackend._sysmem_fallback_risk() is True
        assert calls["n"] == 1


class TestQwen3VL8BOn16GiBCard:
    """The issue-11349 shape, and the near-miss band the fix actually changes.

    Numbers are the measured ones: 36 layers, 8 KV heads, 128/128 head dims, so f16 KV is
    144 KiB/token, plus 4.80 GiB of weights and a 1.40 GiB mmproj-F16.
    """

    WEIGHTS_MIB = 4.80 * GIB
    MMPROJ_MIB = 1.40 * GIB
    CUDA_CTX_MIB = 320.0
    MMPROJ_SAFETY_MIB = 1.40 * GIB * 0.4
    COMPUTE_MIB = 400.0
    TOTAL_MIB = 16 * GIB
    FREE_MIB = 15400.0  # ~1 GiB held by the Windows desktop

    @staticmethod
    def kv_mib(ctx):
        return 36 * ctx * 8 * (128 + 128) * 2 / (1024 * 1024)

    def footprint(self, ctx):
        return (
            self.WEIGHTS_MIB
            + self.MMPROJ_MIB
            + self.CUDA_CTX_MIB
            + self.MMPROJ_SAFETY_MIB
            + self.COMPUTE_MIB
            + self.kv_mib(ctx)
        )

    def test_kv_matches_the_measured_allocation(self):
        # llama-server asked for exactly 9216.00 MiB at -c 65536 on the real card.
        assert self.kv_mib(65536) == pytest.approx(9216.0)

    def test_65536_was_already_rejected_and_still_is(self):
        # Honesty check: the reporter's exact context does NOT change decision. It
        # overflowed the budget before the fix too, so Studio already handed it to --fit.
        fp = self.footprint(65536)
        assert fp > _vram_usable_mib(self.FREE_MIB, self.TOTAL_MIB, _CTX_FIT_VRAM_FRACTION)
        assert fp > _vram_usable_mib(
            self.FREE_MIB, self.TOTAL_MIB, _CTX_FIT_VRAM_FRACTION, sysmem_fallback = True
        )

    def test_near_miss_context_flips_from_pin_to_fit(self):
        # This is what the fix is for: a footprint that fitted with only ~350 MiB of
        # slack, which any fragmentation spike turns into a silent host-RAM spill.
        fp = self.footprint(49152)
        before = _vram_usable_mib(self.FREE_MIB, self.TOTAL_MIB, _CTX_FIT_VRAM_FRACTION)
        after = _vram_usable_mib(
            self.FREE_MIB, self.TOTAL_MIB, _CTX_FIT_VRAM_FRACTION, sysmem_fallback = True
        )
        assert fp <= before, "precondition: today this pins a full offload"
        assert fp > after, "after the fix it is handed to --fit instead"

    def test_comfortable_context_is_untouched(self):
        # The fix must not disturb loads with real headroom.
        fp = self.footprint(32768)
        assert fp <= _vram_usable_mib(
            self.FREE_MIB, self.TOTAL_MIB, _CTX_FIT_VRAM_FRACTION, sysmem_fallback = True
        )

    def test_flip_band_is_bounded(self):
        # Bounding the regression: only footprints inside this band change behaviour,
        # and the band is the difference between the two floors, nothing more.
        before = _vram_usable_mib(self.FREE_MIB, self.TOTAL_MIB, _CTX_FIT_VRAM_FRACTION)
        after = _vram_usable_mib(
            self.FREE_MIB, self.TOTAL_MIB, _CTX_FIT_VRAM_FRACTION, sysmem_fallback = True
        )
        assert before - after == pytest.approx(
            _LLAMA_FIT_TARGET_DEFAULT_MIB
            - min(_VRAM_FLOOR_RESERVE_MIB, (1 - _CTX_FIT_VRAM_FRACTION) * self.TOTAL_MIB)
        )

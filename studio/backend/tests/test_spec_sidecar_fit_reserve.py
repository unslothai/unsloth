# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unsizeable DSpark/DFlash sidecar must be kept free by the child's ``--fit-target``.

Both drafters borrow the target's token_embd/output, so llama.cpp's fitter cannot build a
standalone draft context to measure them: llama-context.cpp throws "requires ctx_other"
and common/fit.cpp logs "fitting without it", then packs the target up to --fit-target.
Studio already prices the drafter (weights + draft KV + compute) for its own context
choice, but the child never heard about it, so on a Mac whose target needed partial
offload the draft allocated past the margin and the first prefill died with Metal
``kIOGPUCommandBufferCallbackErrorOutOfMemory`` (DeepSeek-V4.1 UD-IQ1_S + the 8 GB DSpark
sidecar, --fit-target 6464 MiB). These cells pin:

  * the reserve is added on top of every other margin, whatever path emits the flag;
  * a CPU-pinned drafter and --fit off reserve nothing;
  * on Metal, Auto drops a sidecar whose reserve is what stops the target fitting, and
    an explicit DSpark request keeps it (with the reserve).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as _placement  # noqa: E402
import test_metal_explicit_context_guard as _metal  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_CAPS = {
    "supports_dspark": True,
    "supports_dflash": True,
    "supports_fit_target": True,
    "supports_fit_ctx": True,
    "supports_kv_unified": True,
    "spec_draft_n_max_flag": "--spec-draft-n-max",
}


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    # load_model writes its API key and child logs under the Studio home.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


class TestIntegrityFlags:
    def test_reserve_rides_on_top_of_the_auto_fit_margin(self):
        base = LlamaCppBackend._ctx_integrity_flags(1, True, True, 0, 0, _CAPS)
        flags = LlamaCppBackend._ctx_integrity_flags(
            1, True, True, 0, 0, _CAPS, fit_target_reserve_mib = 7630
        )
        assert int(_flag(flags, "--fit-target")) == int(_flag(base, "--fit-target")) + 7630

    def test_reserve_rides_on_top_of_the_metal_floor(self):
        flags = LlamaCppBackend._ctx_integrity_flags(
            1, True, False, 0, 0, _CAPS, fit_target_floor_mib = 6464, fit_target_reserve_mib = 7630
        )
        assert int(_flag(flags, "--fit-target")) == 6464 + 7630

    def test_reserve_makes_the_legacy_path_speak(self):
        # The legacy path is silent at llama.cpp's own default; a reserve is not that.
        assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(
            1, True, False, 0, 0, _CAPS
        )
        flags = LlamaCppBackend._ctx_integrity_flags(
            1, True, False, 0, 0, _CAPS, fit_target_reserve_mib = 100
        )
        assert int(_flag(flags, "--fit-target")) == 1024 + 100

    @pytest.mark.parametrize("use_fit, caps", [(False, _CAPS), (True, {})])
    def test_no_fit_or_no_flag_emits_nothing(self, use_fit, caps):
        assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(
            1, use_fit, False, 0, 0, caps, fit_target_reserve_mib = 100
        )


class TestReserveArithmetic:
    def _backend(self, file_bytes):
        backend = LlamaCppBackend()
        backend._get_gguf_size_bytes = lambda _path: file_bytes
        return backend

    def test_planner_reserve_wins_when_larger(self):
        assert self._backend(8 * _GIB)._sidecar_fit_reserve_mib(9 * _GIB, "/d.gguf") == 9 * 1024

    def test_file_size_floors_a_flat_fallback(self):
        assert self._backend(8 * _GIB)._sidecar_fit_reserve_mib(0, "/d.gguf") == 8 * 1024

    def test_rounds_up(self):
        assert self._backend(0)._sidecar_fit_reserve_mib(_MIB + 1, "/d.gguf") == 2

    def test_nothing_known_is_zero(self):
        assert self._backend(0)._sidecar_fit_reserve_mib(0, None) == 0


def _cuda_launch(
    tmp_path,
    *,
    kind = "dspark",
    use_fit = True,
    extra_args = None,
):
    backend, gguf = _placement._backend_non_vulkan(tmp_path, memory = [(0, 24_000, 24_000)])
    sidecar = tmp_path / f"{kind}-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._select_gpus = lambda *a, **k: (None, True) if use_fit else ([0], False)
    backend.probe_server_capabilities = lambda _binary = None: dict(_CAPS)
    backend._get_gguf_size_bytes = lambda path: 8 * _GIB if kind in str(path) else 1024
    result = _placement._launch(
        backend,
        gguf,
        speculative_type = kind,
        extra_args = extra_args,
        **{f"{kind}_draft_path": str(sidecar)},
    )
    return result["cmd"], str(sidecar)


class TestDiscreteLaunch:
    @pytest.mark.parametrize("kind", ["dspark", "dflash"])
    def test_fit_on_keeps_the_sidecar_free(self, tmp_path, kind):
        cmd, sidecar = _cuda_launch(tmp_path, kind = kind)
        assert _flag(cmd, "--model-draft") == sidecar
        # llama.cpp's own 1024 MiB plus at least the 8 GiB sidecar.
        assert int(_flag(cmd, "--fit-target")) >= 1024 + 8 * 1024

    def test_fit_off_has_no_fitter_to_tell(self, tmp_path):
        cmd, sidecar = _cuda_launch(tmp_path, use_fit = False)
        assert _flag(cmd, "--model-draft") == sidecar
        assert "--fit-target" not in cmd

    def test_a_cpu_pinned_drafter_reserves_nothing(self, tmp_path):
        cmd, sidecar = _cuda_launch(tmp_path, extra_args = ["--spec-draft-ngl", "0"])
        assert _flag(cmd, "--model-draft") == sidecar
        assert int(_flag(cmd, "--fit-target") or 1024) < 1024 + 8 * 1024


def _metal_launch(tmp_path, monkeypatch, *, mode, budget_gib):
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    real_init = GgufLoadIntent.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("dspark_draft_path", str(sidecar))
        kwargs.setdefault("speculative_type", mode)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(GgufLoadIntent, "__init__", init)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: dict(_CAPS)),
    )
    backend = LlamaCppBackend()
    result = _metal._launch(
        tmp_path,
        monkeypatch,
        n_ctx = 0,
        backend = backend,
        real_fit = True,
        budget_bytes = int(budget_gib * _GIB),
        wired_bytes = int(budget_gib * _GIB),
        weights_bytes = 4 * _GIB,
    )
    return result["cmd"], backend, str(sidecar)


class TestMetalLaunch:
    # In this harness every GGUF reads 4 GiB, so the sidecar does too; the target with
    # its compute and soft overhead prices ~9.2 GiB. 12 GiB holds the target alone but
    # not the target plus the sidecar; 16 GiB holds both.

    def test_auto_drops_a_sidecar_that_does_not_fit(self, tmp_path, monkeypatch):
        cmd, backend, _ = _metal_launch(tmp_path, monkeypatch, mode = "auto", budget_gib = 12)
        assert "--model-draft" not in cmd
        assert backend.spec_fallback_reason == "drafter_no_vram"

    def test_explicit_dspark_keeps_it_with_the_reserve(self, tmp_path, monkeypatch):
        cmd, backend, sidecar = _metal_launch(tmp_path, monkeypatch, mode = "dspark", budget_gib = 12)
        assert _flag(cmd, "--model-draft") == sidecar
        assert int(_flag(cmd, "--fit-target")) >= 1024 + 4 * 1024

    def test_auto_keeps_a_sidecar_that_fits(self, tmp_path, monkeypatch):
        cmd, backend, sidecar = _metal_launch(tmp_path, monkeypatch, mode = "auto", budget_gib = 16)
        assert _flag(cmd, "--model-draft") == sidecar
        assert int(_flag(cmd, "--fit-target")) >= 1024 + 4 * 1024
        assert backend.spec_fallback_reason is None


def test_reserve_is_read_after_the_budget_price():
    """The reserve must not make an unpriced child look budget-priced: it joins the
    rebuild after ``_fit_target_priced`` is read, like the Metal floor."""
    import core.inference.llama_cpp as _llama_cpp

    src = Path(_llama_cpp.__file__).read_text()
    priced = src.index('_fit_target_priced = "--fit-target" in _integrity_flags')
    reserve = src.index("fit_target_reserve_mib = _sidecar_fit_reserve_mib")
    assert priced < reserve


def test_multi_gpu_cuda_does_not_broadcast_the_reserve(tmp_path):
    """--fit-target is one margin applied to every device: adding the whole sidecar to
    it would withhold the sidecar on each card and push target layers to the host."""
    backend, gguf = _placement._backend_non_vulkan(
        tmp_path, memory = [(0, 24_000, 24_000), (1, 24_000, 24_000)]
    )
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._select_gpus = lambda *a, **k: (None, True)
    backend.probe_server_capabilities = lambda _binary = None: dict(_CAPS)
    backend._get_gguf_size_bytes = lambda path: 8 * _GIB if "dspark" in str(path) else 1024
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = str(sidecar)
    )["cmd"]
    assert _flag(cmd, "--model-draft") == str(sidecar)
    assert int(_flag(cmd, "--fit-target") or 1024) < 1024 + 8 * 1024


@pytest.mark.parametrize("spec_type", ["draft-dspark", "draft-dflash"])
def test_a_pass_through_sidecar_is_reserved_too(tmp_path, spec_type):
    """Extras owning --spec-type replace Studio's emission, so the reserve has to follow
    the sidecar the child really loads, even with Studio's own speculation off."""
    backend, gguf = _placement._backend_non_vulkan(tmp_path, memory = [(0, 24_000, 24_000)])
    sidecar = tmp_path / "custom-sidecar-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._select_gpus = lambda *a, **k: (None, True)
    backend.probe_server_capabilities = lambda _binary = None: dict(_CAPS)
    backend._get_gguf_size_bytes = lambda path: 8 * _GIB if "sidecar" in str(path) else 1024
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "off",
        extra_args = ["--spec-type", spec_type, "--model-draft", str(sidecar)],
    )["cmd"]
    assert str(sidecar) in cmd
    assert int(_flag(cmd, "--fit-target")) >= 1024 + 8 * 1024


def _metal_ctx(tmp_path, monkeypatch, mode):
    sub = tmp_path / mode
    sub.mkdir()
    sidecar = sub / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    real_init = GgufLoadIntent.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("dspark_draft_path", str(sidecar))
        kwargs.setdefault("speculative_type", mode)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(GgufLoadIntent, "__init__", init)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: dict(_CAPS)),
    )
    backend = LlamaCppBackend()
    cmd = _metal._launch(
        sub,
        monkeypatch,
        n_ctx = 0,
        backend = backend,
        real_fit = True,
        budget_bytes = 12 * _GIB,
        wired_bytes = 12 * _GIB,
        weights_bytes = 4 * _GIB,
        kv_per_token = 128 * 1024,
    )["cmd"]
    monkeypatch.undo()
    return cmd


def test_metal_auto_drop_releases_the_drafter_charges(tmp_path, monkeypatch):
    """Once Auto drops the sidecar the context must match a load with speculation off:
    the flat draft reserve and the draft compute charge go with it."""
    auto = _metal_ctx(tmp_path, monkeypatch, "auto")
    off = _metal_ctx(tmp_path, monkeypatch, "off")
    assert "--model-draft" not in auto and "--model-draft" not in off
    assert _flag(auto, "-c") == _flag(off, "-c")

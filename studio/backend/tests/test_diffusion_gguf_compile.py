# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the compiled GGUF dequant accelerator (``diffusion_gguf_compile.py``).

Covers install/uninstall idempotency + exact reversibility, the kill-switch, and the
on-by-default behaviour. Runs on CPU -- patching the module attribute is lazy
(torch.compile only traces on the first real call).
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")
gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

from core.inference import diffusion_gguf_compile as gc  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    # Always start and end from a clean, unpatched state so tests do not leak the process-wide patch into each other.
    gc.uninstall_all()
    yield
    gc.uninstall_all()


def test_compiled_dequant_install_uninstall_reversible():
    orig = gguf_utils.dequantize_gguf_tensor
    assert gc.is_compiled_dequant_installed() is False

    assert gc.install_compiled_dequant() is True
    assert gc.is_compiled_dequant_installed() is True
    # The module attribute is now a different (compiled) callable...
    assert gguf_utils.dequantize_gguf_tensor is not orig
    # ...idempotent: a second install is a no-op, attribute unchanged.
    patched = gguf_utils.dequantize_gguf_tensor
    assert gc.install_compiled_dequant() is True
    assert gguf_utils.dequantize_gguf_tensor is patched

    gc.uninstall_compiled_dequant()
    assert gc.is_compiled_dequant_installed() is False
    # Exact original restored.
    assert gguf_utils.dequantize_gguf_tensor is orig
    # Uninstall is idempotent.
    gc.uninstall_compiled_dequant()
    assert gguf_utils.dequantize_gguf_tensor is orig


def test_compiled_dequant_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GGUF_COMPILE_DEQUANT", "0")
    orig = gguf_utils.dequantize_gguf_tensor
    assert gc.install_compiled_dequant() is False
    assert gc.is_compiled_dequant_installed() is False
    assert gguf_utils.dequantize_gguf_tensor is orig


def test_compiled_dequant_on_by_default(monkeypatch):
    # The compiled dequant is the real win, so it is ON without any env opt-in.
    monkeypatch.delenv("UNSLOTH_DIFFUSION_GGUF_COMPILE_DEQUANT", raising=False)
    assert gc.install_compiled_dequant() is True
    assert gc.is_compiled_dequant_installed() is True


def test_uninstall_all(monkeypatch):
    orig = gguf_utils.dequantize_gguf_tensor
    gc.install_compiled_dequant()
    assert gc.is_installed() is True
    gc.uninstall_all()
    assert gc.is_installed() is False
    assert gguf_utils.dequantize_gguf_tensor is orig


class TestGgufTrimmedDimsAreRestored:
    """GGUF stores no leading size-1 axes, so ``nn.Parameter(torch.zeros((1, dim)))`` comes back as
    ``(dim,)`` and diffusers' exact shape check refuses the load. Z-Image is the live case: a GGUF
    pick with Precision = Off died with ``cap_pad_token expected shape torch.Size([1, 3840]), but
    got torch.Size([3840])``.
    """

    @staticmethod
    def _model():
        import torch

        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cap_pad_token = torch.nn.Parameter(torch.zeros((1, 8)))
                self.proj = torch.nn.Linear(8, 4)

        return _M()

    def test_a_trimmed_leading_axis_is_put_back(self):
        import torch

        from core.inference.diffusion import _restore_gguf_trimmed_dims

        model = self._model()
        sd = {"cap_pad_token": torch.arange(8, dtype=torch.float32)}
        out = _restore_gguf_trimmed_dims(model, sd)
        assert tuple(out["cap_pad_token"].shape) == (1, 8)
        # The values must survive the reshape in order, or the pad token is silently scrambled.
        assert torch.equal(out["cap_pad_token"].flatten(), torch.arange(8, dtype=torch.float32))

    def test_a_tensor_of_the_wrong_size_is_left_alone(self):
        """The guard must not rescue a genuinely wrong tensor: element count has to match."""
        import torch

        from core.inference.diffusion import _restore_gguf_trimmed_dims

        model = self._model()
        sd = {"cap_pad_token": torch.zeros(7)}
        out = _restore_gguf_trimmed_dims(model, sd)
        assert tuple(out["cap_pad_token"].shape) == (7,)

    def test_a_transposed_tensor_is_left_alone(self):
        """Same element count, but the expected shape is not the stored shape with 1s in front, so
        this is a real mismatch and must still reach diffusers' error rather than be reshaped."""
        import torch

        from core.inference.diffusion import _restore_gguf_trimmed_dims

        model = self._model()
        sd = {"proj.weight": torch.zeros(8, 4)}  # model wants (4, 8)
        out = _restore_gguf_trimmed_dims(model, sd)
        assert tuple(out["proj.weight"].shape) == (8, 4)

    def test_the_shim_reaches_the_name_single_file_model_actually_calls(self):
        """``single_file_model`` imports the loader at module level, so it holds its own reference.
        Patching only the defining module leaves the real call site bound to the original, which is
        exactly the way the first version of this fix silently did nothing on a live load."""
        import torch

        from diffusers.loaders import single_file_model as sfm
        from diffusers.models import model_loading_utils as mlu

        from core.inference.diffusion import _install_gguf_dim_restore

        originals = (mlu.load_model_dict_into_meta, sfm.load_model_dict_into_meta)
        seen: dict = {}

        def _fake(model, state_dict, *args, **kwargs):
            seen["shape"] = tuple(state_dict["cap_pad_token"].shape)
            return []

        mlu.load_model_dict_into_meta = _fake
        sfm.load_model_dict_into_meta = _fake
        try:
            _install_gguf_dim_restore(logging.getLogger("t"))
            first = sfm.load_model_dict_into_meta
            _install_gguf_dim_restore(logging.getLogger("t"))
            assert sfm.load_model_dict_into_meta is first, "shim wrapped itself twice"
            # Call through the module the loader really uses, not the one that defines it.
            sfm.load_model_dict_into_meta(self._model(), {"cap_pad_token": torch.zeros(8)})
            assert seen["shape"] == (1, 8)
        finally:
            mlu.load_model_dict_into_meta, sfm.load_model_dict_into_meta = originals

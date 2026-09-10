# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the vision-mmproj batch/ubatch default."""

import inspect
import os
from types import SimpleNamespace

import pytest

from studio.backend.core.inference.llama_cpp import (
    LlamaCppBackend,
    _batch_ubatch_for_mmproj,
    _MMPROJ_DEFAULT_N_BATCH_UBATCH,
)
from studio.backend.routes.inference import _launch_vision_mmproj


class TestBatchUbatchForMmproj:
    """Tests for _batch_ubatch_for_mmproj."""

    def test_vision_mmproj_without_override_gets_default(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj("mmproj-F16.gguf", None, None, None)
        assert n_batch == _MMPROJ_DEFAULT_N_BATCH_UBATCH
        assert n_ubatch == _MMPROJ_DEFAULT_N_BATCH_UBATCH

    def test_no_projector_is_unchanged(self):
        # Text-only, vision off, a suppressed projector and a missing or
        # family-mismatched file all reach here as None.
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(None, None, None, None)
        assert n_batch is None
        assert n_ubatch is None

    def test_explicit_n_batch_is_preserved(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj("mmproj-F16.gguf", 1024, None, None)
        assert n_batch == 1024
        assert n_ubatch is None

    def test_explicit_n_ubatch_is_preserved(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj("mmproj-F16.gguf", None, 1024, None)
        assert n_batch is None
        assert n_ubatch == 1024

    @pytest.mark.parametrize("flag", ["--batch-size", "--ubatch-size", "-b", "-ub"])
    def test_extra_arg_override_is_respected(self, flag):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj("mmproj-F16.gguf", None, None, [flag, "1024"])
        assert n_batch is None
        assert n_ubatch is None

    @pytest.mark.parametrize("var", ["LLAMA_ARG_BATCH", "LLAMA_ARG_UBATCH"])
    def test_env_override_is_respected(self, var, monkeypatch):
        monkeypatch.setenv(var, "1024")
        n_batch, n_ubatch = _batch_ubatch_for_mmproj("mmproj-F16.gguf", None, None, None)
        assert n_batch is None
        assert n_ubatch is None


class TestLaunchVisionMmproj:
    """What the estimators must price, so panel and admission match the launch."""

    @staticmethod
    def _config(**kwargs):
        fields = {"is_vision": True, "gguf_mmproj_file": "/m/mmproj-F16.gguf"}
        return SimpleNamespace(**{**fields, **kwargs})

    def test_configured_projector(self):
        assert _launch_vision_mmproj(self._config(), None, False) == "/m/mmproj-F16.gguf"

    def test_text_only_model(self):
        assert _launch_vision_mmproj(self._config(is_vision = False), None, False) is None

    def test_vision_switched_off(self):
        # Only an audio-only projector survives the switch, and it produces no image
        # tokens, so nothing here needs the bigger micro-batch.
        assert _launch_vision_mmproj(self._config(), None, True) is None

    def test_no_mmproj_suppresses_the_configured_one(self):
        assert _launch_vision_mmproj(self._config(), ["--no-mmproj"], False) is None

    def test_inherited_url_survives_no_mmproj(self, monkeypatch):
        # --no-mmproj empties the command line without clearing mmproj.path, so the
        # child still opens an inherited projector.
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
        got = _launch_vision_mmproj(self._config(), ["--no-mmproj"], False)
        assert got == "https://example.invalid/mmproj.gguf"

    def test_inherited_path_must_exist(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(tmp_path / "gone.gguf"))
        assert _launch_vision_mmproj(self._config(gguf_mmproj_file = None), None, False) is None
        real = tmp_path / "mmproj.gguf"
        real.write_bytes(b"")
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(real))
        got = _launch_vision_mmproj(self._config(gguf_mmproj_file = None), None, False)
        assert got == str(real)

    def test_extras_mmproj_last_wins(self, tmp_path):
        override = tmp_path / "custom.gguf"
        override.write_bytes(b"")
        got = _launch_vision_mmproj(self._config(), ["--mmproj", str(override)], False)
        assert got == str(override)

    @pytest.fixture(autouse = True)
    def _no_inherited_projector(self, monkeypatch):
        for var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"):
            monkeypatch.delenv(var, raising = False)
        assert not os.environ.get("LLAMA_ARG_MMPROJ")


def test_default_is_decided_from_the_resolved_projector_before_the_fit():
    """Order inside ``load_model``: download, resolve, decide, then price.

    ``_resolve_gguf_load_intent`` leaves ``intent.mmproj_path`` unset for a repo id and
    ``_download_mmproj`` is what assigns it, so deciding at the intent unpack reads None
    on the ordinary loading path and never raises the sizes at all. The decision needs
    the resolved launch path rather than the requested one, and has to land before the
    fit, which prices the compute buffer off the micro-batch that launches.
    """
    source = inspect.getsource(LlamaCppBackend.load_model)
    download = source.index("self._download_mmproj(")
    resolve = source.index("self._resolve_launch_mmproj_path(")
    decide = source.index("_batch_ubatch_for_mmproj(")
    price = source.index("_ubatch_for_slots(n_parallel)")
    assert download < resolve < decide < price

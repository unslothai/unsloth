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
    _child_effective_mmproj,
    _mmproj_opens_images,
    _MMPROJ_DEFAULT_N_BATCH_UBATCH,
)
from studio.backend.routes.inference import _launch_vision_mmproj, _remote_opens_vision_mmproj


class TestBatchUbatchForMmproj:
    """Tests for _batch_ubatch_for_mmproj."""

    def test_vision_mmproj_without_override_gets_default(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(True, None, None, None)
        assert n_batch == _MMPROJ_DEFAULT_N_BATCH_UBATCH
        assert n_ubatch == _MMPROJ_DEFAULT_N_BATCH_UBATCH

    def test_no_projector_is_unchanged(self):
        # Text-only, vision off, an audio-only encoder, a suppressed projector and a
        # missing or family-mismatched file all reach here as False.
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(False, None, None, None)
        assert n_batch is None
        assert n_ubatch is None

    def test_explicit_n_batch_is_preserved(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(True, 1024, None, None)
        assert n_batch == 1024
        assert n_ubatch is None

    def test_explicit_n_ubatch_is_preserved(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(True, None, 1024, None)
        assert n_batch is None
        assert n_ubatch == 1024

    @pytest.mark.parametrize("flag", ["--batch-size", "--ubatch-size", "-b", "-ub"])
    def test_extra_arg_override_is_respected(self, flag):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(True, None, None, [flag, "1024"])
        assert n_batch is None
        assert n_ubatch is None

    @pytest.mark.parametrize("var", ["LLAMA_ARG_BATCH", "LLAMA_ARG_UBATCH"])
    def test_env_override_is_respected(self, var, monkeypatch):
        monkeypatch.setenv(var, "1024")
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(True, None, None, None)
        assert n_batch is None
        assert n_ubatch is None


class TestChildEffectiveMmproj:
    """Whose projector llama-server ends up holding."""

    def test_emitted_wins_over_nothing(self):
        assert _child_effective_mmproj("/m/own.gguf", {}) == "/m/own.gguf"

    def test_url_outranks_the_emitted_flag(self):
        # The URL download overwrites mmproj.path after argv is parsed, so a
        # configured audio-only projector does not decide what the child opens.
        env = {"LLAMA_ARG_MMPROJ_URL": "https://example.invalid/vision.gguf"}
        assert _child_effective_mmproj("/m/audio.gguf", env) == env["LLAMA_ARG_MMPROJ_URL"]

    def test_inherited_path_only_fills_a_gap(self, tmp_path):
        real = tmp_path / "mmproj.gguf"
        real.write_bytes(b"")
        env = {"LLAMA_ARG_MMPROJ": str(real)}
        assert _child_effective_mmproj("/m/own.gguf", env) == "/m/own.gguf"
        assert _child_effective_mmproj(None, env) == str(real)

    def test_inherited_path_must_exist(self, tmp_path):
        env = {"LLAMA_ARG_MMPROJ": str(tmp_path / "gone.gguf")}
        assert _child_effective_mmproj(None, env) is None

    def test_nothing_anywhere(self):
        assert _child_effective_mmproj(None, {}) is None


class TestMmprojOpensImages:
    """ModelConfig calls every discovered projector vision, so ask the file."""

    @staticmethod
    def _answer(monkeypatch, value):
        import utils.models.gguf_metadata as meta
        monkeypatch.setattr(meta, "mmproj_accepts_image", lambda path: value)

    def test_no_projector(self):
        assert _mmproj_opens_images(None) is False
        assert _mmproj_opens_images("") is False

    def test_image_projector(self, monkeypatch):
        self._answer(monkeypatch, True)
        assert _mmproj_opens_images("/m/mmproj-F16.gguf") is True

    def test_audio_only_projector(self, monkeypatch):
        # ultravox, Voxtral, Qwen3-ASR: no image chunk can reach the assertion, so the
        # bigger micro-batch would be reserved against nothing.
        self._answer(monkeypatch, False)
        assert _mmproj_opens_images("/m/mmproj-F16.gguf") is False

    def test_unreadable_stays_image_capable(self, monkeypatch):
        import utils.models.gguf_metadata as meta

        def _boom(path):
            raise OSError("unreadable")

        monkeypatch.setattr(meta, "mmproj_accepts_image", _boom)
        assert _mmproj_opens_images("/m/mmproj-F16.gguf") is True


class TestRemoteOpensVisionMmproj:
    """Nothing is downloaded yet, so charge a vision repo as image-capable."""

    @staticmethod
    def _config(**kwargs):
        return SimpleNamespace(**{"is_vision": True, **kwargs})

    def test_vision_repo(self):
        assert _remote_opens_vision_mmproj(self._config(), None, False) is True

    def test_text_only_repo(self):
        assert _remote_opens_vision_mmproj(self._config(is_vision = False), None, False) is False

    def test_vision_switched_off(self):
        assert _remote_opens_vision_mmproj(self._config(), None, True) is False

    def test_no_mmproj(self):
        assert _remote_opens_vision_mmproj(self._config(), ["--no-mmproj"], False) is False

    def test_no_mmproj_still_counts_an_inherited_projector(self, monkeypatch):
        # --no-mmproj empties the command line; the child still opens the env's one.
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
        assert _remote_opens_vision_mmproj(self._config(), ["--no-mmproj"], False) is True

    @pytest.fixture(autouse = True)
    def _no_inherited_projector(self, monkeypatch):
        for var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"):
            monkeypatch.delenv(var, raising = False)


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
        # child still opens an inherited projector, and a URL outranks even a
        # configured projector this launch would emit.
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
        for extras in (["--no-mmproj"], None):
            got = _launch_vision_mmproj(self._config(), extras, False)
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


def test_resident_files_subtract_the_term_they_were_priced_with():
    """``_gguf_resident_file_gb`` is exact only while both arms stay paired.

    It reports files as ``_estimate_gguf_required_gb`` minus the context term that
    function added, so a term added at the raised micro-batch and taken away at 512
    would move the weights figure by the difference.
    """
    from studio.backend.routes import inference as routes

    required = inspect.getsource(routes._estimate_gguf_required_gb)
    resident = inspect.getsource(routes._gguf_resident_file_gb)
    for arm in ("_launch_vision_mmproj(", "_remote_opens_vision_mmproj("):
        assert arm in required
        assert arm in resident

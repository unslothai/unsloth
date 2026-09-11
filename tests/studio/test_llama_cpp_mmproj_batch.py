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
    _mmproj_needs_bigger_ubatch,
    _mmproj_opens_images,
    _MMPROJ_DEFAULT_N_BATCH_UBATCH,
)
from studio.backend.routes.inference import _launch_vision_mmproj, _remote_opens_vision_mmproj


class TestBatchUbatchForMmproj:
    """Tests for _batch_ubatch_for_mmproj."""

    def test_vision_mmproj_without_override_gets_default(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(True, None, None, None, {})
        assert n_batch == _MMPROJ_DEFAULT_N_BATCH_UBATCH
        assert n_ubatch == _MMPROJ_DEFAULT_N_BATCH_UBATCH

    def test_no_projector_is_unchanged(self):
        # Text-only, vision off, an audio-only encoder, a suppressed projector and a
        # missing or family-mismatched file all reach here as False.
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(False, None, None, None, {})
        assert n_batch is None
        assert n_ubatch is None

    def test_explicit_n_ubatch_is_preserved(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(True, None, 1024, None, {})
        assert n_batch is None
        assert n_ubatch == 1024

    @pytest.mark.parametrize("flag", ["--ubatch-size", "-ub"])
    def test_a_named_micro_batch_is_left_alone(self, flag):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(True, None, None, [flag, "1024"], {})
        assert n_batch is None
        assert n_ubatch is None

    def test_env_micro_batch_is_left_alone(self):
        env = {"LLAMA_ARG_UBATCH": "1024"}
        assert _batch_ubatch_for_mmproj(True, None, None, None, env) == (None, None)

    @pytest.mark.parametrize(
        "batch, expected",
        [(4096, (4096, _MMPROJ_DEFAULT_N_BATCH_UBATCH)), (1024, (1024, 1024))],
    )
    def test_a_named_batch_caps_the_raise(self, batch, expected):
        # A batch-only setting says nothing about the micro-batch, and leaving that at
        # 512 is the abort this exists to stop. mtmd cuts the image into n_batch
        # chunks, so the batch caps how big the micro-batch has to be.
        assert _batch_ubatch_for_mmproj(True, batch, None, None, {}) == expected

    @pytest.mark.parametrize("batch", ["-1", "-2"])
    def test_a_negative_batch_is_read_as_llama_cpp_reads_it(self, batch):
        # common_params stores the batch signed and llama_context_params casts it to
        # uint32_t, so -1 reaches the child as 4294967295 and caps nothing. Taking the
        # raw value would read as a batch under the ubatch and skip the raise.
        assert _batch_ubatch_for_mmproj(True, None, None, ["-b", batch], {}) == (
            None,
            _MMPROJ_DEFAULT_N_BATCH_UBATCH,
        )

    def test_a_negative_env_batch_is_normalized_too(self):
        env = {"LLAMA_ARG_BATCH": "-1"}
        assert _batch_ubatch_for_mmproj(True, None, None, None, env) == (
            None,
            _MMPROJ_DEFAULT_N_BATCH_UBATCH,
        )

    def test_a_small_batch_already_holds_the_chunk(self):
        # -b 256 makes every chunk 256, which the llama.cpp default 512 holds.
        assert _batch_ubatch_for_mmproj(True, None, None, ["-b", "256"], {}) == (None, None)

    def test_a_batch_in_the_extras_caps_the_raise(self):
        # The field stays None so Unsloth emits no --batch-size; the extras keep theirs.
        assert _batch_ubatch_for_mmproj(True, None, None, ["-b", "1024"], {}) == (None, 1024)


class TestChildEffectiveMmproj:
    """Whose projector llama-server ends up holding."""

    def test_emitted_wins_over_nothing(self):
        assert _child_effective_mmproj("/m/own.gguf", None, {}) == "/m/own.gguf"

    def test_url_outranks_the_emitted_flag(self):
        # The URL download overwrites mmproj.path after argv is parsed, so a
        # configured audio-only projector does not decide what the child opens.
        env = {"LLAMA_ARG_MMPROJ_URL": "https://example.invalid/vision.gguf"}
        assert _child_effective_mmproj("/m/audio.gguf", None, env) == env["LLAMA_ARG_MMPROJ_URL"]

    def test_inherited_path_only_fills_a_gap(self, tmp_path):
        real = tmp_path / "mmproj.gguf"
        real.write_bytes(b"")
        env = {"LLAMA_ARG_MMPROJ": str(real)}
        assert _child_effective_mmproj("/m/own.gguf", None, env) == "/m/own.gguf"
        assert _child_effective_mmproj(None, None, env) == str(real)

    def test_inherited_path_must_exist(self, tmp_path):
        env = {"LLAMA_ARG_MMPROJ": str(tmp_path / "gone.gguf")}
        assert _child_effective_mmproj(None, None, env) is None

    def test_pass_through_flag_beats_the_emitted_one(self, tmp_path):
        # Extras are appended after the managed flags, so they last-win at the child.
        override = tmp_path / "custom.gguf"
        override.write_bytes(b"")
        got = _child_effective_mmproj("/m/own.gguf", ["--mmproj", str(override)], {})
        assert got == str(override)

    def test_url_beats_the_pass_through_flag(self, tmp_path):
        override = tmp_path / "custom.gguf"
        override.write_bytes(b"")
        env = {"LLAMA_ARG_MMPROJ_URL": "https://example.invalid/vision.gguf"}
        got = _child_effective_mmproj(None, ["--mmproj", str(override)], env)
        assert got == env["LLAMA_ARG_MMPROJ_URL"]

    def test_nothing_anywhere(self):
        assert _child_effective_mmproj(None, None, {}) is None


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


class TestMmprojNeedsBiggerUbatch:
    """Only the projectors that can actually abort the server pay for the raise."""

    @staticmethod
    def _projector(monkeypatch, tmp_path, value):
        import utils.models.gguf_metadata as meta

        path = tmp_path / "mmproj.gguf"
        path.write_bytes(b"")
        monkeypatch.setattr(meta, "read_mmproj_vision_projector_type", lambda p: value)
        return str(path)

    @pytest.mark.parametrize(
        "projector, n_embd, expected",
        [
            # gemma4uv is non-causal for every text size; 1120 tokens clears 512.
            ("gemma4uv", 3840, True),
            # gemma4v is non-causal EXCEPT on E2B (1536) and E4B (2560).
            ("gemma4v", 3840, True),
            ("gemma4v", 2560, False),
            ("gemma4v", 1536, False),
            # Non-causal but capped under the stock ubatch: 256 and 384 tokens.
            ("gemma3", 2560, False),
            ("deepseek4v", 4096, False),
            # Causal, so no image size reaches the assert.
            ("qwen3vl_merger", 2048, False),
            ("youtuvl", 4096, False),
            ("pixtral", 4096, False),
        ],
    )
    def test_projector_families(self, monkeypatch, tmp_path, projector, n_embd, expected):
        path = self._projector(monkeypatch, tmp_path, projector)
        assert _mmproj_needs_bigger_ubatch(path, n_embd) is expected

    def test_no_projector(self):
        assert _mmproj_needs_bigger_ubatch(None, 4096) is False

    def test_an_unnamed_family_keeps_the_raise(self, monkeypatch, tmp_path):
        # A header without the key could still be a Gemma 4, and guessing wrong there
        # crashes the server rather than costing an offload.
        path = self._projector(monkeypatch, tmp_path, None)
        assert _mmproj_needs_bigger_ubatch(path, 3840) is True

    def test_an_unfetched_url_keeps_the_raise(self, monkeypatch):
        import utils.models.gguf_metadata as meta
        monkeypatch.setattr(meta, "read_mmproj_vision_projector_type", lambda p: None)
        assert _mmproj_needs_bigger_ubatch("https://example.invalid/mmproj.gguf", 3840) is True

    def test_an_audio_only_projector_does_not_pay(self, monkeypatch, tmp_path):
        path = self._projector(monkeypatch, tmp_path, "ultravox")
        assert _mmproj_needs_bigger_ubatch(path, 4096) is False

    def test_mmproj_auto_with_nothing_resolved_keeps_the_raise(self):
        # --mmproj-auto leaves llama-server discovering an adjacent projector that
        # never reached the command line, so nothing here can open it and classify it.
        assert _mmproj_needs_bigger_ubatch(None, 3840, ["--mmproj-auto"]) is True
        # Last-wins, exactly as llama-server parses the trio.
        assert _mmproj_needs_bigger_ubatch(None, 3840, ["--mmproj-auto", "--no-mmproj"]) is False
        assert _mmproj_needs_bigger_ubatch(None, 3840, ["--no-mmproj", "--mmproj-auto"]) is True

    def test_mmproj_auto_on_a_text_only_model_pays_nothing(self):
        assert _mmproj_needs_bigger_ubatch(None, 3840, ["--mmproj-auto"], is_vision = False) is False


class TestRemoteOpensVisionMmproj:
    """Nothing is downloaded yet, so charge a vision repo as image-capable."""

    @staticmethod
    def _config(**kwargs):
        return SimpleNamespace(**{"is_vision": True, "gguf_hf_repo": "owner/repo", **kwargs})

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

    def test_the_repo_projector_outranks_an_inherited_path(self, monkeypatch, tmp_path):
        # The download emits it, so a plain inherited path only fills a gap; charging
        # 512 here would admit a load that then launches at 2048.
        import utils.models.gguf_metadata as meta

        real = tmp_path / "audio.gguf"
        real.write_bytes(b"")
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(real))
        # Only the inherited file is audio-only; the unfetched repo one cannot be read.
        monkeypatch.setattr(meta, "mmproj_accepts_image", lambda path: path != str(real))
        assert _remote_opens_vision_mmproj(self._config(), None, False) is True
        # ...and it is the inherited one that answers once no repo projector is emitted.
        assert _remote_opens_vision_mmproj(self._config(), ["--no-mmproj"], False) is False

    def test_an_inherited_audio_projector_is_classified(self, monkeypatch, tmp_path):
        # It is on this disk, so ask it rather than reserving for images it cannot make.
        import utils.models.gguf_metadata as meta

        real = tmp_path / "mmproj.gguf"
        real.write_bytes(b"")
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(real))
        monkeypatch.setattr(meta, "mmproj_accepts_image", lambda path: False)
        assert _remote_opens_vision_mmproj(self._config(), ["--no-mmproj"], False) is False
        monkeypatch.setattr(meta, "mmproj_accepts_image", lambda path: True)
        assert _remote_opens_vision_mmproj(self._config(), ["--no-mmproj"], False) is True

    @pytest.fixture(autouse = True)
    def _no_inherited_projector(self, monkeypatch):
        for var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"):
            monkeypatch.delenv(var, raising = False)


class TestLaunchVisionMmproj:
    """What the estimators must price, so panel and admission match the launch."""

    @pytest.fixture(autouse = True)
    def _files(self, monkeypatch, tmp_path):
        import utils.models.model_config as mc

        monkeypatch.setattr(mc, "mmproj_matches_model_family", lambda model, mmproj: True)
        for var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"):
            monkeypatch.delenv(var, raising = False)
        self.model = tmp_path / "model.gguf"
        self.model.write_bytes(b"")
        self.mmproj = tmp_path / "mmproj-F16.gguf"
        self.mmproj.write_bytes(b"")

    def _config(self, **kwargs):
        fields = {
            "is_vision": True,
            "gguf_file": str(self.model),
            "gguf_mmproj_file": str(self.mmproj),
        }
        return SimpleNamespace(**{**fields, **kwargs})

    def test_configured_projector(self):
        assert _launch_vision_mmproj(self._config(), None, False) == str(self.mmproj)

    def test_text_only_model(self):
        assert _launch_vision_mmproj(self._config(is_vision = False), None, False) is None

    def test_a_projector_that_is_not_on_disk(self, tmp_path):
        gone = self._config(gguf_mmproj_file = str(tmp_path / "gone.gguf"))
        assert _launch_vision_mmproj(gone, None, False) is None

    def test_a_projector_from_the_wrong_family(self, monkeypatch):
        import utils.models.model_config as mc
        monkeypatch.setattr(mc, "mmproj_matches_model_family", lambda model, mmproj: False)
        assert _launch_vision_mmproj(self._config(), None, False) is None

    def test_vision_switched_off(self):
        # Only an audio-only projector survives the switch, and it produces no image
        # tokens, so nothing here needs the bigger micro-batch.
        assert _launch_vision_mmproj(self._config(), None, True) is None

    def test_no_mmproj_suppresses_the_configured_one(self):
        assert _launch_vision_mmproj(self._config(), ["--no-mmproj"], False) is None

    def test_a_pass_through_projector_survives_the_switch(self, tmp_path):
        # The switch never strips the extras, so the child still opens this one.
        override = tmp_path / "custom.gguf"
        override.write_bytes(b"")
        got = _launch_vision_mmproj(self._config(), ["--mmproj", str(override)], True)
        assert got == str(override)

    def test_inherited_url_survives_no_mmproj(self, monkeypatch):
        # --no-mmproj empties the command line without clearing mmproj.path, so the
        # child still opens an inherited projector, and a URL outranks even a
        # configured projector this launch would emit.
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
        for extras in (["--no-mmproj"], None):
            got = _launch_vision_mmproj(self._config(), extras, False)
            assert got == "https://example.invalid/mmproj.gguf"

    def test_the_switch_scrubs_the_inherited_pair(self, monkeypatch):
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
        assert _launch_vision_mmproj(self._config(), None, True) is None

    def test_inherited_path_must_exist(self, monkeypatch, tmp_path):
        bare = self._config(gguf_mmproj_file = None)
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(tmp_path / "gone.gguf"))
        assert _launch_vision_mmproj(bare, None, False) is None
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(self.mmproj))
        assert _launch_vision_mmproj(bare, None, False) == str(self.mmproj)

    def test_extras_mmproj_last_wins(self, tmp_path):
        override = tmp_path / "custom.gguf"
        override.write_bytes(b"")
        got = _launch_vision_mmproj(self._config(), ["--mmproj", str(override)], False)
        assert got == str(override)


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


def test_the_estimators_classify_the_projector_as_the_launch_does():
    """Same four inputs on both sides, or the panel prices a micro-batch nothing runs.

    ``_mmproj_needs_bigger_ubatch`` reads the extras for ``--mmproj-auto`` and the
    vision state for whether discovery can find anything, so an estimator that passes
    only the path answers differently from ``load_model`` on a load that has no
    configured projector but lets llama-server go looking for one.
    """
    from studio.backend.routes import inference as routes

    runtime = inspect.getsource(routes._gguf_runtime_bytes)
    assert "llama_extra_args," in runtime.split("_mmproj_needs_bigger_ubatch(")[1][:220]
    assert "is_vision = launch_is_vision" in runtime
    for fn in (
        routes._estimate_gguf_required_gb,
        routes._gguf_resident_file_gb,
        routes._gguf_memory_breakdown,
    ):
        assert "launch_is_vision" in inspect.getsource(fn), fn.__name__


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

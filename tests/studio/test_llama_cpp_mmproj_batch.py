# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the Gemma 4 micro-batch default.

llama.cpp aborts a non-causal image decode when the chunk mtmd cuts exceeds n_ubatch.
Only the Gemma 4 towers produce such a chunk at the stock 512, so only they get the
raise; see ``_MMPROJ_DEFAULT_N_BATCH_UBATCH``.
"""

import inspect

import pytest

from studio.backend.core.inference.llama_cpp import (
    LlamaCppBackend,
    _batch_ubatch_for_mmproj,
    _launch_needs_bigger_ubatch,
    _mmproj_emits_oversized_chunks,
    _MMPROJ_DEFAULT_N_BATCH_UBATCH,
)


@pytest.fixture
def projector(monkeypatch, tmp_path):
    """A readable mmproj whose family and image capability the test chooses."""

    def _make(
        family,
        *,
        accepts_image = True,
        name = "mmproj.gguf",
    ):
        import utils.models.gguf_metadata as meta

        path = tmp_path / name
        path.write_bytes(b"")
        families = getattr(_make, "_families", {})
        images = getattr(_make, "_images", {})
        families[str(path)] = family
        images[str(path)] = accepts_image
        _make._families, _make._images = families, images
        monkeypatch.setattr(
            meta, "read_mmproj_vision_projector_type", lambda p: families.get(str(p))
        )
        monkeypatch.setattr(meta, "mmproj_accepts_image", lambda p: images.get(str(p), True))
        return str(path)

    return _make


class TestMmprojEmitsOversizedChunks:
    """Which projector families can produce a chunk the 512 default cannot hold."""

    @pytest.mark.parametrize(
        "family, n_embd, expected",
        [
            # Non-causal at every text size, and 1120 tokens per image clears 512.
            ("gemma4uv", 3840, True),
            # Non-causal EXCEPT on E2B (n_embd 1536) and E4B (2560).
            ("gemma4v", 3840, True),
            ("gemma4v", 2560, False),
            ("gemma4v", 1536, False),
            # Non-causal but capped under the stock ubatch: 256 and 384.
            ("gemma3", 2560, False),
            ("deepseek4v", 4096, False),
            # Causal, so no image size reaches the assert.
            ("qwen3vl_merger", 2048, False),
            ("youtuvl", 4096, False),
            ("pixtral", 4096, False),
        ],
    )
    def test_families(self, projector, family, n_embd, expected):
        assert _mmproj_emits_oversized_chunks(projector(family), n_embd) is expected

    def test_no_projector(self):
        assert _mmproj_emits_oversized_chunks(None) is False

    def test_an_audio_only_encoder_makes_no_image_chunk(self, projector):
        # ModelConfig calls every discovered mmproj vision, so these reach here as
        # vision and must still pay nothing.
        path = projector("ultravox", accepts_image = False)
        assert _mmproj_emits_oversized_chunks(path, 4096) is False

    def test_an_unnamed_family_is_assumed_oversized(self, projector):
        # It could be a Gemma 4, and guessing wrong costs a crash, not an offload.
        assert _mmproj_emits_oversized_chunks(projector(None), 3840) is True

    def test_an_unreadable_file_is_assumed_oversized(self, monkeypatch):
        import utils.models.gguf_metadata as meta

        def _boom(path):
            raise OSError("unreadable")

        monkeypatch.setattr(meta, "mmproj_accepts_image", _boom)
        assert _mmproj_emits_oversized_chunks("/m/mmproj-F16.gguf", 3840) is True


class TestLaunchNeedsBiggerUbatch:
    """Every projector source that can reach the child, asked as one question."""

    def test_the_models_own_projector(self, projector):
        assert _launch_needs_bigger_ubatch(projector("gemma4uv"), 3840, env = {}) is True
        assert _launch_needs_bigger_ubatch(projector("qwen3vl_merger"), 2048, env = {}) is False

    def test_a_text_only_model(self, projector):
        got = _launch_needs_bigger_ubatch(None, 4096, is_vision = False, env = {})
        assert got is False

    def test_the_vision_switch_drops_the_models_own(self, projector):
        path = projector("gemma4uv")
        assert _launch_needs_bigger_ubatch(path, 3840, vision_off = True, env = {}) is False

    def test_no_mmproj_drops_the_models_own(self, projector):
        path = projector("gemma4uv")
        assert _launch_needs_bigger_ubatch(path, 3840, ["--no-mmproj"], env = {}) is False

    def test_a_pass_through_projector_survives_both(self, projector):
        # Appended after the managed flags and stripped by neither the switch nor
        # --no-mmproj, so this opens an image tower regardless.
        path = projector("gemma4uv")
        for extra_kwargs in ({}, {"vision_off": True}):
            got = _launch_needs_bigger_ubatch(
                None, 3840, ["--mmproj", path], env = {}, **extra_kwargs
            )
            assert got is True

    def test_a_pass_through_projector_is_still_classified(self, projector):
        path = projector("qwen3vl_merger")
        assert _launch_needs_bigger_ubatch(None, 2048, ["--mmproj", path], env = {}) is False

    def test_an_inherited_projector_survives_no_mmproj(self, projector):
        # --no-mmproj empties the command line without clearing mmproj.path.
        path = projector("gemma4uv")
        got = _launch_needs_bigger_ubatch(
            None, 3840, ["--no-mmproj"], env = {"LLAMA_ARG_MMPROJ": path}
        )
        assert got is True

    def test_an_inherited_projector_is_still_classified(self, projector):
        path = projector("qwen3vl_merger")
        got = _launch_needs_bigger_ubatch(None, 2048, env = {"LLAMA_ARG_MMPROJ": path})
        assert got is False

    def test_an_unfetched_url_counts(self, projector):
        # Nothing has downloaded it, so it cannot be classified.
        env = {"LLAMA_ARG_MMPROJ_URL": "https://example.invalid/mmproj.gguf"}
        got = _launch_needs_bigger_ubatch(projector("qwen3vl_merger"), 2048, env = env)
        assert got is True

    def test_the_switch_scrubs_the_inherited_pair(self, projector):
        for var in ("LLAMA_ARG_MMPROJ_URL", "LLAMA_ARG_MMPROJ"):
            got = _launch_needs_bigger_ubatch(
                None, 3840, vision_off = True, env = {var: "https://example.invalid/x.gguf"}
            )
            assert got is False

    def test_mmproj_auto_with_nothing_resolved(self, projector):
        # Discovery may open an adjacent projector this process was never told about.
        assert _launch_needs_bigger_ubatch(None, 3840, ["--mmproj-auto"], env = {}) is True
        # Last-wins, exactly as llama-server parses the trio.
        assert (
            _launch_needs_bigger_ubatch(None, 3840, ["--mmproj-auto", "--no-mmproj"], env = {})
            is False
        )
        assert (
            _launch_needs_bigger_ubatch(None, 3840, ["--no-mmproj", "--mmproj-auto"], env = {})
            is True
        )

    def test_mmproj_auto_on_a_text_only_model(self):
        got = _launch_needs_bigger_ubatch(None, 3840, ["--mmproj-auto"], is_vision = False, env = {})
        assert got is False


class TestBatchUbatchForMmproj:
    """Turning the answer into the two flags, without undoing a size the user chose."""

    def test_raised_when_nothing_else_sets_one(self):
        assert _batch_ubatch_for_mmproj(True, None, None, None, {}) == (
            _MMPROJ_DEFAULT_N_BATCH_UBATCH,
            _MMPROJ_DEFAULT_N_BATCH_UBATCH,
        )

    def test_untouched_when_no_projector_needs_it(self):
        assert _batch_ubatch_for_mmproj(False, None, None, None, {}) == (None, None)

    def test_an_explicit_micro_batch_is_preserved(self):
        assert _batch_ubatch_for_mmproj(True, None, 1024, None, {}) == (None, 1024)

    @pytest.mark.parametrize("flag", ["--ubatch-size", "-ub"])
    def test_a_named_micro_batch_is_left_alone(self, flag):
        assert _batch_ubatch_for_mmproj(True, None, None, [flag, "1024"], {}) == (None, None)

    def test_an_env_micro_batch_is_left_alone(self):
        assert _batch_ubatch_for_mmproj(True, None, None, None, {"LLAMA_ARG_UBATCH": "1024"}) == (
            None,
            None,
        )

    @pytest.mark.parametrize(
        "batch, expected",
        [(4096, (4096, _MMPROJ_DEFAULT_N_BATCH_UBATCH)), (1024, (1024, 1024))],
    )
    def test_a_named_batch_caps_the_raise(self, batch, expected):
        # mtmd cuts the image into n_batch chunks, so the batch caps how big the
        # micro-batch must be; it does not cancel the raise.
        assert _batch_ubatch_for_mmproj(True, batch, None, None, {}) == expected

    def test_a_small_batch_already_holds_the_chunk(self):
        # -b 256 makes every chunk 256, which the llama.cpp default 512 holds.
        assert _batch_ubatch_for_mmproj(True, None, None, ["-b", "256"], {}) == (None, None)

    def test_a_batch_in_the_extras_caps_the_raise(self):
        # The field stays None so Unsloth emits no --batch-size; the extras keep theirs.
        assert _batch_ubatch_for_mmproj(True, None, None, ["-b", "1024"], {}) == (None, 1024)

    @pytest.mark.parametrize("source", ["extras", "env"])
    def test_a_negative_batch_is_read_as_llama_cpp_reads_it(self, source):
        # common_params stores the batch signed and llama_context_params casts it to
        # uint32_t, so -1 reaches the child as 4294967295.
        args, env = (["-b", "-1"], {}) if source == "extras" else (None, {"LLAMA_ARG_BATCH": "-1"})
        assert _batch_ubatch_for_mmproj(True, None, None, args, env) == (
            None,
            _MMPROJ_DEFAULT_N_BATCH_UBATCH,
        )


def test_the_decision_lands_after_the_download_and_before_the_fit():
    """Order inside ``load_model``: download, resolve, decide, then price.

    ``_resolve_gguf_load_intent`` leaves ``intent.mmproj_path`` unset for a repo id and
    ``_download_mmproj`` assigns it, so deciding at the intent unpack reads None on the
    ordinary loading path and never raises at all. The decision also has to land before
    the fit, which prices the compute buffer off the micro-batch that launches.
    """
    source = inspect.getsource(LlamaCppBackend.load_model)
    download = source.index("self._download_mmproj(")
    decide = source.index("_batch_ubatch_for_mmproj(")
    price = source.index("_ubatch_for_slots(n_parallel)")
    assert download < decide < price
    # And it decides from the resolved projector, not the requested one: a missing or
    # family-mismatched file launches a text-only server that must not pay for images.
    decision = source[decide : source.index("\n\n", decide)]
    assert "self._resolve_launch_mmproj_path(" in decision


def test_both_sides_read_the_embedding_length_the_same_way():
    """GGUF does not guarantee KV order, and the two readers disagree when it varies.

    ``_read_gguf_metadata`` only matches ``{arch}.`` keys once ``general.architecture``
    has gone past, so a file writing ``embedding_length`` first leaves it unset;
    ``read_gguf_embedding_length`` buffers instead. Only the E2B/E4B test reads this,
    so a split would have the launch raise while the panel priced 512.
    """
    source = inspect.getsource(LlamaCppBackend.load_model)
    decision = source[
        source.index("_batch_ubatch_for_mmproj(") : source.index(
            "\n\n", source.index("_batch_ubatch_for_mmproj(")
        )
    ]
    assert "_read_gguf_embedding_length(" in decision
    assert "self._embedding_length" not in decision


def test_the_estimators_ask_the_same_question_as_the_launch():
    """One helper, called with a config on one side and load state on the other.

    ``_gguf_resident_file_gb`` reports files as ``_estimate_gguf_required_gb`` minus the
    context term that function added, branching on the same local-vs-remote condition
    to stay paired: a term added at 2048 and taken away at 512 would move the weights
    figure by the difference.
    """
    from studio.backend.routes import inference as routes

    assert "_launch_needs_bigger_ubatch" in inspect.getsource(routes._launch_raises_ubatch)
    for fn in (routes._estimate_gguf_required_gb, routes._gguf_resident_file_gb):
        body = inspect.getsource(fn)
        assert "_launch_raises_ubatch(" in body, fn.__name__
        assert "_remote_raises_ubatch(" in body, fn.__name__
    assert "_launch_raises_ubatch(" in inspect.getsource(routes._gguf_memory_breakdown)

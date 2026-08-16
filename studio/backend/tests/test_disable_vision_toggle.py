# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The disable_vision toggle's contract: defaults, reload, and state that must not leak.

The toggle emits no llama.cpp flag. It ORs into the mmproj gates so that
``effective_is_vision`` goes False, which is what greys out the frontend attach
button through the runtime ``is_vision`` echo. So most of the contract is about
defaults, the reload the flag has to force, and the reset paths -- the gates
themselves are covered against the real load_model statements in
test_metal_paravirtual_guard.py and test_mmproj_cpu_pin_policy.py.

Back-compat is the point of the first half: the field is new, so every persisted
blob and every older client omits it, and all of those must behave exactly as
before.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Installs the module stubs the backend import needs. Before core.inference.
from test_llama_cpp_placement import _backend, _launch, _write_gguf  # noqa: E402,F401

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402
from models.inference import (  # noqa: E402
    LoadRequest,
    ValidateModelRequest,
    _InferenceRuntimeFields,
)


class TestDefaultsPreserveTodaysBehaviour:
    """An old client that never heard of the field must load exactly as before."""

    def test_load_request_defaults_to_vision_enabled(self):
        assert LoadRequest(model_path = "m.gguf").disable_vision is False

    def test_validate_request_defaults_to_vision_enabled(self):
        assert ValidateModelRequest(model_path = "m.gguf").disable_vision is False

    def test_intent_defaults_to_vision_enabled(self):
        assert GgufLoadIntent(model_identifier = "m").disable_vision is False

    def test_runtime_fields_default_to_vision_enabled(self):
        fields = _InferenceRuntimeFields()
        assert fields.disable_vision is False
        assert fields.vision_disabled_by_user is False
        assert fields.vision_on_cpu is False


class TestTheThreeEchoesAreIndependent:
    """Three different outcomes the client must be able to tell apart: the setting as
    sent, image input that is off BY REQUEST rather than for want of a projector, and
    a projector that is present but slow. A UI conflating them would tell a user
    images are off when they merely got slower, or send them hunting an mmproj that
    is sitting right there."""

    def test_the_request_echo_carries(self):
        assert LoadRequest(model_path = "m.gguf", disable_vision = True).disable_vision is True
        assert GgufLoadIntent(model_identifier = "m", disable_vision = True).disable_vision is True

    def test_a_slow_projector_is_not_a_disabled_one(self):
        fields = _InferenceRuntimeFields(vision_on_cpu = True)
        assert fields.vision_on_cpu is True
        assert fields.vision_disabled_by_user is False
        assert fields.disable_vision is False

    def test_a_text_only_gguf_carrying_the_flag_is_not_reported_as_switched_off(self):
        # disable_vision echoes what was sent; vision_disabled_by_user is
        # is_vision AND disable_vision, so it stays False here and the client falls
        # through to the generic "cannot accept images" rather than offering a
        # toggle that would change nothing.
        fields = _InferenceRuntimeFields(disable_vision = True, vision_disabled_by_user = False)
        assert fields.disable_vision is True
        assert fields.vision_disabled_by_user is False


class TestPersistedBlobBackCompat:
    def test_an_override_blob_without_the_key_normalizes_to_absent(self):
        from utils.openai_auto_switch_settings import normalize_model_override
        entry = normalize_model_override({"model_id": "owner/repo", "n_ctx": 4096})
        assert "disable_vision" not in entry

    def test_a_stored_true_reaches_the_load_kwargs(self):
        from utils.openai_auto_switch_settings import (
            model_override_load_kwargs,
            normalize_model_override,
        )

        entry = normalize_model_override({"model_id": "owner/repo", "disable_vision": True})
        assert entry["disable_vision"] is True
        assert model_override_load_kwargs(entry, is_gguf = True)["disable_vision"] is True

    def test_a_stored_false_is_not_persisted_at_all(self):
        # Falsey flags are not content: storing them would make an override that
        # says nothing survive the removal sweep.
        from utils.openai_auto_switch_settings import normalize_model_override
        entry = normalize_model_override({"model_id": "owner/repo", "disable_vision": False})
        assert "disable_vision" not in entry


class TestFlippingTheToggleForcesAReload:
    """Toggling changes which files the child opens, so a live server launched the
    other way cannot satisfy the request. Driven through the real comparator: a
    stale True here would leave the user staring at a dead attach button on a
    server that never reloaded."""

    def _loaded(self, tmp_path, *, disable_vision: bool):
        backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 40_000, 48_000)])
        _launch(backend, gguf, disable_vision = disable_vision)
        return backend, gguf

    def _intent(self, gguf, **overrides):
        kwargs = dict(model_identifier = "test", gguf_path = str(gguf))
        kwargs.update(overrides)
        return GgufLoadIntent(**kwargs)

    def test_an_identical_request_still_takes_the_fast_path(self, tmp_path):
        backend, gguf = self._loaded(tmp_path, disable_vision = False)
        assert backend.adopt_load_intent_if_matched(self._intent(gguf)) is True

    def test_turning_vision_off_against_a_vision_server_reloads(self, tmp_path):
        backend, gguf = self._loaded(tmp_path, disable_vision = False)
        assert (
            backend.adopt_load_intent_if_matched(self._intent(gguf, disable_vision = True)) is False
        )

    def test_turning_vision_back_on_reloads_too(self, tmp_path):
        # Both directions: a projector-free server cannot serve an image request.
        backend, gguf = self._loaded(tmp_path, disable_vision = True)
        assert backend.adopt_load_intent_if_matched(self._intent(gguf)) is False
        assert backend.adopt_load_intent_if_matched(self._intent(gguf, disable_vision = True)) is True


class TestTheStateDoesNotLeakToTheNextModel:
    """A stale True makes the NEXT model report a projector state it never had. The
    fields are echoed straight to the UI, so the leak shows up as a greyed-out
    attach button on a model whose vision is perfectly fine."""

    def test_unload_clears_every_vision_field(self):
        backend = LlamaCppBackend()
        backend._is_vision = True
        backend._disable_vision = True
        backend._vision_disabled_by_user = True
        backend._vision_on_cpu = True

        backend.unload_model()

        assert backend._is_vision is False
        assert backend._disable_vision is False
        assert backend._vision_disabled_by_user is False
        assert backend._vision_on_cpu is False

    def test_a_fresh_backend_starts_with_vision_enabled_and_on_gpu(self):
        backend = LlamaCppBackend()
        assert (
            backend._disable_vision,
            backend._vision_disabled_by_user,
            backend._vision_on_cpu,
        ) == (False, False, False)

    def test_the_diffusion_handoff_clears_them_beside_is_vision(self):
        # The diffusion path returns before the assignment that records the vision
        # state, and only unload clears it, so a projector state from the previous
        # GGUF would linger on a model that has no projector concept at all.
        src = inspect.getsource(LlamaCppBackend._start_diffusion_server)
        at = src.index("self._is_vision = False")
        tail = src[at : at + 400]
        assert "self._disable_vision = False" in tail
        assert "self._vision_disabled_by_user = False" in tail
        assert "self._vision_on_cpu = False" in tail

    def test_the_vision_strip_retry_stops_reporting_a_cpu_projector(self):
        # That retry relaunches without --mmproj at all, so there is no projector
        # left running anywhere, on the CPU or otherwise.
        src = inspect.getsource(LlamaCppBackend.load_model)
        # The retry is the block that re-records the spawned argv before relaunching;
        # anchoring on the vision clear alone would match the unrelated resets.
        at = src.rindex("_last_spawn_cmd = list(cmd)")
        block = src[at : src.index("self._start_llama_process(cmd, env)", at)]
        assert "self._is_vision = False" in block
        assert "self._vision_on_cpu = False" in block

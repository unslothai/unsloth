# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A direct holder (completions, embeddings, caption/OCR) has no admission lease, so nothing
keeps its controller from being retired between the lookup and the register: another backend's
first lookup drops every idle controller. Registered on that orphan, the holder is invisible to
the chats on its own key. The lookup and the register are one step under the registry lock."""

from __future__ import annotations

import threading
from types import SimpleNamespace
import uuid

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import get_preemption_controller
import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def test_a_first_lookup_elsewhere_cannot_orphan_a_direct_holder(monkeypatch):
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_openai_llama_preemption_will_apply", lambda *a, **k: True)
    monkeypatch.setattr(inference, "_openai_llama_admission_budget", lambda *a, **k: 16384)
    monkeypatch.setattr(inference, "_refresh_residency_before_planning", lambda controller: None)
    fake = SimpleNamespace(
        base_url = "http://127.0.0.1:65077", _admission_key = f"direct-race-{uuid.uuid4().hex}"
    )
    key = inference._preempt_key(fake)
    other_key = f"other-backend-{uuid.uuid4().hex}"
    real_register = preemption.PreemptionController.register

    def register(self, gen_id, **kwargs):
        if gen_id.startswith("embd-holder-"):
            # Another backend's first lookup lands between this holder's lookup and register.
            other = threading.Thread(target = get_preemption_controller, args = (other_key,))
            other.start()
            other.join(timeout = 0.5)
        return real_register(self, gen_id, **kwargs)

    monkeypatch.setattr(preemption.PreemptionController, "register", register)
    holder = None
    try:
        holder = inference._openai_llama_count_embeddings_holder(fake, "x" * 400)
        assert holder is not None
        assert get_preemption_controller(key).participant(holder) is not None, (
            "the holder registered on a controller later lookups no longer find"
        )
    finally:
        if holder is not None:
            inference._openai_llama_release_completions_holder(fake, holder)
        with preemption._CONTROLLERS_LOCK:
            preemption._CONTROLLERS.pop(key, None)
            preemption._CONTROLLERS.pop(other_key, None)

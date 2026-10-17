# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A diffusion runner records `off` for exact concurrency, not being llama-server and unable to
apply the setting. Compared against an effective `auto` or `on`, the settings route told the user
to reload the diffusion model forever, and the reload recorded `off` again."""

import routes.inference as inference
import routes.settings as settings


class _Runner:
    is_active = True
    is_diffusion = True
    requested_exact_concurrency = "off"


class _Llama(_Runner):
    is_diffusion = False


def test_a_diffusion_runner_does_not_need_a_reload(monkeypatch):
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Runner())
    assert settings._exact_concurrency_reload_required("auto") is False
    assert settings._exact_concurrency_reload_required("on") is False


def test_a_llama_server_child_still_does(monkeypatch):
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Llama())
    assert settings._exact_concurrency_reload_required("auto") is True
    assert settings._exact_concurrency_reload_required("off") is False


class _Idle:
    is_active = False
    is_diffusion = False
    requested_exact_concurrency = "auto"


def test_an_extra_slot_loaded_under_the_old_setting_still_does(monkeypatch):
    # The unrouted primary is idle (or already matches); an extra model slot was loaded under
    # the previous setting and keeps it for every chat routed to it.
    import core.inference.llama_cpp as llama_mod

    extra = _Llama()
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Idle())
    monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [extra])
    assert settings._exact_concurrency_reload_required("auto") is True
    assert settings._exact_concurrency_reload_required("off") is False
    extra.is_diffusion = True
    assert settings._exact_concurrency_reload_required("auto") is False

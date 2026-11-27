# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cache-usability checks must not hit the hub: subdir layout is a fact of the on-disk snapshot."""

import pytest

from hub.utils.hf_cache_state import with_load_subdirs


_BICODEC = "unsloth/Spark-TTS-0.5B"
_PLAIN = "unsloth/Llama-3.2-1B-Instruct"


@pytest.fixture
def detector_spy(monkeypatch):
    """Record how detect_audio_type is called, without touching the network."""
    import utils.models.model_config as model_config

    calls = []

    def fake_detect(
        model_name,
        hf_token = None,
        local_files_only = False,
        revision = None,
    ):
        calls.append(
            {
                "model_name": model_name,
                "local_files_only": local_files_only,
                "revision": revision,
            }
        )
        return "bicodec" if model_name == _BICODEC else None

    monkeypatch.setattr(model_config, "detect_audio_type", fake_detect)
    return calls


def test_cache_resolution_asks_for_the_offline_answer(detector_spy):
    """The regression: a cache probe must not be able to block on the hub."""
    with_load_subdirs(_BICODEC, ("config.json",))

    assert detector_spy, "detect_audio_type was not consulted at all"
    assert all(call["local_files_only"] is True for call in detector_spy), (
        "a cached-snapshot probe reached detect_audio_type without local_files_only, so "
        "resolving an on-disk snapshot can now block on a network read"
    )


def test_the_offline_answer_is_still_the_right_answer(detector_spy):
    """Going offline must not cost the fix its whole point."""
    assert with_load_subdirs(_BICODEC, ("config.json",)) == (
        "config.json",
        "LLM/config.json",
    )
    assert with_load_subdirs(_PLAIN, ("config.json",)) == ("config.json",)


def test_the_security_scanner_keeps_its_network_capable_default(detector_spy):
    """Only the cache path is pinned offline; the scanner wants the remote answer."""
    from utils.security import security_load_subdirs

    assert security_load_subdirs(_BICODEC) == ("LLM",)
    assert detector_spy[-1]["local_files_only"] is False


def test_a_detector_failure_still_degrades_to_root_only(monkeypatch):
    """A raising detector degrades to root-only and skips the YAML fallback, pinned as current behaviour."""
    import utils.models.model_config as model_config

    def boom(*args, **kwargs):
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(model_config, "detect_audio_type", boom)

    assert with_load_subdirs(_BICODEC, ("config.json",)) == ("config.json",)


def test_going_offline_makes_the_yaml_fallback_more_reachable(monkeypatch):
    """Offline detection reports nothing for an uncached repo, letting the YAML registry default answer."""
    import utils.models.model_config as model_config

    monkeypatch.setattr(
        model_config,
        "detect_audio_type",
        lambda model_name, hf_token = None, local_files_only = False, revision = None: None,
    )

    assert with_load_subdirs(_BICODEC, ("config.json",)) == (
        "config.json",
        "LLM/config.json",
    )
    assert with_load_subdirs(_PLAIN, ("config.json",)) == ("config.json",)

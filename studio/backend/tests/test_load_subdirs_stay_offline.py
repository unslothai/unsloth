# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deciding whether a cache on disk is usable must not touch the network.

``590ac9f22`` taught the cached-snapshot resolvers about load subdirectories by routing
them through ``security_load_subdirs``, which calls ``detect_audio_type``. That function
only skips its remote tokenizer fetch when ``local_files_only`` is set, and the new
callers did not set it -- so ``_resolve_model_snapshot`` and the two cache-pin sites, all
previously pure filesystem work, gained a hub round trip with no timeout in front of
them. On a slow or hung hub that turns "is this snapshot already here?" into a stall.

The subdir layout is a property of the snapshot sitting on disk, so the local answer is
also the correct one. ``security_load_subdirs`` keeps its network-capable default for the
security scanner, which genuinely wants the remote answer.
"""

import pytest

from hub.utils.hf_cache_state import with_load_subdirs


_BICODEC = "unsloth/Spark-TTS-0.5B"
_PLAIN = "unsloth/Llama-3.2-1B-Instruct"


@pytest.fixture
def detector_spy(monkeypatch):
    """Record how detect_audio_type is called, without touching the network."""
    calls = []

    import utils.models.model_config as model_config

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
    """A raising detector is a soft failure, not a crash.

    Note it degrades all the way to root-only rather than reaching the YAML fallback:
    ``security_load_subdirs`` wraps both branches in one ``try``, so an exception in
    ``detect_audio_type`` skips the ``load_model_defaults`` check that its own comment
    says is there for exactly that case. That mismatch is byte-identical on
    ``b41b819a4`` and is not this PR's to fix -- pinned here so it is a decision rather
    than a surprise.
    """
    import utils.models.model_config as model_config

    def boom(*args, **kwargs):
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(model_config, "detect_audio_type", boom)

    assert with_load_subdirs(_BICODEC, ("config.json",)) == ("config.json",)


def test_going_offline_makes_the_yaml_fallback_more_reachable(monkeypatch):
    """The upside of pinning the cache path offline.

    A network failure used to raise straight past the YAML fallback. Asked with
    ``local_files_only``, detection simply reports nothing for an uncached repo, so the
    registry default gets its turn and a known bicodec repo is still identified.
    """
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

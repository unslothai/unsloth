# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""utils.prebuilt.llama_backend agrees with install_llama_prebuilt.py.

The installer owns backend selection and writes the marker; the backend reads that
marker directly on paths where spawning the installer is not an option (the
model-load recovery gate runs per load, the status endpoints per poll). Two
implementations of one contract drift, so this compares them on the same inputs
rather than trusting a comment.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_STUDIO = _BACKEND.parent
for _path in (str(_BACKEND), str(_STUDIO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

ilp = importlib.import_module("install_llama_prebuilt")
from utils.prebuilt import llama_backend as backend_marker  # noqa: E402

# Marker shapes shared by the installer and backend reader.
MARKERS = [
    {},
    {"asset": "app-b1-linux-x64-cuda12-older.tar.gz"},
    {"asset": "app-b1-linux-x64-cpu.tar.gz", "force_cpu": True},
    {"asset": "app-b1-linux-x64-cpu.tar.gz", "force_cpu": False},
    {"asset": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz"},
    {"asset": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz", "llama_backend": None},
    {"asset": "app-b1-win-vulkan.zip", "llama_backend": "auto"},
    {"asset": "app-b1-win-vulkan.zip", "llama_backend": "vulkan"},
    {"asset": "app-b1-win-vulkan.zip", "llama_backend": ""},
    {"asset": "x.tar.gz", "llama_backend": "sycl"},
    {"asset": "x.tar.gz", "llama_backend": 7},
    {"backend": "cuda", "backend_request": "auto"},
    {"backend": "cpu", "backend_request": "cpu", "force_cpu": True},
    {"backend": "vulkan", "backend_request": "vulkan"},
    {"backend": "rocm", "backend_request": "hip"},
    {"backend": "sycl", "backend_request": "sycl"},
]


@pytest.fixture(autouse = True)
def _no_ambient_backend_env(monkeypatch):
    for name in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN"):
        monkeypatch.delenv(name, raising = False)


@pytest.mark.parametrize("marker", MARKERS, ids = range(len(MARKERS)))
def test_both_read_the_same_choice_from_a_marker(tmp_path, marker):
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"release_tag": "b1", **marker}), encoding = "utf-8"
    )
    assert backend_marker.marker_backend_request(marker) == ilp.persisted_backend_request(tmp_path)


def test_the_install_kind_maps_are_identical():
    assert backend_marker.INSTALL_KIND_BACKENDS == ilp.INSTALL_KIND_BACKENDS


def test_the_requestable_backends_are_identical():
    assert backend_marker.REQUESTABLE_BACKENDS == ilp.REQUESTABLE_BACKENDS


@pytest.mark.parametrize(
    "primary,legacy,expected",
    [
        (None, None, None),
        (None, "on", "vulkan"),
        ("auto", "on", "auto"),
        ("cpu", "on", "cpu"),
        ("hip", None, "rocm"),
        ("metal", "on", "vulkan"),
        ("unknown", "true", "vulkan"),
    ],
)
def test_public_backend_selector_outranks_the_legacy_flag(primary, legacy, expected):
    assert backend_marker.environment_backend_override(primary, legacy) == expected


def test_the_api_offers_exactly_the_requestable_backends():
    """The route's Literal is what FastAPI validates and documents, so a backend
    added to the installer must reach the picker rather than 422 on the way in."""
    from typing import get_args

    from routes.llama import LlamaBackendRequest

    field = LlamaBackendRequest.model_fields["backend"]
    assert set(get_args(field.annotation)) == set(ilp.REQUESTABLE_BACKENDS)


def test_the_api_reports_an_unreadable_newer_backend_request_verbatim():
    """A choice written by a newer Unsloth survives the response model.

    Coercing it to "auto" would tell the picker this install is detecting when it
    is not, and the picker would then happily overwrite the newer choice.
    """
    from routes.llama import LlamaBackendStatusResponse

    response = LlamaBackendStatusResponse(backend_request = "sycl")

    assert response.backend_request == "sycl"


@pytest.mark.parametrize(
    "marker, chosen",
    [
        # Detected, so crash recovery may still fall back to CPU placement.
        ({}, False),
        ({"llama_backend": None}, False),
        ({"llama_backend": ""}, False),
        ({"llama_backend": "auto"}, False),
        ({"backend_request": "auto"}, False),
        # Legacy Vulkan stays eligible for automatic recovery.
        ({"asset": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz"}, False),
        # Chosen.
        ({"llama_backend": "vulkan"}, True),
        ({"force_cpu": True}, True),
        ({"backend_request": "vulkan"}, True),
        ({"backend_request": "cpu"}, True),
        # Unreadable is chosen: undoing a choice we cannot name is the wrong guess.
        ({"llama_backend": "sycl"}, True),
        ({"backend_request": "sycl"}, True),
    ],
)
def test_was_chosen_separates_a_pinned_install_from_a_detected_one(marker, chosen):
    assert backend_marker.marker_backend_was_chosen(marker) is chosen

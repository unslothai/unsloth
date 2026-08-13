# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The flag catalogue the extra-arguments editor validates against.

The catalogue comes from the INSTALLED binary's ``--help``, not a list shipped with
Unsloth, because a custom or newer llama.cpp is exactly the case where a bundled list
would reject a flag that works. These tests pin the two things the editor depends on:
that a failed probe is reported as unverifiable rather than as "no such flag", and
that the managed list it explains rejections with cannot drift from the validator.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

_LSA_PATH = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_server_args.py"
_spec = importlib.util.spec_from_file_location("_lsa_catalog_test", _LSA_PATH)
_lsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lsa)


def _call(
    monkeypatch,
    capabilities,
    *,
    raises = False,
):
    """Run the route with a stubbed probe, returning the response model."""
    import routes.inference as inference_route

    class _Backend:
        @staticmethod
        def probe_server_capabilities():
            if raises:
                raise RuntimeError("no binary")
            return capabilities

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    return asyncio.run(inference_route.get_llama_flags(current_subject = "test"))


def test_a_parsed_catalogue_is_returned(monkeypatch):
    caps = {"found": True, "flags": {"--top-k": "top-k sampling", "--numa": "NUMA policy"}}
    result = _call(monkeypatch, caps)

    assert result.probe_ok is True
    assert result.flags["--top-k"] == "top-k sampling"
    assert "--numa" in result.flags


def test_a_failed_probe_is_unverifiable_not_empty_of_flags(monkeypatch):
    # The distinction the editor turns into copy: probe_ok False means "cannot
    # check", and every argument has to be let through. Reporting no flags as if
    # the binary supported none would mark every correct flag as a typo.
    result = _call(monkeypatch, {}, raises = True)

    assert result.probe_ok is False
    assert result.flags == {}
    # Still worth answering: the managed list needs no binary, so a rejection can
    # still be explained before the request is made.
    assert "--parallel" in result.managed


def test_a_missing_binary_is_also_unverifiable(monkeypatch):
    # found=False is the no-binary answer, which is not the same as a binary whose
    # help would not parse, but the editor can do nothing different about it.
    result = _call(monkeypatch, {"found": False, "flags": {}})

    assert result.probe_ok is False


def test_a_binary_whose_help_did_not_parse_is_unverifiable(monkeypatch):
    # A binary that ran but produced nothing parseable must not read as verified,
    # or every flag becomes unknown.
    result = _call(monkeypatch, {"found": True, "flags": {}})

    assert result.probe_ok is False


def test_the_managed_list_cannot_drift_from_the_validator(monkeypatch):
    # The editor explains a rejection using this list, and the load is refused by
    # validate_extra_args. If they disagree, the UI accepts an argument the load
    # then refuses, or warns about one that would have worked.
    result = _call(monkeypatch, {"found": True, "flags": {"--top-k": "x"}})

    for flag in result.managed:
        assert _lsa.is_managed_flag(flag), flag
    for flag in ("--parallel", "--model", "--api-key", "--agent", "--host"):
        assert flag in result.managed


@pytest.mark.parametrize("value", [123, None])
def test_help_text_is_coerced_to_a_string(monkeypatch, value):
    # The probe's dict is typed as object, and a non-string description would fail
    # response validation at request time rather than here.
    result = _call(monkeypatch, {"found": True, "flags": {"--top-k": value}})

    assert isinstance(result.flags["--top-k"], str)

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


def test_a_help_run_that_failed_partway_is_unverifiable(monkeypatch):
    # The catalogue is non-empty, which is exactly the trap: --help exited nonzero
    # after printing some of itself, so everything below the failure point is
    # missing and would be reported as "not in this build".
    result = _call(
        monkeypatch,
        {"found": True, "flags": {"--top-k": "x"}, "help_probe_ok": False},
    )

    assert result.probe_ok is False


def test_an_older_probe_without_the_field_is_still_trusted(monkeypatch):
    # The key is new; a capability dict that predates it (or a stub in another test)
    # must not read as a failed probe.
    result = _call(monkeypatch, {"found": True, "flags": {"--top-k": "x"}})

    assert result.probe_ok is True


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


# --- what the probe publishes ---------------------------------------------------
# The route only forwards; these pin the map it forwards, driven through the real
# --help parser rather than a stub of it.


def _probe_with_help(
    monkeypatch,
    tmp_path,
    help_text: str,
    returncode: int = 0,
):
    """Run the real capability probe over a canned --help."""
    import types

    from core.inference.llama_cpp import LlamaCppBackend

    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    def _run(_cmd, **kwargs):
        return types.SimpleNamespace(stdout = help_text, stderr = "", returncode = returncode)

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", _run)
    # Keyed on (path, mtime, size), so a fresh file per test is a fresh probe.
    return LlamaCppBackend.probe_server_capabilities(str(binary))


_HELP_WITH_A_REMOVAL_STUB = """\
--top-k N                               top-k sampling (default: 40)
--draft, --draft-n, --draft-max N       the argument has been removed. use --spec-draft-n-max
--numa TYPE                             attempt optimizations that help on some NUMA systems
"""


def test_a_removed_flag_is_not_published_as_supported(monkeypatch, tmp_path):
    # llama.cpp keeps the old names in --help only to say they are gone. Publishing
    # them would make the editor stay quiet about a flag the load then refuses,
    # which is the one thing the catalogue exists to prevent.
    caps = _probe_with_help(monkeypatch, tmp_path, _HELP_WITH_A_REMOVAL_STUB)

    assert "--top-k" in caps["flags"]
    assert "--numa" in caps["flags"]
    for removed in ("--draft", "--draft-n", "--draft-max"):
        assert removed not in caps["flags"], removed


def test_short_aliases_are_published_too(monkeypatch, tmp_path):
    # -t is as valid as --threads, and the catalogue is what the editor checks a
    # typed flag against, so publishing only long names warned that a correct flag
    # was not in this build.
    caps = _probe_with_help(
        monkeypatch,
        tmp_path,
        "-t, --threads N                         number of threads (default: -1)\n"
        "-fa, --flash-attn on|off|auto           set Flash Attention use\n",
    )

    assert "--threads" in caps["flags"]
    assert "-t" in caps["flags"]
    assert caps["flags"]["-t"] == caps["flags"]["--threads"]
    assert "-fa" in caps["flags"]
    # A value placeholder is not a flag.
    assert "-1" not in caps["flags"]
    assert "N" not in caps["flags"]


def test_a_help_that_exited_nonzero_says_so(monkeypatch, tmp_path):
    # Partial output parses fine, so nothing else in the dict would reveal that the
    # rest of the catalogue is missing.
    caps = _probe_with_help(monkeypatch, tmp_path, _HELP_WITH_A_REMOVAL_STUB, returncode = 1)

    assert caps["help_probe_ok"] is False


def test_a_clean_help_run_says_so(monkeypatch, tmp_path):
    caps = _probe_with_help(monkeypatch, tmp_path, _HELP_WITH_A_REMOVAL_STUB)

    assert caps["help_probe_ok"] is True


def test_the_denylist_can_be_read_without_probing(monkeypatch):
    # The panel sanitizes a stored list with this before turning it into an explicit
    # request, and a cold --help takes up to ten seconds. Waiting for the probe would
    # leave a flag denied since that list was saved sitting in the request.
    import asyncio

    import routes.inference as inference_route

    probed = False

    class _Backend:
        @staticmethod
        def probe_server_capabilities():
            nonlocal probed
            probed = True
            return {"found": True, "flags": {"--top-k": "x"}}

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    result = asyncio.run(inference_route.get_llama_flags(managed_only = True, current_subject = "test"))

    assert probed is False
    assert "--agent" in result.managed
    assert result.flags == {}
    # Nothing was checked against a binary, so nothing may be called a typo.
    assert result.probe_ok is False


def test_the_published_slot_default_is_the_effective_one(monkeypatch):
    # A build without --kv-unified serves one slot however many are configured, and
    # load_model clamps to that before launch. An editor sizing its batch floor from
    # the raw default would refuse "--batch-size 2" against a command that runs it.
    import routes.inference as inference_route

    monkeypatch.setattr(
        inference_route.LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(lambda *a, **k: {"found": True, "supports_kv_unified": False}),
    )
    assert inference_route._effective_parallel_slots(4) == 1
    # With the flag the ask stands, and one slot is already the floor.
    monkeypatch.setattr(
        inference_route.LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(lambda *a, **k: {"found": True, "supports_kv_unified": True}),
    )
    assert inference_route._effective_parallel_slots(4) == 4
    assert inference_route._effective_parallel_slots(1) == 1
    # The diffusion runner receives no --parallel at all.
    assert inference_route._effective_parallel_slots(4, diffusion_kind = True) == 1


def test_an_unreadable_probe_keeps_the_asked_for_slot_count(monkeypatch):
    # Refusing to answer is not a reason to clamp: every other caller of the probe
    # here keeps the ask when it cannot be read.
    import routes.inference as inference_route

    def _boom(*_a, **_k):
        raise RuntimeError("no binary")

    monkeypatch.setattr(
        inference_route.LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(_boom),
    )
    assert inference_route._effective_parallel_slots(4) == 4


def test_the_slot_probe_never_runs_on_the_event_loop(monkeypatch):
    # The clamp asks the binary whether it supports --kv-unified, and on a cold cache
    # that is `llama-server --help` with a ten second timeout. Computed inline it
    # stalled every other request on the first open of the panel after an update, and
    # the managed-only answer too, which exists precisely to avoid waiting for a probe.
    import asyncio
    import threading

    import routes.inference as inference_route

    loop_thread = None
    probe_thread = None

    def _probe(*_a, **_k):
        nonlocal probe_thread
        probe_thread = threading.current_thread()
        return {"found": True, "supports_kv_unified": True}

    monkeypatch.setattr(
        inference_route.LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(_probe),
    )
    monkeypatch.setattr(inference_route, "_resolve_parallel_slots", lambda *a, **k: 4)

    async def _run():
        nonlocal loop_thread
        loop_thread = threading.current_thread()
        return await inference_route._effective_default_slots(None)

    assert asyncio.run(_run()) == (4, False)
    assert probe_thread is not None, "the clamp did not consult the binary at all"
    assert probe_thread is not loop_thread, "the --help probe ran on the event loop"


def test_a_single_slot_default_still_reports_the_clamp(monkeypatch):
    # One slot cannot be clamped below one, so the default needs no probe. Whether
    # this build clamps is a different question and still has to be answered: the
    # editor sizes an EXPLICIT Slots value the user may raise without re-reading
    # this route, and off the loop like the rest of it.
    import asyncio
    import threading

    import routes.inference as inference_route

    loop_thread = None
    probe_thread = None

    def _probe(*_a, **_k):
        nonlocal probe_thread
        probe_thread = threading.current_thread()
        return {"found": True, "supports_kv_unified": False}

    monkeypatch.setattr(
        inference_route.LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(_probe),
    )
    monkeypatch.setattr(inference_route, "_resolve_parallel_slots", lambda *a, **k: 1)

    async def _run():
        nonlocal loop_thread
        loop_thread = threading.current_thread()
        return await inference_route._effective_default_slots(None)

    assert asyncio.run(_run()) == (1, True)
    assert probe_thread is not loop_thread, "the --help probe ran on the event loop"


def test_the_clamp_is_read_from_the_same_helper_the_load_uses(monkeypatch):
    import routes.inference as inference_route

    monkeypatch.setattr(
        inference_route.LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(lambda *a, **k: {"found": True, "supports_kv_unified": True}),
    )
    assert inference_route._parallel_slots_are_clamped() is False
    monkeypatch.setattr(
        inference_route.LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(lambda *a, **k: {"found": True, "supports_kv_unified": False}),
    )
    assert inference_route._parallel_slots_are_clamped() is True
    # An unreadable probe keeps the ask here too, so nothing is refused over it.
    monkeypatch.setattr(
        inference_route.LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no binary"))),
    )
    assert inference_route._parallel_slots_are_clamped() is False

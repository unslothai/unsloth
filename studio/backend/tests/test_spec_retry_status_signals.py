# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The status fields a client reads before declining the resident-model shortcut.

``spec_binary_fallback_can_retry`` needs a different llama-server installed before an
identical /load can repair a binary stand-down. The chat UI cannot see that, so it
reloaded (and prompted to stop running chats) for every re-pick of a model whose drafter
stood down, for a load the backend would have deduplicated.

``_spec_fallback_binary_changed`` publishes it. It is answered ONLY for the two binary
reasons: /api/inference/status is polled from first paint, and neither the binary lookup
nor the capability probe has business running on every poll of a healthy runtime.

Two more arms of ``_runtime_matches_intent`` reject an identical load while leaving
``spec_fallback_reason`` null entirely, so a client reading only the reason adopts a
degraded runtime and nothing ever retries it: a retryable DFlash sidecar fetch, and a
capability probe that has started answering since a launch it degraded.
``_spec_dflash_retry_pending`` and ``_spec_probe_retry_pending`` publish those.

The helper is extracted from the route module's source rather than imported, so the test
costs nothing and does not drag FastAPI in behind it.
"""

from __future__ import annotations

import ast
import sys
import types as _types
from pathlib import Path
from typing import Optional

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# conftest's autouse fixture imports core.inference.llama_cpp, which wants these.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

_ROUTE = Path(__file__).resolve().parent.parent / "routes" / "inference.py"
_NAME = "_spec_fallback_binary_changed"


def _load_helper(name = _NAME):
    tree = ast.parse(_ROUTE.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace: dict = {"Optional": Optional}
            exec(compile(ast.Module([node], []), str(_ROUTE), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} is gone; the status no longer reports it")


class _Backend:
    def __init__(
        self,
        reason,
        changed = False,
        raises = False,
    ):
        self.spec_fallback_reason = reason
        self._changed = changed
        self._raises = raises
        self.calls = 0

    def spec_binary_fallback_can_retry(self):
        self.calls += 1
        if self._raises:
            raise RuntimeError("binary lookup failed")
        return self._changed


def test_answers_only_for_the_two_binary_reasons():
    helper = _load_helper()
    for reason in (
        None,
        "drafter_not_found",
        "drafter_no_vram",
        "runtime_error",
        "mla_mtp_disabled",
    ):
        backend = _Backend(reason)
        assert helper(backend) is None
        # The point of the gate: no binary lookup on a poll that cannot need one.
        assert backend.calls == 0


def test_reports_the_whole_retry_predicate():
    # The predicate, not just its revision half: binary_no_mtp also asks whether the
    # replacement advertises what the drafter kind needs, and a replacement that still
    # lacks it never repairs, so a half answer would prompt on every later re-pick.
    helper = _load_helper()
    for reason in ("binary_no_mtp", "binary_outdated"):
        assert helper(_Backend(reason, changed = False)) is False
        assert helper(_Backend(reason, changed = True)) is True


def test_an_unreadable_binary_is_unknown_not_false():
    # False would tell the client the drafter cannot be repaired and suppress the reload
    # an update was meant to enable; None leaves it with the coarser answer.
    helper = _load_helper()
    assert helper(_Backend("binary_no_mtp", raises = True)) is None


class _SpecBackend:
    """The three attributes the other two helpers read."""

    def __init__(
        self,
        *,
        dflash = False,
        inconclusive = False,
        probe = None,
        raises = False,
    ):
        self._dflash_retry_needed = dflash
        self._capability_probe_inconclusive = inconclusive
        self._is_diffusion = False
        self._probe = probe
        self._raises = raises
        self.probes = 0

    def probe_server_capabilities(self):
        self.probes += 1
        if self._raises:
            raise RuntimeError("probe failed")
        return {"mtp_probe_inconclusive": self._probe}


def test_the_probe_arm_only_probes_once_a_launch_was_degraded():
    helper = _load_helper("_spec_probe_retry_pending")
    settled = _SpecBackend(inconclusive = False)
    assert helper(settled) is False
    # The status is polled from first paint; a healthy runtime must not pay for this.
    assert settled.probes == 0
    # Still inconclusive: nothing has changed, so an identical load would dedupe.
    assert helper(_SpecBackend(inconclusive = True, probe = True)) is False
    # Answering now: the degraded runtime is re-derived once.
    assert helper(_SpecBackend(inconclusive = True, probe = False)) is True


def test_the_probe_arm_skips_diffusion_and_survives_a_failed_probe():
    helper = _load_helper("_spec_probe_retry_pending")
    diffusion = _SpecBackend(inconclusive = True, probe = False)
    diffusion._is_diffusion = True
    assert helper(diffusion) is False
    assert helper(_SpecBackend(inconclusive = True, raises = True)) is None


def test_the_dflash_arm_reports_the_retry_flag():
    helper = _load_helper("_spec_dflash_retry_pending")
    assert helper(_SpecBackend(dflash = True)) is True
    assert helper(_SpecBackend(dflash = False)) is False


def test_a_backend_without_the_dflash_flag_is_unknown():
    helper = _load_helper("_spec_dflash_retry_pending")

    class _Bare:
        # Attribute access raises rather than returning a default, so the helper's
        # try/except is what keeps the status route answering at all.
        def __getattr__(self, name):
            raise AttributeError(name)

    assert helper(_Bare()) is None

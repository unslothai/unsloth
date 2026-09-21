# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a failed image generation can still say WHY once it is no longer running.

A POST lost past the proxy's ~100s window leaves the client polling generate-progress as
its only channel. Both engines clear ``_gen`` in a ``finally``, so without a retained
reason that poll answers "not running" for a failure, and the settling path reads it as
success and can advance a batch past an output that never arrived.

Static plus behavioural: generating needs a GPU, so what runs here is the part deciding
what a CALLER sees, the route's classification, while the engines' retention, clearing and
publishing are asserted against their own source.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

ENGINES = ("core/inference/diffusion.py", "core/inference/sd_cpp_backend.py")


def _src(rel: str) -> str:
    return (Path(_BACKEND_DIR) / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("engine", ENGINES)
def test_an_engine_retains_the_reason_past_the_clear_of_gen(engine):
    src = _src(engine)
    assert "self._last_generate_error = str(exc) or type(exc).__name__" in src, (
        f"{engine} no longer retains a failed generation's reason, so idle progress "
        "cannot distinguish a failure from a finished run"
    )
    # Cleared when the NEXT run starts, or a later poll would read a stale failure as
    # belonging to the generation now in flight.
    assert re.search(
        r"self\._gen = _(?:Gen|Sd)[A-Za-z]*\(total_steps[^\n]*\n\s*#[^\n]*\n(?:\s*#[^\n]*\n)?\s*self\._last_generate_error = None",
        src,
    ), f"{engine} does not clear the retained reason when a generation starts"


@pytest.mark.parametrize("engine", ENGINES)
def test_the_idle_branch_is_what_publishes_it(engine):
    """On the idle branch only. Published while active it would describe the previous run."""
    src = _src(engine)
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_progress":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    keys = [k.value for k in sub.keys if isinstance(k, ast.Constant)]
                    if "active" in keys:
                        found.append(keys)
    assert found, f"{engine} has no generate_progress dict to check"
    idle = [k for k in found if "error" in k]
    assert len(idle) == 1, f"{engine}: expected exactly one branch to carry error, got {found}"


def test_the_route_classifies_it_and_never_relays_engine_text(monkeypatch):
    """The engine keeps the raw string; only the route decides what a caller may see.

    Otherwise the field becomes a hole for the engine's own text, which carries local paths
    and argv -- the thing _generate_failure_detail exists to keep out of a message.
    """
    import routes.inference as route

    raw = (
        "CUDA out of memory. Tried to allocate 2.00 GiB while loading "
        "/home/somebody/.unsloth/studio/models/secret-project/model.safetensors "
        "with --argv-that-should-not-ship"
    )
    classified = route._generate_failure_detail(raw)
    assert "/home/somebody" not in classified, "a local path reached a client-visible message"
    assert "--argv" not in classified
    assert classified != raw and classified, "the raw text was relayed unchanged"
    # And it still says something specific rather than collapsing to the bare fallback,
    # which is the whole reason this PR touched these messages.
    assert classified != route._GENERATE_FAILURE_FALLBACK, (
        "an out-of-memory failure classified to the bare fallback"
    )

    # Nothing to say when nothing failed.
    assert route._generate_failure_detail("") == route._GENERATE_FAILURE_FALLBACK


def test_the_progress_route_puts_the_classified_reason_on_the_response():
    """Wiring, read from the route's own source: the response must carry the CLASSIFIED
    value, not the engine's."""
    src = _src("routes/inference.py")
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 2000]
    assert '"error": _generate_failure_detail(raw_error) if raw_error else None' in body, (
        "the progress route no longer classifies the retained reason"
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_an_engine_identifies_the_reason_with_the_attempt_that_caused_it(engine):
    """A retained reason is only useful if a caller can tell WHOSE run it came from.

    A counter only answers "later", and later includes a concurrent client's run as well as
    this attempt's, while a generation whose POST never reached the backend started no run at
    all. So the engine keeps the id the request carried, and a caller matches it exactly.
    """
    src = _src(engine)
    assert "self._last_generate_attempt = attempt_id" in src, (
        f"{engine} does not keep the attempt id, so a reason cannot be identified"
    )
    assert '"generation_attempt": getattr(self, "_last_generate_attempt", None),' in src, (
        f"{engine} does not publish the attempt id beside the reason"
    )
    assert "attempt_id: Optional[str] = None," in src, (
        f"{engine} does not accept an attempt id from the route"
    )
    # Recorded where the run STARTS, beside the clear, or the id and the reason would
    # describe different moments.
    recorded = src.index("self._last_generate_attempt = attempt_id")
    clear = src.index("self._last_generate_error = None")
    assert 0 < recorded - clear < 600, (
        f"{engine} records the attempt id away from where the reason is cleared"
    )
    # And no counter left behind: a monotonic lower bound is what this replaced, and leaving
    # it published invites the comparison it cannot support.
    assert "generation_seq" not in src, f"{engine} still publishes the superseded counter"


def test_the_route_forwards_the_attempt_id_and_sends_it_only_beside_a_reason():
    """Forwarded from the request, returned only with the failure it dates.

    Without a reason there is nothing to attribute, and only the client that sent the id can
    match it -- so a concurrent client has no business reading which attempt last failed.
    """
    src = _src("routes/inference.py")
    at = src.index("async def generate_diffusion_image")
    assert "attempt_id = request.attempt_id," in src[at : at + 4000], (
        "the generate route no longer forwards the attempt id to the engine"
    )
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 2500]
    assert 'if not progress.get("error"):' in body
    assert 'progress.pop("generation_attempt", None)' in body


def test_the_attempt_id_is_bounded_and_patterned_on_the_way_in():
    """It comes off a request and goes back out on a response, so it is validated there."""
    from models.inference import DiffusionGenerateRequest

    ok = DiffusionGenerateRequest(prompt = "p", attempt_id = "a-Z_09")
    assert ok.attempt_id == "a-Z_09"
    assert DiffusionGenerateRequest(prompt = "p").attempt_id is None
    for bad in ("has space", "semi;colon", "<script>", "x" * 65, "sl/ash"):
        with pytest.raises(Exception):
            DiffusionGenerateRequest(prompt = "p", attempt_id = bad)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a failed image generation can still say WHY once it is no longer running.

A generation whose POST is lost past the proxy's ~100s window leaves the client polling
generate-progress as its only channel. Both engines clear ``_gen`` in a ``finally``, so
without a retained reason that poll answers "not running" for a failure -- identical to a
run that finished -- and the client's settling path reads it as success and can advance a
multi-run batch past an output that never arrived.

Static plus behavioural, deliberately. The engines need a GPU to generate, so what is
exercised here by executing is the part that decides what a CALLER sees: the route's
classification. That the reason is retained at all, cleared at the start of the next run,
and published on the idle branch is asserted against the engines' own source, which is
cheap and cannot drift silently.
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
def test_an_engine_dates_the_reason_with_a_run_counter(engine):
    """A retained reason is only useful if a caller can tell WHICH run it belongs to.

    A generation whose POST never reached the backend started no run, so the counter has not
    moved since before that POST; without it the client attributes the previous run's failure
    to its own lost request and skips the gallery probe that would have said the request
    never arrived.
    """
    src = _src(engine)
    assert 'self._generate_seq = getattr(self, "_generate_seq", 0) + 1' in src, (
        f"{engine} does not advance a run counter, so a reason cannot be dated"
    )
    assert '"generation_seq": getattr(self, "_generate_seq", 0),' in src, (
        f"{engine} does not publish the run counter beside the reason"
    )
    # Bumped where the run STARTS, beside the clear, or the counter and the reason would
    # describe different moments.
    bump = src.index('self._generate_seq = getattr(self, "_generate_seq", 0) + 1')
    clear = src.index("self._last_generate_error = None")
    assert 0 < bump - clear < 400, (
        f"{engine} bumps the counter away from where the reason is cleared"
    )


def test_the_route_sends_the_counter_only_beside_a_reason():
    """On its own it is an internal counter with no meaning to a client, and shipping it
    unconditionally invites exactly the correlation this is meant to make possible to be
    done against a number that was never qualified."""
    src = _src("routes/inference.py")
    at = src.index("async def diffusion_generate_progress")
    body = src[at : at + 2500]
    assert 'if not progress.get("error"):' in body
    assert 'progress.pop("generation_seq", None)' in body

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Returning from Compare to a ``/chat?new=<uuid>`` chat must re-attach its thread id.

The shared provider is HIDDEN rather than unmounted when Compare opens, which is the point of
#9129 -- but ``enterCompare`` clears ``activeThreadId`` on the way in, and on the way back
nothing puts it back for this view:

  * ``ThreadNewChatSwitch`` returns immediately, because the nonce never changed;
  * ``ActiveThreadSync`` is disabled outright whenever a nonce is present;
  * ``ProjectLanding`` restores on resume, and the single-chat path had no equivalent.

``ThreadScopedSettingsSync`` is NOT nonce-gated, so the chat came back live but detached: it
runs on the installation defaults, and an edit made in it moves THOSE rather than its own
snapshot. Title, context usage and the model notice are keyed on the same id.

``NonceThreadResumeRestore``'s effect is sliced verbatim out of the provider and replayed
through the same React-effect emulator the sibling suites use, so wiring drift breaks this
rather than silently un-testing it.
"""

from __future__ import annotations

import re
import textwrap

import pytest

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

PROVIDER = source_path("studio/frontend/src/features/chat/runtime-provider.tsx")
TEMP = WORKDIR / "temp" / "nonce_chat_resume_restore"
SOURCES = (PROVIDER,)


def _restore_effect() -> str:
    """The component's single ``useEffect`` body, verbatim."""
    component = slice_between(
        read(PROVIDER),
        "function NonceThreadResumeRestore(",
        "\n// A thread read that fails leaves the chat unpaired",
    )
    match = re.search(r"useEffect\(\(\) => \{\n(.*?)\n  \}, \[([^\]]*)\]\);", component, re.S)
    assert match, "NonceThreadResumeRestore effect not found"
    deps = [name.strip() for name in match.group(2).split(",") if name.strip()]
    assert deps == ["enabled", "mainThreadId"], f"emulator does not bind {deps}"
    return match.group(1)


def _harness_source() -> str:
    return f"""
export const state = {{ activeThreadId: null }};
export const sets = [];
const useChatRuntimeStore = {{
  getState: () => ({{
    ...state,
    setActiveThreadId: (id) => {{
      state.activeThreadId = id;
      sets.push(id);
    }},
  }}),
}};
// The real predicate: assistant-ui mints local ids for a thread with no row behind it yet.
const isAssistantLocalThreadId = (id) =>
  typeof id === "string" && id.startsWith("__LOCALID_");

// One mounted component: a ref that survives renders, and an effect re-run only when a
// dependency actually changed.
export function mount(initialEnabled) {{
  const wasEnabledRef = {{ current: initialEnabled }};
  let last = null;
  return function render(enabled, mainThreadId) {{
    const deps = [enabled, mainThreadId];
    if (last !== null && last[0] === deps[0] && last[1] === deps[1]) return;
    last = deps;
    (() => {{
{_restore_effect()}
    }})();
  }};
}}

export function report(extra) {{
  console.log(JSON.stringify({{
    activeThreadId: state.activeThreadId,
    sets,
    ...extra,
  }}));
}}
"""


def _run(body: str) -> dict:
    require_node(SOURCES)
    script = textwrap.dedent(
        f"""
        // @ts-nocheck
        import {{ mount, report, state, sets }} from "./harness.ts";
        {textwrap.dedent(body)}
        """
    )
    return run_harness(TEMP, _harness_source(), script)


def test_returning_from_compare_reattaches_the_materialized_thread() -> None:
    """The bug, end to end: hide the view, clear the id as Compare does, show it again."""
    out = _run(
        """
        const render = mount(true);
        render(true, "thread-42");        // the chat, materialized and attached
        state.activeThreadId = "thread-42";
        render(false, "thread-42");       // Compare opens: hidden, and it clears the id
        state.activeThreadId = null;
        sets.length = 0;
        render(true, "thread-42");        // back again
        report({});
        """
    )
    assert (
        out["activeThreadId"] == "thread-42"
    ), "a chat that comes back visible must be attached to the thread it is showing"
    assert out["sets"] == ["thread-42"], "exactly one restore, on the resume"


def test_a_resume_never_overwrites_an_id_somebody_else_already_set() -> None:
    """The restore only fills a hole. Racing whoever owns the id would be worse than the gap."""
    out = _run(
        """
        const render = mount(true);
        render(false, "thread-42");
        state.activeThreadId = "thread-99";
        sets.length = 0;
        render(true, "thread-42");
        report({});
        """
    )
    assert out["sets"] == []
    assert out["activeThreadId"] == "thread-99"


def test_a_fresh_new_chat_is_not_attached_on_its_first_render() -> None:
    """Mounting is not resuming. An empty New Chat has no thread yet, and publishing one
    would price and pair a chat the user has not started."""
    out = _run(
        """
        const render = mount(true);
        render(true, "thread-42");
        report({});
        """
    )
    assert out["sets"] == []
    assert out["activeThreadId"] is None


def test_an_unpersisted_runtime_id_is_not_published() -> None:
    """A local id has no row, so ThreadScopedSettingsSync discards it again -- and the
    round trip it would open can only 404."""
    out = _run(
        """
        const render = mount(true);
        render(false, "__LOCALID_7");
        sets.length = 0;
        render(true, "__LOCALID_7");
        report({});
        """
    )
    assert out["sets"] == []
    assert out["activeThreadId"] is None


def test_staying_hidden_never_restores() -> None:
    """A backgrounded provider must not write the active thread the visible view owns."""
    out = _run(
        """
        const render = mount(true);
        render(false, "thread-42");
        state.activeThreadId = null;
        sets.length = 0;
        render(false, "thread-7");
        report({});
        """
    )
    assert out["sets"] == []


def test_the_provider_gates_the_restore_on_a_nonce_view_that_is_visible() -> None:
    """Structural. The emulator above replays the effect but not the props the JSX hands it,
    so the enable expression is only pinned here. Every clause carries its own reason:
    ``newThreadNonce`` because a saved thread already has ActiveThreadSync, ``initialThreadId``
    for the same reason, ``pairId``/``base`` because a compare pane has no single chat to
    attach, and ``backgrounded`` because a hidden view must not claim the visible one's id."""
    jsx = slice_between(
        read(PROVIDER),
        "<NonceThreadResumeRestore",
        "<ThreadScopedSettingsSync",
    )
    for clause in (
        'modelType === "base"',
        "!pairId",
        "!!newThreadNonce",
        "!initialThreadId",
        "!backgrounded",
    ):
        assert clause in jsx, f"NonceThreadResumeRestore must be gated on {clause}"


def test_the_harness_binds_every_name_the_effect_reaches() -> None:
    """The effect is replayed verbatim, so a name this harness does not define becomes a
    ReferenceError -- which reads as "the restore never fired", a bug that is not there."""
    body = _restore_effect()
    for name in re.findall(r"\b([A-Za-z_$][\w$]*)\s*\(", body):
        if name in {"if", "return", "for", "while", "switch", "catch"}:
            continue
        assert name in _harness_source(), f"the effect calls {name}(), which the harness omits"

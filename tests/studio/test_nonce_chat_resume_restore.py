# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Returning from Compare to a ``/chat?new=<uuid>`` chat must re-attach its thread id.

The provider is HIDDEN rather than unmounted when Compare opens (the point of #9129), but
``enterCompare`` clears ``activeThreadId`` and nothing put it back for this view:
``ThreadNewChatSwitch`` returns immediately since the nonce never changed, ``ActiveThreadSync``
is disabled whenever a nonce is present, and only ``ProjectLanding`` restored on resume.

``ThreadScopedSettingsSync`` is NOT nonce-gated, so the chat came back live but detached, on
installation defaults, with an edit moving THOSE rather than its own snapshot. Title, context
usage and the model notice are keyed on the same id.

``NonceThreadResumeRestore``'s effect is sliced verbatim and replayed through the same
React-effect emulator the sibling suites use, so wiring drift breaks this.
"""

from __future__ import annotations

import re
import textwrap

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
    assert deps == ["aui", "enabled", "mainThreadId"], f"emulator does not bind {deps}"
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

// Which threads have been persisted. A thread NOT listed here is a placeholder the
// runtime minted and nothing was ever written for -- what an untouched project landing
// holds after a compare round trip. Keyed by thread id, exactly as getItemById is.
export const remoteIds = {{}};
const aui = {{
  threads: () => ({{
    __internal_getAssistantRuntime: () => ({{
      threads: {{
        getItemById: (id) => ({{
          getState: () => ({{ remoteId: remoteIds[id] }}),
        }}),
      }},
    }}),
  }}),
}};

// One mounted component: a ref that survives renders, and an effect re-run only when a
// dependency actually changed.
export function mount(initialEnabled) {{
  const wasEnabledRef = {{ current: initialEnabled }};
  let last = null;
  return function render(enabled, mainThreadId) {{
    const deps = [aui, enabled, mainThreadId];
    if (last !== null && last[1] === deps[1] && last[2] === deps[2]) return;
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
        import {{ mount, report, state, sets, remoteIds }} from "./harness.ts";
        {textwrap.dedent(body)}
        """
    )
    return run_harness(TEMP, _harness_source(), script, sources = SOURCES)


def test_returning_from_compare_reattaches_the_materialized_thread() -> None:
    """The bug, end to end: hide the view, clear the id as Compare does, show it again."""
    out = _run(
        """
        remoteIds["thread-42"] = "remote-42";   // materialized: a row exists for it
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


def test_a_materialized_chat_keeping_its_runtime_id_is_still_restored() -> None:
    """The ordinary case, and the one an id-prefix guard would have skipped.

    ``createStudioDbAdapter.initialize()`` writes the row under whatever id assistant-ui
    minted and returns it, so a ``?new=`` chat that has been sent to keeps its
    ``__LOCALID_`` id, and refusing that refuses almost every chat this component exists
    for. Published raw, as ActiveThreadSync does elsewhere; consumers that need a
    persisted id filter for themselves."""
    out = _run(
        """
        // initialize() wrote the row under the minted id and handed it back, so this
        // thread IS persisted despite its shape. That is the whole point of the test.
        remoteIds["__LOCALID_7"] = "__LOCALID_7";
        const render = mount(true);
        render(false, "__LOCALID_7");
        state.activeThreadId = null;
        sets.length = 0;
        render(true, "__LOCALID_7");
        report({});
        """
    )
    assert out["sets"] == ["__LOCALID_7"]
    assert out["activeThreadId"] == "__LOCALID_7"


def test_a_view_with_no_thread_at_all_publishes_nothing() -> None:
    """Nothing to attach to. `null` is already the state, and writing it again would
    clear the contextUsage the store keys off it."""
    out = _run(
        """
        const render = mount(true);
        render(false, null);
        sets.length = 0;
        render(true, null);
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
    """Structural. The emulator replays the effect but not the props the JSX hands it, so
    the enable expression is only pinned here. ``newThreadNonce`` and ``initialThreadId``
    because a saved thread already has ActiveThreadSync, ``pairId``/``base`` because a
    compare pane has no single chat to attach, ``backgrounded`` because a hidden view must
    not claim the visible one's id."""
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


def test_an_untouched_landing_is_not_read_into_a_chat_on_resume() -> None:
    """A project landing that was never typed in still holds a blank placeholder thread
    after a compare round trip, because the runtime mints one eagerly. Publishing it makes
    ProjectLanding set pendingNewThreadId and swap the project overview for an empty
    Thread, so the user comes back to a chat they never started.

    Note what is NOT the discriminator: the placeholder here has the same `__LOCALID_`
    shape as the materialized chat two tests up. Only the absence of a row tells them
    apart.
    """
    out = _run(
        """
        const render = mount(true);
        render(false, "__LOCALID_9");   // hidden by Compare, nothing ever written for it
        state.activeThreadId = null;
        sets.length = 0;
        render(true, "__LOCALID_9");    // back to the landing
        report({});
        """
    )
    assert out["sets"] == [], "an unpersisted placeholder must not be published"
    assert out["activeThreadId"] is None, "so the landing stays a landing"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One ChatRuntimeProvider spans the project and single views, so switching between them is
a prop change rather than a remount (#8908).

That keeps a project chat's run attached, and it also breaks the guards the old shape got for
free: a remounted provider had a fresh ref and composer every time, so a shared one has to
decide for itself whether a nonce is new (``activeNonce``), whether the composer still holds
someone else's staged attachment (``hasSwitched``), and whether it is on screen (``paused``).
``ThreadAutoSwitch`` takes ``paused`` too, since the stop it requests names every temporary
queue on the page rather than its own provider's.

The five effects that decide those -- ``ThreadAutoSwitch``'s two, ``ThreadNewChatSwitch``'s two
and the implicit-new-chat marker -- are sliced VERBATIM out of ``runtime-provider.tsx`` and
replayed through a React-effect emulator (per-effect dependency arrays, re-run only on change,
memo cleared on unmount). ``requestTemporaryPromptQueueStop`` is sliced verbatim too, because
whose queue it stops is one of the questions here.

Stubbed: the JSX wiring the props onto the two children (pinned by
``test_the_provider_wires_the_pause_and_the_shared_ref``), assistant-ui's thread runtime, and
``refreshContextUsage`` (a counter; the recount itself is pinned by
``test_new_chat_context_recount.py``). ``ActiveThreadSync`` is not modelled: it is off whenever
either child is rendered.
"""

from __future__ import annotations

import json
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
STORE = source_path("studio/frontend/src/features/chat/stores/chat-runtime-store.ts")
QUEUE = source_path("studio/frontend/src/features/chat/utils/prompt-queue-boundary.ts")

TEMP = WORKDIR / "temp" / "project_chat_view_switch"

SOURCES = (PROVIDER, STORE, QUEUE)

# Every name the emulator can supply to a sliced dependency array.
BOUND_NAMES = {
    "aui",
    "checkpoint",
    "loadedContextLength",
    "initialThreadId",
    "isLoading",
    "mainThreadId",
    "modelLoading",
    "newThreadNonce",
    "newThreadSwitchStateRef",
    "nonce",
    "onSwitchFailed",
    "paused",
    "runActive",
    "syncActiveThreadId",
    "threadId",
}

EFFECT_PATTERN = r"useEffect\(\(\) => \{\n(.*?)\n  \}, \[([^\]]*)\]\);"


def _effects(text: str, label: str) -> list[tuple[list[str], str]]:
    """Every ``useEffect`` in a slice as (dependency names, verbatim body)."""
    matches = re.findall(EFFECT_PATTERN, text, re.S)
    assert matches, f"{label}: no effects found; the slice markers have moved"
    effects = [
        ([name.strip() for name in deps.split(",") if name.strip()], body) for body, deps in matches
    ]
    for deps, _body in effects:
        unknown = set(deps) - BOUND_NAMES
        assert not unknown, f"{label}: emulator does not bind {sorted(unknown)}"
    return effects


def _auto_switch_source() -> str:
    return slice_between(
        read(PROVIDER), "function ThreadAutoSwitch(", "\nfunction ThreadNewChatSwitch("
    )


def _new_chat_source() -> str:
    return slice_between(
        read(PROVIDER), "function ThreadNewChatSwitch(", "\nfunction ActiveThreadSync("
    )


def _provider_switch_state_source() -> str:
    """The provider's shared switch-state ref and the effect that marks it, verbatim."""
    return slice_between(
        read(PROVIDER),
        "  const newThreadSwitchStateRef = useRef<NewThreadSwitchState>({",
        "\n  return (",
    )


def _provider_ref_initial_value() -> str:
    """The object literal the provider's ``useRef`` is created with, verbatim."""
    match = re.search(
        r"useRef<NewThreadSwitchState>\((\{.*?\})\);", _provider_switch_state_source(), re.S
    )
    assert match, "the shared switch-state ref is no longer created with an object literal"
    return match.group(1)


def _pending_switch_cap() -> str:
    """The bound on outstanding saved-thread claims, verbatim, so the emulator enforces the
    number the source actually uses rather than a copy that can drift."""
    match = re.search(r"^const MAX_PENDING_SAVED_THREAD_SWITCHES = \d+;$", read(PROVIDER), re.M)
    assert match, "the pending-switch cap is no longer a module-level const"
    return match.group(0)


def _provider_jsx() -> str:
    """The provider subtree that wires the two children, for the wiring guard."""
    return slice_between(
        read(PROVIDER),
        "      <ToolPaneScopeContext.Provider",
        "        {/* The view stays mounted",
    )


def _set_active_thread_id_reducer() -> str:
    """The one store reducer both components write through, verbatim."""
    return slice_between(
        read(STORE),
        "setActiveThreadId: (activeThreadId) =>",
        "applyThreadScopedSettings:",
    ).strip()


# `export { X } from "./y"` and `import type { X } from "./y"`, at the top level.
# The harness inlines one file, so a surviving specifier is a module Node cannot
# resolve in the temp dir; the names themselves are stubbed in the prelude.
_REEXPORT = re.compile(
    r"^(?:export|import)\s+(?:type\s+)?\{[^}]*\}\s+from\s+[\"'][^\"']+[\"'];\s*$",
    re.MULTILINE,
)


def _prompt_queue_boundary_body() -> str:
    """Everything in prompt-queue-boundary.ts after its import block, verbatim.

    Re-export statements are dropped rather than replayed. They are module
    plumbing, not body, and the slice marker is only the last plain import, so a
    re-export written below it would otherwise be inlined into the harness and
    fail to resolve.
    """
    text = read(QUEUE)
    marker = 'from "./prompt-queue-model-boundary";'
    body = text[text.index(marker) + len(marker) :]
    return _REEXPORT.sub("", body)


HARNESS_PRELUDE = """
// @ts-nocheck
// Fixtures the sliced source reads through. Everything below the PRELUDE marker is
// copied verbatim out of the studio sources.
export const world: any = {
  // What assistant-ui reports as the main thread. The runtime is not modelled; both
  // switch calls move this the way the real ones do, which is all either effect reads.
  mainThreadId: null as string | null,
  switchedToNewThread: 0,
  switchedToThread: [] as string[],
  clearedAttachments: 0,
  refreshes: 0,
  // Every event the verbatim requestTemporaryPromptQueueStop dispatched, and who fired it.
  stopEvents: [] as any[],
  // Materialized prompt queues, keyed by thread, as the queue UI store holds them.
  promptQueues: {} as Record<string, any>,
  // Window listeners still registered: a per-switch leak would show up here.
  eventListeners: 0,
  // Set to a promise to hold switchToNewThread() open across a nonce change.
  newThreadGate: null as Promise<void> | null,
  // Holds switchToThread open, so a saved-chat switch can still be in flight when the
  // route moves on. That ordering is the whole point of the stale-arrival correction.
  switchToThreadGate: null as Promise<void> | null,
  newThreadRejects: false,
  switchToThreadRejects: false,
  composerThrows: false,
  clearAttachmentsRejects: false,
  // switchToNewThread() calls still awaiting the gate. Must return to 0.
  pendingNewThreadSwitches: 0,
  // #9251's shell-release signal, counted so a superseded failure can be shown to still
  // fire it.
  switchFailedSignals: 0,
  // Threads a row exists for. A recorded nonce thread is only worth returning to once the
  // user has actually sent to it; a blank placeholder is not, and minting a fresh one is
  // the older behaviour that must survive.
  remoteIds: {} as Record<string, string>,
  // Threads the runtime store no longer holds an entry for, so getItemById throws.
  missingThreadIds: {} as Record<string, boolean>,
  // Which component's effects are replaying right now, so a module-level call can be
  // attributed to a caller. Set by the emulator, never read by the sliced source.
  component: "none",
};

const state: any = {
  activeThreadId: null,
  activeThreadEpoch: 0,
  contextUsage: null,
  contextUsageByThreadId: {},
  params: { checkpoint: "", systemPrompt: "", systemVariables: "" },
  loadedContextLength: null,
  modelLoading: false,
  runningByThreadId: {},
};

function set(updater: any): void {
  const patch = typeof updater === "function" ? updater(state) : updater;
  Object.assign(state, patch);
}

const actions: any = {
  __SET_ACTIVE_THREAD_ID__
};

export const useChatRuntimeStore: any = {
  getState: () => ({ ...state, ...actions }),
};

export function seed(patch: any): void {
  Object.assign(state, patch);
}

export function snapshot(): any {
  return {
    activeThreadId: state.activeThreadId,
    activeThreadEpoch: state.activeThreadEpoch,
    contextUsage: state.contextUsage,
  };
}

// A thread assistant-ui already holds as main: the chat whose run was left going in the
// background, so coming back to it is a no-op switch rather than a fresh one.
export function seedMainThread(threadId: string | null): void {
  world.mainThreadId = threadId;
}

// node has no DOM. Only what the sliced boundary module touches is provided, and the
// dispatch is recorded rather than delivered so a test can read the detail it built.
class CustomEvent {
  type: string;
  detail: any;
  constructor(type: string, init?: any) {
    this.type = type;
    this.detail = init?.detail;
  }
}
const window: any = {
  addEventListener: (): void => {
    world.eventListeners += 1;
  },
  removeEventListener: (): void => {
    world.eventListeners -= 1;
  },
  dispatchEvent: (event: any): boolean => {
    world.stopEvents.push({
      type: event.type,
      detail: event.detail,
      firedBy: world.component,
    });
    return true;
  },
};

// The two names prompt-queue-boundary.ts imports. The queue UI store is read by the
// verbatim requestTemporaryPromptQueueStop to pick the temporary threads to stop, so it
// holds real entries; the model boundary belongs to requestLocalPromptQueueStop, which
// nothing here calls.
const usePromptQueueUI: any = { getState: () => ({ byThreadId: world.promptQueues }) };
const localPromptQueueModelBoundary: any = { advance: (): void => {} };

// The recount is pinned by test_new_chat_context_recount.py. Here it only has to show
// whether the pause gate on the second effect lets it fire.
async function refreshContextUsage(): Promise<void> {
  world.refreshes += 1;
}

const auiFixture: any = {
  threads: () => ({
    __internal_getAssistantRuntime: () => ({
      threads: {
        // Throws for an unknown id rather than returning undefined, as assistant-ui does:
        // getThreadListItemState returns SKIP_UPDATE and ShallowMemoizeSubject's constructor
        // turns that into "Entry not available in the store". A fixture that returned
        // undefined would quietly excuse a caller that cannot survive the real one.
        getItemById: (id: string) => {
          if (world.missingThreadIds[id]) {
            throw new Error("Entry not available in the store");
          }
          return { getState: () => ({ remoteId: world.remoteIds[id] }) };
        },
      },
    }),
    // Both outcomes go through the gate, so a failure can be held open as long as a
    // success can: a rejection that lands after the user has moved on is the ordering
    // the two-arm handler has to survive, and a synchronous throw could not produce it.
    switchToNewThread: (): Promise<void> => {
      world.switchedToNewThread += 1;
      const openedThreadId = `local-${world.switchedToNewThread}`;
      const rejects = world.newThreadRejects;
      world.pendingNewThreadSwitches += 1;
      return Promise.resolve(world.newThreadGate ?? undefined).then(() => {
        world.pendingNewThreadSwitches -= 1;
        if (rejects) {
          throw new Error("switchToNewThread failed");
        }
        world.mainThreadId = openedThreadId;
      });
    },
    switchToThread: (threadId: string): Promise<void> => {
      world.switchedToThread.push(threadId);
      const rejects = world.switchToThreadRejects;
      const gate = world.switchToThreadGate;
      if (!gate) {
        if (rejects) {
          return Promise.reject(new Error("switchToThread failed"));
        }
        world.mainThreadId = threadId;
        return Promise.resolve();
      }
      // Assistant-ui assigns mainThreadId when the switch settles, with no idea whether
      // the route still wants it.
      return gate.then(() => {
        if (rejects) {
          throw new Error("switchToThread failed");
        }
        world.mainThreadId = threadId;
      });
    },
  }),
  // A composer only exists once a thread is mounted; the switch reaches for it before
  // that on the very first mount, which is what its try/catch is for.
  composer: () => {
    if (world.composerThrows) {
      throw new Error("no composer mounted");
    }
    return {
      // Not a bare counter: clearAttachments() removes each staged file through the
      // attachment adapter, so a remove() that fails rejects the promise it returns.
      clearAttachments: async (): Promise<void> => {
        world.clearedAttachments += 1;
        if (world.clearAttachmentsRejects) {
          throw new Error("attachment remove() failed");
        }
      },
    };
  },
};

// ---- PRELUDE ENDS: verbatim studio source follows ----
"""


HARNESS_RENDER = """

// Replays sliced effects with React's dependency rule: an effect re-runs only when one of
// its own dependencies changed since the last render, and a component that unmounts loses
// its memo, so remounting re-runs everything.
function replayEffects(effects: any[], scope: any, memo: any[]): void {
  effects.forEach((effect: any, index: number) => {
    const next = effect.deps.map((name: string) => scope[name]);
    const previous = memo[index];
    if (
      previous != null &&
      previous.length === next.length &&
      previous.every((value: any, i: number) => Object.is(value, next[i]))
    ) {
      return;
    }
    memo[index] = next;
    effect.run();
  });
}

__PENDING_SWITCH_CAP__

const newThreadSwitchStateRef: any = { current: __REF_INITIAL_VALUE__ };

// The four fields the exact-state assertions below were written to lock. `nonceThread` and
// `landedAttempt` are deliberately NOT among them: both are bookkeeping for the reattach
// correction, their values are whatever thread and attempt a given test happens to be on,
// and folding them into twenty unrelated pins would add churn rather than coverage. Each
// has its own accessor, and its own tests.
export function switchState(): any {
  const {
    nonceThread: _ignored,
    landedAttempt: _alsoIgnored,
    ...pinned
  } = newThreadSwitchStateRef.current;
  return pinned;
}

export function nonceThreadId(): any {
  return newThreadSwitchStateRef.current.nonceThread?.threadId ?? null;
}

// Whether the nonce's own switch has landed, which is what makes the current main thread
// its own rather than the one the user came from.
export function nonceOwnershipIsSettled(): boolean {
  const { landedAttempt, attempt } = newThreadSwitchStateRef.current;
  return landedAttempt === attempt;
}

const providerMemo: any[] = [];
const autoSwitchMemo: any[] = [];
const newChatMemo: any[] = [];
let autoSwitchMounted = false;
let newChatMounted = false;

// One aui for the session, as a provider that is never remounted holds.
const aui = auiFixture;

// Each component's effect list is rebuilt per render so the sliced bodies close over that
// render's props, the way the real closures do. The memo arrays outlive the call.
// #9251: releases the retained reload shell. The provider hands this to the base pane's
// ThreadAutoSwitch, and a failed switch has to fire it whether or not a newer switch has
// superseded this one -- the shell is waiting either way.
//
// Module scope, because it is a dependency of the switch effect and the real one is a
// useCallback. A fresh closure per render would re-run that effect every time and start a
// switch per render, which is the emulator lying rather than the component misbehaving.
const onSwitchFailed = () => {
  world.switchFailedSignals += 1;
};

function renderThreadAutoSwitch(props: any): void {
  const threadId: string = props.initialThreadId;
  const syncActiveThreadId = (props.syncActiveThreadId ?? true) && !props.backgrounded;
  const paused = Boolean(props.backgrounded);
  const isLoading = props.isLoading ?? false;
  // Subscribed through useAuiState, so a render sees whatever assistant-ui holds now.
  const mainThreadId = world.mainThreadId;
  const scope: any = {
    aui,
    isLoading,
    mainThreadId,
    newThreadSwitchStateRef,
    onSwitchFailed,
    paused,
    syncActiveThreadId,
    threadId,
  };
  const effects: any[] = [
__AUTO_SWITCH_EFFECTS__
  ];
  replayEffects(effects, scope, autoSwitchMemo);
}

function renderThreadNewChatSwitch(props: any): void {
  const nonce: string = props.newThreadNonce;
  const paused = Boolean(props.backgrounded);
  const isLoading = props.isLoading ?? false;
  // Read through useChatRuntimeStore selectors, so a re-render sees the store as it stands.
  const checkpoint = state.params.checkpoint;
  const loadedContextLength = state.loadedContextLength;
  const modelLoading = state.modelLoading;
  const runActive = Object.values(state.runningByThreadId ?? {}).some(Boolean);
  // The stale-switch correction subscribes to it, exactly as ThreadAutoSwitch does.
  const mainThreadId = world.mainThreadId;
  const scope: any = {
    aui,
    isLoading,
    mainThreadId,
    newThreadSwitchStateRef,
    nonce,
    paused,
    checkpoint,
    loadedContextLength,
    modelLoading,
    runActive,
  };
  const effects: any[] = [
__NEW_CHAT_EFFECTS__
  ];
  replayEffects(effects, scope, newChatMemo);
}

function renderProviderBody(props: any): void {
  const initialThreadId = props.initialThreadId ?? undefined;
  const newThreadNonce = props.newThreadNonce ?? undefined;
  const scope: any = { initialThreadId, newThreadNonce };
  const effects: any[] = [
__PROVIDER_EFFECTS__
  ];
  replayEffects(effects, scope, providerMemo);
}

/**
 * One render of ChatRuntimeProvider with the props ChatPage hands it.
 *
 * The two child conditions and the derived props are restated from the provider's JSX
 * rather than sliced (JSX is not replayable here); they are pinned verbatim by
 * test_the_provider_wires_the_pause_and_the_shared_ref, so a change to either side of
 * that wiring fails rather than quietly diverging.
 *
 * Children's effects run before the parent's, as React commits them.
 */
export function renderProvider(props: any = {}): void {
  const autoSwitchRendered = Boolean(props.initialThreadId);
  const newChatRendered = !props.initialThreadId && Boolean(props.newThreadNonce);
  if (!autoSwitchRendered && autoSwitchMounted) {
    autoSwitchMounted = false;
    autoSwitchMemo.length = 0;
  }
  if (!newChatRendered && newChatMounted) {
    newChatMounted = false;
    newChatMemo.length = 0;
  }

  if (autoSwitchRendered) {
    autoSwitchMounted = true;
    world.component = "ThreadAutoSwitch";
    renderThreadAutoSwitch(props);
  }
  if (newChatRendered) {
    newChatMounted = true;
    world.component = "ThreadNewChatSwitch";
    renderThreadNewChatSwitch(props);
  }
  world.component = "ChatRuntimeProvider";
  renderProviderBody(props);
  world.component = "none";
}

/**
 * A render, then the re-render the app does once the switch settles: both switch calls
 * move assistant-ui's mainThreadId, which is a subscribed value.
 */
export async function renderSettled(props: any = {}): Promise<void> {
  renderProvider(props);
  await new Promise((resolve) => setTimeout(resolve, 0));
  renderProvider(props);
  await new Promise((resolve) => setTimeout(resolve, 0));
}
"""


def _rendered_effects(effects: list[tuple[list[str], str]]) -> str:
    blocks = []
    for deps, body in effects:
        blocks.append(
            "    {\n"
            f"      deps: {json.dumps(deps)},\n"
            "      run: () => {\n"
            f"{body}\n"
            "      },\n"
            "    },"
        )
    return "\n".join(blocks)


def _harness_source() -> str:
    prelude = HARNESS_PRELUDE.replace("__SET_ACTIVE_THREAD_ID__", _set_active_thread_id_reducer())
    render = (
        HARNESS_RENDER.replace("__PENDING_SWITCH_CAP__", _pending_switch_cap())
        .replace("__REF_INITIAL_VALUE__", _provider_ref_initial_value())
        .replace(
            "__PROVIDER_EFFECTS__",
            _rendered_effects(_effects(_provider_switch_state_source(), "ChatRuntimeProvider")),
        )
        .replace(
            "__AUTO_SWITCH_EFFECTS__",
            _rendered_effects(_effects(_auto_switch_source(), "ThreadAutoSwitch")),
        )
        .replace(
            "__NEW_CHAT_EFFECTS__",
            _rendered_effects(_effects(_new_chat_source(), "ThreadNewChatSwitch")),
        )
    )
    return prelude + _prompt_queue_boundary_body() + render


# A rejected switchToNewThread() is only caught on the deferred path, so node would abort
# the whole process on the immediate one. Recorded here instead, and asserted on.
SCRIPT_HEADER = """
const unhandled: string[] = [];
process.on("unhandledRejection", (reason: any) => {
  unhandled.push(String(reason));
});
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
"""


def _run(imports: str, body: str) -> dict:
    require_node(SOURCES)
    script = textwrap.dedent(
        f"""
        // @ts-nocheck
        import {{ {imports} }} from "./harness.ts";
        {SCRIPT_HEADER}
        {textwrap.dedent(body)}
        """
    )
    return run_harness(TEMP, _harness_source(), script, sources = SOURCES)


# A resident GGUF, so the second effect has something it could price. Only the tests about
# the pause gate need it; everywhere else the bar has nothing to count and stands down.
LOADED_MODEL = """
    seed({ params: { checkpoint: "unsloth/gguf-model" }, loadedContextLength: 8192 });
"""


# ---------------------------------------------------------------------------
# Structural guards for what the emulator restates rather than replays.
# ---------------------------------------------------------------------------


def test_the_provider_wires_the_pause_and_the_shared_ref():
    """Structural. ``renderProvider`` restates the provider's JSX, so the JSX has to say what
    it restates: one ref handed to both children, ``paused`` driven by ``backgrounded``, and
    ``syncActiveThreadId`` stood down while backgrounded. Without this guard the behavioural
    tests below would keep passing against wiring that no longer exists."""
    jsx = _provider_jsx()
    assert "{initialThreadId && (" in jsx, "ThreadAutoSwitch must render only for a saved thread"
    assert "{!initialThreadId && newThreadNonce && (" in jsx, (
        "ThreadNewChatSwitch must render only for a new-chat nonce, and never alongside "
        "ThreadAutoSwitch: both write the same shared switch state"
    )
    # Three, not two: ThreadBackendAutosave reads the same ref so its active-thread
    # publication can tell "this pane is on screen" from "a switch away from it has not
    # landed yet". Both switch children still share it, which is what the count protects.
    assert jsx.count("newThreadSwitchStateRef={newThreadSwitchStateRef}") == 3, (
        "both switch children must share ONE ref, or leaving a new chat for a saved one "
        "cannot tell the next new chat that the composer is no longer fresh -- and the "
        "autosave must read that same ref rather than a copy of its own"
    )
    for child in ("ThreadAutoSwitch", "ThreadNewChatSwitch", "ThreadBackendAutosave"):
        opening = jsx.index(f"<{child}")
        element = jsx[opening : jsx.index("/>", opening)]
        assert (
            "newThreadSwitchStateRef={newThreadSwitchStateRef}" in element
        ), f"{child} must be handed the shared switch state ref"
    assert jsx.count("paused={backgrounded}") == 2, (
        "BOTH children must be paused while backgrounded. ThreadAutoSwitch's first effect "
        "reaches requestTemporaryPromptQueueStop, which names every temporary queue on the "
        "page rather than this provider's, so an unpaused one stops a queue the view on "
        "screen owns"
    )
    assert (
        "syncActiveThreadId={syncActiveThreadId && !backgrounded}" in jsx
    ), "a backgrounded provider must not write the active thread the visible view owns"


def test_the_harness_stubs_every_name_the_queue_boundary_imports():
    """Structural. ``_prompt_queue_boundary_body()`` replays that module with its imports
    stripped, so an import this harness does not define becomes a ReferenceError the moment
    a stop is requested -- and that reads as "the switch never fired" rather than as a
    missing stub."""
    text = read(QUEUE)
    imported = re.findall(r"import \{ ([^}]+) \} from", text)
    names = [name.strip() for block in imported for name in block.split(",") if name.strip()]
    assert names, "parsed an empty import list; this guard would check nothing"
    with open(__file__, encoding = "utf-8") as handle:
        harness = handle.read()
    missing = [name for name in names if f"const {name}" not in harness]
    assert not missing, f"prompt-queue-boundary.ts imports {missing}, which this harness omits"


# ---------------------------------------------------------------------------
# (a) Nonce transitions.
# ---------------------------------------------------------------------------


def test_a_first_new_chat_switches_once_and_clears_nothing():
    """The provider is shared now, so the first nonce it ever sees still has to open a fresh
    thread -- and must not clear a composer the user may already have staged a file into,
    since nothing has been carried over yet."""
    out = _run(
        "renderSettled, snapshot, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        console.log(JSON.stringify({
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
          activeThreadId: snapshot().activeThreadId,
          switchState: switchState(),
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["switched"] == 1
    assert out["cleared"] == 0, "the first switch has no outgoing composer to clear"
    assert out["stops"] == 1
    assert out["activeThreadId"] is None
    assert out["switchState"] == {
        "activeNonce": "n1",
        "hasSwitched": True,
        "attempt": 1,
        "pendingSavedThreadIds": [],
    }
    assert out["unhandled"] == 0


def test_re_rendering_at_the_same_nonce_never_switches_again():
    """A shared provider re-renders constantly (a store write anywhere above it is enough).
    Only the nonce marks a new chat, so an unchanged one must be inert -- re-switching would
    throw away the thread the user is typing into."""
    out = _run(
        "renderSettled, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ newThreadNonce: "n1" });
        console.log(JSON.stringify({
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
        }));
        """,
    )
    assert out["switched"] == 1, "an unchanged nonce must not open a second thread"
    assert out["cleared"] == 0
    assert out["stops"] == 1, "nor re-stop the temporary prompt queue"


def test_a_nonce_rotation_while_visible_switches_and_clears_at_once():
    """New Chat clicked while the view is on screen. ``activeNonce`` is not null, so the
    clear does not wait for the switch: the composer being reused is the one on screen, and
    it is emptied before the new thread arrives rather than after."""
    out = _run(
        "renderProvider, renderSettled, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        let release: any;
        world.newThreadGate = new Promise((resolve) => { release = resolve; });
        renderProvider({ newThreadNonce: "n2" });
        const midSwitch = {
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          pending: world.pendingNewThreadSwitches,
        };
        release();
        await tick();
        console.log(JSON.stringify({
          midSwitch,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
          pending: world.pendingNewThreadSwitches,
        }));
        """,
    )
    assert out["midSwitch"] == {
        "switched": 2,
        "cleared": 1,
        "pending": 1,
    }, "with a live composer the clear is immediate, not deferred behind the switch"
    assert out["switched"] == 2
    assert out["cleared"] == 1, "one rotation must clear once, not once per render"
    assert out["stops"] == 2
    assert out["pending"] == 0


def test_a_nonce_that_rotates_while_paused_is_honoured_when_the_view_returns():
    """Compare keeps this provider mounted so a project run stays attached; the project
    landing behind it can rotate its nonce meanwhile (a new project, a created chat left
    behind). The switch owes that rotation once the pause lifts, and owes it exactly once."""
    out = _run(
        "renderSettled, seed, world",
        f"""
        {LOADED_MODEL}
        await renderSettled({{ newThreadNonce: "n1" }});
        // Compare opens.
        await renderSettled({{ newThreadNonce: "n1", backgrounded: true }});
        // The landing rotates its nonce while off screen.
        await renderSettled({{ newThreadNonce: "n2", backgrounded: true }});
        const paused = {{
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
          refreshes: world.refreshes,
        }};
        // Compare closes.
        await renderSettled({{ newThreadNonce: "n2" }});
        console.log(JSON.stringify({{
          paused,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
        }}));
        """,
    )
    assert out["paused"]["switched"] == 1, "a paused view must not open a thread"
    assert out["paused"]["cleared"] == 0
    assert out["paused"]["stops"] == 1
    assert out["switched"] == 2, "the rotation the pause swallowed must be honoured on return"
    assert out["cleared"] == 1, "and the composer must not carry the staged file into it"
    assert out["stops"] == 2


def test_resuming_at_the_same_nonce_does_not_switch_again():
    """Entering and leaving compare with nothing else happening. The ref survives the pause,
    so the returning view recognises its own nonce and leaves the thread it already opened
    alone -- which is the point of keeping the provider mounted in the first place."""
    out = _run(
        "renderSettled, snapshot, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ newThreadNonce: "n1", backgrounded: true });
        await renderSettled({ newThreadNonce: "n1" });
        console.log(JSON.stringify({
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
          switchState: switchState(),
          activeThreadId: snapshot().activeThreadId,
        }));
        """,
    )
    assert out["switched"] == 1, "a round trip through compare must not open a second thread"
    assert out["cleared"] == 0, "nor empty the composer the user left something staged in"
    assert out["stops"] == 1, "nor re-stop the temporary prompt queue"
    assert out["switchState"] == {
        "activeNonce": "n1",
        "hasSwitched": True,
        "attempt": 1,
        "pendingSavedThreadIds": [],
    }
    assert out["activeThreadId"] is None


def test_a_paused_new_chat_does_not_touch_the_visible_views_thread():
    """Mounted is not on screen. While it is paused the switch leaves the shared single-chat
    state to the view the user is actually looking at."""
    out = _run(
        "renderSettled, seed, snapshot, world",
        f"""
        {LOADED_MODEL}
        seed({{ activeThreadId: "thread-on-screen" }});
        await renderSettled({{ newThreadNonce: "n1", backgrounded: true }});
        console.log(JSON.stringify({{
          switched: world.switchedToNewThread,
          stops: world.stopEvents.length,
          activeThreadId: snapshot().activeThreadId,
        }}));
        """,
    )
    assert out["switched"] == 0
    assert out["stops"] == 0
    assert (
        out["activeThreadId"] == "thread-on-screen"
    ), "a paused switch must not blank the active thread the visible view is using"


def test_a_paused_new_chat_does_not_price_the_shared_context_bar():
    """The pause gate is on the recount effect too. Nothing else would hold it back here --
    a resident model, no active thread and a blank bar are exactly its firing conditions --
    so without it a view hidden behind compare would put its own empty prompt on the bar the
    visible one owns. Deferred, not skipped: the count is owed once the pause lifts."""
    out = _run(
        "renderSettled, seed, world",
        f"""
        {LOADED_MODEL}
        seed({{ activeThreadId: null, contextUsage: null }});
        await renderSettled({{ newThreadNonce: "n1", backgrounded: true }});
        const paused = {{ refreshes: world.refreshes, switched: world.switchedToNewThread }};
        await renderSettled({{ newThreadNonce: "n1" }});
        console.log(JSON.stringify({{
          paused,
          refreshes: world.refreshes,
          switched: world.switchedToNewThread,
        }}));
        """,
    )
    assert (
        out["paused"]["refreshes"] == 0
    ), "a paused view must not price a prompt onto the bar the visible view owns"
    assert out["paused"]["switched"] == 0
    assert out["refreshes"] == 1, "releasing the pause must price the empty prompt once"
    assert out["switched"] == 1


# ---------------------------------------------------------------------------
# (b) Staged attachments, immediate and deferred.
# ---------------------------------------------------------------------------


def test_an_implicit_new_chat_defers_the_clear_until_the_new_thread_arrives():
    """``/chat`` with no thread and no nonce is a new chat too, so the provider marks the
    composer used. When a nonce then appears there is no ``activeNonce`` to switch away
    from, so the clear is deferred: the composer that has to be emptied is the one the new
    thread brings, and clearing before the switch would empty the outgoing one instead."""
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({});
        const implicit = { switched: world.switchedToNewThread, state: switchState() };

        let release: any;
        world.newThreadGate = new Promise((resolve) => { release = resolve; });
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const midSwitch = {
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
        };
        release();
        await tick();
        console.log(JSON.stringify({
          implicit,
          midSwitch,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["implicit"]["switched"] == 0, "an implicit new chat opens no thread of its own"
    assert out["implicit"]["state"] == {
        "activeNonce": None,
        "hasSwitched": True,
        "attempt": 0,
        "pendingSavedThreadIds": [],
    }
    assert out["midSwitch"]["switched"] == 1
    assert out["midSwitch"]["cleared"] == 0, "the deferred clear must wait for the switch"
    assert out["midSwitch"]["stops"] == 1
    assert out["cleared"] == 1, "and must run once the new thread's composer is the live one"
    assert out["unhandled"] == 0


def test_a_composer_that_is_not_mounted_yet_does_not_break_the_switch():
    """The clear reaches for a composer that may not exist. Its try/catch has to hold on
    both paths, or the switch below it never runs and the view is stranded on the old
    thread with a blank bar."""
    out = _run(
        "renderSettled, world",
        """
        world.composerThrows = true;
        await renderSettled({});
        await renderSettled({ newThreadNonce: "n1" });
        const deferred = { switched: world.switchedToNewThread, cleared: world.clearedAttachments };
        await renderSettled({ newThreadNonce: "n2" });
        console.log(JSON.stringify({
          deferred,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["deferred"] == {"switched": 1, "cleared": 0}
    assert out["switched"] == 2, "a missing composer must not stop the immediate path switching"
    assert out["cleared"] == 0
    assert out["unhandled"] == 0


@pytest.mark.parametrize(
    ("setup", "path", "attempts"),
    [
        # activeNonce is non-null when the second nonce arrives, so the clear is the
        # synchronous one, outside any promise chain of its own.
        pytest.param(
            'await renderSettled({ newThreadNonce: "n0" });', "immediate", 2, id = "immediate"
        ),
        # activeNonce is null, so the clear is the one the switch's success arm calls.
        pytest.param("await renderSettled({});", "deferred", 1, id = "deferred"),
    ],
)
def test_an_attachment_remove_that_fails_is_not_an_unhandled_rejection(setup, path, attempts):
    """``clearAttachments()`` is not a bare call: it removes each staged file through the
    attachment adapter, so a remove() that fails rejects the promise it returns. The
    surrounding try/catch cannot see that -- it only guards the synchronous reach for the
    composer -- so the call has to be chained. The switch itself must still complete either
    way: an attachment that would not delete is not a reason to strand the view."""
    nonce = "n1"
    out = _run(
        "renderSettled, snapshot, switchState, world",
        f"""
        {setup}
        world.clearAttachmentsRejects = true;
        await renderSettled({{ newThreadNonce: "{nonce}" }});
        console.log(JSON.stringify({{
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
          state: switchState(),
          activeThreadId: snapshot().activeThreadId,
          unhandled: unhandled.length,
          reasons: unhandled,
        }}));
        """,
    )
    assert out["cleared"] == 1, f"the {path} path still attempts the clear"
    assert out["unhandled"] == 0, (
        f"a rejecting attachment remove() on the {path} path became an unhandled "
        f"rejection: {out['reasons']}"
    )
    assert out["state"] == {
        "activeNonce": nonce,
        "hasSwitched": True,
        "attempt": attempts,
        "pendingSavedThreadIds": [],
    }
    assert out["activeThreadId"] is None, "and the switch below it still ran"


# ---------------------------------------------------------------------------
# (c) ThreadAutoSwitch resets the shared nonce.
# ---------------------------------------------------------------------------


def test_opening_a_saved_thread_releases_the_nonce_so_the_same_one_switches_again():
    """The sidebar hands the same landing nonce back after a saved chat. Without the reset
    the returning view would recognise its own nonce, decline to switch, and leave the user
    looking at the saved thread's messages under a New Chat header."""
    out = _run(
        "renderSettled, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ initialThreadId: "thread-a" });
        const opened = { state: switchState(), switchedTo: world.switchedToThread.slice() };
        await renderSettled({ newThreadNonce: "n1" });
        console.log(JSON.stringify({
          opened,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          state: switchState(),
        }));
        """,
    )
    assert out["opened"]["state"] == {
        "activeNonce": None,
        "hasSwitched": True,
        "attempt": 2,
        "pendingSavedThreadIds": [],
    }
    assert out["opened"]["switchedTo"] == ["thread-a"]
    assert out["switched"] == 2, "back to the same nonce must restore the new thread"
    assert out["state"] == {
        "activeNonce": "n1",
        "hasSwitched": True,
        "attempt": 3,
        "pendingSavedThreadIds": [],
    }


def test_new_chat_then_a_saved_thread_then_new_chat_clears_the_staged_attachment():
    """The full round trip, which is what a shared provider made possible and what makes
    the composer stale: the new thread it switches back to is the same uninitialized one,
    still holding whatever was staged before the detour."""
    out = _run(
        "renderProvider, renderSettled, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ initialThreadId: "thread-a" });

        let release: any;
        world.newThreadGate = new Promise((resolve) => { release = resolve; });
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const midSwitch = { cleared: world.clearedAttachments };
        release();
        await tick();
        console.log(JSON.stringify({
          midSwitch,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          switchedTo: world.switchedToThread,
        }));
        """,
    )
    assert (
        out["midSwitch"]["cleared"] == 0
    ), "after a saved thread the clear is the deferred one: activeNonce was reset to null"
    assert out["cleared"] == 1, "the reused new-thread composer must lose its attachment"
    assert out["switched"] == 2


def test_a_saved_thread_that_is_already_the_main_one_still_releases_the_nonce():
    """Coming back to the chat whose run was left going: assistant-ui already holds it as
    main, so the switch branch is skipped entirely. The nonce reset has to sit ABOVE that
    branch or this route back leaves the stale nonce marked active and the next New Chat
    silently does nothing."""
    out = _run(
        "renderSettled, seedMainThread, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        // The saved chat is the one assistant-ui already has open.
        seedMainThread("thread-a");
        const stopsBefore = world.stopEvents.length;
        await renderSettled({ initialThreadId: "thread-a" });
        const opened = {
          state: switchState(),
          switchedTo: world.switchedToThread.slice(),
          stops: world.stopEvents.length - stopsBefore,
        };
        await renderSettled({ newThreadNonce: "n1" });
        console.log(JSON.stringify({
          opened,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
        }));
        """,
    )
    assert out["opened"]["switchedTo"] == [], "a thread already open must not be re-opened"
    assert out["opened"]["stops"] == 0, "nor stop a queue on a switch that did not happen"
    assert out["opened"]["state"] == {
        "activeNonce": None,
        "hasSwitched": True,
        "attempt": 1,
        "pendingSavedThreadIds": [],
    }, "the nonce reset must not sit behind the mainThreadId guard"
    assert out["switched"] == 2, "the returning New Chat must still open a fresh thread"
    assert out["cleared"] == 1


# ---------------------------------------------------------------------------
# (d) Rapid switching, faster than switchToNewThread resolves.
# ---------------------------------------------------------------------------


def test_a_deferred_clear_for_a_nonce_that_moved_on_is_dropped():
    """The deferred clear lands after an await, by which time the user may be two views
    further on. Clearing then would empty a composer they have since staged a file into,
    so the guard re-reads the shared ref and drops the callback."""
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({});
        let release: any;
        world.newThreadGate = new Promise((resolve) => { release = resolve; });

        // Deferred clear armed for n1, still awaiting its switch.
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        // The user moves on before it resolves.
        renderProvider({ newThreadNonce: "n2" });
        await tick();
        const beforeRelease = {
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
        };

        release();
        await tick();
        await tick();
        console.log(JSON.stringify({
          beforeRelease,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          state: switchState(),
          pending: world.pendingNewThreadSwitches,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["beforeRelease"] == {
        "switched": 2,
        "cleared": 1,
    }, "n2 takes the immediate path and clears before its own switch"
    assert (
        out["cleared"] == 1
    ), "the stale deferred clear must not fire: n1's composer is two views behind"
    assert out["switched"] == 2
    assert out["state"] == {
        "activeNonce": "n2",
        "hasSwitched": True,
        "attempt": 2,
        "pendingSavedThreadIds": [],
    }
    assert out["pending"] == 0, "no switch callback may be left waiting"
    assert out["unhandled"] == 0


def test_three_nonces_faster_than_the_switch_resolves_clear_once_each_at_most():
    """Clicking New Chat repeatedly. Each nonce owes one thread; only the first is deferred,
    and only the last deferral could still be the live one."""
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({});
        let release: any;
        world.newThreadGate = new Promise((resolve) => { release = resolve; });
        renderProvider({ newThreadNonce: "n1" });
        renderProvider({ newThreadNonce: "n2" });
        renderProvider({ newThreadNonce: "n3" });
        await tick();
        const beforeRelease = {
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
        };
        release();
        await tick();
        await tick();
        console.log(JSON.stringify({
          beforeRelease,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
          state: switchState(),
          pending: world.pendingNewThreadSwitches,
        }));
        """,
    )
    assert out["switched"] == 3, "every nonce owes exactly one thread"
    assert out["stops"] == 3
    assert (
        out["cleared"] == 2
    ), "n2 and n3 clear immediately; n1's deferred clear is dropped as stale"
    assert out["beforeRelease"]["cleared"] == 2, "neither immediate clear waited for a switch"
    assert out["state"] == {
        "activeNonce": "n3",
        "hasSwitched": True,
        "attempt": 3,
        "pendingSavedThreadIds": [],
    }
    assert out["pending"] == 0


# ---------------------------------------------------------------------------
# (e) A switchToNewThread() that rejects.
# ---------------------------------------------------------------------------


def test_a_rejected_switch_releases_the_nonce_so_the_same_one_can_be_retried():
    """The state is mutated before the switch is attempted, so a rejection would otherwise
    leave the guard believing a nonce that never opened a thread is the live one -- and the
    nonce is the only thing that marks a New Chat, so the same New Chat could never be
    served again. The rejection arm puts ``activeNonce`` back.

    Retried at the next commit that re-runs the effect, not at the next render: React
    re-runs an effect only when one of its dependencies changed, which is as true of this
    effect as of any other. ``paused`` flipping on a compare round trip is one such commit,
    and it is the one exercised here."""
    out = _run(
        "renderProvider, renderSettled, snapshot, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        world.newThreadRejects = true;
        renderProvider({ newThreadNonce: "n2" });
        await tick();
        await tick();
        const afterReject = {
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
          state: switchState(),
          activeThreadId: snapshot().activeThreadId,
          mainThreadId: world.mainThreadId,
          unhandled: unhandled.length,
        };

        // Compare opens and closes: paused is a dependency, so the effect runs again.
        world.newThreadRejects = false;
        await renderSettled({ newThreadNonce: "n2", backgrounded: true });
        await renderSettled({ newThreadNonce: "n2" });
        console.log(JSON.stringify({
          afterReject,
          switched: world.switchedToNewThread,
          state: switchState(),
          mainThreadId: world.mainThreadId,
          unhandled: unhandled.length,
        }));
        """,
    )
    state = out["afterReject"]
    assert state["switched"] == 2, "the switch was attempted"
    assert state["state"] == {
        "activeNonce": None,
        "hasSwitched": True,
        "attempt": 2,
        "pendingSavedThreadIds": [],
    }, "a nonce whose switch failed must not stay recorded as the live one"
    assert state["activeThreadId"] is None
    assert state["mainThreadId"] == "local-1", "the user is still on the previous thread"
    assert state["cleared"] == 1, "the outgoing composer was already emptied"
    assert state["unhandled"] == 0, (
        "both arms of the handler are attached on every path, so a failed switch is never "
        "an unhandled rejection"
    )
    assert out["switched"] == 3, (
        "the released nonce lets the same New Chat be served in place; without the release "
        "the view stays on the old thread for good"
    )
    assert out["state"] == {
        "activeNonce": "n2",
        "hasSwitched": True,
        "attempt": 3,
        "pendingSavedThreadIds": [],
    }
    assert out["mainThreadId"] == "local-3", "and the retry is the thread the user ends on"
    assert out["unhandled"] == 0


def test_a_rejected_switch_on_the_deferred_path_keeps_the_draft_and_releases_the_nonce():
    """The deferred clear exists to empty the composer the new thread brings. A switch that
    failed brought none, so the user is still looking at the outgoing composer and its
    staged file has to survive -- while the nonce is still released, or the retry that would
    finally move them off it can never run."""
    out = _run(
        "renderSettled, switchState, world",
        """
        await renderSettled({});
        world.newThreadRejects = true;
        await renderSettled({ newThreadNonce: "n1" });
        const afterReject = {
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          state: switchState(),
          unhandled: unhandled.length,
        };

        // The retry, once the runtime is healthy again.
        world.newThreadRejects = false;
        await renderSettled({ newThreadNonce: "n1", backgrounded: true });
        await renderSettled({ newThreadNonce: "n1" });
        console.log(JSON.stringify({
          afterReject,
          switched: world.switchedToNewThread,
          cleared: world.clearedAttachments,
          state: switchState(),
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["afterReject"]["switched"] == 1
    assert out["afterReject"]["unhandled"] == 0, "the rejection arm handles it"
    assert out["afterReject"]["cleared"] == 0, (
        "a failed switch must not clear: the composer still on screen is the outgoing one, "
        "and the user's staged file is still theirs"
    )
    assert out["afterReject"]["state"] == {
        "activeNonce": None,
        "hasSwitched": True,
        "attempt": 1,
        "pendingSavedThreadIds": [],
    }, "the nonce is released on the deferred path too"
    assert out["switched"] == 2, "the same nonce is retried once the effect re-runs"
    assert out["cleared"] == 1, "and the retry that does open a thread clears as it should"
    assert out["state"] == {
        "activeNonce": "n1",
        "hasSwitched": True,
        "attempt": 2,
        "pendingSavedThreadIds": [],
    }
    assert out["unhandled"] == 0


def test_a_late_rejection_cannot_disturb_a_nonce_that_has_since_moved_on():
    """The rejection arm writes to shared state after an await, so it has the same staleness
    problem the deferred clear has and the same guard: it only releases the nonce it was
    started for. A failure for n1 arriving after the user reached n2 must leave n2 alone,
    or the next commit would throw away the thread they are already typing in."""
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({});
        let release: any;
        world.newThreadGate = new Promise((resolve) => { release = resolve; });
        world.newThreadRejects = true;
        renderProvider({ newThreadNonce: "n1" });
        await tick();

        // The user moves on before n1's failure lands.
        world.newThreadRejects = false;
        world.newThreadGate = null;
        renderProvider({ newThreadNonce: "n2" });
        await tick();
        const beforeFailure = { state: switchState(), switched: world.switchedToNewThread };

        release();
        await tick();
        await tick();
        console.log(JSON.stringify({
          beforeFailure,
          state: switchState(),
          switched: world.switchedToNewThread,
          pending: world.pendingNewThreadSwitches,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["beforeFailure"]["state"] == {
        "activeNonce": "n2",
        "hasSwitched": True,
        "attempt": 2,
        "pendingSavedThreadIds": [],
    }
    assert out["beforeFailure"]["switched"] == 2
    assert out["state"] == {
        "activeNonce": "n2",
        "hasSwitched": True,
        "attempt": 2,
        "pendingSavedThreadIds": [],
    }, "n1's late failure must not release the nonce n2 opened a thread for"
    assert out["switched"] == 2, "and must not trigger a switch of its own"
    assert out["pending"] == 0
    assert out["unhandled"] == 0


def test_a_late_failure_cannot_release_a_nonce_a_newer_attempt_owns():
    """Two switches for the SAME nonce can be in flight, so the nonce cannot identify an
    attempt. New Chat starts a switch, a saved chat opens while it runs (releasing the nonce
    without ending the switch), then the same New Chat starts a second switch that succeeds.
    When the first fails, a rejection arm keyed on the nonce alone reads its own nonce back and
    releases it, switching away from the thread the second attempt opened.

    The complement is asserted on the same shared state: a lone failure with nothing overlapping
    must still release, or a New Chat that failed once could never be served again."""
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({});

        // Attempt 1 for n1, held open and destined to fail.
        let release: any;
        world.newThreadGate = new Promise((resolve) => { release = resolve; });
        world.newThreadRejects = true;
        renderProvider({ newThreadNonce: "n1" });
        await tick();

        // A saved chat is opened while it is still going: ThreadAutoSwitch releases the
        // nonce, which is what lets a second switch for the same nonce start at all.
        world.newThreadGate = null;
        world.newThreadRejects = false;
        // renderSettled, not a single render: the app re-renders when the saved switch
        // moves mainThreadId, and that re-render is what lets ThreadAutoSwitch see its own
        // thread arrive. Rendering once leaves the switch looking permanently in flight.
        await renderSettled({ initialThreadId: "thread-a" });

        // Back to the same New Chat: a third attempt for n1, and this one succeeds.
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const beforeFailure = {
          state: switchState(),
          switched: world.switchedToNewThread,
          mainThreadId: world.mainThreadId,
        };

        // Only now does the first attempt fail.
        release();
        await tick();
        await tick();
        const afterFailure = {
          state: switchState(),
          mainThreadId: world.mainThreadId,
          switched: world.switchedToNewThread,
        };

        // The next commit that re-runs the effect: a compare round trip.
        await renderSettled({ newThreadNonce: "n1", backgrounded: true });
        await renderSettled({ newThreadNonce: "n1" });
        const afterNextCommit = {
          state: switchState(),
          switched: world.switchedToNewThread,
          mainThreadId: world.mainThreadId,
        };

        // Complement: a lone failure, nothing overlapping, must release.
        world.newThreadRejects = true;
        await renderSettled({ newThreadNonce: "n2" });
        console.log(JSON.stringify({
          beforeFailure,
          afterFailure,
          afterNextCommit,
          loneFailure: { state: switchState(), switched: world.switchedToNewThread },
          pending: world.pendingNewThreadSwitches,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert (
        out["beforeFailure"]
        == {
            "state": {
                "activeNonce": "n1",
                "hasSwitched": True,
                "attempt": 3,
                "pendingSavedThreadIds": [],
            },
            "switched": 2,
            "mainThreadId": "local-2",
        }
    ), "two new-chat attempts for one nonce, the second of which opened the thread on screen, plus the saved detour between them"
    assert out["afterFailure"]["state"] == {
        "activeNonce": "n1",
        "hasSwitched": True,
        "attempt": 3,
        "pendingSavedThreadIds": [],
    }, "the first attempt's failure must not release a nonce a later one owns"
    assert (
        out["afterFailure"]["mainThreadId"] == "local-2"
    ), "and must not disturb the thread attempt 2 opened"
    assert out["afterFailure"]["switched"] == 2
    assert out["afterNextCommit"] == {
        "state": {
            "activeNonce": "n1",
            "hasSwitched": True,
            "attempt": 3,
            "pendingSavedThreadIds": [],
        },
        "switched": 2,
        "mainThreadId": "local-2",
    }, "a released nonce would have made the next commit switch away from local-2"
    assert out["loneFailure"] == {
        "state": {
            "activeNonce": None,
            "hasSwitched": True,
            "attempt": 4,
            "pendingSavedThreadIds": [],
        },
        "switched": 3,
    }, "with nothing overlapping, a failed attempt must still release its nonce"
    assert out["pending"] == 0
    assert out["unhandled"] == 0


def test_a_late_deferred_clear_cannot_wipe_an_attachment_a_newer_attempt_staged():
    """The success arm has the same two-in-flight problem. The deferred clear is armed only
    once the nonce was released, so: a switch, a detour releasing the nonce, a New Chat held open
    with the clear armed, then a second detour and return starting a newer attempt for the same
    nonce. When the older switch resolves, a success arm keyed on the nonce alone clears the
    composer the newer attempt is using, taking an attachment staged in it.

    The complement is asserted on the same shared state: a detour nulls the nonce and must still
    cancel a pending clear, so the nonce check cannot simply become the attempt check."""
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        // A first switch, so hasSwitched is set.
        await renderSettled({ newThreadNonce: "n0" });
        // Detour releases the nonce, which is what arms the DEFERRED clear next time.
        await renderSettled({ initialThreadId: "thread-a" });

        // Attempt 2 for n1, held open, deferred clear armed.
        let release;
        world.newThreadGate = new Promise((resolve) => { release = resolve; });
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const armed = { state: switchState(), cleared: world.clearedAttachments };

        // Second detour, then back to the same New Chat: attempt 3 opens its thread.
        world.newThreadGate = null;
        renderProvider({ initialThreadId: "thread-a" });
        await tick();
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const beforeLate = {
          state: switchState(),
          cleared: world.clearedAttachments,
          switched: world.switchedToNewThread,
        };

        // Only now does the older switch resolve, with the user settled in attempt 3's thread.
        release();
        await tick();
        await tick();
        const afterLate = { state: switchState(), cleared: world.clearedAttachments };
        console.log(JSON.stringify({
          armed, beforeLate, afterLate, unhandled: unhandled.length,
        }));
        """,
    )
    assert out["armed"]["state"] == {
        "activeNonce": "n1",
        "hasSwitched": True,
        "attempt": 3,
        "pendingSavedThreadIds": [],
    }, "the detour must leave the nonce released so this switch arms the deferred clear"
    assert (
        out["beforeLate"]["state"]["attempt"] == 4
    ), "the return after the second detour must start a newer attempt for the same nonce"
    assert (
        out["afterLate"]["cleared"] == out["beforeLate"]["cleared"]
    ), "the older switch's deferred clear must not wipe the composer the newer attempt opened"
    assert out["afterLate"]["state"]["activeNonce"] == "n1"
    assert out["unhandled"] == 0


def test_every_switch_the_effect_starts_bumps_the_attempt_exactly_once():
    """``attempt`` is what tells two in-flight switches apart, so it has to advance once
    per switch actually started and never on a render that returned at a guard. Driven
    across the whole mix: a first mount, a repeat render, a rotation, a pause, a resume, a
    saved-thread detour and a return.

    Saved-thread switches count too. They used not to, which left two of them sharing one
    token, so a first saved chat rejecting after a second had settled read as current and
    detached the chat on screen."""
    out = _run(
        "renderSettled, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ newThreadNonce: "n2" });
        await renderSettled({ newThreadNonce: "n2", backgrounded: true });
        await renderSettled({ newThreadNonce: "n3", backgrounded: true });
        await renderSettled({ newThreadNonce: "n3" });
        await renderSettled({ initialThreadId: "thread-a" });
        await renderSettled({ newThreadNonce: "n3" });
        console.log(JSON.stringify({
          state: switchState(),
          switched: world.switchedToNewThread,
          switchedToSaved: world.switchedToThread.length,
        }));
        """,
    )
    assert out["switched"] == 4, "n1, n2, n3 and the return to n3 after the saved chat"
    assert out["switchedToSaved"] == 1, "the saved-thread detour"
    assert out["state"]["attempt"] == out["switched"] + out["switchedToSaved"], (
        "attempt must count switches started, of either kind, and not renders: a bump on a "
        "render that returned at a guard would let a stale rejection think it is current, "
        "and a switch that does not bump would let one think it is current too"
    )
    assert out["state"] == {
        "activeNonce": "n3",
        "hasSwitched": True,
        "attempt": 5,
        "pendingSavedThreadIds": [],
    }


def test_a_rejected_saved_thread_switch_blanks_the_bar_only_while_visible():
    """ThreadAutoSwitch's own failure path, for contrast: it catches, and the catch is
    already gated on syncActiveThreadId, so a backgrounded provider cannot blank the active
    thread the visible view owns even when its switch fails."""
    out = _run(
        "renderSettled, seed, snapshot, world",
        """
        world.switchToThreadRejects = true;
        seed({ activeThreadId: "thread-on-screen" });
        await renderSettled({ initialThreadId: "thread-a", backgrounded: true });
        const backgroundedRun = {
          switchedTo: world.switchedToThread.slice(),
          activeThreadId: snapshot().activeThreadId,
        };
        await renderSettled({ initialThreadId: "thread-a" });
        console.log(JSON.stringify({
          backgroundedRun,
          activeThreadId: snapshot().activeThreadId,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert (
        out["backgroundedRun"]["switchedTo"] == []
    ), "a backgrounded provider does not attempt the switch at all"
    assert (
        out["backgroundedRun"]["activeThreadId"] == "thread-on-screen"
    ), "so it cannot blank the visible view's thread, and its catch is gated as well"
    assert out["activeThreadId"] is None, "the same failure while visible does blank it"
    assert out["unhandled"] == 0, "ThreadAutoSwitch catches its own rejection"


# ---------------------------------------------------------------------------
# (f) requestTemporaryPromptQueueStop: how often, and whose queue.
# ---------------------------------------------------------------------------


def test_a_full_compare_round_trip_stops_the_temporary_queue_once():
    """The stop discards an incognito queue that is about to become unreachable. Pausing and
    resuming abandons nothing, so the only stop in the whole cycle is the one the New Chat
    itself owed."""
    out = _run(
        "renderSettled, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ newThreadNonce: "n1", backgrounded: true });
        await renderSettled({ newThreadNonce: "n1", backgrounded: true });
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ newThreadNonce: "n1" });
        console.log(JSON.stringify({
          stops: world.stopEvents.length,
          firedBy: world.stopEvents.map((event: any) => event.firedBy),
        }));
        """,
    )
    assert out["stops"] == 1, "a pause and resume must not re-stop the queue"
    assert out["firedBy"] == ["ThreadNewChatSwitch"]


def test_a_backgrounded_saved_thread_switch_stops_nothing_and_pays_on_resume():
    """requestTemporaryPromptQueueStop names every materialized temporary queue on the page,
    not this provider's, so an off-screen caller stops a queue the visible view owns. The
    pause gate on the first effect is what keeps a backgrounded provider away from it -- and
    it is a dependency, so the stop and the switch are deferred to the resume, not skipped."""
    out = _run(
        "renderSettled, seedMainThread, snapshot, switchState, world",
        """
        // An incognito queue materialized by the pane the user is looking at.
        world.promptQueues = {
          "temp-chat": { temporary: true },
          "saved-chat": { temporary: false },
        };
        seedMainThread("some-other-thread");
        await renderSettled({ initialThreadId: "thread-a", backgrounded: true });
        const backgroundedRun = {
          stops: world.stopEvents.length,
          switchedTo: world.switchedToThread.slice(),
          activeThreadId: snapshot().activeThreadId,
        };

        // Compare closes: this view is on screen and owes both.
        await renderSettled({ initialThreadId: "thread-a" });
        console.log(JSON.stringify({
          backgroundedRun,
          stops: world.stopEvents.length,
          events: world.stopEvents,
          switchedTo: world.switchedToThread,
          activeThreadId: snapshot().activeThreadId,
          state: switchState(),
        }));
        """,
    )
    assert (
        out["backgroundedRun"]["stops"] == 0
    ), "a backgrounded ThreadAutoSwitch must not reach a stop that names the visible view's queue"
    assert out["backgroundedRun"]["switchedTo"] == [], "nor switch a thread off screen"
    assert (
        out["backgroundedRun"]["activeThreadId"] is None
    ), "and the second effect stays gated through syncActiveThreadId"
    assert out["stops"] == 1, "the resume pays the stop the pause deferred, exactly once"
    event = out["events"][0]
    assert event["firedBy"] == "ThreadAutoSwitch"
    assert event["detail"]["temporaryOnly"] is True
    assert event["detail"]["threadIds"] == ["temp-chat"]
    assert out["switchedTo"] == ["thread-a"], "and the switch the pause deferred"
    assert out["activeThreadId"] == "thread-a"
    assert out["state"] == {
        "activeNonce": None,
        "hasSwitched": False,
        "attempt": 1,
        "pendingSavedThreadIds": [],
    }, "the nonce reset is deferred with the rest of the effect, and so is the bump"


def test_a_failed_saved_thread_switch_retries_only_while_the_view_is_on_screen():
    """The route that reaches the gate above without a synthetic mount. A switch that fails
    leaves mainThreadId pointing elsewhere, so the guard the effect would otherwise return
    at stays open, and ``syncActiveThreadId`` flips on every compare open and close -- a
    dependency, so each toggle re-runs the effect. Only the on-screen half of that toggle
    may act: the compare half would be stopping a queue in the pane the user is using."""
    out = _run(
        "renderSettled, seedMainThread, world",
        """
        world.switchToThreadRejects = true;
        world.promptQueues = { "compare-temp": { temporary: true } };
        // The chat the user was reading before clicking the one that is now gone.
        seedMainThread("thread-b");
        await renderSettled({ initialThreadId: "thread-a" });
        const visible = {
          stops: world.stopEvents.length,
          switchedTo: world.switchedToThread.length,
        };
        // Compare opens, then closes.
        await renderSettled({ initialThreadId: "thread-a", backgrounded: true });
        const inCompare = {
          stops: world.stopEvents.length,
          switchedTo: world.switchedToThread.length,
        };
        await renderSettled({ initialThreadId: "thread-a" });
        console.log(JSON.stringify({
          visible,
          inCompare,
          stops: world.stopEvents.length,
          firedBy: world.stopEvents.map((event: any) => event.firedBy),
          switchedTo: world.switchedToThread.length,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["visible"] == {"stops": 1, "switchedTo": 1}, "the first attempt is legitimate"
    assert out["inCompare"] == {"stops": 1, "switchedTo": 1}, (
        "opening compare re-runs the effect, and the pause gate is what stops it from "
        "killing the compare pane's own incognito queue"
    )
    assert out["stops"] == 2, "closing compare retries, on screen, where the retry belongs"
    assert out["firedBy"] == ["ThreadAutoSwitch", "ThreadAutoSwitch"]
    assert out["switchedTo"] == 2
    assert out["unhandled"] == 0


# ---------------------------------------------------------------------------
# (g) Bounded work across many view switches.
# ---------------------------------------------------------------------------


def test_a_hundred_and_twenty_view_switches_stay_bounded():
    """The provider now outlives every view switch, so anything it accumulates accumulates
    for the session: a per-switch listener, a callback left waiting on a switch that already
    resolved, a second stop per pass. Each switch is allowed a fixed, small amount of work
    and nothing else."""
    out = _run(
        "renderSettled, switchState, world",
        """
        const switches = 120;
        for (let i = 0; i < switches; i += 1) {
          await renderSettled(
            i % 2 === 0 ? { newThreadNonce: "n1" } : { initialThreadId: "thread-a" },
          );
        }
        console.log(JSON.stringify({
          switches,
          switchedToNewThread: world.switchedToNewThread,
          switchedToThread: world.switchedToThread.length,
          cleared: world.clearedAttachments,
          stops: world.stopEvents.length,
          listeners: world.eventListeners,
          pending: world.pendingNewThreadSwitches,
          state: switchState(),
          unhandled: unhandled.length,
        }));
        """,
    )
    passes = out["switches"] // 2
    assert out["switchedToNewThread"] == passes, "one new thread per project pass, no more"
    assert out["switchedToThread"] == passes, "one saved-thread switch per single pass"
    assert (
        out["cleared"] == passes - 1
    ), "one clear per return to the landing, and none on the first arrival"
    assert out["stops"] == 2 * passes, "one stop per switch, not a growing number per switch"
    assert out["listeners"] == 0, "no listener may survive a view switch"
    assert out["pending"] == 0, "no switch callback may be left waiting"
    assert out["state"] == {
        "activeNonce": None,
        "hasSwitched": True,
        "attempt": 2 * passes,
        "pendingSavedThreadIds": [],
    }, "one bump per switch started, of either kind, and none per render"
    assert out["unhandled"] == 0


@pytest.mark.parametrize("switches", [20, 120])
def test_the_work_per_view_switch_does_not_grow_with_the_session(switches):
    """The same run at two lengths. Totals that stay linear in the number of switches are
    what rules out per-switch accumulation; a single length cannot tell O(1) from O(n)."""
    out = _run(
        "renderSettled, world",
        f"""
        for (let i = 0; i < {switches}; i += 1) {{
          await renderSettled(
            i % 2 === 0 ? {{ newThreadNonce: "n1" }} : {{ initialThreadId: "thread-a" }},
          );
        }}
        console.log(JSON.stringify({{
          stops: world.stopEvents.length,
          cleared: world.clearedAttachments,
          switched: world.switchedToNewThread,
        }}));
        """,
    )
    passes = switches // 2
    assert out["stops"] == 2 * passes
    assert out["cleared"] == passes - 1
    assert out["switched"] == passes


def test_a_saved_switch_that_lands_after_the_route_moved_does_not_take_the_view():
    """The provider is shared now, so a stale switchToThread has nothing to remount into.

    Project landing, then a saved chat, then back before its switch settles. assistant-ui
    assigns mainThreadId when the promise resolves and cannot know the route moved, so the
    project landing's composer ended up pointed at the saved chat and the next message went
    to the wrong conversation. The nonce view recognises that exact id -- the one
    ThreadAutoSwitch recorded as pending -- and reasserts a fresh thread.
    """
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        const openedByLanding = world.mainThreadId;

        // Open a saved chat, but hold its switch open.
        let release: any;
        world.switchToThreadGate = new Promise((resolve) => { release = resolve; });
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // Back to the project landing before it settles.
        world.switchToThreadGate = null;
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const beforeArrival = world.mainThreadId;

        // Only now does the saved switch land.
        release();
        await tick();
        await tick();
        // The commit React runs because mainThreadId moved.
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        await tick();

        console.log(JSON.stringify({
          openedByLanding,
          beforeArrival,
          mainThreadId: world.mainThreadId,
          switchedTo: world.switchedToThread,
          savedSwitches: world.switchedToThread.filter((id) => id.startsWith("thread-")),
          state: switchState(),
          unhandled: unhandled.length,
        }));
        """,
    )
    # switchedToThread also carries the correction's reattach, so name the saved ones.
    assert out["savedSwitches"] == ["thread-a"], "the saved switch really was started"
    assert out["mainThreadId"] != "thread-a", (
        "a switch that landed after the user returned must not leave the visible project "
        "landing pointed at the saved chat, or the next message goes to the wrong thread"
    )
    assert out["state"]["activeNonce"] == "n1", "and the landing still owns the nonce"
    assert out["state"]["pendingSavedThreadIds"] == [], "the claim is spent, not left armed"
    assert out["unhandled"] == 0


def test_a_saved_switch_that_lands_in_time_is_left_alone():
    """The complement, and the one a blunter guard gets wrong: the user really is on the
    saved chat when it settles, so nothing may switch away from it."""
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        await renderSettled({ initialThreadId: "thread-a" });
        console.log(JSON.stringify({
          mainThreadId: world.mainThreadId,
          switchedToNew: world.switchedToNewThread,
          state: switchState(),
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["mainThreadId"] == "thread-a", "the saved chat is the one on screen"
    assert out["switchedToNew"] == 1, "only the landing's own switch, no correction"
    assert out["state"]["pendingSavedThreadIds"] == [], "the claim is released on arrival"
    assert out["unhandled"] == 0


def test_a_saved_switch_that_fails_late_does_not_detach_the_view_that_replaced_it():
    """The rejection arm writes shared state after an await, so it needs the same staleness
    guard the other arms have. Unguarded, a saved switch failing after the user returned to
    the project landing cleared the active id that landing had just set, detaching a chat
    the failure has nothing to do with."""
    out = _run(
        "renderProvider, renderSettled, seed, snapshot, world",
        """
        await renderSettled({ newThreadNonce: "n1" });

        let release: any;
        world.switchToThreadGate = new Promise((resolve) => { release = resolve; });
        world.switchToThreadRejects = true;
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // Back to the landing, which takes ownership of the active id.
        world.switchToThreadGate = null;
        world.switchToThreadRejects = false;
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        seed({ activeThreadId: "owned-by-the-landing" });

        release();
        await tick();
        await tick();

        console.log(JSON.stringify({
          activeThreadId: snapshot().activeThreadId,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert (
        out["activeThreadId"] == "owned-by-the-landing"
    ), "a superseded failure must not clear an active id a newer view owns"
    assert out["unhandled"] == 0


def test_two_saved_switches_in_flight_are_both_corrected():
    """One scalar claim cannot speak for two outstanding switches.

    Open A, open B before A settles, then return to the project landing. If B lands first
    the correction spends the claim, and A landing afterwards finds nothing recorded: the
    nonce view leaves assistant-ui pointed at A and the next project-composer message is
    appended to a chat the user is not looking at. Every switch this view starts has to be
    tracked, not just the most recent one.
    """
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });

        // A opens first and is held.
        let releaseA: any;
        world.switchToThreadGate = new Promise((resolve) => { releaseA = resolve; });
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // B opens before A settles, on its own gate.
        let releaseB: any;
        world.switchToThreadGate = new Promise((resolve) => { releaseB = resolve; });
        renderProvider({ initialThreadId: "thread-b" });
        await tick();

        // Back to the project landing while both are still out.
        world.switchToThreadGate = null;
        renderProvider({ newThreadNonce: "n1" });
        await tick();

        // B lands first and is corrected.
        releaseB();
        await tick();
        await tick();
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        await tick();
        const afterB = world.mainThreadId;

        // Then A lands, with the claim already spent on B.
        releaseA();
        await tick();
        await tick();
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        await tick();

        console.log(JSON.stringify({
          afterB,
          mainThreadId: world.mainThreadId,
          savedSwitches: world.switchedToThread.filter((id) => id.startsWith("thread-")),
          state: switchState(),
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["savedSwitches"] == ["thread-a", "thread-b"], "both saved switches really started"
    assert out["afterB"] != "thread-b", "the first arrival is corrected, as before"
    assert out["mainThreadId"] != "thread-a", (
        "the second stale arrival must be corrected too, or the visible project landing is "
        "left pointed at a saved chat and the next message goes to it"
    )
    assert out["state"]["activeNonce"] == "n1", "and the landing still owns the nonce"
    assert out["unhandled"] == 0


def test_outstanding_claims_stay_bounded():
    """A claim is only spent when its id arrives, so a switch that never settles never
    spends one. Nothing here is a real session -- it takes that many saved-chat clicks with
    not one switch resolving -- but the list must not grow for the life of the tab."""
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        world.switchToThreadGate = new Promise(() => {});
        for (let i = 0; i < 40; i += 1) {
          renderProvider({ initialThreadId: `thread-${i}` });
          await tick();
        }
        console.log(JSON.stringify({
          claims: switchState().pendingSavedThreadIds,
          started: world.switchedToThread.length,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["started"] == 40, "every switch really was started"
    assert len(out["claims"]) == 16, "the claim list is capped"
    assert out["claims"][-1] == {
        "id": "thread-39",
        "settled": False,
    }, "and it keeps the newest, dropping the oldest"
    assert all(
        claim["settled"] is False for claim in out["claims"]
    ), "none of these switches ever settled, which is why nothing retired them"
    assert out["unhandled"] == 0


def test_the_same_saved_thread_opened_twice_is_corrected_twice():
    """Deduplicating the claim by id loses one of two outstanding switches for that id.

    A, then B, then A again before any settles: both A switches are really started, so both
    can land. The first stale A arrival spends the single claim and the second is accepted
    beneath the project composer, which is the same wrong-conversation bug one layer down.
    A claim belongs to a switch, not to a thread id.
    """
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });

        let releaseA1: any, releaseB: any, releaseA2: any;
        world.switchToThreadGate = new Promise((resolve) => { releaseA1 = resolve; });
        renderProvider({ initialThreadId: "thread-a" });
        await tick();
        world.switchToThreadGate = new Promise((resolve) => { releaseB = resolve; });
        renderProvider({ initialThreadId: "thread-b" });
        await tick();
        world.switchToThreadGate = new Promise((resolve) => { releaseA2 = resolve; });
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // Back to the project landing with all three out.
        world.switchToThreadGate = null;
        renderProvider({ newThreadNonce: "n1" });
        await tick();

        for (const release of [releaseA1, releaseB, releaseA2]) {
          release();
          await tick();
          await tick();
          renderProvider({ newThreadNonce: "n1" });
          await tick();
          await tick();
        }

        console.log(JSON.stringify({
          mainThreadId: world.mainThreadId,
          savedSwitches: world.switchedToThread.filter((id) => id.startsWith("thread-")),
          state: switchState(),
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["savedSwitches"] == ["thread-a", "thread-b", "thread-a"], "all three really started"
    assert out["mainThreadId"] != "thread-a", (
        "the second arrival for the same id must be corrected too, or the visible landing is "
        "left pointed at the saved chat"
    )
    assert out["state"]["activeNonce"] == "n1"
    assert out["unhandled"] == 0


def test_a_saved_switch_failing_late_does_not_detach_the_saved_chat_that_replaced_it():
    """Only new-chat switches advanced the attempt, so two saved switches shared one token.

    A is still loading when the user opens B. B settles and owns the view. A then rejects,
    finds the attempt unchanged, passes the staleness guard and clears the active thread id
    -- detaching the visible B chat from its thread-scoped settings and context state.
    """
    out = _run(
        "renderProvider, renderSettled, snapshot, world",
        """
        await renderSettled({ newThreadNonce: "n1" });

        let releaseA: any;
        world.switchToThreadGate = new Promise((resolve) => { releaseA = resolve; });
        world.switchToThreadRejects = true;
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // B opens while A is still out, and succeeds.
        let releaseB: any;
        world.switchToThreadGate = new Promise((resolve) => { releaseB = resolve; });
        world.switchToThreadRejects = false;
        renderProvider({ initialThreadId: "thread-b" });
        await tick();
        releaseB();
        await tick();
        await tick();
        renderProvider({ initialThreadId: "thread-b" });
        await tick();
        const beforeFailure = snapshot().activeThreadId;

        // Only now does A fail.
        releaseA();
        await tick();
        await tick();

        console.log(JSON.stringify({
          beforeFailure,
          activeThreadId: snapshot().activeThreadId,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["beforeFailure"] == "thread-b", "B really owned the active id before A failed"
    assert (
        out["activeThreadId"] == "thread-b"
    ), "a superseded saved switch failing must not detach the saved chat now on screen"
    assert out["unhandled"] == 0


def test_a_claim_whose_switch_settled_off_view_does_not_outlive_it():
    """A claim has to be retired by its own switch settling, not by a view happening to
    notice that thread.

    A is delayed, B becomes visible, A settles anyway: the B view reasserts B and spends a
    B claim, and A's claim stays armed with nothing left to correct. Opening A normally
    later spends the stale claim instead of the new one, so returning to a project composer
    reads the legitimate A as a stale arrival and starts a SECOND switchToNewThread
    alongside the landing's own -- which can switch away from a thread the user has already
    sent a message on.
    """
    out = _run(
        "renderProvider, renderSettled, switchState, world",
        """
        await renderSettled({ newThreadNonce: "n1" });

        // A is opened and held.
        let releaseA: any;
        world.switchToThreadGate = new Promise((resolve) => { releaseA = resolve; });
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // B is opened and lands, so B is what the user is looking at.
        world.switchToThreadGate = null;
        await renderSettled({ initialThreadId: "thread-b" });

        // A settles anyway, off view. The B view reasserts B.
        releaseA();
        await tick();
        await tick();
        await renderSettled({ initialThreadId: "thread-b" });
        const armedAfterB = switchState().pendingSavedThreadIds.length;

        // Later the user opens A normally, and it lands while on screen.
        await renderSettled({ initialThreadId: "thread-a" });

        // Then goes to the project composer on a fresh nonce.
        const before = world.switchedToNewThread;
        await renderSettled({ newThreadNonce: "n2" });
        await tick();
        await tick();

        console.log(JSON.stringify({
          armedAfterB,
          claims: switchState().pendingSavedThreadIds,
          newThreadSwitches: world.switchedToNewThread - before,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert (
        out["armedAfterB"] == 0
    ), "a switch that settled while another saved chat was visible leaves nothing armed"
    assert out["newThreadSwitches"] == 1, (
        "the landing's own switch and no correction: a legitimate saved chat that the user "
        "opened and left is not a stale arrival"
    )
    assert out["claims"] == []
    assert out["unhandled"] == 0


0


def test_a_stale_arrival_reattaches_the_chat_the_user_started():
    """Correcting a stale arrival with switchToNewThread() is only right while the nonce's
    thread is still blank.

    Once the user has sent a message that thread is materialized and assistant-ui's
    newThreadId has been cleared, so asking for "a new thread" mints a SECOND blank one:
    the composer walks off the conversation they just started, it looks lost, and the next
    message lands in the blank thread. The correction has to reattach the thread this view
    actually owns.
    """
    out = _run(
        "renderProvider, renderSettled, nonceThreadId, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        const ownThread = world.mainThreadId;

        // Open a saved chat and hold it.
        let release: any;
        world.switchToThreadGate = new Promise((resolve) => { release = resolve; });
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // Back to the landing, which reattaches its own thread...
        world.switchToThreadGate = null;
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        // ...and the user sends a message, materializing it.
        world.mainThreadId = "remote-1";
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        await tick();

        const switchesBefore = world.switchedToNewThread;

        // Only now does the saved switch land.
        release();
        await tick();
        await tick();
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        await tick();

        console.log(JSON.stringify({
          ownThread,
          mainThreadId: world.mainThreadId,
          newThreadSwitches: world.switchedToNewThread - switchesBefore,
          savedSwitches: world.switchedToThread.filter((id) => id.startsWith("thread-")),
          reattachedTo: world.switchedToThread[world.switchedToThread.length - 1],
          owned: nonceThreadId(),
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["mainThreadId"] == "remote-1", (
        "the correction must put the user back on the conversation they started, not on a "
        "fresh blank thread"
    )
    assert out["newThreadSwitches"] == 0, "and it must not mint another thread to do it"
    assert out["reattachedTo"] == "remote-1", "it reattaches by id"
    assert out["savedSwitches"] == ["thread-a"], "only the one saved switch was started"
    assert out["owned"] == "remote-1", (
        "and the record follows the thread through materialization, which is why the "
        "reattach can name the persisted id rather than the local one it opened"
    )
    assert out["unhandled"] == 0


def test_a_superseded_failure_still_releases_the_reload_shell():
    """#9251 holds a retained shell over a reload until the initial switch reports in, and
    releases it from ThreadAutoSwitch's rejection arm. This PR added a staleness guard to
    that arm, so the signal has to sit AHEAD of it: a switch that lost the race is still a
    switch that ended, and returning early would leave the shell showing its snapshot for
    ever.

    Cross-PR, and the reason it is asserted here: nothing in #9251's own tests exercises a
    superseded switch, because before this PR there was no guard to be superseded by.
    """
    out = _run(
        "renderProvider, renderSettled, snapshot, world",
        """
        await renderSettled({ newThreadNonce: "n1" });

        // A saved switch that will fail, held open.
        let release: any;
        world.switchToThreadGate = new Promise((resolve) => { release = resolve; });
        world.switchToThreadRejects = true;
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // The route moves on, so a newer switch owns the runtime and supersedes it.
        world.switchToThreadGate = null;
        world.switchToThreadRejects = false;
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const signalsBefore = world.switchFailedSignals;

        release();
        await tick();
        await tick();

        console.log(JSON.stringify({
          signalsBefore,
          signals: world.switchFailedSignals,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["signalsBefore"] == 0, "nothing had failed yet"
    assert out["signals"] == 1, (
        "a superseded failure must still release the reload shell, or a reload that races "
        "a view change hangs on the retained snapshot"
    )
    assert out["unhandled"] == 0


def test_returning_to_a_nonce_reopens_the_chat_started_under_it():
    """A ?new= URL does not change when its chat materializes, so Back from a saved chat
    lands on the same nonce -- and a saved-chat detour released it, so the switch effect
    runs again and used to mint a fresh blank thread. The conversation the user started is
    then hidden and their next message opens yet another chat.

    Only for a nonce whose thread was actually sent to. A blank placeholder is still
    replaced, which is the behaviour the detour test above pins.
    """
    out = _run(
        "renderProvider, renderSettled, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        const opened = world.mainThreadId;

        // The user sends a message: the thread materializes and gets a row.
        world.remoteIds[opened] = "remote-1";

        // Off to a saved chat, which releases the nonce...
        await renderSettled({ initialThreadId: "thread-a" });
        // ...and Back to the very same nonce.
        const before = world.switchedToNewThread;
        await renderSettled({ newThreadNonce: "n1" });

        console.log(JSON.stringify({
          opened,
          mainThreadId: world.mainThreadId,
          mintedOnReturn: world.switchedToNewThread - before,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["mainThreadId"] == out["opened"], (
        "coming back to the nonce must reopen the conversation started under it, not a "
        "blank thread the user has to find their way out of"
    )
    assert out["mintedOnReturn"] == 0, "and it must not mint another thread to do it"
    assert out["unhandled"] == 0


def test_a_reopen_that_lands_after_the_nonce_rotated_does_not_take_the_view():
    """The returning-nonce reopen has to be as cancellable as a saved-thread switch.

    A project landing remounts when the user comes back from a saved chat, and its mount
    effect rotates the nonce -- but the first render still carries the OLD one, so the
    reopen starts before the rotation. Two switches are then in flight, and if the reopen
    resolves last the overview's composer is left on the old conversation and the next
    prompt appends there.
    """
    out = _run(
        "renderProvider, renderSettled, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        const started = world.mainThreadId;
        world.remoteIds[started] = "remote-1";

        // Detour to a saved chat, releasing the nonce.
        await renderSettled({ initialThreadId: "thread-a" });

        // Back on the stale nonce: the reopen starts, held open.
        let release: any;
        world.switchToThreadGate = new Promise((resolve) => { release = resolve; });
        renderProvider({ newThreadNonce: "n1" });
        await tick();

        // The landing's mount effect rotates the nonce, starting a fresh switch.
        world.switchToThreadGate = null;
        renderProvider({ newThreadNonce: "n2" });
        await tick();
        await tick();

        // Only now does the reopen land.
        release();
        await tick();
        await tick();
        renderProvider({ newThreadNonce: "n2" });
        await tick();
        await tick();

        console.log(JSON.stringify({
          started,
          mainThreadId: world.mainThreadId,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["mainThreadId"] != out["started"], (
        "a reopen that lost the race to a rotated nonce must not leave the overview on the "
        "old conversation"
    )
    assert out["unhandled"] == 0


def test_a_nonce_does_not_adopt_the_chat_the_user_came_from():
    """Ownership is only recorded from a thread the nonce's OWN switch opened.

    Entering New Chat from a saved chat leaves that saved chat as ``mainThreadId`` until
    switchToNewThread() resolves, and its claim was already retired while it was on screen,
    so it is unclaimed too. Treating "unclaimed and current" as ownership recorded the chat
    the user had just LEFT: a detour before the fresh switch settled kept that record, and
    coming back to the same ?new= URL reopened it, so the next prompt appended to the wrong
    conversation.
    """
    out = _run(
        "renderProvider, renderSettled, nonceThreadId, nonceOwnershipIsSettled, world",
        """
        // A saved chat that has been sent to, so it has a row worth reopening.
        world.remoteIds["thread-a"] = "remote-a";
        await renderSettled({ initialThreadId: "thread-a" });

        // New Chat, with its switch held open: the view is still on thread-a.
        let releaseNew: any;
        world.newThreadGate = new Promise((resolve) => { releaseNew = resolve; });
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const ownedWhilePending = nonceThreadId();
        const settledWhilePending = nonceOwnershipIsSettled();

        // The user leaves for another saved chat before the fresh switch lands.
        world.newThreadGate = null;
        releaseNew();
        await renderSettled({ initialThreadId: "thread-b" });

        // Back to the same ?new= URL.
        await renderSettled({ newThreadNonce: "n1" });

        console.log(JSON.stringify({
          ownedWhilePending,
          settledWhilePending,
          mainThreadId: world.mainThreadId,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert (
        out["settledWhilePending"] is False
    ), "the nonce's own switch has not landed yet, so nothing it sees is its own"
    assert (
        out["ownedWhilePending"] is None
    ), "the outgoing saved chat must not be recorded as the nonce's own thread"
    assert out["mainThreadId"] != "thread-a", (
        "returning to the nonce must not reopen the chat the user left, or their next "
        "message appends to it"
    )
    assert out["unhandled"] == 0


def test_a_nonce_still_owns_the_thread_its_own_switch_opened():
    """The other half: once the nonce's switch HAS landed, the thread it opened is its own,
    including after materialization renames it. Without this the reopen never arms and
    ``test_returning_to_a_nonce_reopens_the_chat_started_under_it`` is the regression."""
    out = _run(
        "renderSettled, nonceThreadId, nonceOwnershipIsSettled, world",
        """
        world.remoteIds["thread-a"] = "remote-a";
        await renderSettled({ initialThreadId: "thread-a" });

        // New Chat, allowed to land this time.
        await renderSettled({ newThreadNonce: "n1" });
        const opened = world.mainThreadId;
        const ownedAfterLanding = nonceThreadId();
        const settledAfterLanding = nonceOwnershipIsSettled();

        // The user sends: the thread materializes and gets a row.
        world.remoteIds[opened] = "remote-1";
        await renderSettled({ newThreadNonce: "n1" });

        // Detour and back: the reopen must find it.
        await renderSettled({ initialThreadId: "thread-b" });
        const mintedBefore = world.switchedToNewThread;
        await renderSettled({ newThreadNonce: "n1" });

        console.log(JSON.stringify({
          opened,
          ownedAfterLanding,
          settledAfterLanding,
          mainThreadId: world.mainThreadId,
          mintedOnReturn: world.switchedToNewThread - mintedBefore,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["settledAfterLanding"] is True
    assert (
        out["ownedAfterLanding"] == out["opened"]
    ), "the thread this nonce's own switch opened is the one it owns"
    assert out["mainThreadId"] == out["opened"], "and coming back must reopen it"
    assert out["mintedOnReturn"] == 0, "without minting another blank thread to do it"
    assert out["unhandled"] == 0


def test_a_superseded_landing_does_not_hand_ownership_to_a_newer_attempt():
    """A switch that lands after a newer one started must not mark the newer attempt as
    settled: the newer switch is still in flight, and its view is still on whatever the
    older one left behind."""
    out = _run(
        "renderProvider, renderSettled, nonceOwnershipIsSettled, world",
        """
        world.remoteIds["thread-a"] = "remote-a";
        await renderSettled({ initialThreadId: "thread-a" });

        // First New Chat, held on its own gate.
        let releaseFirst: any;
        world.newThreadGate = new Promise((resolve) => { releaseFirst = resolve; });
        renderProvider({ newThreadNonce: "n1" });
        await tick();

        // A second nonce supersedes it, held on a gate of its own so it stays in flight.
        world.newThreadGate = new Promise(() => {});
        renderProvider({ newThreadNonce: "n2" });
        await tick();

        // Only now does the FIRST switch land.
        releaseFirst();
        await tick();
        await tick();

        console.log(JSON.stringify({
          settled: nonceOwnershipIsSettled(),
          unhandled: unhandled.length,
        }));
        """,
    )
    assert (
        out["settled"] is False
    ), "an older switch landing says nothing about the attempt that replaced it"
    assert out["unhandled"] == 0


def test_a_remembered_thread_the_store_has_dropped_does_not_take_the_app_down():
    """``nonceThread`` is a REMEMBERED id, in a ref that outlives every view switch now, and
    the reopen looks it up with ``getItemById``. That call does not return undefined for an
    unknown id -- assistant-ui throws "Entry not available in the store" out of
    ``ShallowMemoizeSubject``'s constructor -- so the optional chain around it catches
    nothing, and an effect that throws with no error boundary above it blanks the app.

    Unsloth deletes chats through storage and tombstones rather than ``runtime.threads.delete()``,
    so nothing evicts an entry today. The reopen must not be the thing that depends on that.
    """
    out = _run(
        "renderSettled, world",
        """
        // Open a nonce chat and send to it, so it is remembered as the nonce's own.
        await renderSettled({ newThreadNonce: "n1" });
        const opened = world.mainThreadId;
        world.remoteIds[opened] = "remote-1";
        await renderSettled({ newThreadNonce: "n1" });

        // Away and back, but the runtime has since dropped the entry.
        await renderSettled({ initialThreadId: "thread-a" });
        world.missingThreadIds[opened] = true;

        let threw: any = null;
        try {
          await renderSettled({ newThreadNonce: "n1" });
        } catch (error: any) {
          threw = String(error?.message ?? error);
        }

        console.log(JSON.stringify({
          threw,
          mainThreadId: world.mainThreadId,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert out["threw"] is None, f"the reopen threw out of the effect: {out['threw']}"
    assert (
        out["mainThreadId"] != "thread-a"
    ), "and it still has to leave the saved chat it came from"
    assert out["unhandled"] == 0


def test_back_from_a_nonce_chats_own_row_keeps_that_chat():
    """The reopen's claim is this view's OWN, and must not read as a stale arrival.

    A materialized ?new= chat can be opened through its own sidebar row, and the ?new= URL
    never changed, so Back returns to the nonce with that chat already current. The switch
    effect still pushes a reopen claim for it, and ``switchToThread()`` early-returns when
    its target is already the main thread, so the claim is still outstanding when the
    correction effect runs in the same commit. Treating it as somebody else's late arrival
    fell through to ``switchToNewThread()`` and replaced the conversation the user had just
    come back to with a blank chat.
    """
    out = _run(
        "renderSettled, world",
        """
        await renderSettled({ newThreadNonce: "n1" });
        const started = world.mainThreadId;
        world.remoteIds[started] = "remote-1";
        await renderSettled({ newThreadNonce: "n1" });

        // Opened through its own row, which is how a materialized new chat is reachable.
        await renderSettled({ initialThreadId: started });

        const mintedBefore = world.switchedToNewThread;
        await renderSettled({ newThreadNonce: "n1" });

        console.log(JSON.stringify({
          started,
          mainThreadId: world.mainThreadId,
          mintedOnReturn: world.switchedToNewThread - mintedBefore,
          unhandled: unhandled.length,
        }));
        """,
    )
    assert (
        out["mainThreadId"] == out["started"]
    ), "Back must leave the user in the conversation they came back to, not a blank chat"
    assert (
        out["mintedOnReturn"] == 0
    ), "the remembered chat was already current, so nothing needed opening at all"
    assert out["unhandled"] == 0

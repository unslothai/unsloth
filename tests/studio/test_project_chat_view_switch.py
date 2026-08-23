# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One ChatRuntimeProvider spans the project and single views, so switching between them
is now a prop change rather than a remount (#8908).

That is what keeps a project chat's run attached when the user opens an old chat and comes
back, and it is also what breaks the two guards the old shape got for free. A remounted
provider had a fresh ref and a fresh composer every time; a shared one carries both across
every view switch, so ``ThreadNewChatSwitch`` has to decide for itself whether a nonce it is
seeing is new (``newThreadSwitchStateRef.activeNonce``), whether the composer it is about to
reuse still holds someone else's staged attachment (``hasSwitched``), and whether it is even
on screen (``paused``). ``ThreadAutoSwitch`` takes ``paused`` for the same reason: the stop it
requests names every temporary queue on the page rather than its own provider's.

The five effects that make those decisions -- ``ThreadAutoSwitch``'s two, ``ThreadNewChatSwitch``'s
two and ``ChatRuntimeProvider``'s implicit-new-chat marker -- are sliced VERBATIM out of
``runtime-provider.tsx`` and replayed through a React-effect emulator: per-effect dependency
arrays, re-run only when a dependency changed, memo cleared when the component unmounts.
``requestTemporaryPromptQueueStop`` is sliced verbatim too, out of ``prompt-queue-boundary.ts``,
because whose queue it stops is one of the questions here and a counter could not answer it.

Stubbed, and named so a reader knows what is not being measured: the JSX that wires the props
onto the two children (restated in ``renderProvider`` and pinned by
``test_the_provider_wires_the_pause_and_the_shared_ref``), assistant-ui's thread runtime, and
``refreshContextUsage`` (a counter -- the recount itself is pinned by
``test_new_chat_context_recount.py``; here it only has to show whether the pause gate lets it
fire). ``ActiveThreadSync`` is not modelled: it is off whenever either child is rendered.
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
    "ggufContextLength",
    "initialThreadId",
    "isLoading",
    "mainThreadId",
    "modelLoading",
    "newThreadNonce",
    "newThreadSwitchStateRef",
    "nonce",
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


def _prompt_queue_boundary_body() -> str:
    """Everything in prompt-queue-boundary.ts after its import block, verbatim."""
    text = read(QUEUE)
    marker = 'from "./prompt-queue-model-boundary";'
    return text[text.index(marker) + len(marker) :]


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
  newThreadRejects: false,
  switchToThreadRejects: false,
  composerThrows: false,
  clearAttachmentsRejects: false,
  // switchToNewThread() calls still awaiting the gate. Must return to 0.
  pendingNewThreadSwitches: 0,
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
  ggufContextLength: null,
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
      if (world.switchToThreadRejects) {
        return Promise.reject(new Error("switchToThread failed"));
      }
      world.mainThreadId = threadId;
      return Promise.resolve();
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

const newThreadSwitchStateRef: any = { current: __REF_INITIAL_VALUE__ };

export function switchState(): any {
  return { ...newThreadSwitchStateRef.current };
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
  const ggufContextLength = state.ggufContextLength;
  const modelLoading = state.modelLoading;
  const runActive = Object.values(state.runningByThreadId ?? {}).some(Boolean);
  const scope: any = {
    aui,
    isLoading,
    newThreadSwitchStateRef,
    nonce,
    paused,
    checkpoint,
    ggufContextLength,
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
        HARNESS_RENDER.replace("__REF_INITIAL_VALUE__", _provider_ref_initial_value())
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
    return run_harness(TEMP, _harness_source(), script)


# A resident GGUF, so the second effect has something it could price. Only the tests about
# the pause gate need it; everywhere else the bar has nothing to count and stands down.
LOADED_MODEL = """
    seed({ params: { checkpoint: "unsloth/gguf-model" }, ggufContextLength: 8192 });
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
    assert jsx.count("newThreadSwitchStateRef={newThreadSwitchStateRef}") == 2, (
        "both children must share ONE ref, or leaving a new chat for a saved one cannot "
        "tell the next new chat that the composer is no longer fresh"
    )
    assert jsx.count("paused={backgrounded}") == 2, (
        "BOTH children must be paused while backgrounded. ThreadAutoSwitch's first effect "
        "reaches requestTemporaryPromptQueueStop, which names every temporary queue on the "
        "page rather than this provider's, so an unpaused one stops a queue the view on "
        "screen owns"
    )
    assert "syncActiveThreadId={syncActiveThreadId && !backgrounded}" in jsx, (
        "a backgrounded provider must not write the active thread the visible view owns"
    )


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
    assert out["switchState"] == {"activeNonce": "n1", "hasSwitched": True, "attempt": 1}
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
    assert out["midSwitch"] == {"switched": 2, "cleared": 1, "pending": 1}, (
        "with a live composer the clear is immediate, not deferred behind the switch"
    )
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
    assert out["switchState"] == {"activeNonce": "n1", "hasSwitched": True, "attempt": 1}
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
    assert out["activeThreadId"] == "thread-on-screen", (
        "a paused switch must not blank the active thread the visible view is using"
    )


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
    assert out["paused"]["refreshes"] == 0, (
        "a paused view must not price a prompt onto the bar the visible view owns"
    )
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
    assert out["implicit"]["state"] == {"activeNonce": None, "hasSwitched": True, "attempt": 0}
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
    assert out["state"] == {"activeNonce": nonce, "hasSwitched": True, "attempt": attempts}
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
    assert out["opened"]["state"] == {"activeNonce": None, "hasSwitched": True, "attempt": 1}
    assert out["opened"]["switchedTo"] == ["thread-a"]
    assert out["switched"] == 2, "back to the same nonce must restore the new thread"
    assert out["state"] == {"activeNonce": "n1", "hasSwitched": True, "attempt": 2}


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
    assert out["midSwitch"]["cleared"] == 0, (
        "after a saved thread the clear is the deferred one: activeNonce was reset to null"
    )
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
    assert out["opened"]["state"] == {"activeNonce": None, "hasSwitched": True, "attempt": 1}, (
        "the nonce reset must not sit behind the mainThreadId guard"
    )
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
    assert out["beforeRelease"] == {"switched": 2, "cleared": 1}, (
        "n2 takes the immediate path and clears before its own switch"
    )
    assert out["cleared"] == 1, (
        "the stale deferred clear must not fire: n1's composer is two views behind"
    )
    assert out["switched"] == 2
    assert out["state"] == {"activeNonce": "n2", "hasSwitched": True, "attempt": 2}
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
    assert out["cleared"] == 2, (
        "n2 and n3 clear immediately; n1's deferred clear is dropped as stale"
    )
    assert out["beforeRelease"]["cleared"] == 2, "neither immediate clear waited for a switch"
    assert out["state"] == {"activeNonce": "n3", "hasSwitched": True, "attempt": 3}
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
    assert state["state"] == {"activeNonce": None, "hasSwitched": True, "attempt": 2}, (
        "a nonce whose switch failed must not stay recorded as the live one"
    )
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
    assert out["state"] == {"activeNonce": "n2", "hasSwitched": True, "attempt": 3}
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
    }, "the nonce is released on the deferred path too"
    assert out["switched"] == 2, "the same nonce is retried once the effect re-runs"
    assert out["cleared"] == 1, "and the retry that does open a thread clears as it should"
    assert out["state"] == {"activeNonce": "n1", "hasSwitched": True, "attempt": 2}
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
    assert out["beforeFailure"]["state"] == {"activeNonce": "n2", "hasSwitched": True, "attempt": 2}
    assert out["beforeFailure"]["switched"] == 2
    assert out["state"] == {"activeNonce": "n2", "hasSwitched": True, "attempt": 2}, (
        "n1's late failure must not release the nonce n2 opened a thread for"
    )
    assert out["switched"] == 2, "and must not trigger a switch of its own"
    assert out["pending"] == 0
    assert out["unhandled"] == 0


def test_a_late_failure_cannot_release_a_nonce_a_newer_attempt_owns():
    """Two switches for the SAME nonce can be in flight at once, so the nonce cannot
    identify an attempt. The route: New Chat starts a switch, the user opens a saved chat
    while it is still going (which releases the nonce without ending the switch), then
    comes back to the same New Chat, which starts a second switch that succeeds. When the
    first one finally fails, a rejection arm keyed on the nonce alone would read its own
    nonce back and release it -- and the next commit would then switch away from the thread
    the second attempt opened and the user may already be typing in.

    The complement is asserted in the same test, on the same shared state, because it is
    the pair that pins the guard: a lone failure with nothing overlapping must still
    release, or a New Chat that failed once could never be served again."""
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
        renderProvider({ initialThreadId: "thread-a" });
        await tick();

        // Back to the same New Chat: attempt 2 for n1, and this one succeeds.
        renderProvider({ newThreadNonce: "n1" });
        await tick();
        const beforeFailure = {
          state: switchState(),
          switched: world.switchedToNewThread,
          mainThreadId: world.mainThreadId,
        };

        // Only now does attempt 1 fail.
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
    assert out["beforeFailure"] == {
        "state": {"activeNonce": "n1", "hasSwitched": True, "attempt": 2},
        "switched": 2,
        "mainThreadId": "local-2",
    }, "two attempts for one nonce, the second of which opened the thread on screen"
    assert out["afterFailure"]["state"] == {
        "activeNonce": "n1",
        "hasSwitched": True,
        "attempt": 2,
    }, "attempt 1's failure must not release a nonce attempt 2 owns"
    assert out["afterFailure"]["mainThreadId"] == "local-2", (
        "and must not disturb the thread attempt 2 opened"
    )
    assert out["afterFailure"]["switched"] == 2
    assert out["afterNextCommit"] == {
        "state": {"activeNonce": "n1", "hasSwitched": True, "attempt": 2},
        "switched": 2,
        "mainThreadId": "local-2",
    }, "a released nonce would have made the next commit switch away from local-2"
    assert out["loneFailure"] == {
        "state": {"activeNonce": None, "hasSwitched": True, "attempt": 3},
        "switched": 3,
    }, "with nothing overlapping, a failed attempt must still release its nonce"
    assert out["pending"] == 0
    assert out["unhandled"] == 0


def test_every_switch_the_effect_starts_bumps_the_attempt_exactly_once():
    """``attempt`` is what tells two in-flight switches apart, so it has to advance once
    per switch actually started and never on a render that returned at a guard. Driven
    across the whole mix: a first mount, a repeat render, a rotation, a pause, a resume, a
    saved-thread detour and a return."""
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
        }));
        """,
    )
    assert out["switched"] == 4, "n1, n2, n3 and the return to n3 after the saved chat"
    assert out["state"]["attempt"] == out["switched"], (
        "attempt must count switches started, not renders: a bump on a render that "
        "returned at a guard would let a stale rejection think it is current"
    )
    assert out["state"] == {"activeNonce": "n3", "hasSwitched": True, "attempt": 4}


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
    assert out["backgroundedRun"]["switchedTo"] == [], (
        "a backgrounded provider does not attempt the switch at all"
    )
    assert out["backgroundedRun"]["activeThreadId"] == "thread-on-screen", (
        "so it cannot blank the visible view's thread, and its catch is gated as well"
    )
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
    assert out["backgroundedRun"]["stops"] == 0, (
        "a backgrounded ThreadAutoSwitch must not reach a stop that names the visible "
        "view's queue"
    )
    assert out["backgroundedRun"]["switchedTo"] == [], "nor switch a thread off screen"
    assert out["backgroundedRun"]["activeThreadId"] is None, (
        "and the second effect stays gated through syncActiveThreadId"
    )
    assert out["stops"] == 1, "the resume pays the stop the pause deferred, exactly once"
    event = out["events"][0]
    assert event["firedBy"] == "ThreadAutoSwitch"
    assert event["detail"]["temporaryOnly"] is True
    assert event["detail"]["threadIds"] == ["temp-chat"]
    assert out["switchedTo"] == ["thread-a"], "and the switch the pause deferred"
    assert out["activeThreadId"] == "thread-a"
    assert out["state"] == {"activeNonce": None, "hasSwitched": False, "attempt": 0}, (
        "the nonce reset is deferred with the rest of the effect"
    )


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
    assert out["cleared"] == passes - 1, (
        "one clear per return to the landing, and none on the first arrival"
    )
    assert out["stops"] == 2 * passes, "one stop per switch, not a growing number per switch"
    assert out["listeners"] == 0, "no listener may survive a view switch"
    assert out["pending"] == 0, "no switch callback may be left waiting"
    assert out["state"] == {"activeNonce": None, "hasSwitched": True, "attempt": 60}
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

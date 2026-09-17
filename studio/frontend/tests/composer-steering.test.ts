import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";
import {
  localPromptQueueModelBoundary,
  shouldAbortPendingQueueForModelBoundary,
  shouldAbortPendingQueueForSettingsChange,
} from "../src/features/chat/utils/prompt-queue-model-boundary.ts";
import { snapshotQueuedChatRunSettings } from "../src/features/chat/utils/queued-chat-run-settings.ts";
import { reorderPromptQueueItems } from "../src/features/chat/utils/prompt-queue-reorder.ts";
import { steeringInsertionIndex } from "../src/features/chat/utils/composer-preferences.ts";
import { chatModelLifecycleGate } from "../src/features/chat/utils/model-lifecycle-gate.ts";
import { parseExternalModelId } from "../src/features/chat/external-providers.ts";
import {
  planUserPromptQueueStop,
  userStopTargetCancelMode,
} from "../src/features/chat/utils/prompt-queue-user-stop.ts";

// Run the production queue engine with controlled stores and transport.
const source = ts.createSourceFile(
  "thread.tsx",
  readSrc("components/assistant-ui/thread.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const names = [
  "startPromptQueue",
  "steerPromptQueueTarget",
  "steerPromptQueueItem",
  "findPromptQueueRunByItemId",
  "pausePromptQueueRun",
  "resumePromptQueueRun",
  "getPromptQueueRunsForThreadIds",
  "getActivePromptQueueItem",
  "createQueuedPrompt",
  "getPromptQueueTargetIds",
  "getPromptQueueRunTargetIds",
  "promptQueueRunMatchesThreadIds",
  "findPromptQueueRunByTarget",
  "findPromptQueueRunByThreadIds",
  "isPromptQueueTargetRunning",
  "isActivePromptQueueItem",
  "dispatchQueuedPrompt",
  "isPromptQueueRunReadyToDispatch",
  "handlePromptQueueRunState",
  "isPromptQueueRunTargetRunning",
  "advancePromptQueue",
  "handlePromptQueueRunFailed",
  "retainPendingPromptQueueItemsAfterFailure",
];
const declarations = names
  .map((name) => {
    const node = source.statements.find(
      (node) => ts.isFunctionDeclaration(node) && node.name?.text === name,
    );
    assert.ok(node, `Missing production function ${name}`);
    return node.getText(source);
  })
  .join("\n");
const js = ts.transpileModule(declarations, {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.None,
  },
}).outputText;

type Target = ReturnType<typeof makeTarget>;
type Item = { id: string; prompt: string; target: Target; dispatched: boolean };
type Run = {
  id: string;
  items: Item[];
  index: number;
  generation: number;
  paused: boolean;
  waitingForTargetIdle: boolean;
};
function makeTarget(id: string, running = true) {
  return {
    getRunningThreadIds: () => (id ? [id] : []),
    getDocumentThreadId: () => id || null,
    running,
    usesLocalModel: true,
    researchStarted: () => false,
    complete: () => undefined,
    cancels: 0,
    permanentCancels: 0,
    deepResearch: 0,
    isRunning() {
      return this.running;
    },
    cancelActiveRun() {
      this.cancels += 1;
    },
    cancel() {
      this.permanentCancels += 1;
    },
    consumeDeepResearch() {
      this.deepResearch += 1;
    },
  };
}
function world() {
  const runs = new Map<string, Run>();
  const appended: string[] = [];
  const scopedCancels: string[][] = [];
  let serial = 0;
  let modelLoading = false;
  let dispatchRetries = 0;
  let indexing: () => Promise<boolean> = async () => false;
  const noop = () => undefined;
  const deps = {
    promptQueueRuns: runs,
    promptQueueRunOrder: [],
    promptQueueActiveRunIds: new Set(),
    promptQueueDispatchingRunIds: new Set(),
    compactIds: (ids: unknown[]) => [...new Set(ids.filter(Boolean))],
    createPromptQueueRunId: () => `run-${++serial}`,
    createPromptQueueItemId: () => `item-${++serial}`,
    planUserPromptQueueStop,
    userStopTargetCancelMode,
    steeringInsertionIndex,
    cancelPreStreamRunForThreadIds: (ids: string[]) => scopedCancels.push(ids),
    useChatRuntimeStore: {
      getState: () => ({ runningByThreadId: {}, modelLoading }),
    },
    syncPromptQueueUI: noop,
    ensurePromptQueueSubscription: noop,
    requestPromptQueuePump: noop,
    requestPromptQueuePumpIfReady: noop,
    clearPromptQueueRetryTimer: noop,
    schedulePromptQueueTargetStatePoll: noop,
    scheduleQueuedPromptDispatch: () => {
      dispatchRetries += 1;
    },
    PROMPT_QUEUE_DISPATCH_RETRY_MS: 500,
    PROMPT_QUEUE_INDEXING_RETRY_MS: 1,
    deletePromptQueueRun: (run: Run) => runs.delete(run.id),
    toast: { info: noop },
    discardQueuedChatRunSettingsForThread: noop,
    targetHasIndexingDocuments: () => indexing(),
    appendQueuedPrompt: (_run: Run, item: Item) => {
      appended.push(item.prompt);
      item.dispatched = true;
    },
  };
  const engine = new Function(
    ...Object.keys(deps),
    `${js}\nreturn {startPromptQueue, steerPromptQueueItem, dispatchQueuedPrompt, isPromptQueueRunReadyToDispatch, handlePromptQueueRunState, handlePromptQueueRunFailed, resumePromptQueueRun};`,
  )(...Object.values(deps)) as {
    handlePromptQueueRunFailed: (
      threadId?: string,
      localOnly?: boolean,
    ) => void;
    resumePromptQueueRun: (threadIds?: string[]) => void;
    startPromptQueue: (
      items: string[],
      target: Target,
      wait?: boolean,
      behavior?: "queue" | "steer",
    ) => void;
    steerPromptQueueItem: (id: string) => boolean;
    dispatchQueuedPrompt: (
      run: Run,
      item: Item,
      generation?: number,
    ) => Promise<void>;
    isPromptQueueRunReadyToDispatch: (run: Run) => boolean;
    handlePromptQueueRunState: (
      run: Run,
      runningByThreadId: Record<string, boolean>,
    ) => void;
  };
  return {
    ...engine,
    runs,
    appended,
    scopedCancels,
    setModelLoading: (loading: boolean) => {
      modelLoading = loading;
    },
    dispatchRetries: () => dispatchRetries,
    setIndexing: (probe: () => Promise<boolean>) => {
      indexing = probe;
    },
    run: () => [...runs.values()][0]!,
  };
}

test("steer interrupts the dispatched prompt, preserves later prompts and completed history", async () => {
  const w = world();
  const old = makeTarget("chat");
  const fresh = makeTarget("chat");
  const sibling = makeTarget("sibling");
  w.startPromptQueue(["completed", "active", "later A", "later B"], old);
  const run = w.run();
  run.items[0].dispatched = true;
  run.items[1].dispatched = true;
  run.index = 1;
  w.startPromptQueue(["sibling prompt"], sibling);
  w.startPromptQueue([" urgent "], fresh, true, "steer");
  assert.deepEqual(
    run.items.map((i) => i.prompt),
    ["completed", "urgent", "later A", "later B"],
  );
  assert.equal(run.index, 1);
  assert.equal(run.paused, false);
  assert.equal(run.generation, 1);
  assert.equal(old.cancels, 1);
  assert.equal(old.permanentCancels, 0);
  assert.equal(fresh.cancels, 1);
  assert.equal(sibling.cancels, 0);
  assert.deepEqual(w.scopedCancels, [["chat"]]);
  await w.dispatchQueuedPrompt(run, run.items[1]);
  assert.deepEqual(w.appended, [], "wait for the actual response to stop");
  fresh.running = false;
  run.waitingForTargetIdle = false;
  await w.dispatchQueuedPrompt(run, run.items[1]);
  assert.deepEqual(w.appended, ["urgent"]);
});

test("steer starts before every pending item while a queue is waiting or paused", () => {
  for (const paused of [true, false]) {
    const w = world();
    const target = makeTarget("chat");
    w.startPromptQueue(["pending A", "pending B"], target, true);
    const run = w.run();
    run.paused = paused;
    w.startPromptQueue(["urgent A", "urgent B"], target, true, "steer");
    assert.deepEqual(
      run.items.map((i) => i.prompt),
      ["urgent A", "urgent B", "pending A", "pending B"],
    );
    assert.equal(run.index, 0);
    assert.equal(run.paused, false);
    assert.equal(w.isPromptQueueRunReadyToDispatch(run), true);
  }
});

test("an in-flight indexing probe cannot dispatch the old item after a steer", async () => {
  const w = world();
  const target = makeTarget("chat", false);
  w.startPromptQueue(["old first", "old second"], target);
  const run = w.run();
  const old = run.items[0];
  let resolve!: (value: boolean) => void;
  w.setIndexing(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  const staleDispatch = w.dispatchQueuedPrompt(run, old);
  w.startPromptQueue(["urgent"], target, false, "steer");
  resolve(false);
  await staleDispatch;
  assert.deepEqual(w.appended, []);
  assert.deepEqual(
    run.items.map((i) => i.prompt),
    ["urgent", "old first", "old second"],
  );
  w.setIndexing(async () => false);
  await w.dispatchQueuedPrompt(run, run.items[0]);
  assert.deepEqual(w.appended, ["urgent"]);
});

test("normal queue follow-ups append without interrupting", () => {
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["first"], target, true);
  w.startPromptQueue(["second"], target, true, "queue");
  assert.deepEqual(
    w.run().items.map((i) => i.prompt),
    ["first", "second"],
  );
  assert.equal(target.cancels, 0);
  assert.deepEqual(w.scopedCancels, []);
});

test("row steer retains the selected item and history, cancelling its shared target once", async () => {
  const w = world();
  const target = makeTarget("chat");
  const sibling = makeTarget("other chat");
  w.startPromptQueue(["completed", "active", "later A", "later B"], target);
  const run = w.run();
  run.items[0].dispatched = true;
  run.items[1].dispatched = true;
  run.index = 1;
  const selected = run.items[3];
  w.startPromptQueue(["untouched"], sibling);

  assert.equal(w.steerPromptQueueItem(selected.id), true);
  assert.deepEqual(
    run.items.map((item) => item.prompt),
    ["completed", "later B", "later A"],
  );
  assert.equal(run.items[1], selected);
  assert.equal(run.index, 1);
  assert.equal(run.generation, 1);
  assert.equal(target.cancels, 1);
  assert.equal(target.permanentCancels, 0);
  assert.equal(sibling.cancels, 0);
  assert.deepEqual(w.scopedCancels, [["chat"]]);
  await w.dispatchQueuedPrompt(run, selected);
  assert.deepEqual(w.appended, []);
  target.running = false;
  w.handlePromptQueueRunState(run, {});
  await w.dispatchQueuedPrompt(run, selected);
  assert.deepEqual(w.appended, ["later B"]);
  assert.equal(w.steerPromptQueueItem(selected.id), false);
  assert.equal(target.cancels, 1);
});

test("row steer resumes paused and waiting queues with the chosen prompt first", () => {
  for (const paused of [true, false]) {
    for (const selectedIndex of [0, 2]) {
      const w = world();
      const target = makeTarget("chat");
      w.startPromptQueue(["first", "second", "third"], target, true);
      const run = w.run();
      run.paused = paused;
      const before = [...run.items];
      const selected = before[selectedIndex];
      assert.equal(w.steerPromptQueueItem(selected.id), true);
      assert.deepEqual(run.items, [
        selected,
        ...before.filter((i) => i !== selected),
      ]);
      assert.equal(run.index, 0);
      assert.equal(run.paused, false);
      assert.equal(target.cancels, 1);
    }
  }
});

test("row steer handles a sole pending prompt without dropping or duplicating it", () => {
  const w = world();
  const target = makeTarget("chat", false);
  w.startPromptQueue(["only prompt"], target);
  const run = w.run();
  const selected = run.items[0];
  assert.equal(w.steerPromptQueueItem(selected.id), true);
  assert.deepEqual(run.items, [selected]);
  assert.equal(run.index, 0);
  assert.equal(run.paused, false);
});

test("row steer rejects stale, dispatched, unavailable and active research prompts", () => {
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["active", "pending"], target);
  const run = w.run();
  run.items[0].dispatched = true;
  const before = [...run.items];
  assert.equal(w.steerPromptQueueItem("missing"), false);
  assert.equal(w.steerPromptQueueItem(run.items[0].id), false);
  target.researchStarted = () => true;
  assert.equal(w.steerPromptQueueItem(run.items[1].id), false);
  target.researchStarted = () => false;
  target.getRunningThreadIds = () => [];
  target.getDocumentThreadId = () => null;
  assert.equal(w.steerPromptQueueItem(run.items[1].id), false);
  assert.deepEqual(run.items, before);
  assert.equal(target.cancels, 0);
  assert.deepEqual(w.scopedCancels, []);
});

test("row steer invalidates an in-flight document probe and waits for model loading", async () => {
  const w = world();
  const target = makeTarget("chat", false);
  w.startPromptQueue(["first", "second", "third"], target);
  const run = w.run();
  const selected = run.items[2];
  let resolve!: (value: boolean) => void;
  w.setIndexing(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  const staleDispatch = w.dispatchQueuedPrompt(run, run.items[0]);
  w.setModelLoading(true);
  assert.equal(w.steerPromptQueueItem(selected.id), true);
  resolve(false);
  await staleDispatch;
  assert.deepEqual(w.appended, []);
  w.setIndexing(async () => false);
  await w.dispatchQueuedPrompt(run, selected);
  assert.deepEqual(w.appended, []);
  assert.equal(selected.dispatched, false);
  w.setModelLoading(false);
  await w.dispatchQueuedPrompt(run, selected);
  assert.equal(w.isPromptQueueRunReadyToDispatch(run), false);
  assert.deepEqual(w.appended, ["third"]);
  assert.deepEqual(
    run.items.map((item) => item.prompt),
    ["third", "first", "second"],
  );
});

test("a new steering queue waits for cancellation and rejects an unidentified target", async () => {
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["urgent"], target, true, "steer");
  assert.equal(w.run().index, 0);
  assert.equal(target.cancels, 1);
  await w.dispatchQueuedPrompt(w.run(), w.run().items[0]);
  assert.deepEqual(w.appended, []);
  assert.throws(
    () => w.startPromptQueue(["wrong chat"], makeTarget(""), true, "steer"),
    /no longer available/,
  );
  assert.equal(w.runs.size, 1);
  assert.deepEqual(w.scopedCancels, [["chat"]]);
});

// Exercise the production queue factory with controlled settings hydration.
function composerCallbackJs(name: string) {
  let factory: ts.Expression | undefined;
  function visit(node: ts.Node) {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === name &&
      node.initializer &&
      ts.isCallExpression(node.initializer)
    ) {
      factory = node.initializer.arguments[0];
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.ok(factory, `Missing production callback ${name}`);
  return ts.transpileModule(`return (${factory.getText(source)});`, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.None,
    },
  }).outputText;
}
const factoryJs = composerCallbackJs("startHydratedPromptQueue");
const targetFactoryJs = composerCallbackJs("createPromptQueueTarget");

async function targetForSelection(
  checkpoint: string,
  modelLoading: boolean,
  incoming: string | null,
  selectionSuperseded = false,
) {
  let settings!: ReturnType<typeof snapshotQueuedChatRunSettings>;
  const runtime = {
    params: { checkpoint, temperature: 0.4 },
    activeGgufVariant: "old-Q4.gguf",
    loadingModelPick: incoming ? { id: incoming, selectionSuperseded } : null,
    modelLoading,
    permissionMode: "ask",
    toolsEnabled: true,
    ragEnabled: false,
    incognito: false,
    hydratePersistedSettings: async () => undefined,
  };
  const deps = {
    aui: {
      threads: () => ({}),
      threadListItem: () => ({ getState: () => ({ id: "chat", remoteId: "chat" }) }),
    },
    referenceThreadId: "chat",
    chatHistoryClearBoundary: { capture: () => 0 },
    promptQueueTargetMountedRef: { current: true },
    indexingActiveRef: { current: false },
    useChatRuntimeStore: { getState: () => runtime },
    compactIds: (ids: unknown[]) => [...new Set(ids.filter(Boolean))],
    snapshotQueuedChatRunSettings: (...args: Parameters<typeof snapshotQueuedChatRunSettings>) => {
      settings = snapshotQueuedChatRunSettings(...args);
      return settings;
    },
    parseExternalModelId,
    hasPreStreamRunReservation: () => false,
  };
  const create = new Function(...Object.keys(deps), targetFactoryJs)(
    ...Object.values(deps),
  ) as () => Promise<Target>;
  return { target: Object.assign(makeTarget("chat", false), await create()), settings };
}

function hydratedFactory(
  w: ReturnType<typeof world>,
  target: Target,
  hydrate = () => Promise.resolve(target),
) {
  const pending = new Map();
  const deps = {
    referenceThreadId: "chat",
    promptQueueStartPendingRef: { current: pending },
    pendingQueueStartIsStale: () => false,
    useChatRuntimeStore: {
      getState: () => ({
        modelLoading: true,
        queuedSettingsEpoch: 0,
        incognito: false,
      }),
    },
    localPromptQueueModelBoundary,
    shouldAbortPendingQueueForModelBoundary,
    shouldAbortPendingQueueForSettingsChange,
    createPromptQueueTarget: hydrate,
    startPromptQueue: w.startPromptQueue,
    toast: { error: (message: string) => assert.fail(message) },
  };
  return new Function(...Object.keys(deps), factoryJs)(
    ...Object.values(deps),
  ) as (
    prompts: string[],
    wait: boolean,
    onStarted?: () => void,
    onAborted?: () => void,
    capturedAt?: {
      localModelBoundaryGeneration: number;
      queuedSettingsEpoch: number;
      temporary: boolean;
    },
    behavior?: "queue" | "steer",
  ) => boolean;
}

for (const firstBehavior of ["queue", "steer"] as const) {
  for (const latestBehavior of ["queue", "steer"] as const) {
    for (const latestWait of [true, false]) {
      test(`pending follow-up intent: ${firstBehavior} to ${latestBehavior}, wait=${latestWait}`, async () => {
        const w = world();
        const target = makeTarget("chat");
        const resolves: ((target: Target) => void)[] = [];
        const accept = hydratedFactory(
          w,
          target,
          () => new Promise((resolve) => resolves.push(resolve)),
        );
        let cleared = 0;
        const onStarted = () => {
          cleared++;
        };
        accept(
          ["same draft"],
          true,
          onStarted,
          undefined,
          undefined,
          firstBehavior,
        );
        accept(
          ["same draft"],
          latestWait,
          onStarted,
          undefined,
          undefined,
          latestBehavior,
        );
        for (const resolve of resolves) resolve(target);
        await new Promise<void>((resolve) => setImmediate(resolve));
        assert.deepEqual(
          w.run().items.map((item) => item.prompt),
          ["same draft"],
        );
        assert.equal(resolves.length, 1, "hydrate the same pending draft only once");
        assert.equal(cleared, 1);
        assert.equal(target.cancels, Number(latestBehavior === "steer"));
        if (latestBehavior === "queue") {
          assert.equal(w.run().index, latestWait ? -1 : 0);
        }
      });
    }
  }
}

test("pending follow-up intent preserves distinct drafts and later repeat submissions", async () => {
  const w = world();
  const target = makeTarget("chat");
  const accept = hydratedFactory(w, target);
  let cleared = 0;
  const onStarted = () => {
    cleared++;
  };
  accept(["first draft"], true, onStarted);
  accept(["second draft"], true, onStarted);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(accept(["first draft"], true, onStarted), true);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(
    w.run().items.map((item) => item.prompt),
    ["first draft", "second draft", "first draft"],
  );
  assert.equal(cleared, 3);
});

test("three follow-ups are accepted and reorderable during loading, then dispatch in order after the first response", async () => {
  const w = world();
  w.setModelLoading(true);
  const target = makeTarget("chat");
  const accept = hydratedFactory(w, target);
  const cleared: string[] = [];
  for (const text of ["second", "third", "fourth"]) {
    assert.equal(
      accept([text], true, () => cleared.push(text)),
      true,
    );
  }
  await Promise.resolve();
  assert.deepEqual(
    cleared,
    ["second", "third", "fourth"],
    "the composer clears each accepted prompt so another can be entered",
  );
  const run = w.run();
  assert.deepEqual(
    run.items.map((i) => i.prompt),
    cleared,
  );
  assert.ok(run.items.every((i) => !i.dispatched));
  assert.equal(run.index, -1);
  run.items = reorderPromptQueueItems(run.items, 2, 0)!;
  assert.deepEqual(
    run.items.map((i) => i.prompt),
    ["fourth", "second", "third"],
  );
  await w.dispatchQueuedPrompt(run, run.items[0]);
  assert.deepEqual(w.appended, []);
  assert.equal(w.dispatchRetries(), 1);
  w.setModelLoading(false);
  w.handlePromptQueueRunState(run, {});
  assert.equal(
    run.index,
    -1,
    "finishing the model load must not skip the first response",
  );
  target.running = false;
  w.handlePromptQueueRunState(run, {});
  assert.equal(run.index, 0);
  for (const text of ["fourth", "second", "third"]) {
    const item: Item = run.items[run.index];
    assert.equal(item.prompt, text);
    await w.dispatchQueuedPrompt(run, item);
    target.running = true;
    w.handlePromptQueueRunState(run, {});
    target.running = false;
    w.handlePromptQueueRunState(run, {});
  }
  assert.deepEqual(w.appended, ["fourth", "second", "third"]);
  assert.equal(w.runs.size, 0);
});

test("model preparation keeps a follow-up draft until the final switch boundary", async () => {
  const w = world();
  const target = makeTarget("chat");
  const lease = chatModelLifecycleGate.tryAcquire("preparing");
  assert.notEqual(lease, null);
  try {
    const accept = hydratedFactory(w, target);
    let cleared = false;
    accept(["keep this draft"], true, () => {
      cleared = true;
    });
    await Promise.resolve();
    assert.equal(
      cleared,
      false,
      "preflight must not clear a draft into a queue the switch will delete",
    );
    assert.equal(w.runs.size, 0);
    localPromptQueueModelBoundary.advance();
    localPromptQueueModelBoundary.advance();
    chatModelLifecycleGate.markLoading(lease!);
    await new Promise<void>((resolve) => setImmediate(resolve));
    accept(["keep this draft"], true, () => {
      cleared = true;
    });
    await Promise.resolve();
    assert.equal(cleared, true);
    assert.deepEqual(
      w.run().items.map((item) => item.prompt),
      ["keep this draft"],
    );
  } finally {
    chatModelLifecycleGate.release(lease!);
  }
});

test("a failed initial load pauses accepted follow-ups for recovery", async () => {
  for (const waitForInitial of [true, false]) {
    const w = world();
    const target = makeTarget("chat", false);
    w.startPromptQueue(
      ["recover first", "recover second"],
      target,
      waitForInitial,
    );
    const run = w.run();
    const ids = run.items.map((item) => item.id);
    w.handlePromptQueueRunFailed("chat");
    assert.equal(w.runs.size, 1, "accepted prompts must survive a failed load");
    assert.equal(run.paused, true);
    assert.equal(w.isPromptQueueRunReadyToDispatch(run), false);
    assert.deepEqual(
      run.items.map((item) => item.id),
      ids,
    );
    w.handlePromptQueueRunFailed("chat");
    assert.deepEqual(
      run.items.map((item) => item.id),
      ids,
    );
    w.resumePromptQueueRun(["chat"]);
    assert.equal(run.index, 0);
    await w.dispatchQueuedPrompt(run, run.items[0]);
    assert.deepEqual(w.appended, ["recover first"]);
  }
});

test("a failed manual load cannot send queued local prompts on the rollback model", async () => {
  const w = world();
  const local = makeTarget("chat", false);
  const external = makeTarget("external", false);
  external.usesLocalModel = false;
  w.startPromptQueue(["wait for chosen model"], local);
  w.startPromptQueue(["external still works"], external);
  const run = w.run();
  w.handlePromptQueueRunFailed(undefined, true);
  await w.dispatchQueuedPrompt(run, run.items[0]);
  assert.deepEqual(
    w.appended,
    [],
    "rollback must not silently change the queued model",
  );
  const sibling = [...w.runs.values()][1];
  await w.dispatchQueuedPrompt(sibling, sibling.items[0]);
  assert.deepEqual(w.appended, ["external still works"]);
  w.resumePromptQueueRun(["chat"]);
  await w.dispatchQueuedPrompt(run, run.items[0]);
  assert.deepEqual(w.appended, [
    "external still works",
    "wait for chosen model",
  ]);
});

test("a mixed queue finishes its external response before blocking the local follow-up", async () => {
  const w = world();
  const external = makeTarget("chat", false);
  external.usesLocalModel = false;
  const local = makeTarget("chat", false);
  w.startPromptQueue(["external response"], external);
  w.startPromptQueue(["local follow-up"], local);
  const run = w.run();
  await w.dispatchQueuedPrompt(run, run.items[0]);
  external.running = true;
  w.handlePromptQueueRunState(run, {});
  w.handlePromptQueueRunFailed(undefined, true);
  assert.equal(external.cancels, 0);
  assert.equal(external.permanentCancels, 0);
  external.running = false;
  w.handlePromptQueueRunState(run, {});
  assert.equal(run.index, 1);
  assert.equal(w.isPromptQueueRunReadyToDispatch(run), false);
  await w.dispatchQueuedPrompt(run, run.items[1]);
  assert.deepEqual(w.appended, ["external response"]);
  w.resumePromptQueueRun(["chat"]);
  await w.dispatchQueuedPrompt(run, run.items[1]);
  assert.deepEqual(w.appended, ["external response", "local follow-up"]);
});

test("resuming a blocked local follow-up preserves the pending external document check", async () => {
  const w = world();
  const external = makeTarget("chat", false);
  external.usesLocalModel = false;
  w.startPromptQueue(["external response"], external);
  w.startPromptQueue(["local follow-up"], makeTarget("chat", false));
  const run = w.run();
  let resolve!: (indexing: boolean) => void;
  w.setIndexing(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  const pending = w.dispatchQueuedPrompt(run, run.items[0]);
  w.handlePromptQueueRunFailed(undefined, true);
  w.resumePromptQueueRun(["chat"]);
  w.resumePromptQueueRun(["chat"]);
  resolve(false);
  await pending;
  assert.deepEqual(w.appended, ["external response"]);
  external.running = true;
  w.handlePromptQueueRunState(run, {});
  external.running = false;
  w.handlePromptQueueRunState(run, {});
  w.setIndexing(async () => false);
  await w.dispatchQueuedPrompt(run, run.items[1]);
  assert.deepEqual(w.appended, ["external response", "local follow-up"]);
});

test("load failure invalidates a local document probe without duplicate dispatch on resume", async () => {
  const w = world();
  const target = makeTarget("chat", false);
  w.startPromptQueue(["pending"], target);
  const run = w.run();
  let resolve!: (indexing: boolean) => void;
  w.setIndexing(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  const pending = w.dispatchQueuedPrompt(run, run.items[0]);
  w.handlePromptQueueRunFailed(undefined, true);
  w.resumePromptQueueRun(["chat"]);
  w.setIndexing(async () => false);
  await w.dispatchQueuedPrompt(run, run.items[0]);
  resolve(false);
  await pending;
  assert.deepEqual(w.appended, ["pending"]);
});

test("steering during loading leaves the queued prompt pending until loading finishes", async () => {
  const w = world();
  w.setModelLoading(true);
  const target = makeTarget("chat", false);
  w.startPromptQueue(["steer next"], target, false, "steer");
  const run = w.run();
  await w.dispatchQueuedPrompt(run, run.items[0]);
  assert.deepEqual(w.appended, []);
  assert.equal(run.items[0].dispatched, false);
  w.setModelLoading(false);
  await w.dispatchQueuedPrompt(run, run.items[0]);
  assert.deepEqual(w.appended, ["steer next"]);
});

for (const phase of ["legacy", "preparing", "loading", "unloading"] as const) {
  for (const local of [true, false]) {
    for (const boundaryChanges of [0, 1, 2]) {
      test(`hydration: ${phase}, local=${local}, boundaries=${boundaryChanges}`, async () => {
        const w = world();
        const target = makeTarget("chat");
        target.usesLocalModel = local;
        const lease = chatModelLifecycleGate.tryAcquire(
          phase === "legacy" ? undefined : phase,
        )!;
        let resolve!: (target: Target) => void;
        try {
          const accept = hydratedFactory(
            w,
            target,
            () =>
              new Promise((r) => {
                resolve = r;
              }),
          );
          let started = 0;
          let aborted = 0;
          accept(
            ["saved follow-up"],
            true,
            () => {
              started++;
            },
            () => {
              aborted++;
            },
          );
          accept(
            ["saved follow-up"],
            true,
            () => {
              started++;
            },
            () => {
              aborted++;
            },
          );
          for (let n = 0; n < boundaryChanges; n++)
            localPromptQueueModelBoundary.advance();
          resolve(target);
          await new Promise<void>((r) => setImmediate(r));
          const accepted =
            !local || (phase === "loading" && boundaryChanges === 0);
          assert.equal(started, Number(accepted));
          assert.equal(aborted, Number(!accepted));
          assert.equal(w.runs.size, Number(accepted));
        } finally {
          chatModelLifecycleGate.release(lease);
        }
      });
    }
  }
}

test("an external queue still dispatches while the local model loads", async () => {
  const w = world();
  w.setModelLoading(true);
  const target = makeTarget("external", false);
  target.usesLocalModel = false;
  w.startPromptQueue(["external prompt"], target);
  await w.dispatchQueuedPrompt(w.run(), w.run().items[0]);
  assert.deepEqual(w.appended, ["external prompt"]);
  assert.equal(w.dispatchRetries(), 0);
});

test("a real model boundary still rejects a pending factory and preserves its draft", async () => {
  const w = world();
  const target = makeTarget("chat");
  let resolve!: (target: Target) => void;
  const accept = hydratedFactory(
    w,
    target,
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  let cleared = false;
  let aborted = false;
  accept(
    ["old model"],
    true,
    () => {
      cleared = true;
    },
    () => {
      aborted = true;
    },
  );
  localPromptQueueModelBoundary.advance();
  resolve(target);
  await Promise.resolve();
  assert.equal(cleared, false);
  assert.equal(aborted, true);
  assert.equal(w.runs.size, 0);
});

test("loading defers model resolution while retaining queued sampling and permission preferences", () => {
  const state = {
    params: { checkpoint: "outgoing-model", temperature: 0.4 },
    activeGgufVariant: "old-Q4.gguf",
    permissionMode: "ask",
    toolsEnabled: true,
  };
  const typed = state as unknown as Parameters<
    typeof snapshotQueuedChatRunSettings
  >[0];
  const loading = snapshotQueuedChatRunSettings(typed, {
    deferModelResolution: true,
  });
  assert.equal(loading.params.checkpoint, "");
  assert.equal(loading.activeGgufVariant, null);
  assert.equal(loading.params.temperature, 0.4);
  assert.equal(loading.permissionMode, "ask");
  assert.equal(loading.toolsEnabled, true);
  assert.equal(state.params.checkpoint, "outgoing-model");
  const ready = snapshotQueuedChatRunSettings(typed);
  assert.equal(ready.params.checkpoint, "outgoing-model");
  assert.equal(ready.activeGgufVariant, "old-Q4.gguf");
});

for (const behavior of ["queue", "steer"] as const) {
  for (const failed of [false, true]) {
    test(`external-to-local switch waits for the incoming model: ${behavior}, failed=${failed}`, async () => {
      const w = world();
      w.setModelLoading(true);
      const { target, settings } = await targetForSelection(
        "external::provider::hosted-model", true, "incoming-local",
      );
      w.startPromptQueue(["wait for the selected model"], target, false, behavior);
      const run = w.run();
      await w.dispatchQueuedPrompt(run, run.items[0]);
      assert.deepEqual(w.appended, [], "never send to the outgoing hosted provider");
      assert.equal(target.usesLocalModel, true);
      assert.equal(settings.params.checkpoint, "");
      assert.equal(settings.activeGgufVariant, null);
      assert.equal(settings.params.temperature, 0.4);
      assert.equal(settings.permissionMode, "ask");
      assert.equal(settings.toolsEnabled, true);
      if (failed) {
        w.handlePromptQueueRunFailed(undefined, true);
        w.setModelLoading(false);
        await w.dispatchQueuedPrompt(run, run.items[0]);
        assert.equal(w.isPromptQueueRunReadyToDispatch(run), false);
        assert.deepEqual(w.appended, []);
        assert.equal(run.items[0].prompt, "wait for the selected model");
      } else {
        w.setModelLoading(false);
        await w.dispatchQueuedPrompt(run, run.items[0]);
        assert.deepEqual(w.appended, ["wait for the selected model"]);
      }
    });
  }
}

test("an external-to-local preparation keeps the follow-up in the composer", async () => {
  const lease = chatModelLifecycleGate.tryAcquire("preparing")!;
  try {
    const w = world();
    const { target } = await targetForSelection(
      "external::provider::hosted-model", true, "incoming-local",
    );
    let cleared = false;
    hydratedFactory(w, target)(["keep this draft"], false, () => { cleared = true; });
    await Promise.resolve();
    assert.equal(cleared, false);
    assert.equal(w.runs.size, 0);
  } finally {
    chatModelLifecycleGate.release(lease);
  }
});

test("external queues retain their provider across unrelated local loading and failure", async () => {
  const outgoing = "external::provider::hosted-model";
  const before = await targetForSelection(outgoing, false, null);
  const unrelated = await targetForSelection(outgoing, true, null);
  const stalePick = await targetForSelection(outgoing, false, "old-local-pick");
  const incoming = await targetForSelection(outgoing, true, "incoming-local");
  assert.equal(incoming.target.usesLocalModel, true);
  for (const { target, settings } of [before, unrelated, stalePick]) {
    assert.equal(target.usesLocalModel, false);
    assert.equal(settings.params.checkpoint, outgoing);
    const w = world();
    w.setModelLoading(true);
    w.startPromptQueue(["hosted follow-up"], target);
    w.handlePromptQueueRunFailed(undefined, true);
    const run = w.run();
    assert.equal(run.paused, false);
    await w.dispatchQueuedPrompt(run, run.items[0]);
    assert.deepEqual(w.appended, ["hosted follow-up"]);
  }
});

for (const checkpoint of ["outgoing-local", ""]) {
  test(`local loading still defers model resolution: checkpoint=${checkpoint || "empty"}`, async () => {
    for (const pick of ["incoming-local", null]) {
      const { target, settings } = await targetForSelection(checkpoint, true, pick);
      assert.equal(target.usesLocalModel, true);
      assert.equal(settings.params.checkpoint, "");
      assert.equal(settings.activeGgufVariant, null);
    }
    const ready = await targetForSelection(checkpoint, false, null);
    assert.equal(ready.settings.params.checkpoint, checkpoint);
    assert.equal(ready.settings.activeGgufVariant, "old-Q4.gguf");
  });
}

test("a model load starting during the document probe defers the pending append", async () => {
  const w = world();
  const target = makeTarget("chat", false);
  w.startPromptQueue(["wait for model"], target);
  let resolve!: (value: boolean) => void;
  w.setIndexing(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  const pending = w.dispatchQueuedPrompt(w.run(), w.run().items[0]);
  w.setModelLoading(true);
  resolve(false);
  await pending;
  assert.deepEqual(w.appended, []);
  assert.equal(w.run().items[0].dispatched, false);
  assert.equal(w.dispatchRetries(), 1);
});

for (const behavior of ["queue", "steer"] as const) {
  test(`an explicit hosted selection during a local load retains its provider: ${behavior}`, async () => {
    const w = world();
    w.setModelLoading(true);
    const selected = "external::new-provider::selected-model";
    const { target, settings } = await targetForSelection(
      selected, true, "incoming-local", true,
    );
    assert.equal(target.usesLocalModel, false);
    assert.equal(settings.params.checkpoint, selected);
    w.startPromptQueue(["use the selected provider"], target, false, behavior);
    w.handlePromptQueueRunFailed(undefined, true);
    const run = w.run();
    await w.dispatchQueuedPrompt(run, run.items[run.index]);
    assert.deepEqual(w.appended, ["use the selected provider"]);
  });
}

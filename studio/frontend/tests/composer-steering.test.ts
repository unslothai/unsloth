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
type Item = { prompt: string; target: Target; dispatched: boolean };
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
    targetHasIndexingDocuments: () => indexing(),
    appendQueuedPrompt: (_run: Run, item: Item) => {
      appended.push(item.prompt);
      item.dispatched = true;
    },
  };
  const engine = new Function(
    ...Object.keys(deps),
    `${js}\nreturn {startPromptQueue, dispatchQueuedPrompt, isPromptQueueRunReadyToDispatch, handlePromptQueueRunState};`,
  )(...Object.values(deps)) as {
    startPromptQueue: (
      items: string[],
      target: Target,
      wait?: boolean,
      behavior?: "queue" | "steer",
    ) => void;
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
let factory: ts.Expression | undefined;
function findFactory(node: ts.Node) {
  if (
    ts.isVariableDeclaration(node) &&
    node.name.getText(source) === "startHydratedPromptQueue" &&
    node.initializer &&
    ts.isCallExpression(node.initializer)
  ) {
    factory = node.initializer.arguments[0];
  }
  ts.forEachChild(node, findFactory);
}
findFactory(source);
assert.ok(factory);
const factoryJs = ts.transpileModule(`return (${factory.getText(source)});`, {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.None,
  },
}).outputText;
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
  ) => boolean;
}

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
  assert.match(
    source.text,
    /deferModelResolution:\s*chatStateAtQueueStart.modelLoading &&\s*parseExternalModelId\(chatStateAtQueueStart.params.checkpoint\) === null/,
  );
});

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

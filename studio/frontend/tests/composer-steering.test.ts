import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";
import { steeringInsertionIndex } from "../src/features/chat/utils/composer-preferences.ts";
import {
  planUserPromptQueueStop,
  userStopTargetCancelMode,
} from "../src/features/chat/utils/prompt-queue-user-stop.ts";

// Exercise the shipped engine, including generation guards and pause/resume. The
// surrounding browser stores, scheduler and inference transport are controlled.
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
    useChatRuntimeStore: { getState: () => ({ runningByThreadId: {} }) },
    syncPromptQueueUI: noop,
    ensurePromptQueueSubscription: noop,
    requestPromptQueuePump: noop,
    requestPromptQueuePumpIfReady: noop,
    clearPromptQueueRetryTimer: noop,
    schedulePromptQueueTargetStatePoll: noop,
    handlePromptQueueRunState: noop,
    scheduleQueuedPromptDispatch: noop,
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
    `${js}\nreturn {startPromptQueue, dispatchQueuedPrompt, isPromptQueueRunReadyToDispatch};`,
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
  };
  return {
    ...engine,
    runs,
    appended,
    scopedCancels,
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

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Queued quote snapshots are immutable, independently reserved, and resolved
// through the dispatch module's authoritative accepted/released outcome. These
// tests run the production queue engine with real quote stores.

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";
import { steeringInsertionIndex } from "../src/features/chat/utils/composer-preferences.ts";
import { planLocalPromptQueueStop } from "../src/features/chat/utils/prompt-queue-model-boundary.ts";
import { reorderPromptQueueItems } from "../src/features/chat/utils/prompt-queue-reorder.ts";
import {
  planUserPromptQueueStop,
  userStopTargetCancelMode,
} from "../src/features/chat/utils/prompt-queue-user-stop.ts";
import type { Attachment } from "@assistant-ui/react";
import {
  attachQueuedQuoteDispatch,
  canQueueTextAttachment,
  normalizeQueuedPrompt,
  queuedPromptHasContent,
  queuedPromptMessage,
  snapshotQueuedTextPrompt,
  takeQueuedQuoteDispatch,
  type QueuedPrompt,
} from "../src/features/chat/utils/queued-text-attachments.ts";
import {
  acceptQuoteDispatchForMessage,
  canReleaseQueuedQuoteDispatch,
  getQuoteDispatch,
  markQuoteDispatchHistoryFailed,
  markQuoteDispatchHistoryPending,
  markQuoteDispatchSendFailed,
  markQuoteDispatchSendSettled,
  markQuoteDispatchSendStarted,
  quoteInfoForDispatch,
  releaseQuoteDispatch,
  reserveQuoteDispatch,
  resetQuoteDispatchForRetry,
  shouldShowQuoteDispatchInPill,
  subscribeQuoteDispatchOutcome,
} from "../src/components/assistant-ui/message-quote-dispatch.ts";
import type {
  CollectedQuote,
  QuoteAnchor,
} from "../src/components/assistant-ui/message-quote-model.ts";
import {
  addQuote,
  getQuotes,
  getReservedQuoteIds,
  removeQuote,
} from "../src/components/assistant-ui/message-quote-store.ts";
import {
  QueueAppendAbandonedError,
  abandonQueueAppend as abandonQueueAppendWaiter,
  createQueueAppendToken,
  failQueueAppend,
  settleQueueAppend,
  waitForQueueAppend,
} from "../src/features/chat/utils/queue-append-lifecycle.ts";

const source = ts.createSourceFile(
  "thread.tsx",
  readSrc("components/assistant-ui/thread.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const names = [
  "startPromptQueue",
  "createQueuedPrompt",
  "beginQueuedAppendAttempt",
  "appendQueuedPrompt",
  "observeQueuedAppendCompletion",
  "handleQueuedPromptAppendFailure",
  "scheduleQueuedQuoteRetry",
  "retireAcceptedQueuedAppend",
  "pauseUnknownQueuedQuoteOutcome",
  "abandonQueuedAppend",
  "discardQueuedPromptItem",
  "releaseAndForgetPromptQueueItem",
  "forgetPromptQueueItemQuotes",
  "dropQueueItem",
  "resetPromptQueues",
  "pausePromptQueueRun",
  "stopLocalPromptQueueRun",
  "releasePromptQueueQuotes",
  "removePromptQueueItem",
  "editPromptQueueItem",
  "movePromptQueueItem",
  "steerPromptQueueItem",
  "findPromptQueueRunByItemId",
  "findPromptQueueRunByThreadIds",
  "findPromptQueueRunByTarget",
  "promptQueueRunMatchesThreadIds",
  "getPromptQueueRunsForThreadIds",
  "getPromptQueueRunTargetIds",
  "getActivePromptQueueItem",
  "isActivePromptQueueItem",
  "isCurrentQueuedAppendAttempt",
  "unsubscribeQueuedItemQuoteOutcome",
  "subscribeQueuedItemQuoteOutcome",
  "advancePromptQueue",
  "deletePromptQueueRun",
  "canEditPromptQueueItem",
  "canRemovePromptQueueItem",
  "getPromptQueueTargetIds",
];
const declarations = names
  .map((name) => {
    const node = source.statements.find(
      (statement) =>
        ts.isFunctionDeclaration(statement) && statement.name?.text === name,
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

function anchorFor(messageId: string, text: string): QuoteAnchor {
  return {
    messageId,
    turn: 1,
    text,
    markdown: text,
    start: 0,
    end: text.length,
    prefix: "",
    suffix: "",
    occurrence: 1,
  };
}

function add(thread: string, id: string, text = id): CollectedQuote {
  return addQuote(thread, anchorFor(`message-${id}`, text), { id });
}

type Target = ReturnType<typeof makeTarget>;
type Item = QueuedPrompt & {
  id: string;
  target: Target;
  dispatched: boolean;
  dispatchRetries: number;
  appendAttempt: number;
  appendToken: string | null;
  quoteOutcome: "accepted" | "released" | null;
  unsubscribeQuoteOutcome: (() => void) | null;
  blockedByModelFailure?: boolean;
};
type Run = {
  id: string;
  items: Item[];
  index: number;
  generation: number;
  paused: boolean;
  waitingForTargetIdle: boolean;
  prevStoreRunning: boolean;
  retryTimer: null;
  deepResearchConsumed: boolean;
};

function makeTarget(id: string) {
  return {
    id,
    running: false,
    appended: [] as unknown[],
    completions: [] as Promise<void>[],
    onCompletion: null as ((completion: Promise<void> | null) => void) | null,
    appendError: null as unknown,
    getRunningThreadIds: () => [id],
    getDocumentThreadId: () => id,
    getQueueProjectId: () => null,
    isRunning() {
      return this.running;
    },
    async append(prompt: unknown, onCompletion?: (completion: Promise<void> | null) => void) {
      if (this.appendError !== null) {
        throw this.appendError;
      }
      this.appended.push(prompt);
      this.onCompletion = onCompletion ?? null;
      let resolve!: () => void;
      const completion = new Promise<void>((r) => {
        resolve = r;
      });
      this.completions.push(completion);
      onCompletion?.(completion);
      return;
    },
    complete: () => undefined,
    cancel: () => undefined,
    cancelActiveRun: () => undefined,
    releaseModel: () => undefined,
    consumeDeepResearch: () => undefined,
    isIndexing: () => false,
    usesThreadDocuments: false,
    usesLocalModel: false,
    usesDeepResearch: false,
    researchStarted: () => false,
    temporary: false,
  };
}

function world() {
  const runs = new Map<string, Run>();
  const order: string[] = [];
  const active = new Set<string>();
  const dispatching = new Set<string>();
  const scheduled: string[] = [];
  const abandoned: string[] = [];
  let serial = 0;
  let syncThrows = false;
  const noop = () => undefined;
  const deps = {
    normalizeQueuedPrompt,
    queuedPromptHasContent,
    promptQueueRuns: runs,
    promptQueueRunOrder: order,
    promptQueueActiveRunIds: active,
    promptQueueDispatchingRunIds: dispatching,
    promptQueueRoundRobinCursor: 0,
    createPromptQueueRunId: () => `run-${++serial}`,
    createPromptQueueItemId: () => `item-${++serial}`,
    reserveQuoteDispatch,
    getQuoteDispatch,
    releaseQuoteDispatch,
    canReleaseQueuedQuoteDispatch,
    attachQueuedQuoteDispatch,
    takeQueuedQuoteDispatch,
    abandonQueueAppend: (token: string) => {
      abandoned.push(token);
      abandonQueueAppendWaiter(token);
    },
    markQuoteDispatchSendFailed,
    markQuoteDispatchSendSettled,
    markQuoteDispatchSendStarted,
    resetQuoteDispatchForRetry,
    subscribeQuoteDispatchOutcome,
    steeringInsertionIndex,
    reorderPromptQueueItems,
    planUserPromptQueueStop,
    userStopTargetCancelMode,
    planLocalPromptQueueStop,
    waitForPromptQueueTargetIdle: noop,
    refreshPromptQueueTargetIdleWait: noop,
    shouldPollPromptQueueTargetState: () => false,
    promptQueueActiveItemChanged: (
      before: readonly unknown[],
      after: readonly unknown[],
      index: number,
    ) => index >= 0 && before[index] !== after[index],
    compactIds: (ids: Array<string | null | undefined>) => [
      ...new Set(ids.filter((id): id is string => Boolean(id))),
    ],
    syncPromptQueueUI: () => {
      if (syncThrows) {
        syncThrows = false;
        throw new Error("ui sync failed");
      }
    },
    ensurePromptQueueSubscription: noop,
    requestPromptQueuePump: noop,
    requestPromptQueuePumpIfReady: noop,
    clearPromptQueueRetryTimer: noop,
    clearPromptQueuePumpTimer: noop,
    stopPromptQueueSubscription: noop,
    scheduleQueuedPromptDispatch: (_run: Run, item: Item) => {
      scheduled.push(item.id);
    },
    schedulePromptQueueTargetStatePoll: noop,
    PROMPT_QUEUE_MAX_DISPATCH_RETRIES: 5,
    PROMPT_QUEUE_DISPATCH_RETRY_MS: 500,
    useChatRuntimeStore: {
      getState: () => ({ runningByThreadId: {}, modelLoading: false }),
    },
    toast: { info: noop, error: noop },
    steerPromptQueueTarget: noop,
  };
  const engine = new Function(
    ...Object.keys(deps),
    `${js}
return {
  startPromptQueue,
  createQueuedPrompt,
  beginQueuedAppendAttempt,
  appendQueuedPrompt,
  observeQueuedAppendCompletion,
  handleQueuedPromptAppendFailure,
  releaseAndForgetPromptQueueItem,
  discardQueuedPromptItem,
  dropQueueItem,
  resetPromptQueues,
  pausePromptQueueRun,
  stopLocalPromptQueueRun,
  releasePromptQueueQuotes,
  removePromptQueueItem,
  editPromptQueueItem,
  movePromptQueueItem,
  steerPromptQueueItem,
  findPromptQueueRunByItemId,
  getActivePromptQueueItem,
};`,
  )(...Object.values(deps)) as {
    startPromptQueue: (
      items: Array<string | QueuedPrompt>,
      target: Target,
      wait?: boolean,
      behavior?: "queue" | "steer",
      quoteThreadId?: string | null,
    ) => { kind: "started" | "refused" | "quote-invalid" };
    createQueuedPrompt: (prompt: QueuedPrompt, target: Target) => Item;
    beginQueuedAppendAttempt: (run: Run, item: Item) => {
      kind: string;
      attempt?: number;
      quotes?: boolean;
    };
    appendQueuedPrompt: (run: Run, item: Item) => void;
    handleQueuedPromptAppendFailure: (
      run: Run,
      item: Item,
      error: unknown,
      attempt: number,
      generation: number,
    ) => void;
    releaseAndForgetPromptQueueItem: (item: Item) => string;
    discardQueuedPromptItem: (item: Item) => string;
    dropQueueItem: (run: Run, item: Item) => string;
    resetPromptQueues: () => void;
    pausePromptQueueRun: (threadIds?: string[]) => void;
    stopLocalPromptQueueRun: (run: Run) => void;
    releasePromptQueueQuotes: (itemId: string) => boolean;
    removePromptQueueItem: (itemId: string) => boolean;
    editPromptQueueItem: (itemId: string, prompt: string) => boolean;
    movePromptQueueItem: (itemId: string, targetId: string) => boolean;
    steerPromptQueueItem: (itemId: string) => boolean;
    findPromptQueueRunByItemId: (
      itemId: string,
    ) => { run: Run; itemIndex: number; item: Item } | null;
    getActivePromptQueueItem: (run: Run) => Item | undefined;
  };
  return {
    ...engine,
    runs,
    scheduled,
    abandoned,
    setSyncThrows: (value: boolean) => {
      syncThrows = value;
    },
    run: () => [...runs.values()][0]!,
  };
}

function queuedMessageFor(item: Item) {
  return queuedPromptMessage(item);
}

test("a queue item owns an immutable queue-origin snapshot", () => {
  const thread = "queued-snapshot";
  add(thread, "q1", "first");
  add(thread, "q2", "second");
  const w = world();
  const target = makeTarget("chat");
  const result = w.startPromptQueue(["queued"], target, false, "queue", thread);
  assert.equal(result.kind, "started");
  const item = w.run().items[0]!;
  assert.equal(item.quoteDispatch?.origin, "queue");
  assert.deepEqual(item.quoteDispatch?.quoteIds, ["q1", "q2"]);
  const contextId = item.quoteDispatch?.contextId;
  const transport = queuedMessageFor(item).metadata?.custom?.quote;
  assert.ok(transport);
  assert.equal(quoteInfoForDispatch(item.quoteDispatch!).text, transport.text);

  add(thread, "q3", "third");
  assert.equal(item.quoteDispatch?.contextId, contextId);
  assert.deepEqual(item.quoteDispatch?.quoteIds, ["q1", "q2"]);
  assert.deepEqual([...getReservedQuoteIds(thread)], ["q1", "q2"]);
  assert.equal(removeQuote(thread, "q1"), false);

  const second = reserveQuoteDispatch(thread, "queue");
  assert.equal(second.kind, "reserved");
  if (second.kind === "reserved") {
    assert.deepEqual(second.dispatch.quoteIds, ["q3"]);
  }
});

test("editing, moving, and steering keep the same snapshot", () => {
  const thread = "queued-edit";
  add(thread, "q1");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["first"], target, false, "queue", thread);
  w.startPromptQueue(["second"], target, false, "queue");
  const run = w.run();
  const first = run.items[0]!;
  const second = run.items[1]!;
  const contextId = first.quoteDispatch?.contextId;
  assert.equal(w.editPromptQueueItem(first.id, "edited"), true);
  assert.equal(w.movePromptQueueItem(first.id, second.id), true);
  assert.equal(w.steerPromptQueueItem(first.id), true);
  assert.equal(run.items[0]?.id, first.id);
  assert.equal(first.quoteDispatch?.contextId, contextId);
  assert.deepEqual([...getReservedQuoteIds(thread)], ["q1"]);
});

test("removing one item releases only its own snapshot", () => {
  const a = "queued-remove-a";
  const b = "queued-remove-b";
  add(a, "qa");
  add(b, "qb");
  const w = world();
  const target = makeTarget("chat");
  const first = w.startPromptQueue(["first"], target, false, "queue", a);
  assert.equal(first.kind, "started");
  const second = w.startPromptQueue(["second"], target, false, "queue", b);
  assert.equal(second.kind, "started");
  const run = w.run();
  assert.equal(run.items.length, 2);
  const firstItem = run.items[0]!;
  assert.equal(w.removePromptQueueItem(firstItem.id), true);
  assert.deepEqual([...getReservedQuoteIds(a)], []);
  assert.deepEqual([...getReservedQuoteIds(b)], ["qb"]);
  assert.deepEqual(getQuotes(a).map((quote) => quote.id), ["qa"]);
});

test("two queued items keep independent snapshots and outcomes", () => {
  const a = "queued-independent-a";
  const b = "queued-independent-b";
  add(a, "qa");
  add(b, "qb");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["first"], target, false, "queue", a);
  w.startPromptQueue(["second"], target, false, "queue", b);
  const run = w.run();
  const first = run.items[0]!;
  const second = run.items[1]!;
  assert.notEqual(first.quoteDispatch?.contextId, second.quoteDispatch?.contextId);
  assert.deepEqual([...getReservedQuoteIds(a)], ["qa"]);
  assert.deepEqual([...getReservedQuoteIds(b)], ["qb"]);
  assert.equal(w.releasePromptQueueQuotes(first.id), true);
  assert.deepEqual([...getReservedQuoteIds(a)], []);
  assert.deepEqual([...getReservedQuoteIds(b)], ["qb"]);
});

test("retryable pre-history failure preserves and resets the same attempt", async () => {
  const thread = "queued-retry";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  target.appendError = new Error("attachment prep failed");
  w.startPromptQueue(["retry me"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  const contextId = item.quoteDispatch?.contextId;
  w.appendQueuedPrompt(run, item);
  await Promise.resolve();
  assert.equal(item.dispatched, false);
  assert.equal(item.quoteDispatch?.contextId, contextId);
  assert.equal(getQuoteDispatch(contextId!)?.sendState, "not-started");
  assert.deepEqual([...getReservedQuoteIds(thread)], ["qa"]);
  assert.deepEqual(w.scheduled, [item.id]);
});

test("a stale attempt rejection cannot touch a newer attempt", () => {
  const thread = "queued-attempts";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["attempt"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  item.appendAttempt = 2;
  const contextId = item.quoteDispatch!.contextId;
  w.handleQueuedPromptAppendFailure(run, item, new Error("stale"), 1, run.generation);
  assert.equal(item.dispatched, false);
  assert.equal(getQuoteDispatch(contextId)?.sendState, "not-started");
});

test("duplicate dispatch while pending does not increment the attempt", () => {
  const thread = "queued-duplicate";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["duplicate"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  const attempt = item.appendAttempt;
  const appended = target.appended.length;
  w.appendQueuedPrompt(run, item);
  assert.equal(item.appendAttempt, attempt);
  assert.equal(target.appended.length, appended);
});

test("manual release plus quote removal is authoritatively released, not accepted", async () => {
  const thread = "queued-manual-release";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["manual"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  const contextId = item.quoteDispatch!.contextId;
  assert.equal(releaseQuoteDispatch(contextId), true);
  assert.equal(item.quoteOutcome, "released");
  assert.equal(removeQuote(thread, "qa"), true);
  w.handleQueuedPromptAppendFailure(
    run,
    item,
    new Error("late failure"),
    item.appendAttempt,
    run.generation,
  );
  assert.equal(item.quoteOutcome, null);
  assert.equal(item.dispatched, false);
  assert.equal(queuedMessageFor(item).metadata, undefined);
  assert.deepEqual(w.scheduled, [item.id]);
});

test("accepted outcome retires once and never appends twice", async () => {
  const thread = "queued-accepted";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["accepted"], target, false, "queue", thread);
  w.startPromptQueue(["next"], target, false, "queue");
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  assert.equal(
    acceptQuoteDispatchForMessage(queuedMessageFor(item)),
    true,
  );
  assert.equal(item.quoteOutcome, "accepted");
  w.handleQueuedPromptAppendFailure(
    run,
    item,
    new Error("late provider failure"),
    item.appendAttempt,
    run.generation,
  );
  assert.equal(run.items[run.index]?.prompt, "next");
  assert.equal(target.appended.length, 1);
});

test("settled send is not accepted without an outcome", () => {
  const thread = "queued-settled";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["settled"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  markQuoteDispatchSendSettled(item.quoteDispatch!.contextId);
  item.dispatched = false;
  const begin = w.beginQueuedAppendAttempt(run, item);
  assert.equal(begin.kind, "settled");
  assert.equal(getQuoteDispatch(item.quoteDispatch!.contextId)?.sendState, "settled");
  assert.equal(item.quoteOutcome, null);
});

test("a failed dispatch is not started again before reset", () => {
  const thread = "queued-failed-start";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["failed"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  markQuoteDispatchSendFailed(item.quoteDispatch!.contextId);
  const begin = w.beginQueuedAppendAttempt(run, item);
  assert.equal(begin.kind, "failed");
  assert.equal(item.appendAttempt, 0);
  assert.equal(target.appended.length, 0);
});

test("dispatched bookkeeping never coexists with a not-started dispatch", () => {
  const thread = "queued-invariant";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["invariant"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  assert.equal(item.dispatched, true);
  assert.notEqual(
    getQuoteDispatch(item.quoteDispatch!.contextId)?.sendState,
    "not-started",
  );
});

test("release gate protects pending and writing states", () => {
  const thread = "queued-gate";
  add(thread, "qa");
  const reserved = reserveQuoteDispatch(thread, "queue");
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  const { contextId } = reserved.dispatch;
  assert.equal(canReleaseQueuedQuoteDispatch(getQuoteDispatch(contextId)), true);
  markQuoteDispatchSendStarted(contextId);
  assert.equal(canReleaseQueuedQuoteDispatch(getQuoteDispatch(contextId)), false);
  markQuoteDispatchSendSettled(contextId);
  assert.equal(canReleaseQueuedQuoteDispatch(getQuoteDispatch(contextId)), false);
  markQuoteDispatchSendFailed(contextId);
  assert.equal(canReleaseQueuedQuoteDispatch(getQuoteDispatch(contextId)), false);
});

test("only a failed pre-history dispatch can reset for retry", () => {
  const thread = "queued-reset";
  add(thread, "qa");
  const reserved = reserveQuoteDispatch(thread, "queue");
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  const { contextId } = reserved.dispatch;
  markQuoteDispatchSendStarted(contextId);
  markQuoteDispatchSendFailed(contextId);
  assert.equal(resetQuoteDispatchForRetry(contextId), true);
  assert.equal(getQuoteDispatch(contextId)?.sendState, "not-started");
  markQuoteDispatchSendStarted(contextId);
  markQuoteDispatchSendSettled(contextId);
  assert.equal(resetQuoteDispatchForRetry(contextId), false);
});

test("outcome subscription records before deletion and cleans up", () => {
  const thread = "queued-outcome";
  add(thread, "qa");
  const reserved = reserveQuoteDispatch(thread, "queue");
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  let outcome: string | null = null;
  subscribeQuoteDispatchOutcome(reserved.dispatch.contextId, (next) => {
    outcome = next;
  });
  releaseQuoteDispatch(reserved.dispatch.contextId);
  assert.equal(outcome, "released");
  assert.equal(getQuoteDispatch(reserved.dispatch.contextId), null);
});

test("ordinary queued messages omit quote metadata entirely", () => {
  const message = queuedPromptMessage({ prompt: "plain" });
  assert.equal("metadata" in message, false);
});

test("queue-origin not-started snapshots stay out of the pill banner", () => {
  const thread = "queued-pill";
  add(thread, "qa");
  const reserved = reserveQuoteDispatch(thread, "queue");
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  assert.equal(shouldShowQuoteDispatchInPill(reserved.dispatch), false);
  markQuoteDispatchSendStarted(reserved.dispatch.contextId);
  const live = getQuoteDispatch(reserved.dispatch.contextId)!;
  assert.equal(shouldShowQuoteDispatchInPill(live), true);
});

test("text attachments keep their content and quote context", () => {
  const thread = "queued-text-attachment";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  const prompt: QueuedPrompt = {
    prompt: "read the note",
    attachments: [
      {
        id: "file-1",
        type: "document",
        name: "note.txt",
        contentType: "text/plain",
        content: [{ type: "text", text: "note body" }],
        status: { type: "complete" },
      },
    ],
  };
  w.startPromptQueue([prompt], target, false, "queue", thread);
  const message = queuedMessageFor(w.run().items[0]!);
  assert.equal(message.attachments?.length, 1);
  assert.equal(message.attachments?.[0]?.name, "note.txt");
  assert.ok(message.metadata?.custom?.quote);
});

test("image, audio, and file attachments stay outside the text queue", () => {
  const file = new File(["body"], "sample.bin", {
    type: "application/octet-stream",
  });
  const attachments = [
    {
      id: "image-1",
      type: "image",
      status: { type: "requires-action", reason: "composer-send" },
      file,
    },
    {
      id: "audio-1",
      type: "audio",
      status: { type: "requires-action", reason: "composer-send" },
      file,
    },
    {
      id: "file-1",
      type: "file",
      status: { type: "requires-action", reason: "composer-send" },
      file,
    },
  ] as unknown as Attachment[];
  for (const attachment of attachments) {
    assert.equal(canQueueTextAttachment(attachment), false);
    assert.equal(snapshotQueuedTextPrompt("", [attachment]), null);
  }
});

test("permanent failure releases through ordinary cleanup", () => {
  const thread = "queued-permanent";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  target.appendError = new Error("setup failed");
  w.startPromptQueue(["permanent"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  item.dispatchRetries = 5;
  w.handleQueuedPromptAppendFailure(
    run,
    item,
    new Error("still failed"),
    item.appendAttempt,
    run.generation,
  );
  assert.equal(w.runs.size, 0);
  assert.deepEqual([...getReservedQuoteIds(thread)], []);
  assert.deepEqual(getQuotes(thread).map((quote) => quote.id), ["qa"]);
});

test("history-failed retry omits quote metadata and keeps the quotes", () => {
  const thread = "queued-history-failed";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["history"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  const contextId = item.quoteDispatch!.contextId;
  markQuoteDispatchHistoryPending(contextId);
  markQuoteDispatchHistoryFailed(contextId);
  w.handleQueuedPromptAppendFailure(
    run,
    item,
    new Error("history failed"),
    item.appendAttempt,
    run.generation,
  );
  assert.equal(item.dispatched, false);
  assert.equal(item.quoteDispatch, undefined);
  assert.equal(queuedMessageFor(item).metadata, undefined);
  assert.deepEqual(getQuotes(thread).map((quote) => quote.id), ["qa"]);
  assert.deepEqual(w.scheduled, [item.id]);
});

test("creation rollback releases a reservation that never attached", () => {
  const thread = "queued-rollback";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.setSyncThrows(true);
  assert.throws(
    () => w.startPromptQueue(["rollback"], target, false, "queue", thread),
    /ui sync failed/,
  );
  assert.equal(w.runs.size, 0);
  assert.deepEqual([...getReservedQuoteIds(thread)], []);
  assert.deepEqual(getQuotes(thread).map((quote) => quote.id), ["qa"]);
});

test("a synchronous post-invocation throw is classified exactly once", async () => {
  const thread = "queued-sync-throw";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  target.append = ((prompt: unknown, onCompletion?: (completion: Promise<void> | null) => void) => {
    target.appended.push(prompt);
    onCompletion?.(Promise.reject(new Error("post-invocation")));
    throw new Error("sync after invocation");
  }) as typeof target.append;
  w.startPromptQueue(["sync throw"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(item.dispatched, false);
  assert.equal(getQuoteDispatch(item.quoteDispatch!.contextId)?.sendState, "not-started");
  assert.deepEqual(w.scheduled, [item.id]);
});

test("history settlement completes a queue append whose wrapper returns void", async () => {
  const thread = "queued-token-settled";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  const token = createQueueAppendToken();
  let completion: Promise<void> | null = null;
  target.append = ((prompt: unknown, onCompletion?: (value: Promise<void> | null) => void) => {
    target.appended.push(prompt);
    completion = waitForQueueAppend(token);
    onCompletion?.(completion);
    settleQueueAppend(token);
    return undefined;
  }) as unknown as typeof target.append;
  w.startPromptQueue(["token"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  await completion;
  await Promise.resolve();
  assert.equal(item.dispatched, true);
  assert.equal(getQuoteDispatch(item.quoteDispatch!.contextId)?.sendState, "settled");
  assert.equal(item.appendToken, null);
  assert.deepEqual(w.scheduled, []);
});

test("history failure rejects the token and reuses the existing retry path", async () => {
  const thread = "queued-token-failed";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  const token = createQueueAppendToken();
  let completion: Promise<void> | null = null;
  target.append = ((prompt: unknown, onCompletion?: (value: Promise<void> | null) => void) => {
    target.appended.push(prompt);
    completion = waitForQueueAppend(token);
    onCompletion?.(completion);
    failQueueAppend(token, new Error("history failed"));
    return undefined;
  }) as unknown as typeof target.append;
  w.startPromptQueue(["token"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  w.appendQueuedPrompt(run, item);
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(item.dispatched, false);
  assert.equal(item.appendToken, null);
  assert.equal(getQuoteDispatch(item.quoteDispatch!.contextId)?.sendState, "not-started");
  assert.deepEqual(w.scheduled, [item.id]);
});

test("cross-thread delivery settles the source quote collection", () => {
  const sourceThread = "queued-cross-thread-source";
  add(sourceThread, "qa");
  const w = world();
  const target = makeTarget("delivery-thread");
  w.startPromptQueue(["cross"], target, false, "queue", sourceThread);
  const item = w.run().items[0]!;
  assert.equal(acceptQuoteDispatchForMessage(queuedMessageFor(item)), true);
  assert.deepEqual(getQuotes(sourceThread), []);
  assert.equal(item.quoteOutcome, "accepted");
});

test("queue UI exposes a quote badge and live-gated release action", () => {
  const list = readSrc("components/assistant-ui/prompt-queue-list.tsx");
  const collection = readSrc("components/assistant-ui/message-quote-collection.tsx");
  const threadSource = readSrc("components/assistant-ui/thread.tsx");
  const store = readSrc("features/chat/stores/prompt-queue-ui-store.ts");
  assert.match(list, /quoteCount/);
  assert.match(list, /onReleaseQuotes/);
  assert.match(list, /messageSelection\.releaseQuotes/);
  assert.match(list, /quoteReservedByQueuedPrompt/);
  assert.match(threadSource, /canReleaseQueuedQuoteDispatch\(quoteDispatch\)/);
  assert.match(collection, /shouldShowQuoteDispatchInPill/);
  assert.match(collection, /reservedByQueuedPrompt/);
  assert.match(store, /canReleaseQuotes/);
});

test("bulk reset abandons an in-flight queued append", async () => {
  const thread = "queued-reset-abandon";
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["first"], target, false, "queue", thread);
  const run = w.run();
  const item = run.items[0]!;
  item.dispatched = true;
  const completion = waitForQueueAppend("reset-token");
  item.appendToken = "reset-token";
  w.resetPromptQueues();
  assert.deepEqual(w.abandoned, ["reset-token"]);
  assert.equal(w.runs.size, 0);
  await assert.rejects(completion, QueueAppendAbandonedError);
  settleQueueAppend("reset-token");
  assert.deepEqual(w.scheduled, []);
});

test("pause abandons a discarded in-flight queued append", async () => {
  const thread = "queued-pause-abandon";
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["first"], target, false, "queue", thread);
  const run = w.run();
  const first = run.items[0]!;
  const second = w.createQueuedPrompt({ prompt: "second" }, target);
  run.items.push(second);
  run.index = 0;
  first.dispatched = true;
  const completion = waitForQueueAppend("pause-token");
  first.appendToken = "pause-token";
  w.pausePromptQueueRun(["chat"]);
  assert.deepEqual(w.abandoned, ["pause-token"]);
  assert.deepEqual(run.items, [second]);
  await assert.rejects(completion, QueueAppendAbandonedError);
  // A late history settlement after abandonment has no waiter and cannot
  // re-dispatch the removed item or advance the queue.
  settleQueueAppend("pause-token");
  w.appendQueuedPrompt(run, first);
  assert.deepEqual(w.scheduled, []);
});

test("local-model stop abandons a discarded in-flight queued append", async () => {
  const thread = "queued-stop-abandon";
  const w = world();
  const target = makeTarget("chat");
  target.usesLocalModel = true;
  w.startPromptQueue(["first"], target, false, "queue", thread);
  const run = w.run();
  const first = run.items[0]!;
  const second = w.createQueuedPrompt({ prompt: "second" }, target);
  run.items.push(second);
  run.index = 0;
  first.dispatched = true;
  const completion = waitForQueueAppend("stop-token");
  first.appendToken = "stop-token";
  w.stopLocalPromptQueueRun(run);
  assert.deepEqual(w.abandoned, ["stop-token"]);
  assert.deepEqual(run.items, [second]);
  await assert.rejects(completion, QueueAppendAbandonedError);
  settleQueueAppend("stop-token");
  assert.deepEqual(w.scheduled, []);
});

test("dropQueueItem is the centralized cleanup path", () => {
  const thread = "queued-drop";
  add(thread, "qa");
  const w = world();
  const target = makeTarget("chat");
  w.startPromptQueue(["first"], target, false, "queue", thread);
  w.startPromptQueue(["second"], target, false, "queue");
  const run = w.run();
  const first = run.items[0]!;
  first.appendToken = "drop-token";
  assert.equal(w.dropQueueItem(run, first), "released");
  assert.equal(run.items.length, 1);
  assert.deepEqual(w.abandoned, ["drop-token"]);
  assert.deepEqual([...getReservedQuoteIds(thread)], []);
});

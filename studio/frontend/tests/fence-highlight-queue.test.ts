// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createFenceHighlightQueue,
  type FenceHighlightQueueOptions,
  selectionInFence,
} from "../src/components/assistant-ui/fence-highlight-queue.ts";
import { readSrc } from "./helpers/kit.ts";

const harness = (overrides: Partial<FenceHighlightQueueOptions> = {}) => {
  const callbacks = new Set<() => void>();
  const allCallbacks: (() => void)[] = [];
  let blocked = false;
  let commits = 0;
  const queue = createFenceHighlightQueue({
    schedule: (callback) => {
      callbacks.add(callback);
      allCallbacks.push(callback);
      return () => { callbacks.delete(callback); };
    },
    blocked: () => blocked,
    commit: (callback) => { commits += 1; callback(); },
    ...overrides,
  });
  return {
    queue,
    callbacks,
    allCallbacks,
    commits: () => commits,
    block: (value: boolean) => { blocked = value; },
    run: () => {
      const callback = callbacks.values().next().value;
      assert.ok(callback, "a callback is scheduled");
      callbacks.delete(callback);
      callback();
    },
  };
};

test("batches run one per task and rotate unfinished fences fairly", () => {
  const h = harness();
  const calls: string[] = [];
  let first = 0;
  h.queue.add(() => { calls.push("first"); return ++first === 2; }, () => assert.fail());
  h.queue.add(() => { calls.push("second"); return true; }, () => assert.fail());
  assert.deepEqual(calls, []);
  assert.equal(h.callbacks.size, 1);
  assert.equal(h.queue.pending(), 2);
  h.run();
  assert.deepEqual(calls, ["first"]);
  assert.equal(h.queue.pending(), 2);
  h.run();
  assert.deepEqual(calls, ["first", "second"]);
  assert.equal(h.queue.pending(), 1);
  h.run();
  assert.deepEqual(calls, ["first", "second", "first"]);
  assert.equal(h.commits(), 3);
  assert.equal(h.queue.pending(), 0);
  assert.equal(h.callbacks.size, 0);
});

test("blocked work reschedules without committing or advancing a fence", () => {
  const h = harness();
  let calls = 0;
  h.queue.add(() => { calls += 1; return true; }, () => assert.fail());
  h.block(true);
  h.run();
  h.run();
  assert.equal(calls, 0);
  assert.equal(h.commits(), 0);
  assert.equal(h.callbacks.size, 1);
  h.block(false);
  h.run();
  assert.equal(calls, 1);
  assert.equal(h.queue.pending(), 0);
});

test("cancellation removes only its entry and cancels the last scheduled callback", () => {
  const h = harness();
  let calls = 0;
  const cancelFirst = h.queue.add(() => assert.fail(), () => assert.fail());
  const cancelSecond = h.queue.add(() => { calls += 1; return false; }, () => assert.fail());
  cancelFirst();
  cancelFirst();
  assert.equal(h.queue.pending(), 1);
  h.run();
  assert.equal(calls, 1);
  const stale = h.allCallbacks.at(-1)!;
  cancelSecond();
  assert.equal(h.callbacks.size, 0);
  stale();
  assert.equal(calls, 1);
  assert.equal(h.queue.pending(), 0);
});

test("a stale canceled callback cannot run newly queued work", () => {
  const h = harness();
  const cancel = h.queue.add(() => assert.fail(), () => assert.fail());
  const stale = h.allCallbacks[0];
  cancel();
  let calls = 0;
  h.queue.add(() => { calls += 1; return true; }, () => assert.fail());
  stale();
  assert.equal(calls, 0);
  assert.equal(h.callbacks.size, 1);
  h.run();
  assert.equal(calls, 1);
});

test("flush bypasses blocking and finishes pending fences in one commit", () => {
  const h = harness();
  const finished: string[] = [];
  h.queue.add(() => assert.fail(), () => { finished.push("first"); });
  h.queue.add(() => assert.fail(), () => { finished.push("second"); });
  const stale = h.allCallbacks[0];
  h.block(true);
  h.queue.flush();
  assert.deepEqual(finished, ["first", "second"]);
  assert.equal(h.commits(), 1);
  assert.equal(h.queue.pending(), 0);
  assert.equal(h.callbacks.size, 0);
  stale();
  h.queue.flush();
  assert.equal(h.commits(), 1);
});

test("a canceled entry stays canceled when its running step returns unfinished", () => {
  const h = harness();
  let cancel = () => {};
  cancel = h.queue.add(() => { cancel(); return false; }, () => assert.fail());
  h.run();
  assert.equal(h.queue.pending(), 0);
  assert.equal(h.callbacks.size, 0);
});

test("a running step can flush without requeueing itself or canceled siblings", () => {
  const h = harness();
  const finished: string[] = [];
  h.queue.add(() => { h.queue.flush(); return false; }, () => { finished.push("first"); });
  h.queue.add(() => assert.fail(), () => { finished.push("second"); });
  h.run();
  assert.deepEqual(finished, ["first", "second"]);
  assert.equal(h.queue.pending(), 0);
  assert.equal(h.callbacks.size, 0);
});

test("flush tolerates cleanup, recursive flush and newly registered fences", () => {
  const h = harness();
  const finished: string[] = [];
  let cancelSecond = () => {};
  h.queue.add(() => assert.fail(), () => {
    finished.push("first");
    cancelSecond();
    h.queue.flush();
    h.queue.add(() => assert.fail(), () => { finished.push("third"); });
  });
  cancelSecond = h.queue.add(() => assert.fail(), () => assert.fail());
  h.queue.flush();
  assert.deepEqual(finished, ["first", "third"]);
  assert.equal(h.commits(), 1);
  assert.equal(h.queue.pending(), 0);
  assert.equal(h.callbacks.size, 0);
});

test("flush includes work registered by the commit lifecycle", () => {
  let added = false;
  const finished: string[] = [];
  const h = harness({ commit: (callback) => {
    callback();
    if (!added) {
      added = true;
      queue.add(() => assert.fail(), () => { finished.push("mounted"); });
    }
  } });
  const queue = h.queue;
  queue.add(() => assert.fail(), () => { finished.push("first"); });
  queue.flush();
  assert.deepEqual(finished, ["first", "mounted"]);
  assert.equal(queue.pending(), 0);
  assert.equal(h.callbacks.size, 0);
});

test("failed steps leave no failed entry and do not strand other fences", () => {
  const h = harness();
  const failure = new Error("step failed");
  h.queue.add(() => { throw failure; }, () => assert.fail());
  let completed = false;
  h.queue.add(() => { completed = true; return true; }, () => assert.fail());
  assert.throws(h.run, failure);
  assert.equal(h.queue.pending(), 1);
  assert.equal(h.callbacks.size, 1);
  h.run();
  assert.equal(completed, true);
  assert.equal(h.queue.pending(), 0);
});

test("failed finishes still flush the other fences and cancel stale work", () => {
  const h = harness();
  const failure = new Error("finish failed");
  let completed = false;
  h.queue.add(() => assert.fail(), () => { throw failure; });
  h.queue.add(() => assert.fail(), () => { completed = true; });
  assert.throws(h.queue.flush, failure);
  assert.equal(completed, true);
  assert.equal(h.queue.pending(), 0);
  assert.equal(h.callbacks.size, 0);
  for (const stale of h.allCallbacks) stale();
});

test("a failed commit or scheduler does not leak pending entries", () => {
  const failure = new Error("failed");
  const brokenCommit = harness({ commit: () => { throw failure; } });
  brokenCommit.queue.add(() => assert.fail(), () => assert.fail());
  assert.throws(brokenCommit.queue.flush, failure);
  assert.equal(brokenCommit.queue.pending(), 0);
  assert.equal(brokenCommit.callbacks.size, 0);
  const brokenSchedule = harness({ schedule: () => { throw failure; } });
  assert.throws(() => brokenSchedule.queue.add(() => assert.fail(), () => assert.fail()), failure);
  assert.equal(brokenSchedule.queue.pending(), 0);
});

test("only a selection that ends inside a fence holds the highlight queue", () => {
  const node = (inFence: boolean) => ({
    nodeType: 3,
    parentElement: { closest: (selector: string) => (inFence && selector.includes("code-block") ? {} : null) },
  }) as unknown as Node;
  const prose = node(false);
  const code = node(true);
  assert.equal(selectionInFence(null), false);
  assert.equal(selectionInFence({ anchorNode: prose, focusNode: prose }), false, "a selection in prose lets fences colour");
  assert.equal(selectionInFence({ anchorNode: prose, focusNode: code }), true, "a selection ending in code is held");
  assert.equal(selectionInFence({ anchorNode: code, focusNode: prose }), true);
  const element = { nodeType: 1, closest: () => ({}) } as unknown as Node;
  assert.equal(selectionInFence({ anchorNode: element, focusNode: null }), true, "an element endpoint is checked itself");
  const defer = readSrc("components/assistant-ui/code-fence-defer.tsx");
  assert.match(defer, /blocked: \(\) => inputPointerHeld\(\) \|\| highlightQuietIn\(\) > 0 \|\| selectionInFence\(liveSelection\(\)\),/);
});

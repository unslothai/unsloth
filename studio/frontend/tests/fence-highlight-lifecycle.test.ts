// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";

import { createFenceHighlightQueue } from "../src/components/assistant-ui/fence-highlight-queue.ts";
import { readSrc } from "./helpers/kit.ts";

const source = readSrc("components/assistant-ui/code-fence-defer.tsx");
const declarations = source.slice(source.indexOf("let highlightWatching ="), source.indexOf("const highlightQuietIn ="));
const lifecycle = source.slice(source.indexOf("const watchHighlighting ="), source.indexOf("const visibleHighlightLine ="));
const compiled = ts.transpileModule(declarations + lifecycle.replace("export function queueFenceWork", "function queueFenceWork"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText;

function harness() {
  const scheduled = new Set<() => void>();
  const microtasks: (() => void)[] = [];
  const resizeListeners = new Set<() => void>();
  const observers: Observer[] = [];
  const element = {};
  let blocked = false;
  let find = false;
  let clock = 0;
  let removals = 0;
  const queue = createFenceHighlightQueue({
    schedule: (callback) => {
      scheduled.add(callback);
      return () => { scheduled.delete(callback); };
    },
    blocked: () => blocked,
    commit: (callback) => { callback(); },
  });
  class Observer {
    connected = false;
    disconnects = 0;
    callback: () => void;
    constructor(callback: () => void) {
      this.callback = callback;
      observers.push(this);
    }
    observe(target: unknown, options: unknown) {
      assert.equal(target, element);
      assert.deepEqual(options, { attributes: true, attributeFilter: ["data-find-bar-open"] });
      this.connected = true;
    }
    disconnect() {
      this.connected = false;
      this.disconnects += 1;
    }
  }
  const context = {
    window: {
      addEventListener(type: string, callback: () => void) {
        assert.equal(type, "resize");
        resizeListeners.add(callback);
      },
      removeEventListener(type: string, callback: () => void) {
        assert.equal(type, "resize");
        assert.ok(resizeListeners.delete(callback));
        removals += 1;
      },
    },
    document: { documentElement: element },
    performance: { now: () => clock },
    MutationObserver: Observer,
    FIND_BAR_FLAG: "data-find-bar-open",
    findBarOpen: () => find,
    highlightQueue: queue,
    printing: false,
    queueMicrotask: (callback: () => void) => { microtasks.push(callback); },
  };
  const api = new Function(...Object.keys(context), compiled + `
    return {
      queue: queueFenceWork,
      resizedAt: () => highlightResizedAt,
      setPrinting: value => { printing = value; },
    };
  `)(...Object.values(context)) as {
    queue: (work: () => void) => () => void;
    resizedAt: () => number;
    setPrinting: (value: boolean) => void;
  };
  const flushMicrotasks = () => {
    let count = 0;
    while (microtasks.length > 0) {
      assert.ok(++count < 100);
      microtasks.shift()!();
    }
  };
  return {
    ...api,
    observers,
    scheduled,
    microtasks,
    resizeListeners,
    pending: queue.pending,
    removals: () => removals,
    block: (value: boolean) => { blocked = value; },
    resize: (at: number) => {
      clock = at;
      for (const callback of resizeListeners) callback();
    },
    setFind: (value: boolean) => {
      find = value;
      microtasks.push(() => {
        for (const observer of observers) if (observer.connected) observer.callback();
      });
    },
    flushMicrotasks,
    run: () => {
      const callback = scheduled.values().next().value;
      assert.ok(callback);
      scheduled.delete(callback);
      callback();
    },
  };
}

test("replacement work survives the last-entry teardown and remains findable", () => {
  const h = harness();
  const cancel = h.queue(() => assert.fail("canceled work must not run"));
  const observer = h.observers[0];
  cancel();
  assert.equal(h.microtasks.length, 1);
  let finished = false;
  h.queue(() => { finished = true; });
  h.resize(123);
  assert.equal(h.resizedAt(), 123);
  assert.equal(h.observers.length, 1);
  h.block(true);
  h.setFind(true);
  h.flushMicrotasks();
  assert.equal(finished, true);
  assert.equal(h.pending(), 0);
  assert.equal(h.scheduled.size, 0);
  assert.equal(observer.disconnects, 1);
  assert.equal(h.removals(), 1);
  assert.equal(h.resizeListeners.size, 0);
});

test("a commit registering its successor keeps the existing observers alive", () => {
  const h = harness();
  const finished: string[] = [];
  h.queue(() => {
    finished.push("first");
    h.queue(() => { finished.push("second"); });
    h.block(true);
  });
  h.run();
  h.flushMicrotasks();
  assert.deepEqual(finished, ["first"]);
  assert.equal(h.pending(), 1);
  assert.equal(h.observers.length, 1);
  assert.equal(h.observers[0].disconnects, 0);
  assert.equal(h.resizeListeners.size, 1);
  h.block(false);
  h.run();
  h.flushMicrotasks();
  assert.deepEqual(finished, ["first", "second"]);
  assert.equal(h.observers[0].disconnects, 1);
  assert.equal(h.resizeListeners.size, 0);
});

test("repeated cancellation coalesces teardown and a later subscription starts cleanly", () => {
  const h = harness();
  const first = h.queue(() => assert.fail());
  const second = h.queue(() => assert.fail());
  first();
  first();
  second();
  assert.equal(h.microtasks.length, 1);
  assert.equal(h.removals(), 0);
  h.flushMicrotasks();
  assert.equal(h.removals(), 1);
  assert.equal(h.observers[0].disconnects, 1);
  let finished = false;
  h.queue(() => { finished = true; });
  assert.equal(h.observers.length, 2);
  h.run();
  h.flushMicrotasks();
  assert.equal(finished, true);
  assert.equal(h.observers[1].disconnects, 1);
  assert.equal(h.removals(), 2);
  assert.equal(h.scheduled.size, 0);
});

test("closing find keeps blocked work pending and opening it flushes nested work", () => {
  const h = harness();
  const finished: string[] = [];
  h.block(true);
  h.queue(() => {
    finished.push("first");
    h.queue(() => { finished.push("nested"); });
  });
  h.setFind(false);
  h.flushMicrotasks();
  assert.deepEqual(finished, []);
  assert.equal(h.pending(), 1);
  assert.equal(h.observers[0].disconnects, 0);
  h.setFind(true);
  h.flushMicrotasks();
  assert.deepEqual(finished, ["first", "nested"]);
  assert.equal(h.pending(), 0);
  assert.equal(h.scheduled.size, 0);
  assert.equal(h.observers.length, 1);
  assert.equal(h.observers[0].disconnects, 1);
});

test("print and an already open find bypass subscriptions and run work synchronously", () => {
  for (const force of ["print", "find"]) {
    const h = harness();
    h.block(true);
    if (force === "print") h.setPrinting(true);
    else h.setFind(true);
    let finished = false;
    const cancel = h.queue(() => { finished = true; });
    assert.equal(finished, true);
    cancel();
    h.flushMicrotasks();
    assert.equal(h.pending(), 0);
    assert.equal(h.observers.length, 0);
    assert.equal(h.scheduled.size, 0);
    assert.equal(h.resizeListeners.size, 0);
  }
});

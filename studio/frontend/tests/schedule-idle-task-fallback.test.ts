// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type IdleModule = {
  scheduleIdleTask: (callback: () => void, timeout?: number) => () => void;
  scheduleQuietIdleTask: (callback: () => void, timeout?: number) => () => void;
  inputQuietIn: () => number;
};

function engine(t: TestContext, idle?: (callback: () => void) => number) {
  let now = 1_000;
  const listeners = new Map<string, () => void>();
  const scope = globalThis as Record<string, unknown>;
  const saved = { window: scope.window, document: scope.document, performance: scope.performance };
  scope.window = idle ? { requestIdleCallback: idle, cancelIdleCallback: () => {} } : {};
  scope.document = {
    addEventListener: (type: string, listener: () => void) => listeners.set(type, listener),
  };
  Object.defineProperty(globalThis, "performance", { value: { now: () => now }, configurable: true, writable: true });
  t.after(() => {
    scope.window = saved.window;
    scope.document = saved.document;
    Object.defineProperty(globalThis, "performance", { value: saved.performance, configurable: true, writable: true });
  });
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const idleModule = loadWithStubs<IdleModule>(new URL("../src/lib/schedule-idle-task.ts", import.meta.url), {});
  const advance = (ms: number) => {
    for (let step = 0; step < ms; step += 1) {
      now += 1;
      t.mock.timers.tick(1);
    }
  };
  const input = (type: string) => listeners.get(type)?.();
  return { idleModule, advance, input, listeners };
}

test("without requestIdleCallback, the shared scheduler keeps its 120 ms timer whatever the input", (t) => {
  const { idleModule, advance, input } = engine(t);
  let ran = 0;
  input("pointerdown");
  idleModule.scheduleIdleTask(() => (ran += 1), 1000);
  advance(119);
  assert.equal(ran, 0);
  advance(2);
  assert.equal(ran, 1, "settings, the hub readme and markdown previews wait no longer than on main");
});

test("without requestIdleCallback, a quiet task runs soon when nothing is happening", (t) => {
  const { idleModule, advance } = engine(t);
  let ran = 0;
  idleModule.scheduleQuietIdleTask(() => (ran += 1), 1000);
  advance(15);
  assert.equal(ran, 0);
  advance(2);
  assert.equal(ran, 1);
});

test("a quiet task waits for input to go quiet, so it never lands in the middle of a scroll", (t) => {
  const { idleModule, advance, input, listeners } = engine(t);
  for (const type of ["wheel", "scroll", "pointerdown", "keydown", "touchstart", "touchmove"]) {
    idleModule.inputQuietIn();
    assert.ok(listeners.has(type), `${type} counts as input`);
  }
  let ran = 0;
  input("wheel");
  idleModule.scheduleQuietIdleTask(() => (ran += 1), 1000);
  advance(200);
  input("scroll");
  advance(299);
  assert.equal(ran, 0, "300 ms after the last input, not after the first");
  advance(2);
  assert.equal(ran, 1);
});

test("continuous input cannot starve a quiet task past its deadline", (t) => {
  const { idleModule, advance, input } = engine(t);
  let ran = 0;
  idleModule.scheduleQuietIdleTask(() => (ran += 1), 500);
  for (let at = 0; at < 480; at += 40) {
    input("wheel");
    advance(40);
  }
  assert.equal(ran, 0);
  input("wheel");
  advance(40);
  assert.equal(ran, 1, "the deadline holds, as requestIdleCallback's timeout does");
});

test("a cancelled task never runs", (t) => {
  const { idleModule, advance, input } = engine(t);
  let ran = 0;
  idleModule.scheduleIdleTask(() => (ran += 1), 1000)();
  idleModule.scheduleQuietIdleTask(() => (ran += 1), 1000)();
  input("wheel");
  const cancel = idleModule.scheduleQuietIdleTask(() => (ran += 1), 1000);
  advance(100);
  cancel();
  advance(2000);
  assert.equal(ran, 0, "including a quiet task already waiting out input");
});

test("engines with requestIdleCallback keep it", (t) => {
  const queued: Array<() => void> = [];
  const { idleModule, advance } = engine(t, (callback) => queued.push(callback));
  let ran = 0;
  idleModule.scheduleIdleTask(() => (ran += 1), 1000);
  idleModule.scheduleQuietIdleTask(() => (ran += 1), 1000);
  advance(1000);
  assert.equal(ran, 0);
  assert.equal(queued.length, 2);
  for (const callback of queued) callback();
  assert.equal(ran, 2);
});

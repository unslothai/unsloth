// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import {
  USER_DISCRETE_EVENTS,
  USER_SCROLL_EVENTS,
  lockCollapseScroll,
} from "../src/hooks/collapse-scroll-lock.ts";

import { readSrc } from "./helpers/kit.ts";

class FakeContainer extends EventTarget {
  scrollTop = 1000;
  writes: Array<{ top: number; behavior: string }> = [];

  scrollTo(options: { top: number; behavior: "instant" }): void {
    this.writes.push(options);
    this.scrollTop = options.top;
    this.dispatchEvent(new Event("scroll"));
  }

  layoutShift(delta: number): void {
    this.scrollTop += delta;
    this.dispatchEvent(new Event("scroll"));
  }

  gesture(type: string): void {
    this.dispatchEvent(new Event(type));
  }
}

test("a layout-driven scroll inside the window is undone with one instant write", () => {
  const container = new FakeContainer();
  const release = lockCollapseScroll(container, 250);
  container.layoutShift(-300);
  assert.deepEqual(container.writes, [{ top: 1000, behavior: "instant" }]);
  assert.equal(container.scrollTop, 1000);
  release();
});

test("the write's own scroll event does not write again", () => {
  const container = new FakeContainer();
  const release = lockCollapseScroll(container, 250);
  container.layoutShift(120);
  container.layoutShift(-45);
  assert.equal(container.writes.length, 2);
  assert.equal(container.scrollTop, 1000);
  release();
});

for (const type of USER_DISCRETE_EVENTS) {
  test(`a ${type} anywhere in the document releases the hold`, () => {
    const container = new FakeContainer();
    const page = new EventTarget();
    lockCollapseScroll(container, 250, page);
    page.dispatchEvent(new Event(type));
    container.layoutShift(360);
    assert.equal(container.writes.length, 0);
    assert.equal(container.scrollTop, 1360);
  });
}

for (const type of USER_SCROLL_EVENTS) {
  test(`a ${type} gesture releases the hold and the user's scroll stands`, () => {
    const container = new FakeContainer();
    lockCollapseScroll(container, 250);
    container.gesture(type);
    container.layoutShift(360);
    assert.equal(container.writes.length, 0);
    assert.equal(container.scrollTop, 1360);
    container.layoutShift(-50);
    assert.equal(container.writes.length, 0);
    assert.equal(container.scrollTop, 1310);
  });
}

for (const type of ["pointermove", "touchmove", "mousemove", "scrollend"]) {
  test(`a ${type} event does not release the hold`, () => {
    const container = new FakeContainer();
    const release = lockCollapseScroll(container, 250);
    container.gesture(type);
    container.layoutShift(-10);
    assert.equal(container.writes.length, 1);
    assert.equal(container.scrollTop, 1000);
    release();
  });
}

test("the hold expires after the animation duration", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const container = new FakeContainer();
    lockCollapseScroll(container, 250);
    mock.timers.tick(249);
    container.layoutShift(-10);
    assert.equal(container.writes.length, 1);
    mock.timers.tick(1);
    container.layoutShift(-10);
    assert.equal(container.writes.length, 1);
    assert.equal(container.scrollTop, 990);
  } finally {
    mock.timers.reset();
  }
});

test("release is idempotent and removes every listener", () => {
  const container = new FakeContainer();
  const release = lockCollapseScroll(container, 250);
  release();
  release();
  for (const type of [...USER_SCROLL_EVENTS, ...USER_DISCRETE_EVENTS]) {
    container.gesture(type);
  }
  container.layoutShift(-10);
  assert.equal(container.writes.length, 0);
});

test("a gesture after expiry is harmless", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const container = new FakeContainer();
    lockCollapseScroll(container, 100);
    mock.timers.tick(100);
    container.gesture("wheel");
    container.layoutShift(40);
    assert.equal(container.writes.length, 0);
    assert.equal(container.scrollTop, 1040);
  } finally {
    mock.timers.reset();
  }
});

test("the hook never assigns scrollTop and delegates to the pure lock", () => {
  const hook = readSrc("hooks/use-collapse-scroll-lock.ts");
  assert.doesNotMatch(hook, /scrollTop\s*=[^=]/);
  assert.ok(hook.includes('from "./collapse-scroll-lock"'));
  assert.match(hook, /lockCollapseScroll\(\s*container,\s*animationDurationMs,\s*container\.ownerDocument,\s*\)/);
  const lock = readSrc("hooks/collapse-scroll-lock.ts");
  assert.doesNotMatch(lock, /scrollTop\s*=[^=]/);
  assert.ok(lock.includes('behavior: "instant"'));
});

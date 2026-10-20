// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createInputPointerState,
  inputPointerHeld,
} from "../src/lib/input-pointer-state.ts";
import { createFenceHighlightQueue } from "../src/components/assistant-ui/fence-highlight-queue.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function environment(pointerEvents = true) {
  const target = Object.assign(new EventTarget(), {
    ...(pointerEvents ? { PointerEvent: function PointerEvent() {} } : {}),
  });
  const document = Object.assign(new EventTarget(), { hidden: false });
  const send = (type: string, fields: Record<string, unknown> = {}, node: EventTarget = target) => {
    node.dispatchEvent(Object.assign(new Event(type), fields));
  };
  const state = createInputPointerState(target as unknown as Window, document as unknown as Document);
  return { target, document, send, state };
}

const mouse = { pointerId: 1, pointerType: "mouse", button: 0, buttons: 1 };

test("pointer state is safe without a document", () => {
  assert.equal(inputPointerHeld(), false);
});

test("the module records presses before the first highlighting subscriber reads it", (t) => {
  const h = environment();
  const scope = globalThis as Record<string, unknown>;
  const saved = { window: scope.window, document: scope.document };
  scope.window = h.target;
  scope.document = h.document;
  t.after(() => {
    scope.window = saved.window;
    scope.document = saved.document;
    h.state.dispose();
  });
  const module = loadWithStubs<{ inputPointerHeld: () => boolean }>(
    new URL("../src/lib/input-pointer-state.ts", import.meta.url),
    {},
  );
  h.send("pointerdown", mouse);
  assert.equal(module.inputPointerHeld(), true);
  h.send("pointerup", { ...mouse, buttons: 0 });
  assert.equal(module.inputPointerHeld(), false);
});

test("replacing the sole queued fence cannot lose a stationary held scrollbar", () => {
  const h = environment();
  const callbacks = new Set<() => void>();
  let commits = 0;
  const queue = createFenceHighlightQueue({
    schedule: (callback) => {
      callbacks.add(callback);
      return () => { callbacks.delete(callback); };
    },
    blocked: h.state.held,
    commit: (callback) => { commits += 1; callback(); },
  });
  const cancel = queue.add(() => true, () => {});
  h.send("pointerdown", mouse);
  cancel();
  assert.equal(queue.pending(), 0);
  queue.add(() => true, () => {});
  const run = () => {
    const callback = callbacks.values().next().value;
    assert.ok(callback);
    callbacks.delete(callback);
    callback();
  };
  run();
  run();
  assert.equal(commits, 0);
  assert.equal(queue.pending(), 1);
  h.send("pointerup", { ...mouse, buttons: 0 });
  run();
  assert.equal(commits, 1);
  assert.equal(queue.pending(), 0);
  h.state.dispose();
});

test("each pointer must release independently and cancellation releases only that pointer", () => {
  const h = environment();
  h.send("pointerdown", mouse);
  h.send("pointerdown", { ...mouse, pointerId: 2, pointerType: "pen" });
  h.send("pointerup", { ...mouse, buttons: 0 });
  assert.equal(h.state.held(), true);
  h.send("pointercancel", { ...mouse, pointerId: 2 });
  assert.equal(h.state.held(), false);
  h.state.dispose();
});

test("mouse and pen movement recover missed presses and releases without treating hover as a hold", () => {
  const h = environment();
  for (const pointerType of ["mouse", "pen"]) {
    h.send("pointermove", { ...mouse, pointerType, buttons: 0 });
    assert.equal(h.state.held(), false);
    h.send("pointermove", { ...mouse, pointerType });
    assert.equal(h.state.held(), true);
    h.send("pointermove", { ...mouse, pointerType, buttons: 0 });
    assert.equal(h.state.held(), false);
  }
  h.send("pointerdown", { ...mouse, button: 2, buttons: 2 });
  assert.equal(h.state.held(), false);
  h.state.dispose();
});

test("touch pointers remain held through movement and until their own release or cancel", () => {
  const h = environment();
  h.send("pointerdown", { ...mouse, pointerType: "touch" });
  h.send("pointermove", { ...mouse, pointerType: "touch", buttons: 0 });
  h.send("pointerdown", { ...mouse, pointerType: "touch", pointerId: 2 });
  h.send("pointercancel", { ...mouse, pointerType: "touch" });
  assert.equal(h.state.held(), true);
  h.send("pointerup", { ...mouse, pointerType: "touch", pointerId: 2 });
  assert.equal(h.state.held(), false);
  h.state.dispose();
});

test("mouse fallback tracks the primary button and recovers a release outside the window", () => {
  const h = environment(false);
  h.send("mousedown", { button: 2, buttons: 2 });
  assert.equal(h.state.held(), false);
  h.send("mousedown", { button: 0, buttons: 1 });
  assert.equal(h.state.held(), true);
  h.send("mousemove", { buttons: 0 });
  assert.equal(h.state.held(), false);
  h.send("mousemove", { buttons: 1 });
  assert.equal(h.state.held(), true);
  h.send("mouseup", { button: 0, buttons: 0 });
  assert.equal(h.state.held(), false);
  h.state.dispose();
});

test("touch fallback retains other fingers through touchend and touchcancel", () => {
  const h = environment(false);
  h.send("touchstart", { touches: [{}, {}] });
  assert.equal(h.state.held(), true);
  h.send("touchmove", { touches: [{}, {}] });
  h.send("touchend", { touches: [{}] });
  assert.equal(h.state.held(), true);
  h.send("touchcancel", { touches: [] });
  assert.equal(h.state.held(), false);
  h.state.dispose();
});

test("native drag remains held after pointer cancellation and releases on drop or dragend", () => {
  for (const end of ["drop", "dragend"]) {
    const h = environment();
    h.send("pointerdown", mouse);
    h.send("dragstart");
    h.send("pointercancel", mouse);
    assert.equal(h.state.held(), true);
    h.send(end);
    assert.equal(h.state.held(), false);
    h.state.dispose();
  }
});

test("blur, pagehide and hidden documents release stale state", () => {
  for (const event of ["blur", "pagehide", "visibilitychange"]) {
    const h = environment();
    h.send("pointerdown", mouse);
    if (event === "visibilitychange") {
      h.send(event, {}, h.document);
      assert.equal(h.state.held(), true);
      h.document.hidden = true;
      h.send(event, {}, h.document);
    } else {
      h.send(event);
    }
    assert.equal(h.state.held(), false);
    h.state.dispose();
  }
});

test("a focused element losing focus does not release the pointer that clicked elsewhere", () => {
  const h = environment();
  h.send("pointerdown", mouse);
  const event = new Event("blur");
  Object.defineProperty(event, "target", { value: new EventTarget() });
  h.target.dispatchEvent(event);
  assert.equal(h.state.held(), true);
  h.send("blur");
  assert.equal(h.state.held(), false);
  h.state.dispose();
});

test("disposing clears the hold and removes every listener", () => {
  const h = environment();
  h.send("pointerdown", mouse);
  h.state.dispose();
  h.state.dispose();
  assert.equal(h.state.held(), false);
  h.send("pointerdown", mouse);
  h.send("dragstart");
  assert.equal(h.state.held(), false);
});

test("a cancelled dragstart does not leave every fence held", async () => {
  const h = environment();
  h.send("pointerdown", mouse);
  const cancelled = new Event("dragstart", { cancelable: true });
  h.target.addEventListener("dragstart", (event) => event.preventDefault());
  h.target.dispatchEvent(cancelled);
  h.send("pointerup", { ...mouse, buttons: 0 });
  assert.equal(h.state.held(), true, "the outcome is only known once dispatch is over");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(h.state.held(), false, "no drag started, so nothing is held after release");
  h.send("dragstart");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(h.state.held(), true, "a drag that starts stays held until it ends");
  h.send("dragend");
  assert.equal(h.state.held(), false);
  h.state.dispose();
});

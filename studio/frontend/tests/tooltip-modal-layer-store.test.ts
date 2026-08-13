// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The shared modal-layer subscription behind every tooltip. Two observers: a cheap one on the
// body's own style attribute that answers "is a modal up", and a document-wide subtree one that
// answers "which layer owns this trigger", attached only while the first says yes.
//
// What these pin: the expensive observer must not exist outside a modal, must not outlive its
// readers, and must come back when a reader does while a modal is still up.

import assert from "node:assert/strict";
import test from "node:test";

type ObserverInit = {
  attributes?: boolean;
  attributeFilter?: string[];
  attributeOldValue?: boolean;
  subtree?: boolean;
};

type FakeObserver = {
  init: ObserverInit;
  connected: boolean;
  deliver: (records: unknown[]) => void;
};

const observers: FakeObserver[] = [];
/** getAttribute calls, so a test can assert the animation path never serialises a style. */
let attributeReads = 0;

function fakeElement(pointerEvents = ""): HTMLElement {
  return {
    style: { pointerEvents },
    getAttribute: (name: string) => {
      attributeReads += 1;
      return name === "style" ? `pointer-events: ${pointerEvents}` : null;
    },
  } as unknown as HTMLElement;
}

const body = fakeElement();

class FakeMutationObserver {
  private entry: FakeObserver;
  constructor(callback: (records: unknown[]) => void) {
    this.entry = {
      init: {},
      connected: false,
      deliver: (records) => callback(records),
    };
    observers.push(this.entry);
  }
  observe(_target: unknown, init: ObserverInit) {
    this.entry.init = init;
    this.entry.connected = true;
  }
  disconnect() {
    this.entry.connected = false;
  }
}

Object.assign(globalThis, {
  document: { body },
  MutationObserver: FakeMutationObserver,
});

const { getModalLayer, subscribeModalLayer } = await import(
  "../src/components/ui/tooltip-modal-layer.ts"
);

/** The observer watching the whole document for stacking changes, if one is attached. */
function stackedObserver(): FakeObserver | undefined {
  return observers
    .filter((entry) => entry.init.subtree && entry.connected)
    .at(-1);
}

function bodyObserver(): FakeObserver | undefined {
  return observers
    .filter((entry) => !entry.init.subtree && entry.connected)
    .at(-1);
}

function openModal(): void {
  body.style.pointerEvents = "none";
  bodyObserver()?.deliver([{ target: body, oldValue: "" }]);
}

function closeModal(): void {
  body.style.pointerEvents = "";
  bodyObserver()?.deliver([{ target: body, oldValue: "pointer-events: none" }]);
}

// Only the counters. The observers list is deliberately kept: module state is global and a
// previously created observer that was never disconnected is exactly the thing under test, so
// throwing the list away between tests would hide it. The helpers above read the last
// *connected* entry, which is what the module is actually using.
function reset(): void {
  attributeReads = 0;
}

test("no modal means no document-wide observer", () => {
  reset();
  body.style.pointerEvents = "";
  const unsubscribe = subscribeModalLayer(() => undefined);
  assert.equal(getModalLayer(), false);
  assert.equal(stackedObserver(), undefined);
  assert.ok(bodyObserver(), "the body's own style attribute is always watched");
  assert.equal(bodyObserver()?.init.subtree, undefined);
  unsubscribe();
});

test("the document-wide observer follows the modal in and out", () => {
  reset();
  let notified = 0;
  const unsubscribe = subscribeModalLayer(() => {
    notified += 1;
  });

  openModal();
  assert.equal(getModalLayer(), true);
  assert.equal(notified, 1);
  const stacked = stackedObserver();
  assert.ok(stacked, "a modal is up, so stacking has to be watched");
  assert.equal(stacked?.init.attributeOldValue, true);

  closeModal();
  assert.equal(getModalLayer(), false);
  assert.equal(notified, 2);
  assert.equal(stackedObserver(), undefined, "it must not outlive the modal");
  unsubscribe();
});

test("both observers go when the last listener goes", () => {
  reset();
  const unsubscribe = subscribeModalLayer(() => undefined);
  openModal();
  assert.ok(stackedObserver());

  unsubscribe();
  // The modal is still up: nothing closed it before the last reader left.
  assert.equal(
    stackedObserver(),
    undefined,
    "a modal that is still up must not keep the subtree observer alive for nobody",
  );
  assert.equal(bodyObserver(), undefined);
});

test("a listener arriving while a modal is up gets the observer back", () => {
  reset();
  const first = subscribeModalLayer(() => undefined);
  openModal();
  first();
  assert.equal(stackedObserver(), undefined);

  // Dropping the observers must not cost the next reader its answer: the modal is still up, so
  // this subscriber has to be told, and stacking has to be watched again on its behalf.
  let notified = 0;
  const second = subscribeModalLayer(() => {
    notified += 1;
  });
  assert.equal(getModalLayer(), true, "the body still says a modal is up");
  assert.ok(stackedObserver(), "stacking must be watched again");
  assert.equal(notified, 1, "the arriving listener is told the layer is up");
  second();
  body.style.pointerEvents = "";
});

test("an animated inline style notifies nobody and serialises nothing", () => {
  reset();
  let notified = 0;
  const unsubscribe = subscribeModalLayer(() => {
    notified += 1;
  });
  openModal();
  notified = 0;
  attributeReads = 0;

  // What a motion/react frame, a popper reposition and a resize drag all look like here.
  const animated = fakeElement();
  stackedObserver()?.deliver([
    { target: animated, oldValue: "transform: translate3d(0px, 0px, 0px)" },
    { target: animated, oldValue: "transform: translate3d(1px, 0px, 0px)" },
    { target: animated, oldValue: "opacity: 0.4" },
    { target: animated, oldValue: null },
  ]);

  assert.equal(notified, 0);
  assert.equal(
    attributeReads,
    0,
    "the live property answers, no getAttribute needed",
  );
  closeModal();
  unsubscribe();
});

test("a layer losing pointer-events auto notifies", () => {
  reset();
  let notified = 0;
  const unsubscribe = subscribeModalLayer(() => {
    notified += 1;
  });
  openModal();
  notified = 0;

  // A second dialog opens over the first: the layer beneath flips auto to none.
  stackedObserver()?.deliver([
    { target: fakeElement("none"), oldValue: "pointer-events: auto" },
  ]);
  assert.equal(notified, 1);

  // And back, when it closes.
  stackedObserver()?.deliver([
    { target: fakeElement("auto"), oldValue: "pointer-events: none" },
  ]);
  assert.equal(notified, 2);
  closeModal();
  unsubscribe();
});

test("a style that only drops pointer-events still notifies", () => {
  reset();
  let notified = 0;
  const unsubscribe = subscribeModalLayer(() => {
    notified += 1;
  });
  openModal();
  notified = 0;

  // The layer stops writing the property at all, which is how Radix ends the modal state.
  stackedObserver()?.deliver([
    { target: fakeElement(""), oldValue: "pointer-events: auto" },
  ]);
  assert.equal(notified, 1);
  closeModal();
  unsubscribe();
});

test("two subscribers sharing a callback survive one of them leaving", () => {
  reset();
  let notifications = 0;
  const listener = () => {
    notifications += 1;
  };
  const first = subscribeModalLayer(listener);
  const second = subscribeModalLayer(listener);
  first();
  // The Set stores identities, so without a per-subscription wrapper both entries are the same
  // one: the first cleanup would drop it, see an empty set, and disconnect the observers under a
  // subscriber that is still reading.
  notifications = 0;
  openModal();
  assert.equal(getModalLayer(), true);
  assert.ok(notifications > 0, "the surviving subscriber stopped being notified");
  second();
  closeModal();
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

type ObserverInit = {
  attributes?: boolean;
  attributeFilter?: string[];
  childList?: boolean;
  subtree?: boolean;
};

type FakeObserver = {
  targets: { target: unknown; init: ObserverInit }[];
  deliver: () => void;
};

class FakeElement {
  private readonly attributes = new Map<string, string>();
  constructor(attributes: Record<string, string> = {}) {
    for (const [name, value] of Object.entries(attributes)) {
      this.attributes.set(name, value);
    }
  }
  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }
  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
  }
  removeAttribute(name: string): void {
    this.attributes.delete(name);
  }
}

const observers: FakeObserver[] = [];
const children: FakeElement[] = [];
const body = { children };

class FakeMutationObserver {
  private readonly entry: FakeObserver;
  constructor(callback: () => void) {
    this.entry = { targets: [], deliver: () => callback() };
    observers.push(this.entry);
  }
  observe(target: unknown, init: ObserverInit) {
    this.entry.targets.push({ target, init });
  }
  disconnect() {
    this.entry.targets = [];
  }
}

Object.assign(globalThis, {
  document: { body },
  MutationObserver: FakeMutationObserver,
});

const { getViewportBackdropOpen, subscribeViewportBackdrop } = await import(
  "../src/components/tauri/viewport-backdrop.ts"
);

function stateObserver(): FakeObserver {
  const entry = observers.at(-2);
  assert.ok(entry);
  return entry;
}

function childObserver(): FakeObserver {
  const entry = observers.at(-1);
  assert.ok(entry);
  return entry;
}

function connected(): FakeObserver[] {
  return observers.filter((entry) => entry.targets.length > 0);
}

function viewportBackdrop(state: string): FakeElement {
  return new FakeElement({
    "data-slot": "dialog-overlay",
    "data-viewport-backdrop": "true",
    "data-state": state,
  });
}

function mount(element: FakeElement): void {
  children.push(element);
  childObserver().deliver();
}

function unmount(element: FakeElement): void {
  children.splice(children.indexOf(element), 1);
  childObserver().deliver();
}

function change(element: FakeElement, name: string, value: string | null): void {
  assert.ok(
    stateObserver().targets.some((entry) => entry.target === element),
    "every child of the body is watched for its backdrop state",
  );
  if (value === null) element.removeAttribute(name);
  else element.setAttribute(name, value);
  stateObserver().deliver();
}

function reset(): void {
  children.length = 0;
}

test("an open viewport backdrop dims the titlebar until it starts closing", (t) => {
  reset();
  let notified = 0;
  t.after(subscribeViewportBackdrop(() => {
    notified += 1;
  }));
  assert.equal(getViewportBackdropOpen(), false);

  const overlay = viewportBackdrop("open");
  mount(overlay);
  assert.equal(getViewportBackdropOpen(), true);
  assert.equal(notified, 1);

  change(overlay, "data-state", "closed");
  assert.equal(getViewportBackdropOpen(), false);
  assert.equal(notified, 2);

  unmount(overlay);
  assert.equal(getViewportBackdropOpen(), false);
  assert.equal(notified, 2);
});

test("stacked viewport backdrops keep the titlebar dimmed until the last one closes", (t) => {
  reset();
  t.after(subscribeViewportBackdrop(() => undefined));
  const lower = viewportBackdrop("open");
  const upper = viewportBackdrop("open");
  mount(lower);
  mount(upper);
  assert.equal(getViewportBackdropOpen(), true);

  change(upper, "data-state", "closed");
  assert.equal(getViewportBackdropOpen(), true);
  unmount(upper);
  assert.equal(getViewportBackdropOpen(), true);

  change(lower, "data-state", "closed");
  assert.equal(getViewportBackdropOpen(), false);
});

test("panel-local overlays and menus never dim the titlebar", (t) => {
  reset();
  t.after(subscribeViewportBackdrop(() => undefined));
  const panelOverlay = new FakeElement({
    "data-slot": "dialog-overlay",
    "data-state": "open",
  });
  const menu = new FakeElement({
    "data-radix-popper-content-wrapper": "",
    "data-state": "open",
  });
  mount(panelOverlay);
  mount(menu);
  assert.equal(getViewportBackdropOpen(), false);

  change(panelOverlay, "data-viewport-backdrop", "true");
  assert.equal(getViewportBackdropOpen(), true);
  change(panelOverlay, "data-viewport-backdrop", null);
  assert.equal(getViewportBackdropOpen(), false);
});

test("a backdrop already open when the titlebar mounts dims it at once", (t) => {
  reset();
  children.push(viewportBackdrop("open"));
  let notified = 0;
  t.after(subscribeViewportBackdrop(() => {
    notified += 1;
  }));
  assert.equal(getViewportBackdropOpen(), true);
  assert.equal(notified, 1);
});

test("only the body's own children are watched, never a subtree", (t) => {
  reset();
  const overlay = viewportBackdrop("open");
  const root = new FakeElement({ id: "root" });
  children.push(root, overlay);
  t.after(subscribeViewportBackdrop(() => undefined));

  assert.deepEqual(childObserver().targets, [
    { target: body, init: { childList: true } },
  ]);
  assert.deepEqual(
    stateObserver().targets.map((entry) => entry.target),
    [root, overlay],
  );
  for (const { init } of stateObserver().targets) {
    assert.deepEqual(init, {
      attributes: true,
      attributeFilter: ["data-state", "data-viewport-backdrop"],
    });
  }
  for (const entry of connected()) {
    for (const { init } of entry.targets) assert.notEqual(init.subtree, true);
  }
});

test("both observers go when the last listener goes", (t) => {
  reset();
  const first = subscribeViewportBackdrop(() => undefined);
  const second = subscribeViewportBackdrop(() => undefined);
  t.after(first);
  t.after(second);
  mount(viewportBackdrop("open"));
  first();
  assert.ok(connected().length > 0);
  assert.equal(getViewportBackdropOpen(), true);

  second();
  assert.deepEqual(connected(), []);
  assert.equal(getViewportBackdropOpen(), false);
});

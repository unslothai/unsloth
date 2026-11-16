// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { surfaceRef, trackedSurfaces, trustSurfaceRegistry } = await import(
  "../src/lib/surface-registry.ts"
);

const KEY = "test-surface";
const SELECTOR = ".test-surface";

function fakeElement(name: string) {
  return { name, isConnected: true, matches: () => true } as unknown as Element;
}

(globalThis as { document?: unknown }).document ??= {};
trustSurfaceRegistry({
  isConnected: true,
  parentElement: null,
} as unknown as Element);

function registered(): Element[] {
  return trackedSurfaces(KEY, SELECTOR) ?? [];
}

test("a ref(null) detach unregisters and clears the forwarded ref", () => {
  const forwarded = { current: null as Element | null };
  const ref = surfaceRef(KEY, forwarded);
  const first = fakeElement("first");
  ref(first);
  assert.deepEqual(registered(), [first]);
  assert.equal(forwarded.current, first);
  ref(null);
  assert.deepEqual(registered(), []);
  assert.equal(forwarded.current, null);
});

test("attaching a new element without a detach releases the old one", () => {
  const ref = surfaceRef(KEY);
  const a = fakeElement("a");
  const b = fakeElement("b");
  ref(a);
  ref(b);
  assert.deepEqual(registered(), [b]);
  ref(null);
  assert.deepEqual(registered(), []);
});

test("the returned cleanup still works and undoes only once", () => {
  const calls: (Element | null)[] = [];
  const ref = surfaceRef(KEY, (value: Element | null) => {
    calls.push(value);
  });
  const el = fakeElement("el");
  const cleanup = ref(el);
  assert.equal(typeof cleanup, "function");
  (cleanup as () => void)();
  assert.deepEqual(registered(), []);
  ref(null);
  (cleanup as () => void)();
  assert.deepEqual(calls, [el, null]);
});

test("another holder of the same element keeps it registered", () => {
  const el = fakeElement("shared");
  const one = surfaceRef(KEY);
  const two = surfaceRef(KEY);
  one(el);
  two(el);
  one(null);
  assert.deepEqual(registered(), [el]);
  two(null);
  assert.deepEqual(registered(), []);
});

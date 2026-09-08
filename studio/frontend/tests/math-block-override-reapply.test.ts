// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installOverrideWatcher } from "../src/components/assistant-ui/math-block-mode.ts";

/*
 * `applyMathBlockContainment()` is called once, before the first render. Before the watcher below,
 * a tester who set `__UNSLOTH_MATH_BLOCK_CONTAINMENT__` from the devtools console AFTER load
 * changed nothing at all: the attribute kept its previous value and the session went on measuring
 * the arm it was already in. That is the worst thing an escape hatch can do, because the number it
 * produces looks like an answer, and it is what these rows exist to stop.
 *
 * The watcher is exercised against a plain object rather than the real global, so the tests do not
 * have to mutate `globalThis` and cannot leak into each other.
 */

const scopeWithSpy = () => {
  const scope: Record<string, unknown> = {};
  const calls: number[] = [];
  const ok = installOverrideWatcher(scope, () => {
    calls.push(1);
    return "off";
  });
  return { scope, calls, ok };
};

test("assigning the override reapplies the mode", () => {
  const { scope, calls } = scopeWithSpy();
  assert.equal(calls.length, 0, "installing the watcher does not itself apply");
  scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__ = "contain";
  assert.equal(calls.length, 1, "the assignment reapplied");
});

test("reading it back returns what was written", () => {
  // Anything that made the property write-only, or that stored somewhere else, would break the
  // documented console workflow in a different way than the bug it is fixing.
  const { scope } = scopeWithSpy();
  scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__ = "contain";
  assert.equal(scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__, "contain");
  scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__ = false;
  assert.equal(scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__, false);
});

test("a value set BEFORE the watcher is installed is preserved", () => {
  // The build flag and the measurement harness both set the global before load. Installing the
  // watcher must not discard that, or the harness would silently fall back to the ship default.
  const scope: Record<string, unknown> = {
    __UNSLOTH_MATH_BLOCK_CONTAINMENT__: "contain",
  };
  installOverrideWatcher(scope, () => "contain");
  assert.equal(scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__, "contain");
});

test("every assignment reapplies, not just the first", () => {
  const { scope, calls } = scopeWithSpy();
  scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__ = "contain";
  scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__ = "off";
  scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__ = true;
  assert.equal(calls.length, 3);
});

test("the property stays enumerable and configurable, so a later redefine is possible", () => {
  const { scope } = scopeWithSpy();
  const descriptor = Object.getOwnPropertyDescriptor(
    scope,
    "__UNSLOTH_MATH_BLOCK_CONTAINMENT__",
  );
  assert.ok(descriptor, "the property exists");
  assert.equal(descriptor.configurable, true);
  assert.equal(descriptor.enumerable, true);
});

test("a frozen scope is reported rather than throwing through startup", () => {
  // This runs from `main.tsx` before the first render. An exception there is a white screen, and
  // the flag still works when set before load, so refusing quietly is the right failure.
  const frozen = Object.freeze({} as Record<string, unknown>);
  let installed: boolean | null = null;
  assert.doesNotThrow(() => {
    installed = installOverrideWatcher(frozen, () => "off");
  });
  assert.equal(installed, false, "and it says so rather than reporting success");
});

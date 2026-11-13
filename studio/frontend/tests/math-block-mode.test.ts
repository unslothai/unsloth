// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  FIND_IN_PAGE_PROBE,
  MATH_BLOCK_CONTAINMENT_ATTRIBUTE,
  MATH_BLOCK_CONTAINMENT_ON,
  SHIP_DEFAULT,
  gateOnEngine,
  isRuntimeForced,
  resolveMathBlockMode,
} from "../src/components/assistant-ui/math-block-mode.ts";

test("an install that has never set the flag gets the ship default", () => {
  assert.equal(
    SHIP_DEFAULT,
    "contain",
    "PRECONDITION: this feature ships ON, gated on the engine. It is +92% at 500K, 3.2 to " +
      "37 fps, with 10 differing pixels across seven screenshots. See the comment on SHIP_DEFAULT " +
      "for what is accepted rather than solved.",
  );
  assert.equal(resolveMathBlockMode(undefined, ""), SHIP_DEFAULT);
  assert.equal(resolveMathBlockMode(null, ""), SHIP_DEFAULT);
});

test("a mistyped build flag turns it OFF, and does not fall back to the ship default", () => {
  // With an ON default, a typo resolves to off since the user was reaching to disable it.
  assert.equal(SHIP_DEFAULT, "contain", "PRECONDITION: this row is about an ON default");
  assert.equal(resolveMathBlockMode(undefined, "conatin"), "off");
  assert.equal(resolveMathBlockMode(undefined, "true"), "off");
  assert.notEqual(resolveMathBlockMode(undefined, "conatin"), SHIP_DEFAULT);
});

test("the build flag turns it on, and only on the values that mean on", () => {
  assert.equal(resolveMathBlockMode(undefined, "contain"), "contain");
  assert.equal(resolveMathBlockMode(undefined, "1"), "contain");
  assert.equal(resolveMathBlockMode(undefined, "off"), "off");
  assert.equal(resolveMathBlockMode(undefined, "0"), "off");
  assert.equal(resolveMathBlockMode(undefined, "conatin"), "off");
  assert.equal(resolveMathBlockMode(undefined, "true"), "off");
});

test("the runtime global overrides the build flag in BOTH directions", () => {
  assert.equal(resolveMathBlockMode(undefined, "contain"), "contain");
  assert.equal(resolveMathBlockMode(undefined, "off"), "off");

  assert.equal(resolveMathBlockMode("off", "contain"), "off");
  assert.equal(resolveMathBlockMode(false, "contain"), "off");
  assert.equal(resolveMathBlockMode("contain", ""), "contain");
  assert.equal(resolveMathBlockMode(true, ""), "contain");
  assert.equal(resolveMathBlockMode("1", ""), "contain");
});

test("a non-string, non-boolean runtime value falls through to the build flag", () => {
  assert.equal(resolveMathBlockMode({}, "contain"), "contain");
  assert.equal(resolveMathBlockMode(0, "off"), "off");
  assert.equal(resolveMathBlockMode(0, ""), SHIP_DEFAULT);
});

test("the attribute the stylesheet reads is the one the stylesheet reads", () => {
  // index.css cannot import this; math-block-containment-wiring.test.ts pins the stylesheet side.
  assert.equal(MATH_BLOCK_CONTAINMENT_ATTRIBUTE, "data-math-block-containment");
  assert.equal(MATH_BLOCK_CONTAINMENT_ON, "on");
});

/* WebKit before Safari 26 cannot find-in-page skipped content (webkit.org/b/283846). */

test("an engine that cannot find skipped content does not get containment", () => {
  assert.equal(gateOnEngine("contain", false, false), "off");
});

test("an engine that can, does", () => {
  assert.equal(gateOnEngine("contain", true, false), "contain");
});

test("the gate never turns anything ON that was already off", () => {
  assert.equal(gateOnEngine("off", true, false), "off");
  assert.equal(gateOnEngine("off", false, true), "off");
});

test("an explicit runtime override beats the gate, because that is what it is for", () => {
  assert.equal(gateOnEngine("contain", false, true), "contain");
});

test("a BUILD flag does not beat the gate", () => {
  assert.equal(gateOnEngine(resolveMathBlockMode(undefined, "1"), false, false), "off");
  assert.equal(gateOnEngine(resolveMathBlockMode(undefined, "1"), true, false), "contain");
});

test("only an explicit ON counts as a runtime force", () => {
  for (const value of [true, "1", "contain"]) {
    assert.equal(isRuntimeForced(value), true, `${String(value)} forces`);
  }
  for (const value of [undefined, null, false, "", "off", "0", "yes", 1]) {
    assert.equal(isRuntimeForced(value), false, `${String(value)} does not force`);
  }
});

test("the probe names a property, so CSS.supports can be handed it directly", () => {
  assert.match(
    FIND_IN_PAGE_PROBE,
    /^[a-z-]+:\s*\S/,
    "a `property: value` string is what CSS.supports takes in its one-argument form",
  );
});

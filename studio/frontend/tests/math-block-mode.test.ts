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

/**
 * The maths-block containment decision table, RUN rather than described.
 *
 * This feature changes what is on screen by one device pixel on KaTeX's vlist sub-structures, so
 * the thing that has to hold is that it is OFF unless somebody deliberately turned it on, in every
 * way it can be addressed. A truth table nobody executes is not evidence of that.
 */

test("an install that has never set the flag gets the ship default", () => {
  assert.equal(
    SHIP_DEFAULT,
    "contain",
    "PRECONDITION: this feature ships ON, gated on the engine. It is +92% at 500K, 3.2 to " +
      "37 fps, with 10 differing pixels across seven screenshots. See the comment on SHIP_DEFAULT " +
      "for what is accepted rather than solved.",
  );
  // Expressed through the constant rather than through its current value, so this row keeps
  // testing "unset means the ship default" when the default next moves.
  assert.equal(resolveMathBlockMode(undefined, ""), SHIP_DEFAULT);
  assert.equal(resolveMathBlockMode(null, ""), SHIP_DEFAULT);
});

test("a mistyped build flag turns it OFF, and does not fall back to the ship default", () => {
  // The asymmetry matters now that the default is on. Someone who reaches for a flag that is
  // already enabled is reaching for it in order to disable it, so a typo landing on "off" does
  // what they meant. Resolving a typo to the default would ignore them silently, which is the
  // hazard `code-fence-mode.ts` invented a third state to avoid.
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
  // A mistyped value must not land on "contain". It lands on "off", which is no longer the same
  // thing as the ship default; the row below this test is where that asymmetry is asserted.
  assert.equal(resolveMathBlockMode(undefined, "conatin"), "off");
  assert.equal(resolveMathBlockMode(undefined, "true"), "off");
});

test("the runtime global overrides the build flag in BOTH directions", () => {
  // PRECONDITION: without the runtime value these two builds disagree, so the assertions below
  // are about the override and not about the build flag being ignored.
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
  // Pinned here because `index.css` cannot import it, so the two are joined only by this pair of
  // literals and by `tests/math-block-containment-wiring.test.ts`, which reads the stylesheet.
  assert.equal(MATH_BLOCK_CONTAINMENT_ATTRIBUTE, "data-math-block-containment");
  assert.equal(MATH_BLOCK_CONTAINMENT_ON, "on");
});

/*
 * THE ENGINE GATE. WebKit below Safari 26 cannot find SKIPPED `content-visibility` content with
 * native find-in-page (webkit.org/b/283846), which `index.css` already refuses to accept for code
 * blocks. These rows run the decision rather than describing it.
 */

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
  // The console global exists so a measurement or a bug report can force an arm. A gate that
  // silently refused would make the flip look like it worked while measuring the other arm, which
  // is the failure mode that produces a confident wrong number.
  assert.equal(gateOnEngine("contain", false, true), "contain");
});

test("a BUILD flag does not beat the gate", () => {
  // A build ships to machines whose engines the builder cannot see, so `forcedByRuntime` is false
  // for it and the gate stands.
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

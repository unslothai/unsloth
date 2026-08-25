// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MATH_BLOCK_CONTAINMENT_ATTRIBUTE,
  MATH_BLOCK_CONTAINMENT_ON,
  SHIP_DEFAULT,
  resolveMathBlockMode,
} from "../src/components/assistant-ui/math-block-mode.ts";

/**
 * The maths-block containment decision table, RUN rather than described.
 *
 * This feature changes what is on screen by one device pixel on KaTeX's vlist sub-structures, so
 * the thing that has to hold is that it is OFF unless somebody deliberately turned it on, in every
 * way it can be addressed. A truth table nobody executes is not evidence of that.
 */

test("an install that has never set the flag gets nothing", () => {
  assert.equal(SHIP_DEFAULT, "off", "PRECONDITION: this feature ships off");
  assert.equal(resolveMathBlockMode(undefined, ""), "off");
  assert.equal(resolveMathBlockMode(null, ""), "off");
});

test("the build flag turns it on, and only on the values that mean on", () => {
  assert.equal(resolveMathBlockMode(undefined, "contain"), "contain");
  assert.equal(resolveMathBlockMode(undefined, "1"), "contain");
  assert.equal(resolveMathBlockMode(undefined, "off"), "off");
  assert.equal(resolveMathBlockMode(undefined, "0"), "off");
  // A mistyped value must not land on "contain". With a default of "off" it cannot land anywhere
  // else either, which is why this file does not pretend to distinguish unset from unrecognised.
  assert.equal(resolveMathBlockMode(undefined, "conatin"), "off");
  assert.equal(resolveMathBlockMode(undefined, "true"), "off");
});

test("the runtime global overrides the build flag in BOTH directions", () => {
  // PRECONDITION: without the runtime value these two builds disagree, so the assertions below
  // are about the override and not about the build flag being ignored.
  assert.equal(resolveMathBlockMode(undefined, "contain"), "contain");
  assert.equal(resolveMathBlockMode(undefined, ""), "off");

  assert.equal(resolveMathBlockMode("off", "contain"), "off");
  assert.equal(resolveMathBlockMode(false, "contain"), "off");
  assert.equal(resolveMathBlockMode("contain", ""), "contain");
  assert.equal(resolveMathBlockMode(true, ""), "contain");
  assert.equal(resolveMathBlockMode("1", ""), "contain");
});

test("a non-string, non-boolean runtime value falls through to the build flag", () => {
  assert.equal(resolveMathBlockMode({}, "contain"), "contain");
  assert.equal(resolveMathBlockMode(0, ""), "off");
});

test("the attribute the stylesheet reads is the one the stylesheet reads", () => {
  // Pinned here because `index.css` cannot import it, so the two are joined only by this pair of
  // literals and by `tests/math-block-containment-wiring.test.ts`, which reads the stylesheet.
  assert.equal(MATH_BLOCK_CONTAINMENT_ATTRIBUTE, "data-math-block-containment");
  assert.equal(MATH_BLOCK_CONTAINMENT_ON, "on");
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  SHIP_DEFAULT,
  resolveFenceMode,
} from "../src/components/assistant-ui/code-fence-mode.ts";

/**
 * The fence mode decision table, RUN rather than described.
 *
 * Every other test around this change is a regex over a source file, because the thing being
 * protected is a React hook over IntersectionObserver and a behavioural test would need a DOM.
 * This one is not: the mode is a pure function of two values, and the whole point of the default
 * moving from "off" to "defer" is that every existing way of overriding it keeps working in BOTH
 * directions. A truth table nobody executes is not evidence that it does.
 */

test("an install that has never set the flag gets the ship default", () => {
  assert.equal(SHIP_DEFAULT, "defer", "the ship default is deferral");
  assert.equal(resolveFenceMode(undefined, ""), "defer");
});

test("the build flag overrides the default in both directions", () => {
  // DOWN: an install that wants exactly the old rendering says so and gets it.
  assert.equal(resolveFenceMode(undefined, "off"), "off");
  assert.equal(resolveFenceMode(undefined, "0"), "off");
  // UP: still selects deferral explicitly, which is what every measurement arm passes.
  assert.equal(resolveFenceMode(undefined, "defer"), "defer");
  assert.equal(resolveFenceMode(undefined, "1"), "defer");
});

test("the runtime global overrides the default in both directions", () => {
  // The string forms.
  assert.equal(resolveFenceMode("off", ""), "off");
  assert.equal(resolveFenceMode("defer", ""), "defer");
  // And the boolean forms a devtools console types. `false` must turn it OFF against a default
  // that is now ON, which is the direction that did not previously have to work.
  assert.equal(resolveFenceMode(false, ""), "off");
  assert.equal(resolveFenceMode(true, ""), "defer");
});

test("the runtime global beats the build flag, both ways round", () => {
  assert.equal(
    resolveFenceMode(false, "defer"),
    "off",
    "global off beats a build that said on",
  );
  assert.equal(resolveFenceMode("off", "defer"), "off");
  assert.equal(
    resolveFenceMode(true, "off"),
    "defer",
    "global on beats a build that said off",
  );
  assert.equal(resolveFenceMode("defer", "off"), "defer");
});

test("a typo degrades to the old behaviour, never to the new default", () => {
  // Someone tried to configure this and misspelled it. Falling through to the DEFAULT would
  // silently ignore the attempt; falling through to `off` gives them the mode where every fence is
  // highlighted at mount, which is never wrong-looking, and is what this flag did before the
  // default moved.
  for (const typo of [
    "defr",
    "DEFER",
    "true",
    "yes",
    "on",
    "2",
    " defer",
    "tokenise",
  ]) {
    assert.equal(
      resolveFenceMode(undefined, typo),
      "off",
      `build flag ${typo}`,
    );
    assert.equal(resolveFenceMode(typo, ""), "off", `runtime global ${typo}`);
  }
});

test("tokenize is measurement only and no default or boolean can reach it", () => {
  assert.equal(
    resolveFenceMode(undefined, "tokenize"),
    "tokenize",
    "explicit string only",
  );
  assert.equal(resolveFenceMode("tokenize", ""), "tokenize");
  // Nothing else may land there. In particular the two shapes that carry no opinion about which
  // mode is wanted -- an absent global and an unset build flag -- must not.
  assert.notEqual(resolveFenceMode(undefined, ""), "tokenize");
  assert.notEqual(resolveFenceMode(true, ""), "tokenize");
  assert.notEqual(resolveFenceMode(false, ""), "tokenize");
  assert.notEqual(resolveFenceMode(1 as unknown, ""), "tokenize");
  assert.notEqual(resolveFenceMode({} as unknown, ""), "tokenize");
});

test("a non-string non-boolean global is ignored rather than coerced", () => {
  // `globalThis.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__ = 1` is a plausible console typo. Coercing it
  // would make a truthy number mean "defer" and a falsy one mean "off", which are two more
  // undocumented spellings. It falls through to the build flag instead.
  assert.equal(resolveFenceMode(1 as unknown, "off"), "off");
  assert.equal(resolveFenceMode(0 as unknown, "defer"), "defer");
  assert.equal(resolveFenceMode(null, ""), SHIP_DEFAULT);
  assert.equal(resolveFenceMode({} as unknown, ""), SHIP_DEFAULT);
});

/**
 * The mode module must stay free of JSX, or this file stops being loadable and every assertion
 * above silently turns back into a comment.
 */
test("the mode module is plain TypeScript", () => {
  const source = readFileSync(
    new URL(
      "../src/components/assistant-ui/code-fence-mode.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.ok(
    !/<[A-Za-z]/.test(source.replace(/^\s*[/*].*$/gm, "")),
    "no JSX in the mode module",
  );
  assert.ok(
    !/\bfrom\s+["']react["']/.test(source),
    "and no react import: this module has to load under --experimental-strip-types",
  );
});

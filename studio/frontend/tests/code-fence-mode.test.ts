// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SHIP_DEFAULT,
  resolveFenceMode,
} from "../src/components/assistant-ui/code-fence-mode.ts";

import { readSrc } from "./helpers/kit.ts";

test("an install that has never set the flag gets the ship default", () => {
  assert.equal(SHIP_DEFAULT, "defer", "the ship default is deferral");
  assert.equal(resolveFenceMode(undefined, ""), "defer");
});

test("the build flag overrides the default in both directions", () => {
  assert.equal(resolveFenceMode(undefined, "off"), "off");
  assert.equal(resolveFenceMode(undefined, "0"), "off");
  assert.equal(resolveFenceMode(undefined, "defer"), "defer");
  assert.equal(resolveFenceMode(undefined, "1"), "defer");
});

test("the runtime global overrides the default in both directions", () => {
  assert.equal(resolveFenceMode("off", ""), "off");
  assert.equal(resolveFenceMode("defer", ""), "defer");
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
  // Unknown values fall back to `off`, never to the default.
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
  assert.notEqual(resolveFenceMode(undefined, ""), "tokenize");
  assert.notEqual(resolveFenceMode(true, ""), "tokenize");
  assert.notEqual(resolveFenceMode(false, ""), "tokenize");
  assert.notEqual(resolveFenceMode(1 as unknown, ""), "tokenize");
  assert.notEqual(resolveFenceMode({} as unknown, ""), "tokenize");
});

test("a non-string non-boolean global is ignored rather than coerced", () => {
  // Numeric 1 is not coerced; it falls through to the build flag.
  assert.equal(resolveFenceMode(1 as unknown, "off"), "off");
  assert.equal(resolveFenceMode(0 as unknown, "defer"), "defer");
  assert.equal(resolveFenceMode(null, ""), SHIP_DEFAULT);
  assert.equal(resolveFenceMode({} as unknown, ""), SHIP_DEFAULT);
});

test("the mode module is plain TypeScript", () => {
  const source = readSrc("components/assistant-ui/code-fence-mode.ts");
  assert.ok(
    !/<[A-Za-z]/.test(source.replace(/^\s*[/*].*$/gm, "")),
    "no JSX in the mode module",
  );
  assert.ok(
    !/\bfrom\s+["']react["']/.test(source),
    "and no react import: this module has to load under --experimental-strip-types",
  );
});

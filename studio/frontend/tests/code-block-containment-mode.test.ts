// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  CODE_BLOCK_CONTAINMENT_ATTRIBUTE,
  CODE_BLOCK_CONTAINMENT_GLOBAL,
  CODE_BLOCK_CONTAINMENT_ON,
  FENCE_HEIGHT_PROPERTY,
  SHIP_DEFAULT,
  codeBlockContainmentMode,
  installCodeBlockContainmentWatcher,
  resolveCodeBlockContainmentMode,
} from "../src/components/assistant-ui/code-block-containment-mode.ts";

import { readSrc } from "./helpers/kit.ts";

test("an install that has never set the flag gets the ship default", () => {
  assert.equal(SHIP_DEFAULT, "contain");
  assert.equal(resolveCodeBlockContainmentMode(undefined, ""), SHIP_DEFAULT);
  assert.equal(resolveCodeBlockContainmentMode(null, ""), SHIP_DEFAULT);
});

test("a mistyped flag turns it off rather than falling back to the default", () => {
  for (const typo of ["conatin", "true", "yes", "on", "2", " contain"]) {
    assert.equal(resolveCodeBlockContainmentMode(undefined, typo), "off", `build ${typo}`);
    assert.equal(resolveCodeBlockContainmentMode(typo, ""), "off", `runtime ${typo}`);
  }
});

test("the build flag and the runtime global work in both directions", () => {
  assert.equal(resolveCodeBlockContainmentMode(undefined, "contain"), "contain");
  assert.equal(resolveCodeBlockContainmentMode(undefined, "1"), "contain");
  assert.equal(resolveCodeBlockContainmentMode(undefined, "off"), "off");
  assert.equal(resolveCodeBlockContainmentMode(undefined, "0"), "off");
  assert.equal(resolveCodeBlockContainmentMode(true, "off"), "contain");
  assert.equal(resolveCodeBlockContainmentMode(false, "contain"), "off");
  assert.equal(resolveCodeBlockContainmentMode("off", "contain"), "off");
  assert.equal(resolveCodeBlockContainmentMode("contain", "off"), "contain");
});

test("the engine gate refuses containment where find-in-page cannot reach skipped content", () => {
  assert.equal(codeBlockContainmentMode(undefined, "", true), "contain");
  assert.equal(codeBlockContainmentMode(undefined, "", false), "off");
  assert.equal(codeBlockContainmentMode(undefined, "contain", false), "off", "a build flag cannot see the engine");
  assert.equal(codeBlockContainmentMode(true, "", false), "contain", "an explicit runtime override does");
  assert.equal(codeBlockContainmentMode("contain", "", false), "contain");
  assert.equal(codeBlockContainmentMode(false, "", true), "off");
});

test("assigning the override reapplies the mode and reads back what was written", () => {
  const scope: Record<string, unknown> = {};
  const calls: number[] = [];
  assert.equal(installCodeBlockContainmentWatcher(scope, () => { calls.push(1); return "off"; }), true);
  assert.equal(calls.length, 0);
  scope[CODE_BLOCK_CONTAINMENT_GLOBAL] = "contain";
  assert.equal(calls.length, 1);
  assert.equal(scope[CODE_BLOCK_CONTAINMENT_GLOBAL], "contain");
  scope[CODE_BLOCK_CONTAINMENT_GLOBAL] = false;
  assert.equal(calls.length, 2);
  assert.equal(scope[CODE_BLOCK_CONTAINMENT_GLOBAL], false);
});

test("a frozen scope cannot take the watcher and says so", () => {
  const frozen = Object.freeze({}) as Record<string, unknown>;
  assert.equal(installCodeBlockContainmentWatcher(frozen, () => "off"), false);
});

test("the names the stylesheet and the component read are the exported ones", () => {
  assert.equal(CODE_BLOCK_CONTAINMENT_ATTRIBUTE, "data-code-block-containment");
  assert.equal(CODE_BLOCK_CONTAINMENT_ON, "on");
  assert.equal(FENCE_HEIGHT_PROPERTY, "--unsloth-fence-height");
});

test("the mode module is plain TypeScript", () => {
  const source = readSrc("components/assistant-ui/code-block-containment-mode.ts");
  assert.ok(!/<\/?[a-z]+[\s>]/i.test(source.replace(/^\s*[/*].*$/gm, "")), "no JSX");
  assert.ok(!/\bfrom\s+["']react["']/.test(source), "no react import");
  assert.ok(!source.includes("import.meta"), "no import.meta: the DOM half owns the build flag");
});

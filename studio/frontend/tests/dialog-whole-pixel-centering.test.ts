// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A dialog is centred by translating it half its own size. WebKit does not snap transforms to pixels,
// so an odd or fractional height drew the whole dialog between pixel rows and blurred its text at 1x.

import assert from "node:assert/strict";
import test from "node:test";

import postcss from "postcss";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const DIALOGS = ':is([data-slot="dialog-content"], [data-slot="alert-dialog-content"])';

function centeringRules() {
  const found: { selector: string; decls: Record<string, string>; guard: string; unlayered: boolean }[] = [];
  postcss.parse(CSS).walkRules((rule) => {
    if (!rule.selector.startsWith(DIALOGS) || !rule.selector.includes("translate")) return;
    const decls: Record<string, string> = {};
    rule.walkDecls((decl) => {
      decls[decl.prop] = decl.value;
    });
    const parent = rule.parent;
    const guard = parent?.type === "atrule" && parent.name === "supports" ? parent.params : "";
    found.push({ selector: rule.selector, decls, guard, unlayered: parent?.parent?.type === "root" });
  });
  return found;
}

test("centred dialogs shift by whole device pixels on both axes", () => {
  const rules = centeringRules();
  assert.deepEqual(
    rules.map((rule) => [rule.selector, rule.decls]),
    [
      [`${DIALOGS}.-translate-x-1\\/2`, { "--tw-translate-x": "round(-50%, var(--device-px, 1px))" }],
      [`${DIALOGS}.-translate-y-1\\/2`, { "--tw-translate-y": "round(-50%, var(--device-px, 1px))" }],
    ],
  );
});

test("the rounding only sets the translate variables, behind a round() check, outside any layer", () => {
  for (const rule of centeringRules()) {
    // An engine without round() would compute translate to none and drop the dialog off-centre.
    assert.equal(rule.guard, "(translate: 0 round(-50%, 1px))", rule.selector);
    // Unlayered, so it outranks the utility that sets the same variable.
    assert.ok(rule.unlayered, `${rule.selector} must not sit inside @layer`);
    // Never the translate property itself: max-sm:translate-none must still win at phone width.
    assert.deepEqual(Object.keys(rule.decls).filter((prop) => !prop.startsWith("--tw-translate-")), [], rule.selector);
  }
});

test("the shared dialogs still centre with the classes the rounding is keyed on", () => {
  for (const file of ["components/ui/dialog.tsx", "components/ui/alert-dialog.tsx"]) {
    assert.match(readSrc(file), /-translate-x-1\/2 -translate-y-1\/2/, file);
  }
  // A dialog that sets its own translate drops the centring class through twMerge and is left alone.
  assert.match(readSrc("components/ui/command.tsx"), /top-1\/3 translate-y-0/);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const attributes = new Set<string>();
(globalThis as { document?: unknown }).document = {
  documentElement: {
    style: { setProperty() {}, removeProperty() {} },
    setAttribute: (name: string) => attributes.add(name),
    removeAttribute: (name: string) => attributes.delete(name),
    toggleAttribute: (name: string, on: boolean) => (on ? attributes.add(name) : attributes.delete(name)),
    classList: { toggle() {} },
  },
};

const { applyCustomizationToDocument, DEFAULT_CUSTOMIZATION, sanitizeCustomization } = await import(
  "../src/features/settings/stores/appearance-custom-store.ts"
);

test("records from before the code settings keep today's code blocks", () => {
  const c = sanitizeCustomization({});
  assert.equal(c.codeLineNumbers, false);
  assert.equal(c.codeWordWrap, false);
  assert.equal(c.codeLigatures, true);
  const odd = sanitizeCustomization({ codeLineNumbers: "yes", codeWordWrap: 1, codeLigatures: "no" });
  assert.deepEqual([odd.codeLineNumbers, odd.codeWordWrap, odd.codeLigatures], [false, false, true]);
});

test("each code setting drives one root attribute the stylesheet keys on", () => {
  applyCustomizationToDocument(
    { ...DEFAULT_CUSTOMIZATION, codeLineNumbers: true, codeWordWrap: true, codeLigatures: false },
    "light",
  );
  for (const name of ["data-code-line-numbers", "data-code-wrap", "data-code-no-ligatures"])
    assert.ok(attributes.has(name), name);
  applyCustomizationToDocument(DEFAULT_CUSTOMIZATION, "light");
  for (const name of ["data-code-line-numbers", "data-code-wrap", "data-code-no-ligatures"])
    assert.ok(!attributes.has(name), name);
  const css = readSrc("index.css");
  assert.match(css, /html\[data-code-wrap\] \[data-streamdown="code-block"\] :is\(pre, code\)/);
  assert.match(css, /html\[data-code-no-ligatures\] :is\(pre, code, kbd, samp\)/);
  // Long fences size their pre to the longest line, which would leave nothing to wrap against.
  assert.match(css, /html\[data-code-wrap\] \.aui-thread-root \[data-unsloth-fence-paint\] pre \{\s*width: auto;/);
  // Chat hides Streamdown's line numbers only while the setting is off.
  assert.match(css, /html:not\(\[data-code-line-numbers\]\) :is\(\.aui-thread-root, \[data-code-preview\]\)/);
});

test("API examples drop their no-wrap overrides while Word wrap is on", () => {
  const source = readSrc("features/settings/components/api-example-parts.tsx");
  for (const utility of ["!whitespace-pre", "!break-normal", "!min-w-max"]) {
    const uses = source.match(new RegExp(`\\S*:${utility.replace("!", "\\!")}(?=\\s|")`, "g")) ?? [];
    assert.ok(uses.length > 0, utility);
    for (const use of uses) assert.ok(use.startsWith("[html:not([data-code-wrap])_&"), use);
  }
});

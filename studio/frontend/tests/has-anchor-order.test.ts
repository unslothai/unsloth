// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { compile } from "tailwindcss";

import { hasAnchorOrder, hoistHasAnchors } from "../vite-plugin-has-anchor.ts";
import { readSrc, readText } from "./helpers/kit.ts";

const HAS_INSIDE_IS = /:is\([^{]*:has\([^{]*\s[~>+]?\s?\*\)/;

test("Tailwind's group-has and peer-has variants lose their :has() inside :is()", async () => {
  const compiler = await compile("@theme { --spacing: 0.25rem; } @tailwind utilities;");
  const generated = compiler.build([
    "group-has-[[data-state=open]]:opacity-0",
    "group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-16",
    "group-has-[>svg]/alert:col-start-2",
    "group-has-[[data-state=open]]:hover:opacity-0",
    "peer-has-[:checked]:opacity-50",
  ]);
  assert.match(generated, HAS_INSIDE_IS, "Tailwind still emits the shape this plugin rewrites");
  const rewritten = hoistHasAnchors(generated);
  assert.doesNotMatch(
    rewritten,
    HAS_INSIDE_IS,
    "a :has() nested in :is() past a combinator makes Chromium restyle the whole app when the thread mounts a node",
  );
  assert.equal(rewritten.split(":has(").length, generated.split(":has(").length);
});

test("the rewrite keeps the match and the specificity, nested or flattened", () => {
  const cases: [string, string][] = [
    [
      ".a {\n  &:is(:where(.group):has(*:is([data-state=open])) *) {\n    opacity: 0%;\n  }\n}",
      ".a {\n  :where(.group):has(*:is([data-state=open])) & {\n    opacity: 0%;\n  }\n}",
    ],
    [
      ".a {\n  &:is(:where(.peer):has(*:is(:checked)) ~ *) {\n    opacity: 50%;\n  }\n}",
      ".a {\n  :where(.peer):has(*:is(:checked)) ~ & {\n    opacity: 50%;\n  }\n}",
    ],
    [".a\\:b:is(:where(.group\\/alert):has(>svg) *){x:1}", ":where(.group\\/alert):has(>svg) .a\\:b{x:1}"],
    [".x:is(:where(.g):has(.y) *):hover,.z:is(:where(.g):has(.y) *){x:1}", ":where(.g):has(.y) .x:hover,:where(.g):has(.y) .z{x:1}"],
    [
      ".a:is(:where(.g)[data-size=default] *):is(:where(.g):has([data-slot=m]) *){x:1}",
      ":where(.g):has([data-slot=m]) .a:is(:where(.g)[data-size=default] *){x:1}",
    ],
    [".a:is(:where(.g):has(.y) *)>svg{x:1}", ":where(.g):has(.y) .a>svg{x:1}"],
    [".a:is(:where(.g):has(.y) > *){x:1}", ":where(.g):has(.y) > .a{x:1}"],
  ];
  for (const [input, output] of cases) {
    assert.equal(hoistHasAnchors(input), output);
  }
});

test("selectors the rewrite cannot keep equivalent are left alone", () => {
  for (const css of [
    ".k:is(.a, .b){x:1}",
    ".k:is(.b, :where(.g):has(.y) *){x:1}",
    ".k:not(:is(:where(.g):has(.y) *)){x:1}",
    ":where(html) .k:is(:where(.g):has(.y) *){x:1}",
    ".k:is(:where(.g):has(.y) .z){x:1}",
    ".in { :where(*:is(.x)) & { &:has(*:is(.y)) { x: 1; } } }",
  ]) {
    assert.equal(hoistHasAnchors(css), css);
  }
});

test("the plugin rewrites stylesheets only, before the viewport gate", () => {
  const hook = hasAnchorOrder().transform as {
    filter: { id: RegExp; code: string };
    handler: (code: string) => { code: string; map: null } | null;
  };
  assert.ok(hook.filter.id.test("/app/src/index.css?direct"));
  assert.ok(!hook.filter.id.test("/app/src/main.tsx"));
  assert.equal(hook.filter.code, ":has(");
  assert.equal(hook.handler(".a:has(.b){}"), null);
  assert.match(readText("../vite.config.ts"), /childSelectorOrder\(\), hasAnchorOrder\(\), viewportMediaGate\(\)/);
});

test("hand-written :has() rules keep a class subject or move off :has()", () => {
  const css = readSrc("index.css");
  assert.doesNotMatch(css, /:hover\s+\.composer-pill-glyph\s*>\s*:not\(\.composer-pill-x\)/);
  assert.match(css, /\.composer-pill-glyph > :not\(\.composer-pill-x\) \{\s*opacity: var\(--pill-glyph-icon-opacity, 1\);/);
  assert.doesNotMatch(css, /\[class\*="hover:bg-sidebar-accent"\]:is\([^)]*:has\(/);
  assert.doesNotMatch(css, /html[^{]*\[data-wallpaper-route="page"\]:has\(/);
});

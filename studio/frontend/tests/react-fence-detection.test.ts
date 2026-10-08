// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  extractHtmlFences,
  extractReactFences,
  reactComponentName,
  reactFenceLang,
} from "../src/features/chat/artifacts/html-fences.ts";

const COMPONENT = 'import { useState } from "react";\nexport default function Counter() {\n  return <b>1</b>;\n}';

test("jsx and tsx fences preview when they have something to mount", () => {
  assert.equal(reactFenceLang("tsx", COMPONENT), "tsx");
  assert.equal(reactFenceLang("jsx", "export default () => <p />;"), "jsx");
  assert.equal(reactFenceLang("TSX", "function App() { return null; }"), "tsx");
  // The info string's first word is the language.
  assert.equal(reactFenceLang("tsx title=App.tsx", COMPONENT), "tsx");
  // A type or helper snippet has nothing to mount, so no card.
  assert.equal(reactFenceLang("tsx", "interface Props { name: string }"), null);
  assert.equal(reactFenceLang("jsx", "const x = <div />;"), null);
});

test("js and ts fences preview only when they import React", () => {
  assert.equal(reactFenceLang("js", COMPONENT), "jsx");
  assert.equal(reactFenceLang("javascript", COMPONENT), "jsx");
  assert.equal(reactFenceLang("ts", COMPONENT), "tsx");
  assert.equal(reactFenceLang("typescript", 'import { createRoot } from "react-dom/client";\nexport default 1;'), "tsx");
  assert.equal(reactFenceLang("ts", "export default function add(a: number, b: number) { return a + b; }"), null);
  assert.equal(reactFenceLang("js", 'import x from "reactive";\nexport default x;'), null);
  assert.equal(reactFenceLang("python", COMPONENT), null);
  assert.equal(reactFenceLang("html", COMPONENT), null);
  assert.equal(reactFenceLang(null, COMPONENT), null);
});

test("extractReactFences finds every closed React fence and skips the rest", () => {
  const text = [
    "Here is a page:",
    "```html",
    "<p>hi</p>",
    "```",
    "And a component:",
    "```tsx",
    COMPONENT,
    "```",
    "A helper:",
    "```ts",
    "export default function add(a: number) { return a; }",
    "```",
    "````jsx",
    "export default function Quoted() {",
    "  return <pre>{'```'}</pre>;",
    "}",
    "````",
    "```tsx",
    "export default function Unclosed() {}",
  ].join("\n");
  const fences = extractReactFences(text);
  assert.deepEqual(
    fences.map((fence) => [fence.lang, fence.index, reactComponentName(fence.source)]),
    [
      ["tsx", 0, "Counter"],
      ["jsx", 1, "Quoted"],
    ],
  );
  assert.equal(fences[0].source, COMPONENT);
  // The HTML scanner sees the same boundaries: its fence is unchanged by the React one.
  assert.deepEqual(
    extractHtmlFences(text).map((fence) => fence.source),
    ["<p>hi</p>"],
  );
});

test("indented fences lose their indent like HTML fences do", () => {
  const fences = extractReactFences("  ```jsx\n  export default function A() {}\n  ```");
  assert.equal(fences[0]?.source, "export default function A() {}");
});

test("reactComponentName reads the default export, then App", () => {
  assert.equal(reactComponentName("export default async function Page() {}"), "Page");
  assert.equal(reactComponentName("export default class Board extends React.Component {}"), "Board");
  assert.equal(reactComponentName("function Chart() {}\nexport default Chart;"), "Chart");
  assert.equal(reactComponentName("export default function () {}"), null);
  assert.equal(reactComponentName("const App = () => null;"), "App");
  assert.equal(reactComponentName("export default () => <p />;"), null);
});

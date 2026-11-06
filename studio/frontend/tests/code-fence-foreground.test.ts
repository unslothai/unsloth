// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { runInNewContext } from "node:vm";
import type { HighlightResult } from "@streamdown/code";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "@typescript/typescript6";
import { fenceLineForeground } from "../src/components/assistant-ui/code-fence-foreground.ts";
import {
  isBlankLine,
  plainLineText,
} from "../src/components/assistant-ui/code-fence-window.ts";
import { readSrc } from "./helpers/kit.ts";

type Token = HighlightResult["tokens"][number][number];
const token = (
  content: string,
  light = "#383A42",
  dark = "#ABB2BF",
  rest: Partial<Token> = {},
): Token => ({
  content,
  offset: 0,
  htmlStyle: { color: light, "--shiki-dark": dark },
  ...rest,
});

const source = readSrc("components/assistant-ui/code-fence-defer.tsx");
const parsed = ts.createSourceFile(
  "code-fence-defer.tsx",
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const wanted = new Set([
  "LINE_CLASS",
  "TOKEN_CLASS",
  "TOKEN_BG_CLASS",
  "tokenStyle",
  "FenceLine",
]);
const declarations = parsed.statements
  .filter(ts.isVariableStatement)
  .flatMap((statement) =>
    [...statement.declarationList.declarations]
      .filter(
        (declaration) =>
          ts.isIdentifier(declaration.name) &&
          wanted.has(declaration.name.text),
      )
      .map((declaration) => `const ${declaration.getText(parsed)};`),
  );
assert.equal(declarations.length, wanted.size);
const { outputText } = ts.transpileModule(
  `${declarations.join("\n")}\nexports.FenceLine = FenceLine;`,
  {
    compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React },
  },
);
const scope = {
  exports: {} as {
    FenceLine: React.ComponentType<{ line: Token[]; windowed: boolean }>;
  },
  React,
  memo: React.memo,
  isBlankLine,
  plainLineText,
  fenceLineForeground,
};
runInNewContext(outputText, scope);
const render = (line: Token[], windowed = true) =>
  renderToStaticMarkup(
    React.createElement(scope.exports.FenceLine, { line, windowed }),
  );
const spanCount = (html: string) => (html.match(/<span\b/g) ?? []).length;

test("a line inherits its first eligible color pair and keeps different light or dark colors", () => {
  const line = [
    token("a"),
    token("b", "#e45649"),
    token("c"),
    token("d", "#383a42", "#e06c75"),
  ];
  assert.deepEqual(fenceLineForeground(line), {
    light: "#383a42",
    dark: "#abb2bf",
    inherited: [true, false, true, false],
  });
  const html = render(line);
  assert.equal(spanCount(html), 3);
  assert.match(html, /style="--sdm-c:#383a42;--shiki-dark:#abb2bf"/);
  assert.match(html, /--sdm-c:#e45649/);
  assert.match(html, /--shiki-dark:#e06c75/);
});

test("font styles and token backgrounds retain their spans", () => {
  const line = [
    token("a"),
    token("b"),
    token("italic", undefined, undefined, { fontStyle: 1 }),
    token("bold", undefined, undefined, {
      htmlStyle: {
        color: "#383A42",
        "--shiki-dark": "#ABB2BF",
        "font-weight": "bold",
      },
    }),
    token("background", undefined, undefined, { bgColor: "#ff0000" }),
    token("dark-background", undefined, undefined, {
      htmlStyle: {
        color: "#383A42",
        "--shiki-dark": "#ABB2BF",
        "--shiki-dark-bg": "#ff0000",
      },
    }),
  ];
  assert.deepEqual(fenceLineForeground(line)?.inherited, [
    true,
    true,
    false,
    false,
    false,
    false,
  ]);
  const html = render(line);
  assert.equal(spanCount(html), 5);
  assert.match(html, /font-weight:bold/);
  assert.match(html, /--sdm-tbg:#ff0000/);
  assert.match(html, /--shiki-dark-bg:#ff0000/);
});

test("attributes and unrecognized styles leave line inheritance unchanged", () => {
  for (const special of [
    token("attributed", undefined, undefined, {
      htmlAttrs: { "data-marker": "retained" },
    }),
    token("custom", undefined, undefined, {
      htmlStyle: {
        color: "#383A42",
        "--shiki-dark": "#ABB2BF",
        all: "inherit",
      },
    }),
  ]) {
    const line = [token("a"), special, token("b")];
    assert.equal(fenceLineForeground(line), null);
    assert.equal(spanCount(render(line)), 4);
  }
  assert.match(
    render([
      token("x", undefined, undefined, {
        htmlAttrs: { "data-marker": "retained" },
      }),
    ]),
    /data-marker="retained"/,
  );
});

test("approximate, uncolored, empty and nonliteral colors never gain a new foreground", () => {
  for (const special of [
    { content: "tail", offset: 0 },
    token("one-theme", undefined, undefined, {
      htmlStyle: { color: "#383A42" },
    }),
    token("empty", ""),
    token("inherited", "inherit"),
    token("custom", "var(--custom)"),
    token("invalid", "#12345"),
  ] as Token[]) {
    const line = [token("colored"), special];
    assert.equal(fenceLineForeground(line), null);
    assert.equal(spanCount(render(line)), 3);
  }
  assert.equal(fenceLineForeground([]), null);
});

test("raw text preserves characters and windowed-out and blank lines keep their existing form", () => {
  const line = [token("<&>"), token(" \tπ"), token("tail")];
  const html = render(line);
  assert.equal(spanCount(html), 1);
  assert.ok(html.endsWith("&lt;&amp;&gt; \tπtail</span>"));
  const plain = render(line, false);
  assert.equal(spanCount(plain), 1);
  assert.equal(plain.includes("style="), false);
  assert.ok(plain.endsWith("&lt;&amp;&gt; \tπtail</span>"));
  for (const blank of [[], [{ content: "", offset: 0 }]] as Token[][]) {
    assert.equal(spanCount(render(blank)), 1);
    assert.ok(render(blank).endsWith("\n</span>"));
  }
});

test("legacy color fields are honored without mutating committed tokens", () => {
  const line = [
    Object.freeze(
      token("x", undefined, undefined, {
        color: "#383A42",
        htmlStyle: Object.freeze({ "--shiki-dark": "#ABB2BF" }),
      }),
    ),
  ];
  assert.deepEqual(fenceLineForeground(line), {
    light: "#383a42",
    dark: "#abb2bf",
    inherited: [true],
  });
  assert.equal(spanCount(render(line)), 1);
});

test("appending a different color cannot replace unchanged prefix token spans", () => {
  const prefix = [token("a"), token("b", "#e45649", "#e06c75"), token("c")];
  const grown = [
    ...prefix,
    token("d", "#e45649", "#e06c75"),
    token("e", "#e45649", "#e06c75"),
  ];
  const before = fenceLineForeground(prefix)!;
  const after = fenceLineForeground(grown)!;
  assert.equal(after.light, before.light);
  assert.equal(after.dark, before.dark);
  assert.deepEqual(after.inherited.slice(0, prefix.length), before.inherited);
  assert.deepEqual(after.inherited, [true, false, true, false, false]);
  const prefixMarkup = render(prefix).slice(0, -"</span>".length);
  assert.ok(render(grown).startsWith(prefixMarkup));
});

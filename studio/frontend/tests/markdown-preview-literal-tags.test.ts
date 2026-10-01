// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";
import ts from "typescript";

import { INERT_MARKDOWN_COMPONENTS } from "../src/components/markdown/inert-components.ts";
import * as markdownDataImages from "../src/lib/markdown-data-images.ts";
import { readSrc } from "./helpers/kit.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const PREVIEW = "components/markdown/markdown-preview.tsx";

function previewRehypePlugins(): unknown {
  const source = ts.createSourceFile(
    PREVIEW,
    readSrc(PREVIEW),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const consts = new Map<string, ts.Expression>();
  let value: ts.Expression | undefined;
  function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && node.initializer) {
      consts.set(node.name.getText(source), node.initializer);
    }
    const tag = openingTag(node);
    if (tag?.tagName.getText(source) === "Streamdown") {
      for (const attr of tag.attributes.properties) {
        if (
          ts.isJsxAttribute(attr) &&
          attr.name.getText(source) === "rehypePlugins" &&
          attr.initializer &&
          ts.isJsxExpression(attr.initializer)
        ) {
          value = attr.initializer.expression;
        }
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  if (!value) {
    return undefined;
  }
  const expression = ts.isIdentifier(value) ? consts.get(value.text) : value;
  assert.ok(expression, `${PREVIEW}: rehypePlugins is not a module constant`);
  const { outputText } = ts.transpileModule(
    `return (${expression.getText(source)});`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  return new Function(...Object.keys(markdownDataImages), outputText)(
    ...Object.values(markdownDataImages),
  );
}

function renderPreview(markdown: string, inert = false) {
  return renderToStaticMarkup(
    createElement(
      Streamdown,
      {
        mode: "static",
        controls: false,
        rehypePlugins: previewRehypePlugins() as never,
        components: inert ? INERT_MARKDOWN_COMPONENTS : undefined,
      },
      markdown,
    ),
  );
}

test("markdown previews keep placeholders and generic types as text", () => {
  for (const inert of [false, true]) {
    const html = renderPreview(
      "Replace <placeholder> with your key.\n\nVec<T> holds values of type T. Option<Box<T>> too.\n\n<your-api-key>",
      inert,
    );
    assert.match(html, /Replace &lt;placeholder&gt; with your key\./);
    assert.match(
      html,
      /Vec&lt;T&gt; holds values of type T\. Option&lt;Box&lt;T&gt;&gt; too\./,
    );
    assert.match(html, /<p>&lt;your-api-key&gt;<\/p>/);
  }
});

test("markdown previews still render allowed tags and keep inert links inert", () => {
  assert.match(
    renderPreview("Press <kbd>Ctrl</kbd>."),
    /<kbd[^>]*>Ctrl<\/kbd>/,
  );
  assert.doesNotMatch(
    renderPreview("a <script>alert(1)</script> <img src=x onerror=alert(1)> b"),
    /<script|onerror=/,
  );
  const inert = renderPreview("See [the docs](https://unsloth.ai).", true);
  assert.doesNotMatch(inert, /<a[\s>]/);
  assert.match(inert, /the docs/);
});

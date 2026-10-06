// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import * as streamdown from "streamdown";
import * as streamdownMath from "@streamdown/math";
import * as streamdownMermaid from "@streamdown/mermaid";
import ts from "typescript";

import type * as ComposerDraftPreviewModule from "../src/components/assistant-ui/composer-draft-preview.tsx";
import { createCodePlugin } from "../src/components/assistant-ui/code-plugin.ts";
import * as inertComponents from "../src/components/markdown/inert-components.ts";
import type * as MarkdownPreviewModule from "../src/components/markdown/markdown-preview.tsx";
import * as markdownDataImages from "../src/lib/markdown-data-images.ts";
import * as markdownPlugins from "../src/lib/markdown-plugins.ts";
import * as safeMarkdownUrl from "../src/lib/safe-markdown-url.ts";
import * as scheduleIdleTask from "../src/lib/schedule-idle-task.ts";
import * as utils from "../src/lib/utils.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const shared = {
  react: React,
  "react/jsx-runtime": jsxRuntime,
  streamdown,
  "@/lib/markdown-data-images": markdownDataImages,
};
const { MarkdownPreview } = loadWithStubs<typeof MarkdownPreviewModule>(
  new URL("../src/components/markdown/markdown-preview.tsx", import.meta.url),
  {
    ...shared,
    "@/components/assistant-ui/shared-code-plugin": {
      codePlugin: createCodePlugin(),
    },
    "@streamdown/math": streamdownMath,
    "@streamdown/mermaid": streamdownMermaid,
    "@/lib/markdown-plugins": markdownPlugins,
    "@/lib/open-link": { openLink: () => false },
    "@/lib/safe-markdown-url": safeMarkdownUrl,
    "@/lib/schedule-idle-task": scheduleIdleTask,
    "@/lib/utils": utils,
    "./inert-components": inertComponents,
    "katex/dist/katex.min.css": {},
  },
);
const { ComposerDraftPreview } = loadWithStubs<
  typeof ComposerDraftPreviewModule
>(
  new URL(
    "../src/components/assistant-ui/composer-draft-preview.tsx",
    import.meta.url,
  ),
  {
    ...shared,
    "@/features/chat": {
      useChatPreferencesStore: (select: (state: object) => unknown) =>
        select({ plainTextComposer: false }),
    },
    "@/i18n": { useT: () => (key: string) => key },
  },
);

const SURFACES = {
  preview: (markdown: string) =>
    renderToStaticMarkup(React.createElement(MarkdownPreview, { markdown })),
  inertPreview: (markdown: string) =>
    renderToStaticMarkup(
      React.createElement(MarkdownPreview, { markdown, inert: true }),
    ),
  composerDraft: (markdown: string) =>
    renderToStaticMarkup(
      React.createElement(ComposerDraftPreview, { text: markdown }),
    ),
};

test("markdown previews keep placeholders and generic types as text", () => {
  for (const [name, render] of Object.entries(SURFACES)) {
    const html = render(
      "Replace <placeholder> with your key.\n\nVec<T> holds values of type T. Option<Box<T>> too.\n\n<your-api-key>",
    );
    assert.match(html, /Replace &lt;placeholder&gt; with your key\./, name);
    assert.match(
      html,
      /Vec&lt;T&gt; holds values of type T\. Option&lt;Box&lt;T&gt;&gt; too\./,
      name,
    );
    assert.match(html, /<p>&lt;your-api-key&gt;<\/p>/, name);
  }
});

test("markdown previews still render rich markdown", () => {
  const html = SURFACES.preview(
    "Press <kbd>Ctrl</kbd>, see [the docs](https://unsloth.ai).\n\n| a |\n| - |\n| x<br>y |\n\n```rust\nlet v: Vec<T> = vec![];\n```\n\n$$x^2$$",
  );
  assert.match(html, /<kbd[^>]*>Ctrl<\/kbd>/);
  assert.match(html, /<a href="https:\/\/unsloth\.ai\/"[^>]*>the docs<\/a>/);
  assert.match(html, /<table[\s\S]*x<br\/>y/);
  assert.match(html, /let v: Vec&lt;T&gt; = vec!\[\];/);
  assert.match(html, /class="katex/);
  const inert = SURFACES.inertPreview("See [the docs](https://unsloth.ai).");
  assert.doesNotMatch(inert, /<a[\s>]/);
  assert.match(inert, /the docs/);
  assert.match(
    SURFACES.composerDraft("Press <kbd>Ctrl</kbd>."),
    /<kbd[^>]*>Ctrl<\/kbd>/,
  );
});

test("markdown previews keep blocking data: images", () => {
  for (const [name, render] of Object.entries({
    preview: SURFACES.preview,
    inertPreview: SURFACES.inertPreview,
  })) {
    const html = render("![x](data:image/png;base64,iVBORw0KGgo=)");
    assert.doesNotMatch(html, /src="data:/, name);
  }
});

test("hostile markup stays inert in markdown previews", () => {
  for (const [name, render] of Object.entries(SURFACES)) {
    for (const markdown of [
      "a <ScRiPt>alert(1)</ScRiPt> b",
      "<script\nalert(1)",
      "x <IMG SRC=x OnError=alert(1)> y",
      "<svg onload=alert(1)>",
      '<iframe src="https://example.com"></iframe>',
      "<style>*{display:none}</style>",
      '[x](javascript:alert(1)) <a href="jav&#x09;ascript:alert(1)">y</a>',
      "<div><svg onload=alert(1)></svg><img src=x onerror=alert(1)></div>",
      "&lt;script&gt;alert(1)&lt;/script&gt;",
    ]) {
      const html = render(markdown);
      assert.doesNotMatch(
        html,
        /<(script|iframe|svg|style|object|embed|form|math)\b/i,
        `${name}: ${html}`,
      );
      assert.doesNotMatch(
        html,
        /<[a-z][^>]*\s(on[a-z]+|srcdoc|style)=|<[a-z][^>]*="\s*javascript:/i,
        `${name}: ${html}`,
      );
    }
  }
});

test("Hub model cards are wired to the literal-tag pipeline", () => {
  const path = "features/hub/catalog/model-readme.tsx";
  const source = ts.createSourceFile(
    path,
    readSrc(path),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const initializers = new Map<string, string>();
  let rehypePlugins: string | undefined;
  (function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && node.initializer) {
      initializers.set(
        node.name.getText(source),
        node.initializer.getText(source),
      );
    }
    const tag = openingTag(node);
    if (tag?.tagName.getText(source) === "Streamdown") {
      for (const attr of tag.attributes.properties) {
        if (
          ts.isJsxAttribute(attr) &&
          attr.name.getText(source) === "rehypePlugins"
        ) {
          rehypePlugins = attr.initializer
            ?.getText(source)
            .replace(/^\{|\}$/g, "");
        }
      }
    }
    ts.forEachChild(node, visit);
  })(source);
  assert.ok(rehypePlugins, `${path}: Streamdown has no rehypePlugins`);
  assert.equal(
    initializers.get(rehypePlugins) ?? rehypePlugins,
    "withLiteralUnknownTags(README_ALLOWED_TAGS)",
  );
});

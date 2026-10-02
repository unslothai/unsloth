// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import * as streamdown from "streamdown";
import * as streamdownCode from "@streamdown/code";
import * as streamdownMath from "@streamdown/math";
import * as streamdownMermaid from "@streamdown/mermaid";

import * as inertComponents from "../src/components/markdown/inert-components.ts";
import type * as MarkdownPreviewModule from "../src/components/markdown/markdown-preview.tsx";
import * as markdownDataImages from "../src/lib/markdown-data-images.ts";
import * as markdownPlugins from "../src/lib/markdown-plugins.ts";
import * as safeMarkdownUrl from "../src/lib/safe-markdown-url.ts";
import * as scheduleIdleTask from "../src/lib/schedule-idle-task.ts";
import * as utils from "../src/lib/utils.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

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
    "@streamdown/code": streamdownCode,
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

const SURFACES = {
  preview: (markdown: string) =>
    renderToStaticMarkup(React.createElement(MarkdownPreview, { markdown })),
  inertPreview: (markdown: string) =>
    renderToStaticMarkup(
      React.createElement(MarkdownPreview, { markdown, inert: true }),
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

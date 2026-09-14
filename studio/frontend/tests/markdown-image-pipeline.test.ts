// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import { markdownSandboxImageSrc } from "../src/components/assistant-ui/sandbox-files.ts";
import { rehypeSandboxImages } from "../src/components/assistant-ui/rehype-sandbox-images.ts";
import { withDataImageSupport } from "../src/lib/markdown-data-images.ts";
import { safeMarkdownUrl } from "../src/lib/safe-markdown-url.ts";

const context = { threadId: "thread-1", projectId: null as string | null };

function render(markdown: string, scope = context) {
  const sources: string[] = [];
  const html = renderToStaticMarkup(
    createElement(Streamdown, {
      mode: "static",
      children: markdown,
      rehypePlugins: withDataImageSupport({}, [[rehypeSandboxImages, scope]]),
      urlTransform: safeMarkdownUrl,
      components: {
        img: ({ src }) => {
          if (src) sources.push(markdownSandboxImageSrc(src, scope) ?? src);
          return null;
        },
      },
    }),
  );
  return { sources, html };
}

test("relative Python images survive the full Markdown pipeline", () => {
  for (const [src, file] of [
    ["line_plot.png", "line_plot.png"],
    ["outputs/line_plot.png", "outputs/line_plot.png"],
    ["./line_plot.png", "line_plot.png"],
    ["./outputs/line_plot.png", "outputs/line_plot.png"],
    ["outputs/loss%20curve%20%231.png", "outputs/loss%20curve%20%231.png"],
  ]) {
    const result = render(`![Plot](${src})`);
    assert.deepEqual(result.sources, [`/api/inference/sandbox/thread-1/${file}`], src);
    assert.doesNotMatch(result.html, /Image blocked/, src);
  }
});

test("relative images use project scope and explicit URLs keep their session", () => {
  const scope = { threadId: "thread-2", projectId: "project-1" };
  assert.deepEqual(render("![Plot](line_plot.png)", scope).sources, [
    "/api/inference/sandbox/project-project-1/line_plot.png",
  ]);
  const src = "/api/inference/sandbox/original/line_plot.png";
  assert.deepEqual(render(`![Plot](${src})`, scope).sources, [src]);
});

test("HTML image sources are resolved after raw HTML is parsed", () => {
  assert.deepEqual(render('<img src="outputs/line_plot.png" alt="Plot">').sources, [
    "/api/inference/sandbox/thread-1/outputs/line_plot.png",
  ]);
});

test("data images still render and remote image URLs stay blocked", () => {
  const data = "data:image/png;base64,iVBORw0KGgo=";
  assert.deepEqual(render(`![Plot](${data})`).sources, [data]);
  for (const src of [
    "https://example.com/plot.png",
    "http://127.0.0.1/plot.png",
    "//example.com/plot.png",
    "file:///tmp/plot.png",
    "javascript:alert%281%29",
    "data:text/html;base64,PGgxPkhlbGxvPC9oMT4=",
  ]) {
    assert.deepEqual(render(`![Plot](${src})`).sources, [], src);
  }
});

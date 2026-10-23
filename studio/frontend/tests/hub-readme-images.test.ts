// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import {
  rehypeReadmeUrls,
  resolveReadmeUrl,
} from "../src/features/hub/lib/rehype-readme-urls.ts";
import { withLiteralUnknownTags } from "../src/lib/markdown-data-images.ts";

const BASE = "https://huggingface.co/unsloth/flint-GGUF/resolve/main/";
const ASSETS = `${BASE}assets`;

function render(markdown: string, baseUrl: string | null = BASE) {
  return renderToStaticMarkup(
    createElement(
      Streamdown,
      {
        mode: "static",
        rehypePlugins: withLiteralUnknownTags(
          {},
          baseUrl ? [[rehypeReadmeUrls, { baseUrl }]] : [],
        ),
      },
      markdown,
    ),
  );
}

test("without the plugin a bare relative card image is blocked", () => {
  assert.match(
    render('<img src="assets/flint-logo.png" alt="Flint logo">', null),
    /Image blocked: Flint logo/,
  );
});

test("relative card images resolve against the repo", () => {
  for (const [markdown, src] of [
    [
      '<img src="assets/size_vs_kld.png" alt="KLD">',
      `${ASSETS}/size_vs_kld.png`,
    ],
    [
      '<p align="center"><img src="assets/flint-logo.png" width="160" alt="Flint logo"></p>',
      `${ASSETS}/flint-logo.png`,
    ],
    ["![KLD](assets/size_vs_kld.png)", `${ASSETS}/size_vs_kld.png`],
    ["![KLD](./assets/size_vs_kld.png)", `${ASSETS}/size_vs_kld.png`],
    [
      "![root](/unsloth/flint-GGUF/resolve/main/a.png)",
      "https://huggingface.co/unsloth/flint-GGUF/resolve/main/a.png",
    ],
  ]) {
    const html = render(markdown);
    assert.ok(!html.includes("Image blocked"), html);
    assert.ok(html.includes(`src="${src}"`), html);
  }
});

test("Hugging Face blob image URLs load through resolve", () => {
  for (const [src, expected] of [
    [
      "https://huggingface.co/RecursiveSI/flint/blob/main/assets/x.png",
      "https://huggingface.co/RecursiveSI/flint/resolve/main/assets/x.png",
    ],
    [
      "https://huggingface.co/datasets/o/d/blob/main/x.png",
      "https://huggingface.co/datasets/o/d/resolve/main/x.png",
    ],
    ["blob/main/x.png", `${BASE}blob/main/x.png`],
  ]) {
    const html = render(`![x](${src})`);
    assert.ok(html.includes(`src="${expected}"`), html);
  }
  // Links keep the file viewer; other hosts are left alone.
  assert.equal(
    resolveReadmeUrl("https://huggingface.co/o/r/blob/main/x.py", BASE, false),
    "https://huggingface.co/o/r/blob/main/x.py",
  );
  assert.equal(resolveReadmeUrl("LICENSE", BASE, false), `${BASE}LICENSE`);
  assert.ok(
    render("![x](https://example.com/o/r/blob/main/x.png)").includes(
      'src="https://example.com/o/r/blob/main/x.png"',
    ),
  );
});

test("unsafe schemes stay blocked", () => {
  const html = render(
    '![x](javascript:alert(1)) <img src="data:text/html,x" alt="d">',
  );
  assert.ok(!html.includes("javascript:"), html);
  assert.ok(!html.includes("data:text/html"), html);
});

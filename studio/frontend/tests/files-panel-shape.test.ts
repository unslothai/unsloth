// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const css = readSrc("index.css");

test("Chat with files: tighter open padding, a pill once collapsed to its header", () => {
  assert.match(
    css,
    /\.unsloth-files-panel \{[^}]*@apply [^;]*\brounded-2xl px-4 pt-2 pb-4;/,
  );
  assert.match(
    css,
    /\.unsloth-files-panel\[data-collapsed="true"\] \{[^}]*@apply gap-2\.5 pt-2\.5 pb-3;/,
  );
  assert.match(
    css,
    /\.unsloth-files-panel\[data-collapsed="true"\]:not\(:has\(\.unsloth-files-panel-strip > \*\)\) \{\s*@apply rounded-full;/,
  );
});

test("the header's Add files pill is text only", () => {
  const bar = readSrc("features/rag/components/thread-documents-bar.tsx");
  assert.match(
    bar,
    /onClick=\{pickFiles\}\s*className="unsloth-files-panel-action"\s*>\s*Add files/,
  );
});

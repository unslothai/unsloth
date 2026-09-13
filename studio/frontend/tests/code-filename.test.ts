// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getCodeFilename } from "../src/features/chat/utils/code-filename.ts";

test("a bare language keeps the snippet name", () => {
  assert.equal(getCodeFilename("python"), "snippet.py");
  assert.equal(getCodeFilename("html"), "snippet.html");
});

test("an unknown language slugs into the extension", () => {
  assert.equal(getCodeFilename("brainfuck"), "snippet.brainfuck");
});

test("no info string falls back to text", () => {
  assert.equal(getCodeFilename(null), "snippet.txt");
  assert.equal(getCodeFilename(""), "snippet.txt");
});

test("a filename in the info string names the download", () => {
  // The multi-file case: three fences that have to agree on names, because the
  // generated index.html references the other two by name.
  assert.equal(getCodeFilename("html index.html"), "index.html");
  assert.equal(getCodeFilename("css style.css"), "style.css");
  assert.equal(getCodeFilename("javascript script.js"), "script.js");
});

test("non-filename metadata does not leak into the extension", () => {
  // Regression: the whole info string used to be slugged, so this downloaded as
  // `snippet.python-startline-10`.
  assert.equal(getCodeFilename("python startLine=10"), "snippet.py");
});

test("a path-shaped token is refused, not sanitised", () => {
  // Model output; an anchor's `download` is where a traversal name would be tried.
  assert.equal(getCodeFilename("html ../../etc/passwd"), "snippet.html");
  assert.equal(getCodeFilename("html /tmp/evil.html"), "snippet.html");
  assert.equal(getCodeFilename("html ..hidden.js"), "snippet.html");
});

test("an overlong offered name is refused", () => {
  const long = `${"a".repeat(80)}.html`;
  assert.equal(getCodeFilename(`html ${long}`), "snippet.html");
});

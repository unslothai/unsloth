// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { classifyDropPaths } from "../src/features/native-intents/drop-paths.ts";

test("a lone gguf is a model drop", () => {
  assert.deepEqual(classifyDropPaths(["C:/models/qwen.gguf"]), {
    kind: "model",
    path: "C:/models/qwen.gguf",
  });
  // Extension casing comes from the filesystem, not from us.
  assert.equal(classifyDropPaths(["/models/Qwen.GGUF"]).kind, "model");
});

test("documents are an attachment drop, one or many", () => {
  const dropped = classifyDropPaths(["/docs/a.pdf", "/docs/b.MD", "/docs/c.docx"]);
  assert.equal(dropped.kind, "docs");
  assert.equal(dropped.kind === "docs" && dropped.paths.length, 3);
});

test("a mixed or unsupported drop is rejected", () => {
  // The regression in #7661: these used to be reported as "GGUF models only".
  assert.equal(classifyDropPaths(["/docs/a.pdf", "/models/q.gguf"]).kind, "unsupported");
  assert.equal(classifyDropPaths(["/models/a.gguf", "/models/b.gguf"]).kind, "unsupported");
  assert.equal(classifyDropPaths(["/docs/a.pdf", "/docs/b.zip"]).kind, "unsupported");
  assert.equal(classifyDropPaths(["/docs/notes.zip"]).kind, "unsupported");
});

test("an empty payload is not a drop target", () => {
  assert.equal(classifyDropPaths([]).kind, "none");
});

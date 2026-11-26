// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  modelErrorParts,
  MODEL_ERROR_SUMMARY_LIMIT,
} from "../src/lib/model-error-message.ts";

test("short actionable errors remain intact", () => {
  for (const message of [
    "Try a smaller model.",
    "Load refused.\nStop the current task first.",
    "Check llama-server output: for details.",
  ])
    assert.deepEqual(modelErrorParts(message), {
      summary: message,
      details: null,
    });
});
test("diagnostics and paths stay available in full, including CRLF", () => {
  for (const newline of ["\n", "\r\n"])
    for (const marker of ["llama-server output:", "Full log: /tmp/model.log"]) {
      const message = `Couldn't allocate memory.${newline}${newline}${marker}${newline}CUDA error: out of memory`;
      assert.deepEqual(modelErrorParts(message), {
        summary: "Couldn't allocate memory.",
        details: message,
      });
    }
});
test("unstructured exceptions and long paths cannot become giant titles", () => {
  for (const message of [
    "x".repeat(5000),
    "Failure\n".repeat(100),
    "x\n".repeat(40),
    "C:\\" + "long-path-".repeat(500),
  ]) {
    const result = modelErrorParts(message);
    assert.ok(result.summary.length <= MODEL_ERROR_SUMMARY_LIMIT);
    assert.equal(result.details, message);
  }
});
test("boundary length and blank errors", () => {
  assert.equal(
    modelErrorParts("x".repeat(MODEL_ERROR_SUMMARY_LIMIT)).details,
    null,
  );
  assert.equal(modelErrorParts(" ").summary, "Model operation failed");
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadFailureToastParts } from "../src/features/chat/lib/load-failure-message.ts";

test("runtime output and log path stay out of the toast", () => {
  const message =
    "Couldn't allocate memory. Try lowering the context length.\n\nllama-server output:\ncudaMalloc failed: out of memory\n\nFull log: /tmp/llama.log";
  assert.deepEqual(loadFailureToastParts(message), {
    summary: "Couldn't allocate memory. Try lowering the context length.",
    detail: "",
  });
  assert.ok(
    message.includes("cudaMalloc failed"),
    "the original error remains copyable",
  );
});

test("a log path without output is also kept out of the toast", () => {
  assert.deepEqual(
    loadFailureToastParts("Couldn't start.\n\nFull log: /tmp/llama.log"),
    {
      summary: "Couldn't start.",
      detail: "",
    },
  );
});

test("ordinary multiline guidance is preserved", () => {
  assert.deepEqual(
    loadFailureToastParts(
      "Load refused.\nTry again after the current task finishes.",
    ),
    {
      summary: "Load refused.",
      detail: "Try again after the current task finishes.",
    },
  );
});

test("only diagnostic block boundaries are stripped", () => {
  const message = "Check llama-server output: for details.";
  assert.deepEqual(loadFailureToastParts(message), {
    summary: message,
    detail: "",
  });
});

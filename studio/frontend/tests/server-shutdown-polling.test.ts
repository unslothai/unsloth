// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clearAppClosing,
  markAppClosing,
} from "../src/components/tauri/closing-signal.ts";
import {
  isServerShuttingDown,
  markServerShuttingDown,
  throwIfServerShuttingDown,
} from "../src/lib/server-shutdown.ts";

test("throwIfServerShuttingDown is a no-op while the server is running", () => {
  assert.equal(isServerShuttingDown(), false);
  assert.doesNotThrow(() => throwIfServerShuttingDown());
});

test("desktop app closing also pauses inference polling", () => {
  markAppClosing();
  try {
    assert.equal(isServerShuttingDown(), true);
    assert.throws(() => throwIfServerShuttingDown(), (error: unknown) => {
      return error instanceof DOMException && error.name === "AbortError";
    });
  } finally {
    clearAppClosing();
  }
});

test("markServerShuttingDown stops inference polling", () => {
  markServerShuttingDown();
  assert.equal(isServerShuttingDown(), true);
  assert.throws(() => throwIfServerShuttingDown(), (error: unknown) => {
    return error instanceof DOMException && error.name === "AbortError";
  });
});

test("the shutdown dialog marks the server as shutting down before POSTing", async () => {
  const dialog = await readFile(
    new URL("../src/components/shutdown-dialog.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    dialog,
    /markServerShuttingDown\(\)/,
    "shutdown must stop inference polls before uvicorn begins closing connections",
  );
});

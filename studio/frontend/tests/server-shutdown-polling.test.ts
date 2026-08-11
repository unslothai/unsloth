// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  clearAppClosing,
  markAppClosing,
} from "../src/components/tauri/closing-signal.ts";
import {
  isServerShuttingDown,
  markServerShuttingDown,
  throwIfServerShuttingDown,
} from "../src/lib/server-shutdown.ts";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

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

test("the shutdown dialog marks the server as shutting down before POSTing", () => {
  const dialog = read("../src/components/shutdown-dialog.tsx");
  assert.match(
    dialog,
    /markServerShuttingDown\(\)/,
    "shutdown must stop inference polls before uvicorn begins closing connections",
  );
  const stopIndex = dialog.indexOf("markServerShuttingDown()");
  const postIndex = dialog.indexOf('authFetch("/api/shutdown"');
  assert.ok(
    stopIndex >= 0 && postIndex > stopIndex,
    "markServerShuttingDown must run before POST /api/shutdown",
  );
});

test("inference status and monitor fetches refuse to run during shutdown", () => {
  const api = read("../src/features/chat/api/chat-api.ts");
  for (const fn of [
    "getInferenceStatus",
    "getApiMonitor",
    "getApiMonitorEntry",
  ]) {
    const start = api.indexOf(`export async function ${fn}`);
    assert.ok(start >= 0, `${fn} must exist`);
    const body = api.slice(start, api.indexOf("\nexport ", start + 1));
    assert.match(
      body,
      /throwIfServerShuttingDown\(\)/,
      `${fn} must guard shutdown polls`,
    );
  }
});

test("loaded-models polling stands down during shutdown", () => {
  const hook = read("../src/features/loaded-models/use-loaded-models.ts");
  assert.match(hook, /isServerShuttingDown\(\)/);
  assert.match(
    hook,
    /if \(!track \|\| isServerShuttingDown\(\)\) return;/,
    "refresh must not start a read while the server is shutting down",
  );
  assert.match(
    hook,
    /if \(document\.hidden \|\| isServerShuttingDown\(\)\) return;/,
    "the interval poll must stand down during shutdown",
  );
});

test("api monitor polling stands down during shutdown", () => {
  const hook = read("../src/features/api-monitor/use-api-monitor.ts");
  const poll = hook.slice(hook.indexOf("function poll(): void"), hook.indexOf("poll();"));
  assert.match(poll, /if \(isServerShuttingDown\(\)\) \{\s*return;\s*\}/);
  assert.match(poll, /if \(cancelled \|\| isServerShuttingDown\(\)\) return;/);
});

test("api monitor overlay polling stands down during shutdown", () => {
  const overlay = read("../src/features/api-monitor/api-monitor-overlay.tsx");
  const poll = overlay.slice(
    overlay.indexOf("function poll(): void"),
    overlay.indexOf("poll();", overlay.indexOf("function poll(): void")),
  );
  assert.match(poll, /if \(isServerShuttingDown\(\)\) \{\s*return;\s*\}/);
  assert.match(poll, /if \(!cancelled && !isServerShuttingDown\(\)\) schedule\(\)/);
});

test("agents tab inference status sync stands down during shutdown", () => {
  const tab = read("../src/features/settings/tabs/agents-tab.tsx");
  const sync = tab.slice(tab.indexOf("const sync = () => {"), tab.indexOf("sync();"));
  assert.match(sync, /if \(isServerShuttingDown\(\)\) return;/);
});

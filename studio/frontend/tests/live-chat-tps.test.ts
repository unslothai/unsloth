// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  formatLiveChatTps,
  readLiveChatTpsSample,
  visibleLiveChatTps,
} from "../src/features/chat/lib/live-chat-tps.ts";
import type { ApiMonitorEntry } from "../src/features/chat/types/api.ts";

function entry(overrides: Partial<ApiMonitorEntry> = {}): ApiMonitorEntry {
  return {
    id: "monitor-1",
    endpoint: "/v1/chat/completions",
    method: "POST",
    status: "running",
    tok_per_sec: 12.34,
    ...overrides,
  } as ApiMonitorEntry;
}

test("accepts only a finite non-negative running sample from the exact request", () => {
  assert.deepEqual(readLiveChatTpsSample(entry(), "monitor-1"), {
    running: true,
    tps: 12.34,
  });
  assert.deepEqual(
    readLiveChatTpsSample(entry({ tok_per_sec: Number.NaN }), "monitor-1"),
    { running: true, tps: null },
  );
  assert.deepEqual(readLiveChatTpsSample(entry({ tok_per_sec: -1 }), "monitor-1"), {
    running: true,
    tps: null,
  });
});

test("a mismatched or terminal monitor stops the request-owned poll", () => {
  assert.deepEqual(readLiveChatTpsSample(entry(), "monitor-2"), {
    running: false,
    tps: null,
  });
  assert.deepEqual(
    readLiveChatTpsSample(entry({ status: "completed" }), "monitor-1"),
    { running: false, tps: null },
  );
});

test("formats a stable one-decimal widget and an explicit unavailable state", () => {
  assert.equal(formatLiveChatTps(12.34), "12.3");
  assert.equal(formatLiveChatTps(0), "0.0");
  assert.equal(formatLiveChatTps(null), "—");
});

test("terminal requests cannot leave a stale TPS sample visible", () => {
  assert.equal(visibleLiveChatTps("running", 12.3), 12.3);
  assert.equal(visibleLiveChatTps("terminal", 12.3), null);
  assert.equal(visibleLiveChatTps(undefined, 12.3), null);
});

test("a transient monitor read failure remains retryable", async () => {
  const source = await readFile(
    new URL("../src/features/chat/hooks/use-live-chat-tps.ts", import.meta.url),
    "utf8",
  );
  const catchStart = source.indexOf("} catch {");
  const retry = source.indexOf(
    "window.setTimeout(poll, POLL_INTERVAL_MS)",
    catchStart,
  );
  const catchBody = source.slice(catchStart, retry);

  assert.ok(catchStart > 0 && retry > catchStart);
  assert.doesNotMatch(catchBody, /finish\(\)/);
  assert.match(catchBody, /controller\.signal\.aborted/);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

import {
  formatLiveChatTps,
  liveChatTpsThreadKey,
  newestRunningLiveChatTpsEntry,
  readLiveChatTpsSample,
  visibleLiveChatTps,
} from "../src/features/chat/lib/live-chat-tps.ts";
import type { ApiMonitorEntry } from "../src/features/chat/types/api.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

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

test("the routed chat owns TPS while the runtime switch is still landing", () => {
  assert.equal(liveChatTpsThreadKey("older", "newer"), "older");
  assert.equal(
    liveChatTpsThreadKey(undefined, "persisted-new-chat"),
    "persisted-new-chat",
  );
  assert.equal(liveChatTpsThreadKey(undefined, null), "__default");
});

test("the chat page gives route ownership to the TPS hook", async () => {
  const source = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /useLiveChatTps\(search\.thread\)/);
});

test("a delayed older header cannot replace the newest running request", () => {
  const older = () => undefined;
  const newer = () => undefined;
  useChatRuntimeStore.setState({ liveTpsByThreadId: {} });
  const store = useChatRuntimeStore.getState();

  store.beginThreadLiveTps("thread", older, "");
  store.beginThreadLiveTps("thread", newer, "monitor-new");
  store.beginThreadLiveTps("thread", older, "monitor-old");

  const entries = useChatRuntimeStore.getState().liveTpsByThreadId.thread;
  assert.equal(entries.length, 2);
  assert.equal(newestRunningLiveChatTpsEntry(entries)?.owner, newer);
  assert.equal(entries[0]?.monitorId, "monitor-old");
});

test("finishing the newest request restores an older active request", () => {
  const older = () => undefined;
  const newer = () => undefined;
  useChatRuntimeStore.setState({ liveTpsByThreadId: {} });
  const store = useChatRuntimeStore.getState();

  store.beginThreadLiveTps("thread", older, "monitor-old");
  store.beginThreadLiveTps("thread", newer, "monitor-new");
  store.finishThreadLiveTps("thread", newer);

  const remaining = useChatRuntimeStore.getState().liveTpsByThreadId.thread;
  assert.equal(newestRunningLiveChatTpsEntry(remaining)?.owner, older);
  store.finishThreadLiveTps("thread", older);
  assert.equal(useChatRuntimeStore.getState().liveTpsByThreadId.thread, undefined);
});

test("a monitor read failure clears only its request sample", () => {
  const owner = () => undefined;
  useChatRuntimeStore.setState({ liveTpsByThreadId: {} });
  const store = useChatRuntimeStore.getState();

  store.beginThreadLiveTps("thread", owner, "monitor");
  store.setThreadLiveTps("thread", owner, "monitor", 15);
  store.clearThreadLiveTpsSample("thread", owner, "other-monitor");
  assert.equal(
    useChatRuntimeStore.getState().liveTpsByThreadId.thread?.[0]?.lastRunningTps,
    15,
  );
  store.clearThreadLiveTpsSample("thread", owner, "monitor");
  assert.equal(
    useChatRuntimeStore.getState().liveTpsByThreadId.thread?.[0]?.lastRunningTps,
    null,
  );
});

test("monitor failures clear stale samples before retrying", async () => {
  const source = await readFile(
    new URL("../src/features/chat/hooks/use-live-chat-tps.ts", import.meta.url),
    "utf8",
  );
  const catchStart = source.indexOf("} catch (error) {");
  const retry = source.indexOf(
    "window.setTimeout(poll, POLL_INTERVAL_MS)",
    catchStart,
  );
  const catchBody = source.slice(catchStart, retry);

  assert.ok(catchStart > 0 && retry > catchStart);
  assert.match(catchBody, /controller\.signal\.aborted/);
  assert.match(catchBody, /clearThreadLiveTpsSample/);
  assert.match(catchBody, /isPermanentApiMonitorEntryError/);
});

test("an unavailable running sample clears the previous rate", async () => {
  const source = await readFile(
    new URL("../src/features/chat/hooks/use-live-chat-tps.ts", import.meta.url),
    "utf8",
  );
  const sampleStart = source.indexOf("const sample = readLiveChatTpsSample");
  const catchStart = source.indexOf("} catch (error) {", sampleStart);
  const sampleBody = source.slice(sampleStart, catchStart);

  assert.match(sampleBody, /if \(sample\.tps !== null\)/);
  assert.match(sampleBody, /else \{[\s\S]*clearThreadLiveTpsSample/);
});

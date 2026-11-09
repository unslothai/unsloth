// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MANAGED_STARTUP_POLL_MS,
  createPortWakeSignal,
  waitForManagedServerPort,
} from "../src/hooks/managed-server-port.ts";

import { readSrc } from "./helpers/kit.ts";

const USE_TAURI_BACKEND = readSrc("hooks/use-tauri-backend.ts");

test("server-port ends the startup wait at once, not at the next poll tick", async () => {
  const signal = createPortWakeSignal();
  let port: number | null = null;
  const started = performance.now();
  const result = waitForManagedServerPort(
    () => port,
    () => true,
    signal.wait,
  );
  setTimeout(() => {
    port = 8888;
    signal.notify();
  }, 20);
  assert.deepEqual(await result, { status: "ready", port: 8888 });
  assert.ok(
    performance.now() - started < MANAGED_STARTUP_POLL_MS - 200,
    "the wait should not run out the poll interval",
  );
});

test("without a notify the wait still polls, so a missed event cannot hang startup", async () => {
  const signal = createPortWakeSignal();
  let port: number | null = null;
  let checks = 0;
  const result = await waitForManagedServerPort(
    () => {
      checks += 1;
      if (checks === 2) port = 8890;
      return port;
    },
    () => true,
    (ms) => signal.wait(Math.min(ms, 5)),
  );
  assert.deepEqual(result, { status: "ready", port: 8890 });
  assert.equal(checks, 2);
});

test("a port already known skips the wait, and an aborted start reports aborted", async () => {
  let waited = false;
  const wait = async () => {
    waited = true;
  };
  assert.deepEqual(await waitForManagedServerPort(() => 8888, () => true, wait), {
    status: "ready",
    port: 8888,
  });
  assert.deepEqual(await waitForManagedServerPort(() => null, () => false, wait), {
    status: "aborted",
  });
  assert.equal(waited, false);
});

test("a notify with nobody waiting is a no-op and does not cut short a later wait", async () => {
  const signal = createPortWakeSignal();
  signal.notify();
  const started = performance.now();
  await signal.wait(40);
  assert.ok(performance.now() - started >= 30);
});

test("the hook wakes the wait from its server-port listener", () => {
  const listener = USE_TAURI_BACKEND.slice(
    USE_TAURI_BACKEND.indexOf('register<number>("server-port"'),
  );
  const body = listener.slice(0, listener.indexOf("});"));
  assert.match(body, /setApiBase\(e\.payload\);\s*portWakeRef\.current\.notify\(\);/);
  assert.match(
    USE_TAURI_BACKEND,
    /await waitForManagedServerPort\(\s*\(\) => portRef\.current,\s*\(\) => startingRef\.current,\s*portWakeRef\.current\.wait,\s*\);/,
  );
});

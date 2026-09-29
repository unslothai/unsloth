// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  signalLlamaJobStarted,
  signalRunningLlamaJob,
  subscribeToLlamaJobStarted,
} from "../src/lib/llama-job-events.ts";

test("job starts notify this window and other-tab storage listeners", (t) => {
  const target = new EventTarget();
  const writes = new Map<string, string>();
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: target,
  });
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      setItem(key: string, value: string) {
        writes.set(key, value);
      },
    },
  });
  t.after(() => {
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  });

  let notifications = 0;
  const unsubscribe = subscribeToLlamaJobStarted(() => {
    notifications += 1;
  });

  signalLlamaJobStarted("2026-08-11T12:00:00Z");
  assert.equal(notifications, 1);
  assert.match(
    writes.get("unsloth_llama_job_started_at") ?? "",
    /^2026-08-11T12:00:00Z:\d+$/,
  );

  const storageEvent = new Event("storage");
  Object.defineProperty(storageEvent, "key", {
    value: "unsloth_llama_job_started_at",
  });
  target.dispatchEvent(storageEvent);
  assert.equal(notifications, 2);

  assert.equal(
    signalRunningLlamaJob({
      state: "running",
      started_at: "2026-08-11T12:01:00Z",
    }),
    true,
  );
  assert.equal(notifications, 3);
  assert.equal(
    signalRunningLlamaJob({
      state: "success",
      started_at: "2026-08-11T12:01:00Z",
    }),
    false,
  );
  assert.equal(notifications, 3);

  unsubscribe();
  signalLlamaJobStarted(null);
  assert.equal(notifications, 3);
});

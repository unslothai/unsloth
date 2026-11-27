// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { register } from "node:module";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/download-lifecycle-resolver.mjs", import.meta.url);

installLocalStorageFake();
type Timer = { fn: () => void; ms: number; cleared: boolean };
const timers: Timer[] = [];
Object.assign(globalThis.window, {
  location: { protocol: "http:", pathname: "/hub" },
  setTimeout: (fn: () => void, ms = 0) =>
    timers.push({ fn, ms, cleared: false }),
  clearTimeout: (id: number) => {
    if (timers[id - 1]) timers[id - 1].cleared = true;
  },
});

const originalFetch = globalThis.fetch;
globalThis.fetch = (async () =>
  Response.json({}, { status: 200 })) as typeof fetch;
after(() => {
  globalThis.fetch = originalFetch;
});

const { getState, jobKeyOf, putJob, removeJob } =
  await import("../src/features/hub/download-manager/download-manager-state.ts");
const { finalize } =
  await import("../src/features/hub/download-manager/poll-loop.ts");
const { ERROR_LINGER_MS } =
  await import("../src/features/hub/download-manager/download-manager-config.ts");

function job(over: Record<string, unknown> = {}) {
  const variant = (over.variant as string | null | undefined) ?? null;
  return {
    key: jobKeyOf("model", "Org/Repo", variant),
    kind: "model",
    repoId: "Org/Repo",
    variant,
    state: "running",
    downloadedBytes: 1024,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 4096,
    fraction: 0.25,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 1,
    ...over,
  } as never;
}

function fireLinger(): void {
  for (const t of timers.splice(0))
    if (!t.cleared && t.ms === ERROR_LINGER_MS) t.fn();
}

test("a failed download stays listed after its linger, marked retained", () => {
  const row = job();
  putJob(row);
  finalize((row as { key: string }).key, "error", {
    error: "Connection reset",
  });
  const key = (row as { key: string }).key;
  assert.equal(getState().jobs[key]?.state, "error");
  assert.notEqual(getState().jobs[key]?.retained, true);

  fireLinger();
  assert.equal(getState().jobs[key]?.state, "error");
  assert.equal(getState().jobs[key]?.retained, true);
  removeJob(key);
});

test("a failure the panel cannot retry is still removed after its linger", () => {
  const row = job({ variant: `@training-${"b".repeat(32)}` });
  const key = (row as { key: string }).key;
  putJob(row);
  finalize(key, "error", { error: "boom" });
  fireLinger();
  assert.equal(getState().jobs[key], undefined);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A mid-transfer failure used to vanish from the Downloads overlay after a few
// seconds, and was never written to storage, so the only way back was to search
// the model again. Failed and cancelled jobs now stay in the list with Resume
// and survive a restart.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { ManagedDownload } from "../src/features/hub/download-manager/download-manager-types.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const PERSIST_KEY = "unsloth.studio.downloads";
let flushPersistedState: (() => void) | undefined;
Object.assign(globalThis.window, {
  addEventListener: (type: string, listener: () => void) => {
    if (type === "pagehide") flushPersistedState = listener;
  },
});

const { getState, jobKeyOf, putJob } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

function read(relative: string): string {
  return readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");
}

const PANEL = read(
  "../src/features/hub/download-manager/download-manager-panel.tsx",
);
const POLL = read("../src/features/hub/download-manager/poll-loop.ts");
const HYDRATE = read("../src/features/hub/download-manager/hydration.ts");
const STATE = read(
  "../src/features/hub/download-manager/download-manager-state.ts",
);

test("a failed or cancelled row offers Resume in Downloads", () => {
  assert.match(PANEL, /aria-label="Resume download"/);
  assert.match(PANEL, /resumeRequestFromJob/);
  assert.match(PANEL, /downloadManager.requestStart/);
  // Playwright and AppImage cancel/retry smokes wait on this exact copy.
  assert.match(PANEL, /Cancelled\. Partial files kept\./);
});

test("failed and cancelled jobs are persisted so a restart can resume them", () => {
  const key = jobKeyOf("model", "org/failed-model", "Q4_K_M");
  const job: ManagedDownload = {
    key,
    kind: "model",
    repoId: "org/failed-model",
    variant: "Q4_K_M",
    state: "error",
    downloadedBytes: 1_024,
    completedBytes: 512,
    completeOnDisk: false,
    expectedBytes: 4_096,
    fraction: 0.25,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: "Download interrupted. Resume from Downloads to continue.",
    startedAt: 9,
    transport: "http",
  };
  putJob(job);
  assert.ok(flushPersistedState);
  flushPersistedState();

  const persisted = JSON.parse(store.get(PERSIST_KEY) ?? "null");
  assert.equal(persisted.state.jobs[key].state, "error");
  assert.equal(persisted.state.jobs[key].repoId, "org/failed-model");
  assert.equal(getState().jobs[key]?.state, "error");
});

test("a completed job is still not persisted", () => {
  const key = jobKeyOf("model", "org/done-model", null);
  putJob({
    key,
    kind: "model",
    repoId: "org/done-model",
    variant: null,
    state: "complete",
    downloadedBytes: 100,
    completedBytes: 100,
    completeOnDisk: true,
    expectedBytes: 100,
    fraction: 1,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 10,
  });
  assert.ok(flushPersistedState);
  flushPersistedState();

  const persisted = JSON.parse(store.get(PERSIST_KEY) ?? "null");
  assert.equal(persisted.state.jobs[key], undefined);
});

test("hydration keeps failed and cancelled jobs instead of dropping them", () => {
  assert.match(HYDRATE, /if \(RESUMABLE_STATES.has\(job.state\)\) continue;/);
});

test("an idle transfer with files on disk becomes a resumable error, not gone", () => {
  assert.match(POLL, /INTERRUPTED_DOWNLOAD_MESSAGE/);
  assert.match(
    POLL,
    /finalize\(key, "error", \{ error: INTERRUPTED_DOWNLOAD_MESSAGE \}\)/,
  );
  assert.doesNotMatch(POLL, /scheduleRemoval\(key, ERROR_LINGER_MS\)/);
  assert.doesNotMatch(POLL, /scheduleRemoval\(key, CANCELLED_LINGER_MS\)/);
});

test("storage writes failed jobs through the shared persistence predicate", () => {
  assert.match(STATE, /isPersistedJobState\(job.state\)/);
  assert.match(STATE, /isPersistedJobState\(state\)/);
});

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

import {
  conflictInfoForOwner,
  type ConflictEntry,
  type ManagedDownload,
} from "../src/features/hub/download-manager/download-manager-types.ts";
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
const REPO_DOWNLOAD = read(
  "../src/features/hub/download-manager/use-repo-download.ts",
);
const POLL = read("../src/features/hub/download-manager/poll-loop.ts");
const HYDRATE = read("../src/features/hub/download-manager/hydration.ts");
const CONFLICT = read(
  "../src/features/hub/download-manager/transport-conflict.ts",
);
const CONTROLLER = read(
  "../src/features/hub/download-manager/download-manager-controller.ts",
);
const STATE = read(
  "../src/features/hub/download-manager/download-manager-state.ts",
);

test("a failed or cancelled row offers Resume in Downloads", () => {
  assert.match(PANEL, /aria-label="Resume download"/);
  assert.match(PANEL, /resumeRequestFromJob/);
  assert.match(PANEL, /inventoryKind: job\.inventoryKind/);
  assert.match(
    PANEL,
    /const resumable = !job\.external && RESUMABLE_STATES\.has\(job\.state\)/,
  );
  assert.match(PANEL, /downloadManager\s*\.requestStart/);
  assert.match(PANEL, /outcome === "busy" && mounted\.current/);
  assert.match(PANEL, /This repository is already downloading/);
  assert.match(PANEL, /disabled=\{resumePending\}/);
  assert.match(
    PANEL,
    /disabled=\{job\.state === "cancelling" \|\| resumePending\}/,
  );
  assert.match(PANEL, /if \(mounted\.current\) setResumePending\(false\)/);
  assert.match(PANEL, /resolveTransportConflict\("resume"\)/);
  assert.match(PANEL, /resolveTransportConflict\("restart"\)/);
  assert.match(PANEL, /void resolution[\s\S]*?outcome === "busy"/);
  assert.match(CONFLICT, /export async function resumeConflict/);
  assert.match(CONFLICT, /export async function restartConflict/);
  assert.match(CONFLICT, /return runWithPendingStartGuard/);
  // Playwright and AppImage cancel/retry smokes wait on this exact copy.
  assert.match(PANEL, /Cancelled\. Partial files kept\./);
});

test("the global Resume path exposes transport conflict resolution", () => {
  assert.match(
    PANEL,
    /conflictInfoForOwner\(state\.conflicts\[jobKey\], "downloads"\)/,
  );
  assert.match(PANEL, /<TransportConflictDialog/);
  assert.match(PANEL, /requestStart\(resumeRequestFromJob\(job\), "downloads"\)/);
  assert.match(PANEL, /resumeConflict\(jobKey, "downloads"\)/);
  assert.match(PANEL, /restartConflict\(jobKey, "downloads"\)/);
  assert.match(PANEL, /cancelConflict\(jobKey, "downloads"\)/);
  assert.match(PANEL, /mounted\.current = false/);
  assert.match(PANEL, /outcome === "conflict" && !mounted\.current/);
});

test("the global resumable count excludes external jobs", () => {
  assert.match(
    PANEL,
    /!job\.external && RESUMABLE_STATES\.has\(job\.state\)/,
  );
});

test("web auth session boundaries clear persisted download rows", () => {
  assert.match(
    CONTROLLER,
    /AUTH_SESSION_CLEARED_EVENT[\s\S]*AUTH_SESSION_STORED_EVENT/,
  );
  assert.match(CONTROLLER, /if \(isTauri\) return/);
  assert.match(CONTROLLER, /resetDownloadManagerState\(\)/);
  assert.match(CONTROLLER, /useDownloadManagerStore\.persist\.clearStorage\(\)/);
  assert.match(CONTROLLER, /event\.key === AUTH_SESSION_MARK_KEY/);
});

test("a transport conflict belongs to exactly one dialog surface", () => {
  const info = { previous: "http", next: "xet", resumable: true } as const;
  const entry: ConflictEntry = {
    owner: "downloads",
    info,
    pending: {
      kind: "model",
      repoId: "org/conflicted-model",
      variant: "Q4_K_M",
      expectedBytes: 4_096,
    },
  };

  assert.equal(conflictInfoForOwner(entry, "downloads"), info);
  assert.equal(conflictInfoForOwner(entry, "caller"), null);
  assert.match(REPO_DOWNLOAD, /conflictInfoForOwner\(exact, "caller"\)/);
  assert.match(REPO_DOWNLOAD, /entry\.owner === "caller"/);
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

test("hydration revalidates failed and cancelled jobs before keeping them", () => {
  assert.match(HYDRATE, /revalidateHydratedResumableJob\(job\.key, job\)/);
  assert.match(
    HYDRATE,
    /RESUMABLE_STATES\.has\(current\.state\)[\s\S]*idleProbeVerdict\([\s\S]*?progressResp\.target_present[\s\S]*?=== "gone"[\s\S]*?removeJob\(key\)/,
  );
  assert.match(
    HYDRATE,
    /applyProgressUpdate\(key, current, progressResp\)[\s\S]*?hasObservedExpectedBytes\(updated\)[\s\S]*?removeJob\(key\)/,
  );
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

test("a measured missing idle target is removed instead of offered for resume", () => {
  assert.match(
    POLL,
    /idleProbeVerdict\(\s*progressResp\.downloaded_bytes,\s*progressResp\.cache_path,\s*progressResp\.target_present,\s*progressResp\.cache_measured,\s*\) === "gone"/,
  );
  assert.match(POLL, /finalize\(key, "gone"\)/);
});

test("storage writes failed jobs through the shared persistence predicate", () => {
  assert.match(STATE, /isPersistedJobState\(job.state\)/);
  assert.match(STATE, /isPersistedJobState\(state\)/);
});

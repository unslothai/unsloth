// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The desktop update stops the backend while chat stays mounted under the update screen, so its
// focus refresh and the documents poll raised "Failed to refresh models" over a healthy update.

import assert from "node:assert/strict";
import test from "node:test";

import {
  followDesktopUpdateScreen,
  isBackendDownForDesktopUpdate,
  isSilencedDesktopUpdateFailure,
} from "../src/lib/desktop-update-activity.ts";
import { readSrc } from "./helpers/kit.ts";

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const RAG = readSrc("features/rag/components/use-rag-documents.ts");

function region(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start);
  assert.ok(start !== -1, `region start not found: ${from}`);
  assert.ok(end > start, `region end not found after start: ${to}`);
  return source.slice(start, end);
}

const transport = () =>
  Object.assign(new Error("Unsloth isn't running -- please relaunch it."), {
    unslothTransportFailure: true,
  });

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => (resolve = r));
  return { promise, resolve };
}

const noResync = () => Promise.resolve();
const settle = () => new Promise((r) => setImmediate(r));

test("only a transport failure under the update screen is silenced", () => {
  assert.equal(isSilencedDesktopUpdateFailure(transport(), false), false);
  const cleanup = followDesktopUpdateScreen(true, false, noResync);
  assert.equal(isSilencedDesktopUpdateFailure(transport(), false), true);
  assert.equal(
    isSilencedDesktopUpdateFailure(new Error("Internal Server Error"), true),
    false,
  );
  cleanup();
  assert.equal(isBackendDownForDesktopUpdate(), false);
});

test("a read issued under the update screen stays silent after the flag drops", () => {
  const cleanup = followDesktopUpdateScreen(true, false, noResync);
  const downWhenIssued = isBackendDownForDesktopUpdate();
  cleanup();
  assert.equal(isSilencedDesktopUpdateFailure(transport(), downWhenIssued), true);
});

// start_server returns once the backend is spawned, before it answers.
test("leaving the update screen holds the flag until the resync settles", async () => {
  followDesktopUpdateScreen(true, false, noResync)();
  const resync = deferred();
  let resyncs = 0;
  followDesktopUpdateScreen(false, true, () => {
    resyncs += 1;
    return resync.promise;
  });
  assert.equal(resyncs, 1);
  assert.equal(isBackendDownForDesktopUpdate(), true);
  resync.resolve();
  await settle();
  assert.equal(isBackendDownForDesktopUpdate(), false);
});

test("a retry during the resync keeps the flag when the resync settles", async () => {
  const resync = deferred();
  const leave = followDesktopUpdateScreen(false, true, () => resync.promise);
  leave();
  const retry = followDesktopUpdateScreen(true, false, noResync);
  resync.resolve();
  await settle();
  assert.equal(isBackendDownForDesktopUpdate(), true);
  retry();
});

test("no resync without a transition off the update screen", () => {
  let resyncs = 0;
  const count = () => {
    resyncs += 1;
    return Promise.resolve();
  };
  followDesktopUpdateScreen(false, false, count)();
  followDesktopUpdateScreen(true, false, count)();
  followDesktopUpdateScreen(true, true, count)();
  assert.equal(resyncs, 0);
});

test("the chat status refresh latches the flag at issue and checks it before reporting", () => {
  const sync = region(
    RUNTIME,
    "async function syncInferenceStatusToStore(",
    "async function refreshAndWaitForServerModel(",
  );
  const catchAt = sync.indexOf("} catch (error) {");
  const latch = sync.indexOf("const downWhenIssued = isBackendDownForDesktopUpdate();");
  assert.ok(latch !== -1 && latch < catchAt);
  const guard = sync.indexOf(
    "if (isSilencedDesktopUpdateFailure(error, downWhenIssued)) return;",
  );
  assert.ok(guard > catchAt);
  assert.ok(guard < sync.indexOf("setModelsError(message)"));
  assert.ok(guard < sync.indexOf("toast.error"));
});

test("the documents refresh latches the flag at issue and checks it before reporting", () => {
  const refresh = region(RAG, "const refresh = useCallback(", "const loadProjectSources");
  const latch = refresh.indexOf("const downWhenIssued = isBackendDownForDesktopUpdate();");
  const guard = refresh.indexOf(
    "if (isSilencedDesktopUpdateFailure(err, downWhenIssued)) return false;",
  );
  assert.ok(latch !== -1 && latch < refresh.indexOf("} catch (err) {"));
  assert.ok(guard !== -1 && guard < refresh.indexOf("toast.error"));
});

test("the update layer drives the flag and resyncs chat when the screen drops", () => {
  const layer = region(
    readSrc("app/provider.tsx"),
    "function TauriUpdateLayer(",
    "const HIDDEN_TITLEBAR_SIDEBAR_ROUTES",
  );
  assert.match(
    layer,
    /const wasUpdating = wasUpdatingRef\.current;\s*wasUpdatingRef\.current = isUpdating;[\s\S]*?return followDesktopUpdateScreen\(\s*isUpdating,\s*wasUpdating,\s*resyncInferenceStatusAfterServerModelChange,\s*\);\s*\}, \[isUpdating\]\);/,
  );
});

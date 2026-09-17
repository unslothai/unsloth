// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The desktop update stops the backend on purpose and emits no server-crashed, so the app stays
// mounted under the update screen. Chat re-reads its status on every window focus, and the
// documents bar polls every 4s while anything is indexing; those reads failing raised
// "Failed to refresh models / Unsloth isn't running" over a healthy update.

import assert from "node:assert/strict";
import test from "node:test";

import {
  isBackendDownForDesktopUpdate,
  isSilencedDesktopUpdateFailure,
  setBackendDownForDesktopUpdate,
} from "../src/lib/desktop-update-activity.ts";
import { readSrc } from "./helpers/kit.ts";

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const PROVIDER = readSrc("app/provider.tsx");
const RAG = readSrc("features/rag/components/use-rag-documents.ts");

function region(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to);
  // A reworded sentinel must fail here, not silently widen the slice to the rest of the file
  // and leave every assertion below passing against code it was never meant to read.
  assert.ok(start !== -1, `region start not found: ${from}`);
  assert.ok(end > start, `region end not found after start: ${to}`);
  return source.slice(start, end);
}

const SYNC = region(
  RUNTIME,
  "async function syncInferenceStatusToStore(",
  "/**\n * Reconcile the UI after the SERVER unloaded",
);
const CATCH = SYNC.slice(SYNC.indexOf("} catch (error) {"));

const transport = () =>
  Object.assign(new Error("Unsloth isn't running -- please relaunch it."), {
    unslothTransportFailure: true,
  });

test("the flag is off until the update screen raises it", () => {
  assert.equal(isBackendDownForDesktopUpdate(), false);
  setBackendDownForDesktopUpdate(true);
  assert.equal(isBackendDownForDesktopUpdate(), true);
  setBackendDownForDesktopUpdate(false);
  assert.equal(isBackendDownForDesktopUpdate(), false);
});

test("a transport failure under the update screen is silenced", () => {
  setBackendDownForDesktopUpdate(true);
  assert.equal(isSilencedDesktopUpdateFailure(transport(), false), true);
  setBackendDownForDesktopUpdate(false);
});

test("only a request that never reached the backend is silenced", () => {
  // A 500 from a backend that is up is a real failure even mid-update.
  setBackendDownForDesktopUpdate(true);
  assert.equal(
    isSilencedDesktopUpdateFailure(new Error("Internal Server Error"), true),
    false,
  );
  setBackendDownForDesktopUpdate(false);
});

test("nothing is silenced when no update is running", () => {
  assert.equal(isSilencedDesktopUpdateFailure(transport(), false), false);
});

// Skip & Restart drops the update screen while a read issued under it is still working through
// the Tauri retry ladder (10.5s) plus the check_backend_present budget (10s). Reading the flag
// at catch time would let that rejection toast at the moment the backend is coming back.
test("a read issued under the update screen stays silent after the screen drops", () => {
  setBackendDownForDesktopUpdate(true);
  const downWhenIssued = isBackendDownForDesktopUpdate();
  setBackendDownForDesktopUpdate(false);
  assert.equal(isSilencedDesktopUpdateFailure(transport(), downWhenIssued), true);
});

test("the chat status refresh latches the flag at issue, not in the catch", () => {
  const latch = SYNC.indexOf("const downWhenIssued = isBackendDownForDesktopUpdate();");
  assert.ok(latch !== -1, "the flag must be latched when the request is issued");
  assert.ok(latch < SYNC.indexOf("} catch (error) {"), "the latch must precede the catch");
  const guard = CATCH.indexOf("isSilencedDesktopUpdateFailure(error, downWhenIssued)");
  assert.ok(guard !== -1);
  assert.ok(guard < CATCH.indexOf("setModelsError("), "the guard must precede the error state");
  assert.ok(guard < CATCH.indexOf("toast.error"), "the guard must precede the error toast");
});

// The documents bar renders inside the chat composer, so it is mounted under the update screen
// too, and its 4s indexing poll toasted the same sentence on every tick.
test("the documents poll is silenced by the same predicate", () => {
  const refresh = region(RAG, "const refresh = useCallback(", "const hasIndexing =");
  assert.match(refresh, /const downWhenIssued = isBackendDownForDesktopUpdate\(\);/);
  const guard = refresh.indexOf("isSilencedDesktopUpdateFailure(err, downWhenIssued)");
  assert.ok(guard !== -1, "the documents catch must use the shared predicate");
  assert.ok(guard < refresh.indexOf("toast.error"), "the guard must precede the error toast");
});

test("the flag follows the update screen and drops with it", () => {
  const layer = region(
    PROVIDER,
    "function TauriUpdateLayer(",
    "const HIDDEN_TITLEBAR_SIDEBAR_ROUTES",
  );
  assert.match(layer, /setBackendDownForDesktopUpdate\(isUpdating\);/);
  assert.match(layer, /return \(\) => setBackendDownForDesktopUpdate\(false\);/);
  assert.match(layer, /\}, \[isUpdating\]\);/);
});

// Skip & Restart and the shell-failure recovery both start a fresh backend with nothing resident
// while the chat page is still mounted holding the old selection.
test("leaving the update screen re-reads the server's model", () => {
  const layer = region(
    PROVIDER,
    "function TauriUpdateLayer(",
    "const HIDDEN_TITLEBAR_SIDEBAR_ROUTES",
  );
  assert.match(
    layer,
    /if \(wasUpdatingRef\.current && !isUpdating\) \{\s*void resyncInferenceStatusAfterServerModelChange\(\);/,
  );
  assert.match(layer, /wasUpdatingRef\.current = isUpdating;/);
});

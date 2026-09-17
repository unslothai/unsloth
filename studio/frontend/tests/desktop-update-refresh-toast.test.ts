// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The desktop update stops the backend on purpose and emits no server-crashed, so the app stays
// mounted under the update screen. Chat re-reads its status on every window focus, and that read
// failing raised "Failed to refresh models / Unsloth isn't running" over a healthy update.

import assert from "node:assert/strict";
import test from "node:test";

import {
  isBackendDownForDesktopUpdate,
  setBackendDownForDesktopUpdate,
} from "../src/lib/desktop-update-activity.ts";
import { readSrc } from "./helpers/kit.ts";

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const PROVIDER = readSrc("app/provider.tsx");

const SYNC = RUNTIME.slice(
  RUNTIME.indexOf("async function syncInferenceStatusToStore("),
  RUNTIME.indexOf("/**\n * Reconcile the UI after the SERVER unloaded"),
);
const CATCH = SYNC.slice(SYNC.indexOf("} catch (error) {"));

test("the flag is off until the update screen raises it", () => {
  assert.equal(isBackendDownForDesktopUpdate(), false);
  setBackendDownForDesktopUpdate(true);
  assert.equal(isBackendDownForDesktopUpdate(), true);
  setBackendDownForDesktopUpdate(false);
  assert.equal(isBackendDownForDesktopUpdate(), false);
});

test("a refresh that fails under the update screen reports nothing", () => {
  const guard = CATCH.indexOf("isBackendDownForDesktopUpdate()");
  assert.ok(guard !== -1);
  assert.ok(guard < CATCH.indexOf("setModelsError("), "the guard must precede the error state");
  assert.ok(guard < CATCH.indexOf("toast.error"), "the guard must precede the error toast");
});

test("only a request that never reached the backend is silenced", () => {
  // A 500 from a backend that is up is a real failure even mid-update.
  assert.match(
    CATCH,
    /isBackendDownForDesktopUpdate\(\) &&[\s\S]{0,120}unslothTransportFailure === true/,
  );
});

test("the flag follows the update screen and drops with it", () => {
  const layer = PROVIDER.slice(
    PROVIDER.indexOf("function TauriUpdateLayer("),
    PROVIDER.indexOf("const HIDDEN_TITLEBAR_SIDEBAR_ROUTES"),
  );
  assert.match(layer, /setBackendDownForDesktopUpdate\(isUpdating\);/);
  assert.match(layer, /return \(\) => setBackendDownForDesktopUpdate\(false\);/);
  assert.match(layer, /\}, \[isUpdating\]\);/);
});

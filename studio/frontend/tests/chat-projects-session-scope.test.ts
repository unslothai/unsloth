// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

// The module registers its logout listener at import time, so record them first.
const listeners = new Map<string, Set<(event: Event) => void>>();
Object.assign(globalThis.window, {
  addEventListener: (type: string, fn: (event: Event) => void) => {
    let handlers = listeners.get(type);
    if (!handlers) {
      handlers = new Set();
      listeners.set(type, handlers);
    }
    handlers.add(fn);
  },
  removeEventListener: (type: string, fn: (event: Event) => void) => {
    listeners.get(type)?.delete(fn);
  },
});

const { AUTH_SESSION_CLEARED_EVENT } = await import(
  "../src/features/auth/session-events.ts"
);
const { resetChatProjectsState } = await import(
  "../src/features/chat/hooks/use-chat-projects.ts"
);

test("clearing the auth session resets the shared projects cache", () => {
  // A web logout only navigates, so nothing else discards the module-level cache: without
  // this listener the next account renders the previous user's project names.
  const handlers = listeners.get(AUTH_SESSION_CLEARED_EVENT);
  assert.ok(
    handlers,
    "no listener registered for the auth session cleared event",
  );
  assert.ok(
    handlers.has(resetChatProjectsState),
    "the auth session cleared event does not reset the projects cache",
  );
});

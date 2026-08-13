// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ProjectRecord } from "../src/features/chat/types.ts";

import {
  installLocalStorageFake,
  registerChatProjectsStorageResolver,
} from "./helpers/kit.ts";

registerChatProjectsStorageResolver();
installLocalStorageFake();

// The module registers its listeners at import time, so give the fake window real dispatch
// before importing it.
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
  dispatchEvent: (event: Event) => {
    for (const fn of [...(listeners.get(event.type) ?? [])]) fn(event);
    return true;
  },
});

const { AUTH_TOKEN_KEY, clearAuthTokens } = await import(
  "../src/features/auth/session.ts"
);
const storage = await import("./helpers/store-stubs/chat-history-storage.ts");
const { getProjectsSnapshot, loadProjects, resetChatProjectsState } =
  await import("../src/features/chat/hooks/use-chat-projects.ts");

function projectRows(...names: string[]): ProjectRecord[] {
  return names.map((name) => ({
    id: name,
    name,
    archived: false,
    createdAt: 1,
    updatedAt: 1,
  }));
}

function freshSession(): void {
  resetChatProjectsState();
  storage.resetListProjectsCalls();
}

const flush = () => new Promise((done) => setImmediate(done));

function fireStorage(key: string, newValue: string | null): void {
  globalThis.window.dispatchEvent(
    Object.assign(new Event("storage"), { key, newValue }),
  );
}

test("a superseded projects request leaves the live request's slot alone", async () => {
  freshSession();
  const stale = loadProjects(true);
  const calls = storage.listProjectsCalls();
  assert.equal(calls.length, 1);

  clearAuthTokens();
  const fresh = loadProjects(true);
  assert.equal(
    calls.length,
    2,
    "the cleared session did not free the request slot",
  );

  // The pre-logout request settles first: it must early-return without touching the cache.
  calls[0].resolve(projectRows("Acme roadmap"));
  assert.deepEqual(await stale, []);
  assert.deepEqual(getProjectsSnapshot(), []);

  // A joiner reuses the live request, which is what proves the stale finally did not
  // release a slot it no longer owned.
  const joined = loadProjects(true);
  assert.equal(
    calls.length,
    2,
    "the stale request released the live request's slot",
  );

  calls[1].resolve(projectRows("Second account"));
  assert.deepEqual(await fresh, projectRows("Second account"));
  assert.deepEqual(await joined, projectRows("Second account"));
  assert.deepEqual(getProjectsSnapshot(), projectRows("Second account"));
});

test("a request that resolves after the next account published does not clobber it", async () => {
  freshSession();
  const stale = loadProjects(true);
  const calls = storage.listProjectsCalls();

  clearAuthTokens();
  const fresh = loadProjects(true);

  calls[1].resolve(projectRows("Second account"));
  assert.deepEqual(await fresh, projectRows("Second account"));

  calls[0].resolve(projectRows("Acme roadmap"));
  assert.deepEqual(await stale, []);
  assert.deepEqual(getProjectsSnapshot(), projectRows("Second account"));
});

test("a superseded request leaves the queued follow-up to the request that owns it", async () => {
  freshSession();
  const stale = loadProjects(true);
  const calls = storage.listProjectsCalls();

  clearAuthTokens();
  const fresh = loadProjects(true);
  // A mutation lands against the new account and queues a follow-up on its request.
  void loadProjects(true, true);

  // Never await the stale request here: consuming the follow-up would send it round the
  // loop again and leave it pending forever.
  calls[0].resolve(projectRows("Acme roadmap"));
  await flush();
  assert.equal(calls.length, 2, "the stale request consumed the follow-up");
  assert.deepEqual(await stale, []);

  calls[1].resolve(projectRows("Second account"));
  await flush();
  assert.equal(calls.length, 3, "the follow-up was never re-fetched");

  calls[2].resolve(projectRows("Second account", "Later project"));
  assert.deepEqual(await fresh, projectRows("Second account", "Later project"));
});

test("clearing the auth session empties the shared cache at once", async () => {
  freshSession();
  const loaded = loadProjects(true);
  storage.listProjectsCalls()[0].resolve(projectRows("Acme roadmap"));
  assert.deepEqual(await loaded, projectRows("Acme roadmap"));

  clearAuthTokens();
  assert.deepEqual(getProjectsSnapshot(), []);
  // One identity for the empty state, so useSyncExternalStore does not see a new array.
  assert.equal(getProjectsSnapshot(), getProjectsSnapshot());
});

test("another tab's logout clears this tab's cache", async () => {
  freshSession();
  const loaded = loadProjects(true);
  storage.listProjectsCalls()[0].resolve(projectRows("Acme roadmap"));
  await loaded;

  // The cleared event never leaves the tab that logged out, so this is all tab B sees.
  fireStorage(AUTH_TOKEN_KEY, null);
  assert.deepEqual(getProjectsSnapshot(), []);
});

test("a cross-tab token refresh or an unrelated key keeps the cache", async () => {
  freshSession();
  const loaded = loadProjects(true);
  storage.listProjectsCalls()[0].resolve(projectRows("Acme roadmap"));
  await loaded;

  fireStorage(AUTH_TOKEN_KEY, "refreshed-access-token");
  assert.deepEqual(getProjectsSnapshot(), projectRows("Acme roadmap"));

  fireStorage("unrelated_key", null);
  assert.deepEqual(getProjectsSnapshot(), projectRows("Acme roadmap"));
});

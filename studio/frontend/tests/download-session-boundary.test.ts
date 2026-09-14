// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/download-lifecycle-resolver.mjs", import.meta.url);

const PERSIST_KEY = "unsloth.studio.downloads";
const SESSION_CLEARED = "unsloth:auth-session-cleared";
const SESSION_MARK_KEY = "unsloth_auth_session_mark";
const SESSION_STORED = "unsloth:auth-session-stored";
const TOKEN_KEY = "unsloth_auth_token";

const { fireWindowEvent, store } = installLocalStorageFake();
Object.assign(globalThis.window, {
  setTimeout: globalThis.setTimeout.bind(globalThis),
  clearTimeout: globalThis.clearTimeout.bind(globalThis),
});

const controller = await import(
  "../src/features/hub/download-manager/download-manager-controller.ts"
);
const { getState, jobKeyOf, putJob, useDownloadManagerStore } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);
const { runtimeRegistry } = await import(
  "../src/features/hub/download-manager/runtime-registry.ts"
);

const boundaries: [
  string,
  string,
  Event | { key: string; newValue: string | null },
][] = [
  ["same-tab sign-out", SESSION_CLEARED, new Event(SESSION_CLEARED)],
  ["same-tab sign-in", SESSION_STORED, new Event(SESSION_STORED)],
  [
    "cross-tab session change",
    "storage",
    { key: SESSION_MARK_KEY, newValue: "next" },
  ],
  ["cross-tab sign-out", "storage", { key: TOKEN_KEY, newValue: null }],
];

for (const [name, eventType, event] of boundaries) {
  test(`${name} clears download state and pending persistence`, async () => {
    const key = jobKeyOf("model", "private-org/private-model", "Q4_K_M");
    putJob({
      key,
      kind: "model",
      repoId: "private-org/private-model",
      variant: "Q4_K_M",
      state: "running",
      downloadedBytes: 1_024,
      completedBytes: 0,
      completeOnDisk: false,
      expectedBytes: 4_096,
      fraction: 0.25,
      bytesPerSec: 0,
      etaSeconds: 0,
      error: null,
      startedAt: 1,
      transport: "http",
    });
    runtimeRegistry.pendingStartRepoKeys.add("model:private-org/private-model");
    store.set(PERSIST_KEY, "prior-session-downloads");

    try {
      fireWindowEvent(eventType, event);
      assert.deepEqual(getState().jobs, {});
      assert.equal(runtimeRegistry.pendingStartRepoKeys.size, 0);
      assert.equal(store.has(PERSIST_KEY), false);
    } finally {
      controller.__resetDownloadManagerForTests();
      await useDownloadManagerStore.persist.clearStorage();
    }
  });
}

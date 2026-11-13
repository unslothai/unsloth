// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { store, fireWindowEvent } = installLocalStorageFake();

const { useExternalProvidersStore } = await import(
  "../src/features/chat/stores/external-providers-store.ts"
);

test("a connection added in another window shows up here", () => {
  assert.deepEqual(useExternalProvidersStore.getState().providers, []);
  store.set(
    "unsloth_chat_external_providers",
    JSON.stringify([
      {
        id: "conn-new",
        providerType: "openrouter",
        name: "Router",
        baseUrl: "https://example.com/v1",
        models: ["vendor/model"],
        createdAt: 1,
        updatedAt: 1,
      },
    ]),
  );
  fireWindowEvent("storage", { key: "unsloth_chat_external_providers" });
  assert.deepEqual(
    useExternalProvidersStore.getState().providers.map((p) => p.id),
    ["conn-new"],
  );
  assert.equal(store.get("unsloth_chat_external_providers")?.includes("conn-new"), true);
});

test("turning connections off in another window applies here", () => {
  assert.equal(useExternalProvidersStore.getState().connectionsEnabled, true);
  store.set("unsloth_chat_connections_enabled", "false");
  fireWindowEvent("storage", { key: "unsloth_chat_connections_enabled" });
  assert.equal(useExternalProvidersStore.getState().connectionsEnabled, false);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
const { store: localStorage } = installLocalStorageFake();

let serverValue: boolean | undefined;
const requests: Array<{ method: string; path: string; body: unknown }> = [];
const API_KEY = "show_model_disclaimer";

type ResponseGate = {
  method: string;
  reached: Promise<void>;
  wait: Promise<void>;
  markReached: () => void;
  release: () => void;
};

let responseGate: ResponseGate | null = null;

function holdNextResponse(method: string): ResponseGate {
  let markReached: () => void = () => undefined;
  let release: () => void = () => undefined;
  const reached = new Promise<void>((resolve) => {
    markReached = resolve;
  });
  const wait = new Promise<void>((resolve) => {
    release = resolve;
  });
  responseGate = { method, reached, wait, markReached, release };
  return responseGate;
}

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const path = String(input);
  const method = init?.method ?? "GET";
  const body = init?.body ? JSON.parse(String(init.body)) : undefined;
  requests.push({ method, path, body });

  if (method === "POST" && path.endsWith("/migrate")) {
    const legacy = (body as Record<string, unknown> | undefined)?.[API_KEY];
    if (serverValue === undefined && typeof legacy === "boolean") {
      serverValue = legacy;
    }
  } else if (method === "PUT") {
    serverValue = (body as Record<string, boolean>)[API_KEY];
  }

  const responseValue = serverValue ?? false;
  const gate = responseGate?.method === method ? responseGate : null;
  if (gate) {
    responseGate = null;
    gate.markReached();
    await gate.wait;
  }

  return new Response(JSON.stringify({ [API_KEY]: responseValue }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);
const {
  hydrateModelDisclaimerPreference,
  readLegacyModelDisclaimer,
  refreshModelDisclaimerPreference,
  saveModelDisclaimerPreference,
} = await import("../src/features/chat/sync-model-disclaimer-preference.ts");
const MISSING_DISCLAIMER_DEFAULT_PATTERN =
  /showModelDisclaimer: saved\?\.showModelDisclaimer \?\? false/;

test("the model disclaimer is hidden by default", () => {
  const fresh = useChatPreferencesStore.getInitialState();
  assert.equal(fresh.showModelDisclaimer, false);
});

test("a saved payload without the model disclaimer key defaults to hidden", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/stores/chat-preferences-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, MISSING_DISCLAIMER_DEFAULT_PATTERN);
});

test("the legacy browser value seeds the installation setting once", async () => {
  serverValue = undefined;
  requests.length = 0;
  localStorage.set(
    "unsloth_chat_preferences",
    JSON.stringify({ state: { showModelDisclaimer: true }, version: 0 }),
  );

  assert.equal(readLegacyModelDisclaimer(), true);
  await Promise.all([
    hydrateModelDisclaimerPreference(),
    refreshModelDisclaimerPreference(),
  ]);
  assert.equal(serverValue, true);
  assert.equal(useChatPreferencesStore.getState().showModelDisclaimer, true);
  assert.deepEqual(requests[0], {
    method: "POST",
    path: "/api/settings/chat-preferences/migrate",
    body: { [API_KEY]: true },
  });
  assert.equal(requests[1]?.method, "GET");

  localStorage.delete("unsloth_chat_preferences");
  useChatPreferencesStore.getState().setShowModelDisclaimer(false);
  localStorage.delete("unsloth_chat_preferences");
  await hydrateModelDisclaimerPreference();
  assert.equal(useChatPreferencesStore.getState().showModelDisclaimer, true);
});

test("a legacy disabled default does not claim the installation setting", () => {
  localStorage.set(
    "unsloth_chat_preferences",
    JSON.stringify({ state: { showModelDisclaimer: false }, version: 0 }),
  );
  assert.equal(readLegacyModelDisclaimer(), undefined);
});

test("a queued refresh cannot revert a newer toggle", async () => {
  serverValue = false;
  requests.length = 0;
  localStorage.delete("unsloth_chat_preferences");
  useChatPreferencesStore.getState().setShowModelDisclaimer(false);
  localStorage.delete("unsloth_chat_preferences");

  const migrationGate = holdNextResponse("POST");
  const migration = hydrateModelDisclaimerPreference();
  await migrationGate.reached;

  const refresh = refreshModelDisclaimerPreference();
  const saveGate = holdNextResponse("PUT");
  const save = saveModelDisclaimerPreference(true);
  migrationGate.release();
  await saveGate.reached;

  assert.equal(useChatPreferencesStore.getState().showModelDisclaimer, true);
  assert.deepEqual(
    requests.map(({ method }) => method),
    ["POST", "GET", "PUT"],
  );
  saveGate.release();
  await Promise.all([migration, refresh, save]);
  assert.equal(serverValue, true);
});

test("saving the switch updates both the browser and installation", async () => {
  serverValue = false;
  requests.length = 0;

  await saveModelDisclaimerPreference(true);

  assert.equal(serverValue, true);
  assert.equal(useChatPreferencesStore.getState().showModelDisclaimer, true);
  assert.deepEqual(requests[0], {
    method: "PUT",
    path: "/api/settings/chat-preferences",
    body: { [API_KEY]: true },
  });
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The API-only toggle rides the shared auto-switch PUT, so it must round-trip
// without dragging the other switches along: an omitted field keeps its stored
// value, and a backend that predates the field must read as off, not on.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

// The settings API modules reach authFetch through the auth barrel, which
// re-exports login-page.tsx. See helpers/auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

const API = {
  enabled: true,
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_idle_seconds: 300,
  // biome-ignore lint/style/useNamingConvention: API schema
  default_enabled: false,
  // biome-ignore lint/style/useNamingConvention: API schema
  idle_unload_active: true,
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_keep_kv: true,
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_download_model: false,
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_api_only: false,
};

let nextBody: Record<string, unknown> = { ...API };
const bodies: string[] = [];

globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
  if (init?.body) bodies.push(String(init.body));
  return new Response(JSON.stringify(nextBody), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const {
  invalidateOpenAIAutoSwitchSettings,
  loadOpenAIAutoSwitchSettings,
  updateOpenAIAutoSwitchSettings,
} = await import("../src/features/settings/api/openai-auto-switch.ts");

test("a backend without the field reads as off", async () => {
  invalidateOpenAIAutoSwitchSettings();
  const { auto_unload_api_only: _omitted, ...older } = API;
  nextBody = older;
  const settings = await loadOpenAIAutoSwitchSettings();
  assert.equal(settings.autoUnloadApiOnly, false);
  nextBody = { ...API };
});

test("the toggle round-trips", async () => {
  invalidateOpenAIAutoSwitchSettings();
  nextBody = { ...API, auto_unload_api_only: true };
  const saved = await updateOpenAIAutoSwitchSettings(
    true,
    undefined,
    undefined,
    undefined,
    true,
  );
  assert.equal(saved.autoUnloadApiOnly, true);
  nextBody = { ...API };
});

test("saving it alone leaves the other switches untouched", async () => {
  invalidateOpenAIAutoSwitchSettings();
  bodies.length = 0;
  await updateOpenAIAutoSwitchSettings(
    true,
    undefined,
    undefined,
    undefined,
    true,
  );
  assert.deepEqual(JSON.parse(bodies[0] ?? "{}"), {
    enabled: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    auto_unload_api_only: true,
  });
});

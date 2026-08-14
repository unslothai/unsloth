// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Image/video auto-switch rides the shared auto-switch PUT but is its own
// setting: saving it must not carry any other field along, and a backend that
// predates it must read as off rather than inheriting the chat toggle.

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
  // biome-ignore lint/style/useNamingConvention: API schema
  media_auto_unload_idle_seconds: 0,
  // biome-ignore lint/style/useNamingConvention: API schema
  media_idle_unload_active: false,
  // biome-ignore lint/style/useNamingConvention: API schema
  media_auto_switch_model: false,
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

test("a backend without the field reads as off, not as the chat toggle", async () => {
  invalidateOpenAIAutoSwitchSettings();
  const { media_auto_switch_model: _switch, ...older } = API;
  nextBody = older;
  const settings = await loadOpenAIAutoSwitchSettings();
  assert.equal(settings.enabled, true);
  assert.equal(settings.mediaAutoSwitchModel, false);
  nextBody = { ...API };
});

test("the toggle round-trips", async () => {
  invalidateOpenAIAutoSwitchSettings();
  // biome-ignore lint/style/useNamingConvention: API schema
  nextBody = { ...API, media_auto_switch_model: true };
  const saved = await updateOpenAIAutoSwitchSettings({
    enabled: true,
    mediaAutoSwitchModel: true,
  });
  assert.equal(saved.mediaAutoSwitchModel, true);
  nextBody = { ...API };
});

test("saving it alone leaves the other switches untouched", async () => {
  invalidateOpenAIAutoSwitchSettings();
  bodies.length = 0;
  await updateOpenAIAutoSwitchSettings({
    enabled: true,
    mediaAutoSwitchModel: true,
  });
  assert.deepEqual(JSON.parse(bodies[0] ?? "{}"), {
    enabled: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    media_auto_switch_model: true,
  });
});

test("a false toggle is sent, not dropped as absent", async () => {
  // Only `undefined` means "leave stored"; turning the switch OFF has to reach the server.
  invalidateOpenAIAutoSwitchSettings();
  bodies.length = 0;
  await updateOpenAIAutoSwitchSettings({
    enabled: true,
    mediaAutoSwitchModel: false,
  });
  assert.deepEqual(JSON.parse(bodies[0] ?? "{}"), {
    enabled: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    media_auto_switch_model: false,
  });
});

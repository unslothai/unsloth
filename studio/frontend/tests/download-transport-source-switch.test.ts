// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

// The transport settings store announces changes on window.
(globalThis as { window?: EventTarget }).window ??= new EventTarget();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const {
  clearDownloadTransportCapabilitiesCache,
  getDownloadTransportCapabilities,
} = await import("../src/features/hub/download-manager/api.ts");
const { updateHubSource } = await import(
  "../src/features/settings/api/hub-settings.ts"
);
const { subscribeDownloadTransportSettings } = await import(
  "../src/features/settings/api/download-transport.ts"
);

const CAPS_URL = "/api/studio/download-transport-capabilities";
let source: "huggingface" | "modelscope" = "modelscope";
let capsCalls = 0;

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

setAuthFetchHandler((input, init) => {
  const url = String(input);
  if (url.startsWith(CAPS_URL)) {
    capsCalls += 1;
    const xet = source === "huggingface";
    return json({
      http: { available: true, reason: null },
      xet: {
        available: xet,
        reason: xet ? null : "ModelScope does not offer Xet",
      },
      auto_resolves_to: xet ? "xet" : "http",
      auto_reason: null,
    });
  }
  if (url === "/api/settings/download-transport") {
    const xet = source === "huggingface";
    return json({
      mode: "auto",
      xet_available: xet,
      xet_unavailable_reason: xet ? null : "ModelScope does not offer Xet",
      auto_resolves_to: xet ? "xet" : "http",
      auto_reason: null,
    });
  }
  if (url === "/api/settings/hub/source") {
    source = JSON.parse(String(init?.body)).source;
    return json({
      hf_endpoint: "",
      datasets_server_follows_endpoint: false,
      source,
      active_source: source,
    });
  }
  return json({});
});

test("switching the model source drops the cached Xet verdict", async () => {
  clearDownloadTransportCapabilitiesCache();
  source = "modelscope";
  assert.equal((await getDownloadTransportCapabilities()).xet.available, false);

  await updateHubSource("huggingface");
  const before = capsCalls;
  const caps = await getDownloadTransportCapabilities();

  assert.equal(capsCalls, before + 1, "the ModelScope verdict was replayed");
  assert.equal(caps.xet.available, true);
});

test("an answer that lands after the switch is not cached", async () => {
  clearDownloadTransportCapabilitiesCache();
  source = "modelscope";
  const stale = getDownloadTransportCapabilities();
  clearDownloadTransportCapabilitiesCache();
  source = "huggingface";
  await stale;

  assert.equal((await getDownloadTransportCapabilities()).xet.available, true);
});

test("switching the model source refreshes mounted transport controls", async () => {
  source = "modelscope";
  const seen: boolean[] = [];
  const unsubscribe = subscribeDownloadTransportSettings((settings) => {
    seen.push(settings.xetAvailable);
  });
  await updateHubSource("huggingface");
  unsubscribe();

  assert.deepEqual(
    seen,
    [true],
    "the Settings row and hub toggle kept the ModelScope verdict",
  );
});

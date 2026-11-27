// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// WebView2 persists a deny with no settings UI, so Allow resets it first (best effort).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { readSrc, readText } from "./helpers/kit.ts";

register("./helpers/tauri-core-resolver.mjs", import.meta.url);

// Use a file:// URL: Windows paths fail ESM import, and ?bust=N only works on a URL.
const MODULE = new URL(
  "../src/features/settings/api/microphone-permission.ts",
  import.meta.url,
).href;

const VOICE_TAB = readSrc("features/settings/tabs/voice-tab.tsx");

const ADAPTER = readSrc("features/chat/adapters/studio-web-speech-dictation-adapter.ts");

const MAIN_RS = readText("../../src-tauri/src/main.rs");

type StubControl = { calls: { command: string }[]; mode: "ok" | "rejects" };

let generation = 0;

async function load(options: { tauri: boolean; mode?: "ok" | "rejects" }) {
  const control: StubControl = { calls: [], mode: options.mode ?? "ok" };
  Object.defineProperty(globalThis, "__TAURI_CORE_STUB__", {
    value: control,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, "window", {
    value: {
      location: { protocol: options.tauri ? "tauri:" : "https:" },
      ...(options.tauri ? { __TAURI_INTERNALS__: {} } : {}),
    },
    configurable: true,
    writable: true,
  });

  generation += 1;
  const module = (await import(`${MODULE}?bust=${generation}`)) as {
    resetMicrophonePermission: () => Promise<void>;
  };
  return { module, control };
}

test("the desktop app clears the saved answer before asking again", async () => {
  const { module, control } = await load({ tauri: true });

  await module.resetMicrophonePermission();

  assert.deepEqual(
    control.calls.map((call) => call.command),
    ["reset_microphone_permission"],
    "without this call WebView2 refuses the request from its stored deny and never prompts",
  );
});

test("a browser tab calls no command", async () => {
  const { module, control } = await load({ tauri: false });

  await module.resetMicrophonePermission();

  assert.deepEqual(control.calls, [], "invoke does not exist outside the desktop app");
});

test("a runtime without the permission API does not break the request", async () => {
  const { module, control } = await load({ tauri: true, mode: "rejects" });

  // Must not reject: the caller goes on to getUserMedia, which may still prompt.
  await module.resetMicrophonePermission();

  assert.equal(control.calls.length, 1);
});

test("Allow microphone resets before it requests", () => {
  const reset = VOICE_TAB.indexOf("resetMicrophonePermission()");
  const request = VOICE_TAB.indexOf("navigator.mediaDevices.getUserMedia");

  assert.ok(reset > 0, "voice-tab no longer clears the saved microphone answer");
  assert.ok(
    reset < request,
    "the reset runs after the request, so the stored deny still rejects it",
  );
});

test("the blocked message stops pointing desktop users at a padlock", () => {
  assert.ok(
    ADAPTER.includes("Open Settings > Voice and click Allow microphone"),
    "the desktop dictation error still names browser site permissions, which WebView2 has no UI for",
  );
  assert.ok(
    ADAPTER.includes("isTauri"),
    "the message must stay browser-accurate off the desktop app",
  );
});

test("the blocked toast only promises another prompt on the desktop", () => {
  assert.match(
    VOICE_TAB,
    /isTauri\s*\?\s*"settings\.voice\.dictation\.micAccessBlockedDesktop"\s*:\s*"settings\.voice\.dictation\.micAccessBlocked"/,
    "the toast must pick the message for the platform it is running on",
  );
});

test("the browser message still points at the page permission", () => {
  const en = readSrc("i18n/locales/en.ts");
  const blocked = en.match(/micAccessBlocked:\s*\n?\s*"([^"]+)"/);
  assert.ok(blocked, "micAccessBlocked is missing");
  assert.match(blocked[1], /for this Unsloth page/);
});

test("every locale carries both blocked messages", () => {
  const locales = [
    "ar", "de", "en", "es", "fr", "hi",
    "it", "ja", "ko", "pt-br", "ru", "sv", "zh-CN",
  ];
  for (const locale of locales) {
    const source = readFileSync(
      fileURLToPath(new URL(`../src/i18n/locales/${locale}.ts`, import.meta.url)),
      "utf8",
    );
    // Strict parity is enforced in CI, so a key added to en only fails the build.
    assert.ok(
      source.includes("micAccessBlockedDesktop:"),
      `${locale} is missing micAccessBlockedDesktop`,
    );
    assert.ok(
      source.includes("micAccessBlocked:"),
      `${locale} is missing micAccessBlocked`,
    );
  }
});

test("the command is registered with the app", () => {
  assert.ok(
    MAIN_RS.includes("webview_permissions::reset_microphone_permission"),
    "reset_microphone_permission is missing from the invoke handler",
  );
});

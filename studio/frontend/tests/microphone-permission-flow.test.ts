// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The desktop app asks for the microphone with its own dialog once, remembers the
// answer, and never lets WebView2 show a second, differently styled prompt.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { readSrc, readText } from "./helpers/kit.ts";

register("./helpers/tauri-core-resolver.mjs", import.meta.url);

const MODULE = new URL(
  "../src/features/microphone/permission-store.ts",
  import.meta.url,
).href;

const VOICE_TAB = readSrc("features/settings/tabs/voice-tab.tsx");
const SETTINGS = readSrc(
  "features/settings/components/voice-typing-settings-section.tsx",
);
const AUDIO_SOURCE = readSrc("features/audio/hooks/use-audio-source.ts");
const WEB_SPEECH = readSrc(
  "features/chat/adapters/studio-web-speech-dictation-adapter.ts",
);
const PROVIDER = readSrc("app/provider.tsx");
const CONTROLLER = readSrc("features/voice-typing/voice-typing-controller.ts");
const MAIN_RS = readText("../../src-tauri/src/main.rs");
const WEBVIEW_RS = readText("../../src-tauri/src/webview_permissions.rs");

type StubControl = {
  calls: { command: string; args?: Record<string, unknown> }[];
  mode: "ok" | "rejects";
  result?: unknown;
};

let generation = 0;

async function load(options: { tauri: boolean; state?: string }) {
  const control: StubControl = {
    calls: [],
    mode: "ok",
    result: options.state ?? "unknown",
  };
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
    useMicrophonePermissionStore: {
      getState: () => {
        permission: string;
        dialogOpen: boolean;
        request: (reason?: string) => Promise<boolean>;
        askAgain: (reason?: string) => Promise<boolean>;
        allow: () => Promise<void>;
        deny: () => Promise<void>;
        load: () => Promise<void>;
      };
    };
  };
  return { module, control };
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

test("a grant is remembered and never prompts again", async () => {
  const { module, control } = await load({ tauri: true });
  const store = module.useMicrophonePermissionStore;

  const pending = store.getState().request("voice-typing");
  await wait(10);
  assert.equal(store.getState().dialogOpen, true, "the branded dialog opens");

  await store.getState().allow();
  assert.equal(await pending, true);
  assert.equal(store.getState().permission, "granted");

  const callsAfterGrant = control.calls.length;
  assert.equal(await store.getState().request("voice-typing"), true);
  assert.equal(store.getState().dialogOpen, false, "no second dialog");
  assert.equal(
    control.calls.length,
    callsAfterGrant,
    "a remembered grant does not ask the engine again",
  );
  assert.ok(
    control.calls.some(
      (call) =>
        call.command === "set_microphone_permission" &&
        call.args?.state === "granted",
    ),
    "the answer is persisted for the native gate",
  );
});

test("a deny is remembered and does not reopen the dialog", async () => {
  const { module, control } = await load({ tauri: true, state: "denied" });
  const store = module.useMicrophonePermissionStore;

  assert.equal(await store.getState().request("dictation"), false);
  assert.equal(store.getState().dialogOpen, false);
  assert.deepEqual(
    control.calls.map((call) => call.command),
    ["microphone_permission_state"],
    "a saved deny is read, not asked again",
  );
});

test("ask again clears the saved answer before asking", async () => {
  const { module, control } = await load({ tauri: true, state: "denied" });
  const store = module.useMicrophonePermissionStore;
  await store.getState().load();

  const pending = store.getState().askAgain("settings");
  await wait(10);
  assert.equal(store.getState().dialogOpen, true);
  assert.deepEqual(
    control.calls.map((call) => call.command),
    ["microphone_permission_state", "reset_microphone_permission"],
    "the reset runs before the new question",
  );

  await store.getState().allow();
  assert.equal(await pending, true);
});

test("a browser tab leaves the answer to the browser", async () => {
  const { module, control } = await load({ tauri: false });

  assert.equal(
    await module.useMicrophonePermissionStore.getState().request("dictation"),
    true,
  );
  assert.deepEqual(control.calls, [], "there is no native command to call");
});

test("every recorder uses the shared permission flow", () => {
  assert.ok(
    VOICE_TAB.includes("ensureMicrophonePermission"),
    "the Voice tab asks through the branded dialog",
  );
  assert.ok(
    !VOICE_TAB.includes("resetMicrophonePermission"),
    "the old reset-before-request path must not come back",
  );
  assert.ok(
    AUDIO_SOURCE.includes('ensureMicrophonePermission("audio")'),
    "audio recording asks before the native gate sees getUserMedia",
  );
  assert.ok(
    WEB_SPEECH.includes('ensureMicrophonePermission("dictation")'),
    "browser dictation asks through the same dialog",
  );
  assert.ok(
    PROVIDER.includes("<MicrophonePermissionDialog />"),
    "the dialog is mounted once for the whole app",
  );
  assert.ok(
    VOICE_TAB.includes("askAgain: permission === \"denied\""),
    "the Voice tab offers an explicit retry after a deny",
  );
  assert.ok(
    !SETTINGS.includes("MIC_WARM_ALWAYS") &&
      !SETTINGS.includes("hudEnabled") &&
      !SETTINGS.includes("keepOnClipboard"),
    "removed settings must not linger in the section",
  );
});

test("a shortcut press never opens the consent dialog", () => {
  assert.ok(
    CONTROLLER.includes("const permission = await readMicrophonePermission()"),
    "the shortcut path only reads the saved answer",
  );
  assert.ok(
    CONTROLLER.includes("void this.activate(true)"),
    "switching voice typing on is the explicit gesture that asks once",
  );
});

test("the native gate answers WebView2 from the stored choice", () => {
  assert.ok(
    WEBVIEW_RS.includes("add_PermissionRequested"),
    "WebView2's own prompt is intercepted",
  );
  assert.ok(
    WEBVIEW_RS.includes("COREWEBVIEW2_PERMISSION_KIND_MICROPHONE"),
    "only the microphone request is gated",
  );
  assert.ok(
    WEBVIEW_RS.includes("SetSavesInProfile(false)"),
    "the gate stays authoritative instead of WebView2 remembering a deny",
  );
  assert.ok(
    MAIN_RS.includes("webview_permissions::init"),
    "the gate is installed at startup",
  );
  for (const command of [
    "microphone_permission_state",
    "set_microphone_permission",
    "reset_microphone_permission",
  ]) {
    assert.ok(
      MAIN_RS.includes(`webview_permissions::${command}`),
      `${command} is missing from the invoke handler`,
    );
  }
});

test("every locale carries the permission dialog strings", () => {
  const locales = [
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "pt-br",
    "ru",
    "sv",
    "zh-CN",
  ];
  for (const locale of locales) {
    const source = readFileSync(
      fileURLToPath(
        new URL(`../src/i18n/locales/${locale}.ts`, import.meta.url),
      ),
      "utf8",
    );
    for (const key of [
      "permissionTitle:",
      "permissionAllow:",
      "permissionNotNow:",
      "permissionReasonVoiceTyping:",
      "permissionReasonAudio:",
    ]) {
      assert.ok(source.includes(key), `${locale} is missing ${key}`);
    }
  }
});

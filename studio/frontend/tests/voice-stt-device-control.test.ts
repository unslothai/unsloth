// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const voiceTab = readSrc("features/settings/tabs/voice-tab.tsx");
const adapter = readSrc("features/chat/adapters/studio-model-dictation-adapter.ts");
const voiceStore = readSrc("features/settings/stores/voice-settings-store.ts");
const sttDeviceApi = readSrc("features/settings/api/stt-device.ts");
const root = readSrc("app/routes/__root.tsx");

test("changing the dictation device releases only this tab's model", () => {
  // Unscoped, the unload can tear down a model this tab never owned.
  assert.match(
    voiceTab,
    /void unloadSttModel\(sttEngineFor\(sttModel\), sttModel, \{\s*wait: false,\s*\}\)/,
  );
});

test("switching device never kills a transcription being decoded", () => {
  // The default drains for 30s and then releases, discarding the recording.
  assert.match(voiceTab, /wait: false/);
  assert.match(adapter, /if \(options\?\.wait === false\) params\.set\("wait", "false"\)/);
});

test("the unload API still takes the engine and model that scoping needs", () => {
  assert.match(
    adapter,
    /export function unloadSttModel\(\s*engine\?: SttEngine,\s*model\?: string,\s*options\?: \{ wait\?: boolean \},\s*\)/,
  );
  assert.match(adapter, /if \(model\) params\.set\("model", model\)/);
});

test("the device preference travels with every load and transcribe", () => {
  // An omitted device reads as "no opinion", so the setting never applies.
  assert.match(adapter, /device: resolvedDevice/);
  assert.match(
    adapter,
    /params\.set\("device", options\.device \?\? settings\.sttDevice\)/,
  );
});

test("Load into is synced to the server for cold API STT loads", () => {
  // Browser localStorage alone left /v1/audio/transcriptions on auto/GPU (#15280).
  assert.match(sttDeviceApi, /\/api\/settings\/stt-device/);
  assert.match(sttDeviceApi, /from "@\/features\/auth\/api"/);
  // One save at a time, so a quick Auto then CPU cannot commit Auto last.
  assert.match(sttDeviceApi, /pending = pending\s*\.then\(/);
  assert.match(
    voiceStore,
    /set\(\{ sttDevice \}\);\s*\/\/[^\n]*\n\s*syncSttDevicePreference\(sttDevice\)/,
  );
  // Startup sync waits for sign-in (desktop port known), then the saved server copy wins over this browser's.
  assert.doesNotMatch(voiceStore, /onRehydrateStorage/);
  assert.match(
    root,
    /function SttDeviceSyncMount[\s\S]*?hasSettledAuthSession\(\)[\s\S]*?if \(!settled\) return;[\s\S]*?reconcileSttDevicePreference\(/,
  );
  // Adopting the server copy must not send it back.
  assert.match(root, /useVoiceSettingsStore\.setState\(\{ sttDevice \}\)/);
  assert.match(
    sttDeviceApi,
    /if \(state\.saved\) \{\s*adopt\([\s\S]*?else if \(local !== "auto"\) \{\s*syncSttDevicePreference\(local\)/,
  );
  assert.match(root, /<SttDeviceSyncMount \/>/);
});

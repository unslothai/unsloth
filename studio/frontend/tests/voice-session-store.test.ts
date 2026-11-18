// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type * as SessionStore from "../src/features/chat/voice/voice-session-store.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function load(unloadStt: () => Promise<void>) {
  const calls: string[] = [];
  const runtime = {
    setVadSlotLoaded: (loaded: boolean) => calls.push(`vad slot ${loaded}`),
    setSttSlotLoaded: (loaded: boolean) => calls.push(`stt slot ${loaded}`),
    setSttSlotDownload: (download: unknown) => calls.push(`stt download ${download}`),
  };
  const mod = loadWithStubs<typeof SessionStore>(
    new URL("../src/features/chat/voice/voice-session-store.ts", import.meta.url),
    {
      "@/i18n": { translate: (key: string) => key },
      "@/lib/toast": { toast: { info() {}, error() {} } },
      "../adapters/studio-model-dictation-adapter.ts": {
        unloadSttModel: () => {
          calls.push("unload stt");
          return unloadStt();
        },
      },
      "../stores/chat-runtime-store.ts": { useChatRuntimeStore: { getState: () => runtime } },
      "./turn-end.ts": { unloadTurnModel: async () => void calls.push("unload turn") },
      "./vad-client.ts": { unloadVadModel: async () => void calls.push("unload vad") },
      "./voice-loop-bridge.ts": { loadVoiceMode: async () => ({}) },
    },
  );
  return { mod, calls };
}

test("releasing the listening models unloads the transcriber, the speech detector and the turn judge", () => {
  const { mod, calls } = load(async () => {});
  mod.storeVoiceSessionPorts.releaseListening();
  assert.deepEqual(calls, [
    "unload stt",
    "unload vad",
    "unload turn",
    "vad slot false",
    "stt slot false",
    "stt download null",
  ]);
});

test("a transcriber unload that fails still releases the rest and raises nothing", async () => {
  const { mod, calls } = load(() => Promise.reject(new Error("server gone")));
  mod.releaseVoiceModels();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.ok(calls.includes("unload vad") && calls.includes("unload turn"));
});

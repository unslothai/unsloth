// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import * as lifecycle from "../src/lib/llama-job-lifecycle.ts";
import type * as HookModule from "../src/hooks/use-llama-update-check.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { offerKey, offerKeyMatches } = loadWithStubs<typeof HookModule>(
  new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
  {
    react: {},
    "@/features/auth": {
      authFetch: async () => null,
      getAuthToken: () => null,
    },
    "@/hooks/use-hardware-info": { refreshHardwareInfo: async () => {} },
    "@/lib/model-lifecycle-events": { notifyModelLifecycle() {} },
    "@/lib/llama-job-events": {
      signalRunningLlamaJob() {},
      subscribeToLlamaJobStarted: () => () => {},
    },
    "@/lib/llama-job-lifecycle": lifecycle,
  },
);

const none: HookModule.ComponentOffer = {
  update_available: false,
  installed_tag: null,
  latest_tag: null,
  update_size_bytes: null,
};

function status(audio: HookModule.ComponentOffer | null) {
  return {
    llama: {
      ...none,
      update_available: true,
      installed_tag: "b1",
      latest_tag: "b2",
    },
    whisper: null,
    audio,
    backend_migration_available: false,
  } as unknown as HookModule.LlamaUpdateStatus;
}

test("an audio.cpp offer changes the suppression key, so a dismissed llama offer does not hide it", () => {
  const audio = {
    ...none,
    update_available: true,
    installed_tag: "v0.8.2",
    latest_tag: "v0.9.0",
  };
  assert.notEqual(offerKey(status(audio)), offerKey(status(null)));
  assert.notEqual(
    offerKey(status(audio)),
    offerKey(status({ ...audio, latest_tag: "v0.9.1" })),
  );
});

test("without an audio.cpp offer the key matches keys stored before audio joined the card", () => {
  assert.equal(
    offerKey(status(null)),
    JSON.stringify({ llama: ["b1", "b2"], whisper: null, migration: null }),
  );
  assert.equal(offerKey(status({ ...none })), offerKey(status(null)));
});

test("a new whisper.cpp release does not change the suppression key", () => {
  const withWhisper = {
    ...status(null),
    whisper: { ...none, update_available: true, installed_tag: "v1.9.5-unsloth.4", latest_tag: "v1.9.5-unsloth.5" },
  } as unknown as HookModule.LlamaUpdateStatus;
  assert.equal(offerKey(withWhisper), offerKey(status(null)));
});

test("a suppression stored with the whisper.cpp release still matches the same offer", () => {
  const withWhisper = {
    ...status(null),
    whisper: { ...none, update_available: true, installed_tag: "v1.9.5-unsloth.4", latest_tag: "v1.9.5-unsloth.5" },
  } as unknown as HookModule.LlamaUpdateStatus;
  const stored = JSON.stringify({
    llama: ["b1", "b2"],
    whisper: ["v1.9.5-unsloth.4", "v1.9.5-unsloth.5"],
    migration: null,
  });
  assert.equal(offerKeyMatches(stored, withWhisper), true);
  assert.equal(offerKeyMatches(offerKey(withWhisper), withWhisper), true);
  const newerWhisper = {
    ...withWhisper,
    whisper: { ...withWhisper.whisper, latest_tag: "v1.9.5-unsloth.6" },
  } as unknown as HookModule.LlamaUpdateStatus;
  assert.equal(offerKeyMatches(stored, newerWhisper), true);
  const newerLlama = { ...withWhisper, llama: { ...withWhisper.llama, latest_tag: "b3" } };
  assert.equal(offerKeyMatches(stored, newerLlama), false);
});

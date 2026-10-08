// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import * as lifecycle from "../src/lib/llama-job-lifecycle.ts";
import type * as HookModule from "../src/hooks/use-llama-update-check.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { offerKey } = loadWithStubs<typeof HookModule>(
  new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
  {
    react: {},
    "@/features/auth": {
      authFetch: async () => null,
      getAuthToken: () => null,
    },
    "@/hooks/use-hardware-info": { refreshHardwareInfo: async () => {} },
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

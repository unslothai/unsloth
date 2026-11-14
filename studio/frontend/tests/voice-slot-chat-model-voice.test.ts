// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import * as voiceModels from "../src/features/chat/voice/voice-models.ts";

test("the chat model's voice is checked on the selected model, kept slot or primary", async () => {
  const asked: (string | undefined)[] = [];
  const slot = loadWithStubs<typeof import("../src/features/chat/voice/voice-slot.ts")>(
    new URL("../src/features/chat/voice/voice-slot.ts", import.meta.url),
    {
      "@/features/chat/api/chat-api": {
        getInferenceStatus: async (_signal: unknown, model?: string) => {
          asked.push(model);
          return { speaks: model === "LiquidAI/LFM3-Audio-2B" };
        },
        loadModel: async () => ({}),
        unloadModel: async () => ({}),
      },
      "@/features/chat/stores/chat-runtime-store": {
        useChatRuntimeStore: { getState: () => ({ params: { checkpoint: "LiquidAI/LFM3-Audio-2B" } }) },
      },
      "@/features/hub/stores/hf-token-store": { getHfToken: () => null, hfApiToken: () => null },
      "@/i18n": { translate: (key: string) => key },
      "./voice-models": voiceModels,
    },
  );
  assert.equal(await slot.voiceServing(voiceModels.VOICE_CHAT_MODEL_VOICE), true);
  assert.deepEqual(asked, ["LiquidAI/LFM3-Audio-2B"]);
});

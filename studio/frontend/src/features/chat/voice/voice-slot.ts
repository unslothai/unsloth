// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Voice mode's speaking model, kept loaded beside the chat model in one of the
// backend's extra model slots (LoadRequest.voice keeps it there with multi-model off).

import {
  getInferenceStatus,
  loadModel,
  unloadModel,
} from "@/features/chat/api/chat-api";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { voiceIdServes } from "./voice-models";

export async function voiceServing(id: string): Promise<boolean> {
  const status = await getInferenceStatus();
  return (status.serving ?? status.loaded ?? []).some((s) => voiceIdServes(s, id));
}

export async function loadVoice(id: string): Promise<void> {
  await loadModel(
    {
      model_path: id,
      hf_token: hfApiToken(getHfToken()) ?? null,
      max_seq_length: 0,
      load_in_4bit: false,
      is_lora: false,
      voice: true,
    },
    { runtime: "tts" },
  );
}

export async function unloadVoice(id: string): Promise<void> {
  if (await voiceServing(id)) await unloadModel({ model_path: id });
}

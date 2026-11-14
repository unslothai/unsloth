// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import {
  getInferenceStatus,
  loadModel,
  unloadModel,
} from "@/features/chat/api/chat-api";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { voiceIdServes, voiceLoadNeedsCpu } from "./voice-models";

export async function voiceServing(id: string): Promise<boolean> {
  const status = await getInferenceStatus();
  return (status.serving ?? status.loaded ?? []).some((s) => voiceIdServes(s, id));
}

export async function loadVoice(id: string): Promise<void> {
  const load = (audio_device: "auto" | "cpu") =>
    loadModel(
      {
        model_path: id,
        hf_token: hfApiToken(getHfToken()) ?? null,
        max_seq_length: 0,
        load_in_4bit: false,
        is_lora: false,
        voice: true,
        audio_device,
      },
      // Not "tts": chat treats a tts load as the Audio page taking its slot and drops its model.
      { runtime: "chat" },
    );
  try {
    await load("auto");
  } catch (error) {
    if (!voiceLoadNeedsCpu(error)) throw error;
    await load("cpu");
  }
}

export async function unloadVoice(id: string): Promise<void> {
  // Only voice mode speaks with it, so an in-flight sentence is cancelled rather than a 409.
  if (await voiceServing(id)) await unloadModel({ model_path: id, force_cancel_active: true });
}

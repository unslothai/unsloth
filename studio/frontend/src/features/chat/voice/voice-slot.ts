// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import {
  getInferenceStatus,
  loadModel,
  unloadModel,
} from "@/features/chat/api/chat-api";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { translate } from "@/i18n";
import {
  isChatModelVoice,
  servedVoiceTtsModel,
  voiceIdServes,
  voiceLoadNeedsCpu,
} from "./voice-models";

// Older servers report "loaded" instead of "serving".
const servingList = (status: Awaited<ReturnType<typeof getInferenceStatus>>) =>
  status.serving ?? status.loaded ?? [];

export async function voiceServing(id: string): Promise<boolean> {
  if (isChatModelVoice(id)) {
    const checkpoint = useChatRuntimeStore.getState().params.checkpoint || undefined;
    if (!checkpoint) return false;
    const status = await getInferenceStatus(undefined, checkpoint);
    // An unserved (hosted) model gets the primary's status back: check routing before trusting speaks.
    const want = checkpoint.toLowerCase();
    const routed = [status.model_identifier, status.active_model].some((id) => id?.toLowerCase() === want);
    return routed && Boolean(status.speaks);
  }
  const status = await getInferenceStatus();
  return servingList(status).some((s) => voiceIdServes(s, id));
}

export async function servedVoice(): Promise<string | null> {
  const status = await getInferenceStatus();
  return servedVoiceTtsModel(servingList(status));
}

export async function loadVoice(id: string): Promise<void> {
  if (isChatModelVoice(id)) {
    if (await voiceServing(id)) return;
    throw new Error(translate("voicePanel.chatModelCannotSpeak"));
  }
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
  if (isChatModelVoice(id)) return;
  // Only voice mode speaks with it, so an in-flight sentence is cancelled rather than a 409.
  if (await voiceServing(id)) await unloadModel({ model_path: id, force_cancel_active: true });
}

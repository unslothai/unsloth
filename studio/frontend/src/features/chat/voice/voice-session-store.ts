// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { unloadSttModel } from "../adapters/studio-model-dictation-adapter.ts";
import { useChatRuntimeStore } from "../stores/chat-runtime-store.ts";
import { unloadTurnModel } from "./turn-end.ts";
import { unloadVadModel } from "./vad-client.ts";
import { endVoiceOnSignOut, loadVoiceMode } from "./voice-loop-bridge.ts";
import type { VoiceSessionPorts } from "./voice-session.ts";

// Voice mode's listening models: the transcriber, the speech detector (it has no idle unload) and the turn judge.
export function releaseVoiceModels(): void {
  void unloadSttModel().catch(() => {});
  void unloadVadModel();
  void unloadTurnModel();
  useChatRuntimeStore.getState().setVadSlotLoaded(false);
  useChatRuntimeStore.getState().setSttSlotLoaded(false);
  useChatRuntimeStore.getState().setSttSlotDownload(null);
}

export const storeVoiceSessionPorts: VoiceSessionPorts = {
  state: () => {
    const s = useChatRuntimeStore.getState();
    return {
      voiceMode: s.voiceMode,
      selectedVoiceModelId: s.selectedVoiceModelId,
      checkpoint: s.params.checkpoint,
      modelLoading: s.modelLoading,
    };
  },
  setSelectedVoiceModelId: (id) => useChatRuntimeStore.getState().setSelectedVoiceModelId(id),
  setVoiceSlotLoaded: (loaded) => useChatRuntimeStore.getState().setVoiceSlotLoaded(loaded),
  setVoiceSlotLoading: (loading) => useChatRuntimeStore.getState().setVoiceSlotLoading(loading),
  slot: loadVoiceMode,
  releaseListening: releaseVoiceModels,
  endVoice: () => {
    endVoiceOnSignOut();
    useChatRuntimeStore.getState().setVoiceMode("off");
  },
  chatModelCannotSpeak: () => toast.info(translate("voicePanel.chatModelCannotSpeak")),
  loadFailed: (error) =>
    toast.error(translate("voiceControls.loadFailed"), {
      description: error instanceof Error ? error.message : undefined,
    }),
};

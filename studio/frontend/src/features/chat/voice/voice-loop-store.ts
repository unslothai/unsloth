// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useVoiceSettingsStore } from "../../settings/stores/voice-settings-store.ts";
import { useChatRuntimeStore } from "../stores/chat-runtime-store.ts";
import type { VoiceLoopPorts } from "./voice-loop-ports.ts";

const runtime = () => useChatRuntimeStore.getState();

export const storeVoiceLoopPorts: VoiceLoopPorts = {
  runtime,
  settings: () => useVoiceSettingsStore.getState(),
  onMicDeviceChange: (listener) =>
    useVoiceSettingsStore.subscribe((state, prev) => {
      if (state.micDeviceId !== prev.micDeviceId) listener(state.micDeviceId);
    }),
  setHearing: (hearing) => runtime().setVoiceHearing(hearing),
  setTranscribing: (transcribing) => runtime().setVoiceTranscribing(transcribing),
  setHeldForEdit: (held) => runtime().setVoiceHeldForEdit(held),
  setSttSlotLoaded: (loaded) => runtime().setSttSlotLoaded(loaded),
  setSttSlotDownload: (state) => runtime().setSttSlotDownload(state),
  setVadSlotLoaded: (loaded) => runtime().setVadSlotLoaded(loaded),
  useVadFallback: (id) => useChatRuntimeStore.setState({ selectedVadModelId: id }),
};

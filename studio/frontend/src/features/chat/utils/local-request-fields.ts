// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isServedByLlamaCpp } from "@/features/model-picker/model-config/per-model-config";
import { isNpuModelId } from "@/features/npu";
import type { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { ggufCompactionRequestFields } from "./auto-compaction";

type ChatRuntimeState = ReturnType<typeof useChatRuntimeStore.getState>;

/** The compaction fields a local completion sends for the loaded model; rolling for character chats. */
export function localCompactionFields(
  state: ChatRuntimeState,
  rolling: boolean,
): ReturnType<typeof ggufCompactionRequestFields> {
  return ggufCompactionRequestFields({
    isGguf: isServedByLlamaCpp({
      loadedIsGguf: state.loadedIsGguf,
      activeGgufVariant: state.activeGgufVariant,
      activeNativePathToken: state.activeNativePathToken,
      checkpoint: state.params.checkpoint,
    }),
    isMlx: state.loadedIsMlx === true,
    isManagedEngine: state.loadedEngine !== "auto",
    isNpu: isNpuModelId(state.params.checkpoint),
    autoCompactEnabled: state.autoCompactEnabled,
    rolling,
  });
}

/** A local completion's sampling fields, as the chat sends them. */
export function localSamplingFields(params: ChatRuntimeState["params"]) {
  return {
    temperature: params.temperature,
    top_p: params.topP,
    top_k: params.topK,
    min_p: params.minP,
    repetition_penalty: params.repetitionPenalty,
    presence_penalty: params.presencePenalty,
  };
}

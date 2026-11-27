// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { completeChat } from "@/features/chat/api/chat-api";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import { servedCheckpoint } from "@/features/hub/lib/model-identity";
import { localCompactionFields, localSamplingFields } from "@/features/chat/utils/local-request-fields";
import { getLocale, translate } from "@/i18n";
import { loadParticipants, promptHistoryOf, toPromptCharacter } from "../chat-runtime";
import { buildImpersonationMessages, stripSpeakerPrefix } from "../prompt/build";
import { useCharactersStore } from "../store";

/** Writes the persona's next line with the loaded model, for the user to edit and send. */
export async function writeMyReply(threadId: string, messages: readonly ThreadMessage[], signal?: AbortSignal): Promise<string> {
  const scene = useCharactersStore.getState().scenes[threadId];
  if (!scene) throw new Error(translate("characters.chat.noCast", undefined, getLocale()));
  const { cast, persona } = await loadParticipants(scene);
  // Pinned like a character reply's prompt, so a draft reuses the same cached system block.
  const { input } = promptHistoryOf(threadId, messages);
  const prompt = buildImpersonationMessages({
    cast: cast.map(toPromptCharacter),
    persona: { name: persona.name, description: persona.description },
    scene: scene.settings,
    ...input,
  });
  const state = useChatRuntimeStore.getState();
  const { params } = state;
  const text = await completeChat(
    {
      // A quant loaded beside its repo's other quant answers only to its own name.
      model: servedCheckpoint(state.loadedModels, params.checkpoint, state.activeGgufVariant),
      messages: [{ role: "system", content: prompt.system }, ...prompt.messages],
      max_tokens: 320,
      ...localSamplingFields(params),
      stop: prompt.stop,
      enable_thinking: false,
      // A draft must never run tools: --enable-tools overrides enable_tools, so withdraw the catalog too.
      enable_tools: false,
      tool_choice: "none",
      enabled_tools: [],
      // Fitted like a character reply, minus thread_id: a draft must not touch the thread's compaction state.
      ...localCompactionFields(state, true),
    },
    { signal, failure: translate("characters.chat.writeFailed", undefined, getLocale()) },
  );
  return stripSpeakerPrefix(text, prompt.user).trim();
}

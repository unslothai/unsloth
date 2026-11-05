// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { authFetch } from "@/features/auth";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { ggufCompactionRequestFields } from "@/features/chat/utils/auto-compaction";
import { isServedByLlamaCpp } from "@/features/model-picker/model-config/per-model-config";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { loadParticipants, textOf, toPromptCharacter } from "../chat-runtime";
import { roleplayMetaOf } from "../meta";
import { buildImpersonationMessages, stripSpeakerPrefix } from "../prompt/build";
import { useCharactersStore } from "../store";

/** Writes the persona's next line with the loaded model, for the user to edit and send. */
export async function writeMyReply(threadId: string, messages: readonly ThreadMessage[], signal?: AbortSignal): Promise<string> {
  const scene = useCharactersStore.getState().scenes[threadId];
  if (!scene) throw new Error("This chat has no characters.");
  const { cast, persona } = await loadParticipants(scene);
  const chat = messages.filter((m) => m.role === "user" || m.role === "assistant");
  const prompt = buildImpersonationMessages({
    cast: cast.map(toPromptCharacter),
    persona: { name: persona.name, description: persona.description },
    scene: scene.settings,
    history: chat.map((m) => ({ role: m.role as "user" | "assistant", text: textOf(m), meta: roleplayMetaOf(m) })),
  });
  const state = useChatRuntimeStore.getState();
  const { params } = state;
  const response = await authFetch("/api/inference/chat/completions", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: params.checkpoint,
      messages: [{ role: "system", content: prompt.system }, ...prompt.messages],
      stream: false,
      max_tokens: 320,
      temperature: params.temperature,
      top_p: params.topP,
      stop: prompt.stop,
      enable_thinking: false,
      // A draft must never run tools: --enable-tools overrides enable_tools, so withdraw the catalog too.
      enable_tools: false,
      tool_choice: "none",
      enabled_tools: [],
      // Fitted like a character reply, minus thread_id: a draft must not touch the thread's compaction state.
      ...ggufCompactionRequestFields({
        isGguf: isServedByLlamaCpp({
          loadedIsGguf: state.loadedIsGguf,
          activeGgufVariant: state.activeGgufVariant,
          activeNativePathToken: state.activeNativePathToken,
          checkpoint: params.checkpoint,
        }),
        isMlx: state.loadedIsMlx === true,
        isManagedEngine: state.loadedEngine !== "auto",
        autoCompactEnabled: state.autoCompactEnabled,
        rolling: true,
      }),
    }),
  });
  if (!response.ok) throw new Error(await readFastApiError(response, "Could not write a reply"));
  const data = (await response.json()) as { choices?: { message?: { content?: string | null } }[] };
  const text = data.choices?.[0]?.message?.content ?? "";
  return stripSpeakerPrefix(text, persona.name).trim();
}

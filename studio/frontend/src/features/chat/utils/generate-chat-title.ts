// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { streamChatCompletions } from "../api/chat-api";
import type { OpenAIChatCompletionsRequest } from "../types/api";
import { authFetch } from "@/features/auth";
import {
  encryptProviderApiKey,
  isProviderKeyRotationError,
} from "../api/providers-api";
import {
  getExternalProviderApiKey,
  parseExternalModelId,
  toExternalBackendProviderType,
} from "../external-providers";
import { useExternalProvidersStore } from "../stores/external-providers-store";

type TitleResponse = {
  choices?: Array<{
    finish_reason?: string | null;
    message?: {
      content?: string;
    };
  }>;
};

export async function generateChatTitle(
  conversation: string,
  model: string,
  signal?: AbortSignal,
  purpose: "initial" | "refresh" = "initial",
): Promise<string | null> {
  function normalizeTitle(raw: string): string | null {
    let title = raw.split(/\r?\n/, 1)[0] ?? "";
    title = title.replace(/^\s*title\s*:\s*/i, "");
    title = title.replace(/[^\x20-\x7E]+/g, " ");
    title = title.replace(/["'`]+/g, "");

    // reject echoed role labels before stripping punctuation.
    if (/^\s*(user|assistant|base|lora)\s*:/i.test(title)) {
      return null;
    }

    title = title.replace(/[.!?:;,]+/g, " ");
    title = title.replace(/\s+/g, " ").trim();

    const words = title.split(" ").filter(Boolean).slice(0, 6);
    const joined = words.join(" ").trim();
    if (!joined) return null;
    return joined.length > 60 ? joined.slice(0, 60).trimEnd() : joined;
  }

  const selection = parseExternalModelId(model);
  const providers = useExternalProvidersStore.getState();
  const provider = selection
    ? providers.providers.find(
        (candidate) => candidate.id === selection.providerId,
      )
    : undefined;
  if (selection && (!providers.connectionsEnabled || !provider)) return null;
  const apiKey =
    provider && !provider.hasApiKey
      ? getExternalProviderApiKey(provider.id).trim()
      : "";
  const external =
    provider && selection
      ? {
          provider_id: provider.id,
          provider_type: toExternalBackendProviderType(provider.providerType),
          external_model: selection.modelId,
          provider_base_url: provider.baseUrl || null,
        }
      : {};

  const payload: OpenAIChatCompletionsRequest = {
    model,
    ...external,
    stream: Boolean(provider),
    temperature: 0.2,
    top_p: 0.9,
    max_tokens: 24,
    top_k: 20,
    repetition_penalty: 1.0,
    enable_thinking: false,
    reasoning_effort:
      provider?.providerType === "openai_codex" ? undefined : "none",
    // title generation must not inherit the server's tools-on default.
    enable_tools: false,
    messages: [
      {
        role: "system",
        content:
          "Write 1 concise chat title summarizing the conversation topic, not the user's exact wording. Use the assistant reply as context when provided. " +
          (purpose === "refresh" ? "Reflect how the topic has evolved. " : "") +
          "Rules: 2-6 words, no quotes, no punctuation, ASCII only, do not echo input. Output title only.",
      },
      { role: "user", content: conversation },
    ],
  };
  let raw: string | undefined;
  if (payload.stream) {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      raw = "";
      try {
        if (apiKey) {
          payload.encrypted_api_key = await encryptProviderApiKey(
            apiKey,
            attempt > 0,
          );
        }
        for await (const chunk of streamChatCompletions(
          payload,
          signal ?? new AbortController().signal,
        )) {
          const choice = chunk.choices?.[0];
          if (choice?.finish_reason === "length") return null;
          raw += choice?.delta?.content ?? "";
        }
        break;
      } catch (error) {
        if (attempt > 0 || !apiKey || !isProviderKeyRotationError(error))
          return null;
      }
    }
  } else {
    const response = await authFetch("/v1/chat/completions", {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = (await response
      .json()
      .catch(() => null)) as TitleResponse | null;
    if (!response.ok) return null;
    const choice = body?.choices?.[0];
    if (choice?.finish_reason === "length") return null;
    raw = choice?.message?.content;
  }
  if (!raw || /<\/?think>/i.test(raw)) return null;
  return normalizeTitle(raw);
}

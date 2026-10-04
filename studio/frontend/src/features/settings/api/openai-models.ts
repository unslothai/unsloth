// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

import { chatModelsFromCatalog, type OpenAIModel } from "./openai-model-catalog";

export type { OpenAIModel };

/**
 * The chat-capable models this server can serve: `/v1/models` also lists image,
 * video and speech-to-text models, which `/v1/chat/completions` cannot resolve.
 */
export async function listOpenAIModels(): Promise<OpenAIModel[]> {
  const res = await authFetch("/v1/models");
  if (!res.ok) {
    throw new Error(`Failed to list models (${res.status})`);
  }
  return chatModelsFromCatalog(await res.json());
}

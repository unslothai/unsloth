// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub";
import { readFastApiError } from "@/lib/format-fastapi-error";

export interface ValidateChatTemplateResult {
  valid: boolean;
  error: string | null;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return response.json();
}

export async function validateChatTemplate(
  template: string,
  signal?: AbortSignal,
): Promise<ValidateChatTemplateResult> {
  const response = await authFetch("/api/picker/validate-chat-template", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template }),
    signal,
  });
  return parseJsonOrThrow<ValidateChatTemplateResult>(response);
}

export async function fetchDefaultChatTemplate(
  modelName: string,
  ggufVariant?: string | null,
  hfToken?: string | null,
  signal?: AbortSignal,
): Promise<string | null> {
  const query = ggufVariant
    ? `?gguf_variant=${encodeURIComponent(ggufVariant)}`
    : "";
  const response = await authFetch(
    `/api/picker/chat-template/${encodeURIComponent(modelName)}${query}`,
    { headers: hubTokenHeader(hfToken), signal },
  );
  const data = await parseJsonOrThrow<{
    model_name: string;
    chat_template: string | null;
  }>(response);
  return data.chat_template ?? null;
}

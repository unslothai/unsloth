// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub";
import { consumeNativePathToken } from "@/features/native-intents/api";
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
  nativePathToken?: string | null,
): Promise<string | null> {
  // A native (picked / drag-drop) GGUF lives at a path only its signed lease
  // knows, and the picker chat-template GET has no lease plumbing, so redeem a
  // one-shot validate-model lease and read the embedded template through the
  // lease-aware /api/inference/validate probe instead (mirrors the staged
  // header-dims fetch). Non-native models keep the plain GET path.
  if (nativePathToken) {
    let nativePathLease: string | null = null;
    try {
      nativePathLease = (
        await consumeNativePathToken(nativePathToken, "validate-model")
      ).nativePathLease;
    } catch {
      // Lease expired / revoked: no readable path, so no default template (the
      // subsequent load re-mints its own lease).
      return null;
    }
    const response = await authFetch("/api/inference/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_path: modelName,
        gguf_variant: ggufVariant ?? null,
        hf_token: hfToken ?? null,
        native_path_lease: nativePathLease,
        include_chat_template: true,
      }),
      signal,
    });
    const data = await parseJsonOrThrow<{ chat_template?: string | null }>(
      response,
    );
    return data.chat_template ?? null;
  }

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

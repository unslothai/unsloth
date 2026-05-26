// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export interface ChatTemplateValidationResult {
  valid: boolean;
  error: string | null;
}

export async function validateChatTemplate(
  template: string,
  signal?: AbortSignal,
): Promise<ChatTemplateValidationResult> {
  const response = await authFetch("/api/models/validate-chat-template", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template }),
    signal,
  });
  if (!response.ok) {
    throw new Error(`Failed to validate chat template (${response.status})`);
  }
  const data = (await response.json()) as Partial<ChatTemplateValidationResult>;
  return {
    valid: data.valid === true,
    error: typeof data.error === "string" ? data.error : null,
  };
}

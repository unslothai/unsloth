// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export async function renderConversationPdf(
  title: string,
  html: string,
  model?: string,
): Promise<Uint8Array> {
  const response = await authFetch("/api/chat-export/pdf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, html, model }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response, "PDF export failed"));
  }
  return new Uint8Array(await response.arrayBuffer());
}

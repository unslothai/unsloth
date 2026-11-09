// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export async function renderConversationPdf(
  title: string,
  html: string,
): Promise<Uint8Array> {
  const response = await authFetch("/api/chat-export/pdf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, html }),
  });
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: unknown }) => body?.detail)
      .catch(() => undefined);
    throw new Error(
      typeof detail === "string"
        ? detail
        : `PDF export failed (${response.status})`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}

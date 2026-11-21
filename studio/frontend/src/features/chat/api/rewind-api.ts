// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";
import type { RewindResult } from "../utils/rewind";

/** 409 while a tool call is running or the chat is generating; `detail` says which. */
export async function rewindChatThread(
  threadId: string,
  body: { turn_ids: string[]; dry_run: boolean; force?: boolean },
): Promise<RewindResult> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}/rewind`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      formatApiErrorBody(payload) ?? `Request failed (${response.status})`,
    );
  }
  return payload as RewindResult;
}

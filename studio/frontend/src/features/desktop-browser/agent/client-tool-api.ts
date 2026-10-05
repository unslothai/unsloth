// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type ClientToolImage = { data: string; mimeType: string };

async function post(body: Record<string, unknown>): Promise<boolean> {
  const response = await authFetch("/api/inference/client-tool", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  // a 404 is a request the backend stopped waiting for (Stop, timeout): nothing to retry.
  return response.ok;
}

export function claimClientTool(
  sessionId: string,
  requestId: string,
): Promise<boolean> {
  return post({ session_id: sessionId, request_id: requestId, phase: "claim" });
}

export function sendClientToolResult(
  sessionId: string,
  requestId: string,
  result: string,
  images: ClientToolImage[] = [],
): Promise<boolean> {
  return post({
    session_id: sessionId,
    request_id: requestId,
    phase: "result",
    result,
    ...(images.length > 0 ? { images } : {}),
  });
}

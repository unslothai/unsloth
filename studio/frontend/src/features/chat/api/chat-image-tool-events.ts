// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const CHAT_IMAGE_TOOL_SETTLED = "unsloth:chat-image-tool-settled";

export function notifyChatImageToolSettled(frame: unknown): void {
  if (typeof window === "undefined" || frame === null || typeof frame !== "object") return;
  const event = frame as { type?: unknown; tool_name?: unknown };
  if (event.type === "tool_end" && event.tool_name === "generate_image") {
    window.dispatchEvent(new Event(CHAT_IMAGE_TOOL_SETTLED));
  }
}

export function subscribeChatImageToolSettled(onSettled: () => void): () => void {
  window.addEventListener(CHAT_IMAGE_TOOL_SETTLED, onSettled);
  return () => window.removeEventListener(CHAT_IMAGE_TOOL_SETTLED, onSettled);
}

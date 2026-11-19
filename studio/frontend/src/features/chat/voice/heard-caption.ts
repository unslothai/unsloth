// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { VoiceOrbState } from "./orb-state";

type MessageLike = {
  role: string;
  content: readonly { type: string; text?: string }[];
};

export function lastUserUtterance(messages: readonly MessageLike[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== "user") continue;
    return message.content
      .filter((part) => part.type === "text")
      .map((part) => part.text ?? "")
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
  }
  return "";
}

const ANSWERING: ReadonlySet<VoiceOrbState> = new Set([
  "thinking",
  "generating",
  "synthesizing",
  "speaking",
  "paused",
]);

export function showsHeardCaption(
  state: VoiceOrbState | null | undefined,
): boolean {
  return state != null && ANSWERING.has(state);
}

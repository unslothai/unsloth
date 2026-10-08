// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type GeminiThoughtReplayPart = {
  text: string;
  thoughtSignature: string;
};

type ReplayableMessagePart = {
  type: string;
  text?: unknown;
  _google_thought_signature?: unknown;
};

export function pinGeminiPartThoughtSignature<T extends { type: string }>(
  parts: T[],
  thoughtSignature: string | undefined,
  belongsToThought: boolean,
): T[] {
  if (!thoughtSignature || parts.length === 0) {
    return parts;
  }
  const targetType = belongsToThought ? "reasoning" : "text";
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    if (parts[index].type !== targetType) {
      continue;
    }
    parts[index] = {
      ...parts[index],
      _google_thought_signature: thoughtSignature,
    } as T;
    break;
  }
  return parts;
}

export function geminiThoughtReplayPart(
  part: ReplayableMessagePart,
): GeminiThoughtReplayPart | null {
  if (part.type !== "reasoning") {
    return null;
  }
  const signature = part._google_thought_signature;
  if (typeof signature !== "string" || !signature) {
    return null;
  }
  return {
    text: typeof part.text === "string" ? part.text : "",
    thoughtSignature: signature,
  };
}

export function withGeminiThoughtReplayParts(
  extraContent: unknown,
  thoughtParts: GeminiThoughtReplayPart[],
): unknown {
  if (thoughtParts.length === 0) {
    return extraContent;
  }
  const extra =
    extraContent &&
    typeof extraContent === "object" &&
    !Array.isArray(extraContent)
      ? (extraContent as Record<string, unknown>)
      : {};
  const google =
    extra.google &&
    typeof extra.google === "object" &&
    !Array.isArray(extra.google)
      ? (extra.google as Record<string, unknown>)
      : {};
  return {
    ...extra,
    google: {
      ...google,
      thought_parts: thoughtParts.map((part) => ({
        text: part.text,
        thought_signature: part.thoughtSignature,
      })),
    },
  };
}

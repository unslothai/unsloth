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
  _google_thought_parts?: unknown;
};

export function appendGeminiThoughtReplayPart(
  parts: GeminiThoughtReplayPart[],
  text: string,
  thoughtSignature: string,
): void {
  const latestPart = parts.at(-1);
  if (latestPart?.thoughtSignature === thoughtSignature) {
    latestPart.text += text;
    return;
  }
  parts.push({ text, thoughtSignature });
}

export function pinGeminiTextThoughtSignature<T extends { type: string }>(
  parts: T[],
  thoughtSignature: string | undefined,
): T[] {
  if (!thoughtSignature || parts.length === 0) {
    return parts;
  }
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    if (parts[index].type !== "text") {
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

export function pinGeminiThoughtReplayParts<T extends { type: string }>(
  parts: T[],
  thoughtParts: GeminiThoughtReplayPart[],
): T[] {
  if (thoughtParts.length === 0) {
    return parts;
  }
  let targetIndex = -1;
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    if (parts[index].type === "reasoning") {
      targetIndex = index;
      break;
    }
    if (targetIndex === -1 && parts[index].type === "text") {
      targetIndex = index;
    }
  }
  const replayMetadata = thoughtParts.map((part) => ({ ...part }));
  if (targetIndex === -1) {
    parts.push({
      type: "reasoning",
      text: "",
      _google_thought_parts: replayMetadata,
    } as unknown as T);
    return parts;
  }
  parts[targetIndex] = {
    ...parts[targetIndex],
    _google_thought_parts: replayMetadata,
  } as T;
  return parts;
}

export function geminiThoughtReplayParts(
  part: ReplayableMessagePart,
): GeminiThoughtReplayPart[] {
  if (!Array.isArray(part._google_thought_parts)) {
    return [];
  }
  return part._google_thought_parts.flatMap((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return [];
    }
    const record = entry as Record<string, unknown>;
    return typeof record.text === "string" &&
      typeof record.thoughtSignature === "string" &&
      record.thoughtSignature
      ? [
          {
            text: record.text,
            thoughtSignature: record.thoughtSignature,
          },
        ]
      : [];
  });
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

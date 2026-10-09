// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type GeminiThoughtReplayPart = {
  text: string;
  thoughtSignature: string;
};

export type PositionedGeminiThoughtReplayPart = GeminiThoughtReplayPart & {
  afterToolCalls: number;
};

type ReplayableMessagePart = {
  type: string;
  text?: unknown;
  _google_thought_signature?: unknown;
  _google_thought_parts?: unknown;
};

export function appendGeminiThoughtReplayPart(
  parts: PositionedGeminiThoughtReplayPart[],
  text: string,
  thoughtSignature: string,
  afterToolCalls: number,
): void {
  const latestPart = parts.at(-1);
  if (
    latestPart?.thoughtSignature === thoughtSignature &&
    latestPart.afterToolCalls === afterToolCalls
  ) {
    latestPart.text += text;
    return;
  }
  parts.push({ text, thoughtSignature, afterToolCalls });
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

function replayPlacement<T extends { type: string }>(
  parts: T[],
  afterToolCalls: number,
): { targetIndex: number; insertIndex: number } {
  let toolCallsSeen = 0;
  let reasoningIndex = -1;
  let textIndex = -1;
  let insertIndex = parts.length;
  for (let index = 0; index < parts.length; index += 1) {
    const type = parts[index].type;
    if (type === "tool-call") {
      if (toolCallsSeen === afterToolCalls) {
        insertIndex = index;
      }
      toolCallsSeen += 1;
    } else if (toolCallsSeen === afterToolCalls) {
      if (type === "reasoning") {
        reasoningIndex = index;
      } else if (type === "text") {
        textIndex = index;
      }
      insertIndex = index + 1;
    }
  }
  return {
    targetIndex: reasoningIndex === -1 ? textIndex : reasoningIndex,
    insertIndex,
  };
}

export function pinGeminiThoughtReplayParts<T extends { type: string }>(
  parts: T[],
  thoughtParts: PositionedGeminiThoughtReplayPart[],
): T[] {
  if (thoughtParts.length === 0) {
    return parts;
  }
  const byToolRound = new Map<number, GeminiThoughtReplayPart[]>();
  for (const { afterToolCalls, text, thoughtSignature } of thoughtParts) {
    const round = byToolRound.get(afterToolCalls) ?? [];
    round.push({ text, thoughtSignature });
    byToolRound.set(afterToolCalls, round);
  }
  for (const [afterToolCalls, replayMetadata] of byToolRound) {
    const { targetIndex, insertIndex } = replayPlacement(parts, afterToolCalls);
    if (targetIndex === -1) {
      parts.splice(insertIndex, 0, {
        type: "reasoning",
        text: "",
        _google_thought_parts: replayMetadata,
      } as unknown as T);
    } else {
      parts[targetIndex] = {
        ...parts[targetIndex],
        _google_thought_parts: replayMetadata,
      } as T;
    }
  }
  return parts;
}

export function parseGeminiThoughtReplayParts(
  value: unknown,
): GeminiThoughtReplayPart[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((entry) => {
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

export function geminiThoughtReplayParts(
  part: ReplayableMessagePart,
): GeminiThoughtReplayPart[] {
  return parseGeminiThoughtReplayParts(part._google_thought_parts);
}

export function collectGeminiThoughtReplayParts(
  content: readonly unknown[] | undefined,
): GeminiThoughtReplayPart[] {
  if (!content) {
    return [];
  }
  return content.flatMap((part) =>
    part && typeof part === "object" && !Array.isArray(part)
      ? geminiThoughtReplayParts(part as ReplayableMessagePart)
      : [],
  );
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

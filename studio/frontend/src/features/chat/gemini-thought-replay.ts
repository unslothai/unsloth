// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type GeminiThoughtReplayPart = {
  text: string;
  thoughtSignature: string;
};

export type PositionedGeminiThoughtReplayPart = GeminiThoughtReplayPart & {
  afterToolCalls: number;
};

export type GeminiAnswerReplayPart = {
  text: string;
  thoughtSignature?: string;
};

export type PositionedGeminiAnswerReplayPart = GeminiAnswerReplayPart & {
  afterToolCalls: number;
};

export type GeminiContinuationReplayTurn = {
  text: string;
  thoughtSignature?: string;
  thoughtParts?: GeminiThoughtReplayPart[];
  answerParts?: GeminiAnswerReplayPart[];
};

export type GeminiContinuationReplay = {
  turns: GeminiContinuationReplayTurn[];
  visiblePrefix: string;
  stripVisiblePrefix: boolean;
};

export type GeminiContinuationReplayEntry =
  | { role: "assistant"; turn: GeminiContinuationReplayTurn }
  | { role: "user" };

type ReplayableMessagePart = {
  type: string;
  text?: unknown;
  _google_thought_signature?: unknown;
  _google_thought_parts?: unknown;
  _google_answer_parts?: unknown;
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

export function appendGeminiAnswerReplayPart(
  parts: PositionedGeminiAnswerReplayPart[],
  part: GeminiAnswerReplayPart,
  afterToolCalls: number,
): void {
  parts.push({ ...part, afterToolCalls });
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

export function pinGeminiAnswerReplayParts<T extends { type: string }>(
  parts: T[],
  answerParts: PositionedGeminiAnswerReplayPart[],
): T[] {
  if (answerParts.length === 0) return parts;
  const byToolRound = new Map<number, GeminiAnswerReplayPart[]>();
  for (const { afterToolCalls, text, thoughtSignature } of answerParts) {
    const round = byToolRound.get(afterToolCalls) ?? [];
    round.push({ text, ...(thoughtSignature ? { thoughtSignature } : {}) });
    byToolRound.set(afterToolCalls, round);
  }
  for (const [afterToolCalls, replayMetadata] of byToolRound) {
    let toolCallsSeen = 0;
    let targetIndex = -1;
    let insertIndex = parts.length;
    for (let index = 0; index < parts.length; index += 1) {
      const type = parts[index].type;
      if (type === "tool-call") {
        if (toolCallsSeen === afterToolCalls) insertIndex = index;
        toolCallsSeen += 1;
      } else if (toolCallsSeen === afterToolCalls) {
        if (type === "text") targetIndex = index;
        insertIndex = index + 1;
      }
    }
    if (targetIndex === -1) {
      parts.splice(insertIndex, 0, {
        type: "text",
        text: "",
        _google_answer_parts: replayMetadata,
      } as unknown as T);
    } else {
      parts[targetIndex] = {
        ...parts[targetIndex],
        _google_answer_parts: replayMetadata,
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

export function parseGeminiAnswerReplayParts(
  value: unknown,
): GeminiAnswerReplayPart[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return [];
    const record = entry as Record<string, unknown>;
    if (typeof record.text !== "string") return [];
    const thoughtSignature = record.thoughtSignature;
    if (
      thoughtSignature !== undefined &&
      (typeof thoughtSignature !== "string" || !thoughtSignature)
    ) {
      return [];
    }
    return [{
      text: record.text,
      ...(typeof thoughtSignature === "string" ? { thoughtSignature } : {}),
    }];
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

export function geminiAnswerReplayParts(
  part: ReplayableMessagePart,
): GeminiAnswerReplayPart[] {
  return parseGeminiAnswerReplayParts(part._google_answer_parts);
}

export function collectGeminiAnswerReplayParts(
  content: readonly unknown[] | undefined,
): GeminiAnswerReplayPart[] {
  if (!content) return [];
  return content.flatMap((part) =>
    part && typeof part === "object" && !Array.isArray(part)
      ? geminiAnswerReplayParts(part as ReplayableMessagePart)
      : [],
  );
}

function parseGeminiContinuationReplayTurn(
  value: unknown,
): GeminiContinuationReplayTurn | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  if (typeof record.text !== "string") {
    return null;
  }
  const thoughtParts = parseGeminiThoughtReplayParts(record.thoughtParts);
  const answerParts = parseGeminiAnswerReplayParts(record.answerParts);
  const thoughtSignature = record.thoughtSignature;
  return {
    text: record.text,
    ...(typeof thoughtSignature === "string" && thoughtSignature
      ? { thoughtSignature }
      : {}),
    ...(thoughtParts.length > 0 ? { thoughtParts } : {}),
    ...(answerParts.length > 0 ? { answerParts } : {}),
  };
}

export function parseGeminiContinuationReplayTurns(
  value: unknown,
): GeminiContinuationReplayTurn[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((entry) => {
    const turn = parseGeminiContinuationReplayTurn(entry);
    return turn ? [turn] : [];
  });
}

export function readGeminiContinuationReplay(
  metadata: unknown,
): GeminiContinuationReplay | null {
  const custom = (metadata as { custom?: Record<string, unknown> } | undefined)
    ?.custom;
  const replay = custom?.geminiContinuationReplay as
    | {
        turns?: unknown;
        visiblePrefix?: unknown;
        stripVisiblePrefix?: unknown;
      }
    | undefined;
  const turns = parseGeminiContinuationReplayTurns(replay?.turns);
  return turns.length > 0 && typeof replay?.visiblePrefix === "string"
    ? {
        turns,
        visiblePrefix: replay.visiblePrefix,
        stripVisiblePrefix: replay.stripVisiblePrefix !== false,
      }
    : null;
}

export function continuationGeminiReplayTurns(
  metadata: unknown,
  current: GeminiContinuationReplayTurn,
): GeminiContinuationReplayTurn[] {
  const replay = readGeminiContinuationReplay(metadata);
  const hasSignedCurrentPart = Boolean(
    current.thoughtSignature ||
      (current.thoughtParts?.length ?? 0) > 0 ||
      (current.answerParts?.length ?? 0) > 0,
  );
  if (!replay && !hasSignedCurrentPart) {
    return [];
  }
  const currentText =
    replay?.stripVisiblePrefix && current.text.startsWith(replay.visiblePrefix)
      ? current.text.slice(replay.visiblePrefix.length)
      : current.text;
  const currentTurn = { ...current, text: currentText };
  const hasCurrentTurn = Boolean(
    currentTurn.text ||
      currentTurn.thoughtSignature ||
      (currentTurn.thoughtParts?.length ?? 0) > 0 ||
      (currentTurn.answerParts?.length ?? 0) > 0,
  );
  return [...(replay?.turns ?? []), ...(hasCurrentTurn ? [currentTurn] : [])];
}

export function geminiContinuationReplayEntries(
  turns: GeminiContinuationReplayTurn[],
  includeTrailingUser: boolean,
): GeminiContinuationReplayEntry[] {
  return turns.flatMap((turn, index) => [
    { role: "assistant" as const, turn },
    ...(includeTrailingUser || index < turns.length - 1
      ? [{ role: "user" as const }]
      : []),
  ]);
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

export function withGeminiAnswerReplayParts(
  extraContent: unknown,
  answerParts: GeminiAnswerReplayPart[],
): unknown {
  if (answerParts.length === 0) return extraContent;
  const extra =
    extraContent && typeof extraContent === "object" && !Array.isArray(extraContent)
      ? (extraContent as Record<string, unknown>)
      : {};
  const google =
    extra.google && typeof extra.google === "object" && !Array.isArray(extra.google)
      ? (extra.google as Record<string, unknown>)
      : {};
  return {
    ...extra,
    google: {
      ...google,
      answer_parts: answerParts.map((part) => ({
        text: part.text,
        ...(part.thoughtSignature ? { thought_signature: part.thoughtSignature } : {}),
      })),
    },
  };
}

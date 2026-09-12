// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface CodexReasoningLedger {
  byToolCall: Record<string, unknown[]>;
  final?: unknown[];
}


export function codexLocalToolRoundId(provenance: unknown): number | null {
  if (!provenance || typeof provenance !== "object" || Array.isArray(provenance)) {
    return null;
  }
  const value = provenance as { source?: unknown; round_id?: unknown };
  return value.source === "local" && typeof value.round_id === "number"
    ? value.round_id
    : null;
}


export function startsNewCodexToolRound(
  currentRoundId: number | null,
  nextRoundId: number | null,
): boolean {
  return (
    currentRoundId !== null &&
    nextRoundId !== null &&
    currentRoundId !== nextRoundId
  );
}

export function shouldReplayAssistantReasoning(input: {
  enabled: boolean;
  reasoningContent: string;
  hasContent: boolean;
  hasToolCalls: boolean;
  incomplete: boolean;
}): boolean {
  return (
    input.enabled &&
    input.reasoningContent.length > 0 &&
    (input.hasContent || input.hasToolCalls || !input.incomplete)
  );
}

export function addCodexReasoning(
  ledger: CodexReasoningLedger,
  items: unknown[],
  toolCallIds: string[],
): CodexReasoningLedger {
  if (items.length === 0) return ledger;
  const toolCallId = toolCallIds[0];
  if (!toolCallId) return { ...ledger, final: items };
  return {
    ...ledger,
    byToolCall: { ...ledger.byToolCall, [toolCallId]: items },
  };
}

export function readCodexReasoning(
  metadata: unknown,
): CodexReasoningLedger | undefined {
  if (!metadata || typeof metadata !== "object") return undefined;
  const custom = (metadata as { custom?: unknown }).custom;
  if (!custom || typeof custom !== "object") return undefined;
  const value = (custom as Record<string, unknown>).openaiCodexReasoning;
  if (Array.isArray(value)) {
    return value.length > 0 ? { byToolCall: {}, final: value } : undefined;
  }
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  const byToolCall: Record<string, unknown[]> = {};
  const rawByToolCall = record.byToolCall;
  if (rawByToolCall && typeof rawByToolCall === "object") {
    for (const [id, items] of Object.entries(rawByToolCall)) {
      if (Array.isArray(items) && items.length > 0) byToolCall[id] = items;
    }
  }
  const final = Array.isArray(record.final) && record.final.length > 0
    ? record.final
    : undefined;
  return Object.keys(byToolCall).length > 0 || final
    ? { byToolCall, ...(final ? { final } : {}) }
    : undefined;
}

export function codexReasoningForToolCalls(
  ledger: CodexReasoningLedger | undefined,
  toolCallIds: string[],
): unknown[] | undefined {
  if (!ledger) return undefined;
  for (const id of toolCallIds) {
    const items = ledger.byToolCall[id];
    if (items) return items;
  }
  return undefined;
}

/** Which replay exchange each tool-call part of one assistant message lands in, by the
 *  serializer's own rule: a round id that changes starts a new exchange, and a part with
 *  no round id that completes a local pair is an exchange of its own. The MCP image
 *  bound groups a message's results by this, so it shares one marker turn's allowance
 *  across exactly the results the backend will see as one batch -- grouping the whole
 *  message let a later round's picture evict an earlier round's. */
export function localToolExchangeIndexes<TPart>(
  parts: readonly TPart[],
  roundIdOf: (part: TPart) => number | null,
  flushesPair: (part: TPart) => boolean,
): number[] {
  const indexes: number[] = [];
  let exchange = 0;
  let pending = 0;
  let pendingRoundId: number | null = null;
  for (const part of parts) {
    const roundId = roundIdOf(part);
    if (pending > 0 && startsNewCodexToolRound(pendingRoundId, roundId)) {
      exchange += 1;
      pending = 0;
    }
    if (roundId !== null) pendingRoundId = roundId;
    const flushLocalPair = roundId === null && flushesPair(part);
    if (flushLocalPair && pending > 0) {
      exchange += 1;
      pending = 0;
    }
    indexes.push(exchange);
    pending += 1;
    if (flushLocalPair) {
      exchange += 1;
      pending = 0;
    }
  }
  return indexes;
}

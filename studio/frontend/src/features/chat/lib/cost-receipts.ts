// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { nonnegativeDecimal } from "./model-pricing";

export interface CostReceipt {
  provider: "openrouter";
  attemptId: string;
  generationId?: string;
  requestedModel: string;
  servedModel?: string;
  cost: number | null;
  usage?: Record<string, unknown>;
}

export function readCostReceipts(value: unknown): CostReceipt[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (r): r is CostReceipt =>
        r &&
        r.provider === "openrouter" &&
        typeof r.attemptId === "string" &&
        typeof r.requestedModel === "string",
    )
    .map((r) => ({ ...r, cost: nonnegativeDecimal(r.cost) }));
}

/** Each upstream request starts pending. Its final usage updates the same receipt, never adds a charge twice. */
export function createCostRecorder(
  attemptId: string,
  requestedModel: string,
  seed?: unknown,
  resume = false,
) {
  let receipts = readCostReceipts(seed);
  let activeAttempt =
    (resume ? receipts.at(-1)?.attemptId : undefined) ?? attemptId;
  let lastGeneration = resume ? receipts.at(-1)?.generationId : undefined;
  const pending = () => {
    if (!receipts.some((r) => r.attemptId === activeAttempt))
      receipts.push({
        provider: "openrouter",
        attemptId: activeAttempt,
        requestedModel,
        cost: null,
      });
  };
  pending();
  return {
    snapshot: () => receipts.map((r) => ({ ...r })),
    observe(chunk: {
      id?: string;
      model?: string;
      usage?: Record<string, unknown>;
      choices?: unknown[];
      _openrouterAttempt?: string;
      _openrouterReceipt?: {
        id?: string;
        model?: string;
        usage?: Record<string, unknown>;
      };
    }) {
      if (chunk._openrouterReceipt) chunk = chunk._openrouterReceipt;
      if (typeof chunk._openrouterAttempt === "string") {
        // The local placeholder has not reached an upstream request yet.
        receipts = receipts.filter(
          (r) =>
            !(r.attemptId === attemptId && !r.generationId && r.cost == null),
        );
        activeAttempt = chunk._openrouterAttempt;
        lastGeneration = undefined;
        pending();
        return true;
      }
      // Server tool/status frames have synthetic ids and are not billable generations.
      if (Object.keys(chunk).some((key) => key.startsWith("_"))) return false;
      if (!chunk.usage && !chunk.choices?.length) return false;
      const generationId =
        typeof chunk.id === "string" && chunk.id ? chunk.id : lastGeneration;
      lastGeneration = generationId;
      let receipt = generationId
        ? receipts.find((r) => r.generationId === generationId)
        : undefined;
      receipt ??= receipts.find(
        (r) => r.attemptId === activeAttempt && !r.generationId,
      );
      const changed =
        !receipt ||
        receipt.generationId !== generationId ||
        (typeof chunk.model === "string" &&
          receipt.servedModel !== chunk.model) ||
        !!chunk.usage;
      if (!receipt) {
        receipt = {
          provider: "openrouter",
          attemptId: activeAttempt,
          requestedModel,
          cost: null,
        };
        receipts.push(receipt);
      }
      receipt.generationId = generationId;
      if (typeof chunk.model === "string") receipt.servedModel = chunk.model;
      if (chunk.usage) {
        receipt.usage = { ...receipt.usage, ...chunk.usage };
        const cost = nonnegativeDecimal(chunk.usage.cost);
        if (cost != null) receipt.cost = cost;
      }
      return changed;
    },
  };
}

export function sumCostReceipts(receipts: readonly CostReceipt[]) {
  const unique = new Map<string, CostReceipt>();
  for (const receipt of receipts) {
    const key = receipt.generationId ?? `attempt:${receipt.attemptId}`;
    const previous = unique.get(key);
    if (!previous || receipt.cost != null) unique.set(key, receipt);
  }
  // A fork or continuation can retain an older pending snapshot of a now-known generation.
  const resolvedAttempts = new Set(
    [...unique.values()].filter((r) => r.generationId).map((r) => r.attemptId),
  );
  for (const [key, r] of unique)
    if (!r.generationId && resolvedAttempts.has(r.attemptId))
      unique.delete(key);
  const values = [...unique.values()];
  return {
    receipts: values,
    total: values.reduce((sum, r) => sum + (r.cost ?? 0), 0),
    known: values.some((r) => r.cost != null),
    incomplete: values.some((r) => r.cost == null),
  };
}

export function messageCost(custom: unknown) {
  const data = custom as
    | { costReceipts?: unknown; responseDetails?: { providerType?: string } }
    | undefined;
  const receipts = readCostReceipts(data?.costReceipts);
  return {
    ...sumCostReceipts(receipts),
    relevant:
      receipts.length > 0 ||
      data?.responseDetails?.providerType === "openrouter",
    historical:
      receipts.length === 0 &&
      data?.responseDetails?.providerType === "openrouter",
  };
}

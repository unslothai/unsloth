// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import type { MessageRecord } from "../types";
import { isExternalModelId } from "../external-providers.ts";
import { estimateContextUsage, estimateMessagesTokenCount } from "./estimate-chat-tokens.ts";

type BranchRecord = Pick<MessageRecord, "id" | "role" | "content" | "metadata">;

export type BranchContextUsage = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  contextTokens?: number;
  cachedTokens: number;
  cacheWriteTokens?: number;
  modelId?: string;
  messageCount?: number;
  estimatedTokens?: number;
  revision?: number;
  estimated?: boolean;
};

export function isSummedToolLoopUsage(
  usage: { modelId?: unknown; contextTokens?: unknown } | null | undefined,
  content: unknown,
): boolean {
  return (
    typeof usage?.modelId === "string" &&
    isExternalModelId(usage.modelId) &&
    usage.contextTokens === undefined &&
    Array.isArray(content) &&
    content.some((part) => {
      const toolCall = part as { type?: unknown; provenance?: { source?: unknown } } | null;
      return toolCall?.type === "tool-call" && toolCall.provenance?.source === "local";
    })
  );
}

function revisionOf(custom: Record<string, unknown> | undefined): number {
  const revision = custom?.revision;
  return typeof revision === "number" && Number.isInteger(revision) && revision > 0 ? revision : 0;
}

export function revisionTotal(
  messages: readonly { metadata?: { custom?: Record<string, unknown> } }[],
): number {
  let total = 0;
  for (const message of messages) total += revisionOf(message.metadata?.custom);
  return total;
}

export function threadBranchRecords(messages: readonly ThreadMessage[]): BranchRecord[] {
  return messages.map((message) => ({
    id: message.id,
    role: message.role,
    content: message.content,
    metadata: message.metadata?.custom,
  }));
}

function inBranchOrder(branch: readonly Pick<BranchRecord, "id" | "role" | "content">[]) {
  return branch.map(
    ({ id, role, content }, index) => ({ id, role, content, createdAt: index }) as MessageRecord,
  );
}

function branchTokenEstimate(
  branch: readonly Pick<BranchRecord, "id" | "role" | "content">[],
): number | null {
  return estimateMessagesTokenCount(inBranchOrder(branch));
}

export type ContextUsageMarkers = {
  messageCount: number;
  revision: number;
  estimatedTokens: number;
};

export function contextUsageMarkers(branch: readonly BranchRecord[]): ContextUsageMarkers {
  let revision = 0;
  for (const message of branch) revision += revisionOf(message.metadata);
  return {
    messageCount: branch.length,
    revision,
    estimatedTokens: branchTokenEstimate(branch) ?? 0,
  };
}

export function estimateBranch(branch: readonly BranchRecord[]) {
  return estimateContextUsage(inBranchOrder(branch));
}

function estimatedUsage(total: number): BranchContextUsage {
  return {
    promptTokens: total,
    completionTokens: 0,
    totalTokens: total,
    cachedTokens: 0,
    estimated: true,
  };
}

export function branchContextUsage(
  branch: readonly BranchRecord[],
  checkpoint: string | null | undefined,
  localWindow: number | null | undefined,
  ggufWindow: number | null | undefined,
): BranchContextUsage | null {
  let revisions = 0;
  let anchor = -1;
  let calibration: { context: number; estimated: number } | null = null;
  branch.forEach((message, index) => {
    revisions += revisionOf(message.metadata);
    if (message.role !== "assistant") return;
    const usage = message.metadata?.contextUsage as BranchContextUsage | undefined;
    const context = usage?.contextTokens ?? usage?.totalTokens;
    if (!usage || typeof context !== "number" || !Number.isFinite(context) || context < 0) return;
    if (ggufWindow && context > ggufWindow) return;
    const modelMatches = usage.modelId
      ? usage.modelId === checkpoint
      : localWindow != null && localWindow > 0;
    if (!modelMatches || isSummedToolLoopUsage(usage, message.content)) return;
    const current =
      (usage.revision ?? 0) === revisions &&
      (usage.messageCount === undefined || usage.messageCount === index + 1);
    if (current) {
      anchor = index;
      calibration = null;
    } else if (
      typeof usage.estimatedTokens === "number" &&
      Number.isFinite(usage.estimatedTokens) &&
      usage.estimatedTokens > 0
    ) {
      calibration = { context, estimated: usage.estimatedTokens };
    }
  });
  if (calibration) {
    const { context, estimated } = calibration;
    const now = branchTokenEstimate(branch) ?? 0;
    const total = context >= estimated ? now + context - estimated : (now * context) / estimated;
    return estimatedUsage(Math.max(1, Math.round(total)));
  }
  if (anchor < 0) return null;
  const saved = branch[anchor].metadata?.contextUsage as BranchContextUsage;
  if (anchor === branch.length - 1) return saved;
  const base = saved.contextTokens ?? saved.totalTokens;
  return estimatedUsage(base + (branchTokenEstimate(branch.slice(anchor + 1)) ?? 0));
}

export function sameContextUsage(
  shown: BranchContextUsage | null | undefined,
  usage: BranchContextUsage,
  checkpoint: string | null | undefined,
): boolean {
  return (
    shown != null &&
    !shown.estimated === !usage.estimated &&
    shown.promptTokens === usage.promptTokens &&
    shown.completionTokens === usage.completionTokens &&
    shown.totalTokens === usage.totalTokens &&
    shown.contextTokens === usage.contextTokens &&
    shown.cachedTokens === usage.cachedTokens &&
    shown.cacheWriteTokens === usage.cacheWriteTokens &&
    shown.modelId === (usage.modelId ?? checkpoint)
  );
}

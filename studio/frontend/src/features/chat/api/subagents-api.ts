// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { SubagentRecord } from "../types/subagents";
import { resolveToolConfirmation } from "./chat-api";

export type {
  SubagentPendingApproval,
  SubagentRecord,
  SubagentStatus,
  SubagentToolStatus,
  SubagentTranscriptItem,
} from "../types/subagents";

/** Carries the HTTP status so callers can tell "older backend" (404) from a transient failure. */
export class SubagentsApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "SubagentsApiError";
    this.status = status;
  }
}

async function subagentsJson<T>(response: Response): Promise<T> {
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new SubagentsApiError(
      typeof detail === "string"
        ? detail
        : `Request failed (${response.status})`,
      response.status,
    );
  }
  return body as T;
}

/** Summaries (no transcript) of every subagent the thread has spawned, oldest first. */
export async function listSubagents(
  parentThreadId: string,
  signal?: AbortSignal,
): Promise<SubagentRecord[]> {
  const response = await authFetch(
    `/api/subagents?parent_thread_id=${encodeURIComponent(parentThreadId)}`,
    { signal },
  );
  const body = await subagentsJson<{ subagents?: SubagentRecord[] }>(response);
  return Array.isArray(body.subagents) ? body.subagents : [];
}

/** One subagent with its ordered transcript. */
export async function getSubagent(
  subagentId: string,
  signal?: AbortSignal,
): Promise<SubagentRecord> {
  const response = await authFetch(
    `/api/subagents/${encodeURIComponent(subagentId)}`,
    { signal },
  );
  return subagentsJson<SubagentRecord>(response);
}

/** False when the run already finished or belongs to nobody. */
export async function cancelSubagent(subagentId: string): Promise<boolean> {
  const response = await authFetch(
    `/api/subagents/${encodeURIComponent(subagentId)}/cancel`,
    { method: "POST" },
  );
  const body = await subagentsJson<{ cancelled?: boolean }>(response);
  return body.cancelled === true;
}

/**
 * Approves or denies a subagent's pending tool call through the same endpoint the main chat
 * uses: the subagent shares the parent's session, so only the approval id differs. With
 * `fullAccess` the server also upgrades the whole running chat, parent and subagents, to Full
 * access, and resolves true only when it granted that. Throws `ToolApprovalGoneError` when the
 * call is no longer pending.
 */
export function resolveSubagentApproval(
  sessionId: string,
  approvalId: string,
  decision: "allow" | "deny",
  fullAccess = false,
): Promise<boolean> {
  return resolveToolConfirmation(sessionId, approvalId, decision, fullAccess);
}

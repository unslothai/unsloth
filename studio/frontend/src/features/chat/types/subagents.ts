// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Wire shapes of /api/subagents (studio/SUBAGENTS.md, "Record" and "TranscriptItem").

export type SubagentStatus = "running" | "completed" | "failed" | "cancelled";

export type SubagentToolStatus = "running" | "awaiting_approval" | "done";

export type SubagentPendingApproval = {
  approvalId: string;
  toolName: string;
  toolCallId: string;
  arguments: Record<string, unknown>;
};

export type SubagentTranscriptItem =
  | { kind: "text"; text: string }
  | { kind: "reasoning"; text: string }
  | {
      kind: "tool";
      toolCallId: string;
      toolName: string;
      arguments: Record<string, unknown>;
      status: SubagentToolStatus;
      approvalId: string | null;
      result: string | null;
    }
  | { kind: "status"; text: string }
  | { kind: "error"; text: string };

export type SubagentRecord = {
  id: string;
  parentThreadId: string | null;
  childThreadId: string;
  sessionId: string | null;
  description: string;
  prompt: string;
  model: string | null;
  tools: string[];
  status: SubagentStatus;
  error: string | null;
  resultPreview: string | null;
  toolCallCount: number;
  /** Live runs only. */
  pendingApproval: SubagentPendingApproval | null;
  startedAt: number;
  finishedAt: number | null;
  /** Only on detail reads. */
  transcript?: SubagentTranscriptItem[];
};

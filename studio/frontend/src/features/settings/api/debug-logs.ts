// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { DebugLogRequestError } from "../lib/debug-log-error";

export { DebugLogRequestError } from "../lib/debug-log-error";

export type DebugLogStatus =
  | "ok"
  | "empty"
  | "missing"
  | "unreadable"
  | "disabled";

export interface DebugLogSource {
  id: string;
  family: string;
  label: string;
  realpath: string;
  sizeBytes: number;
  modifiedAt: number;
  isCurrent: boolean;
}

export interface DebugLogSources {
  sources: DebugLogSource[];
  defaultSourceId: string | null;
  fileLoggingDisabled: boolean;
}

export interface DebugLogPage {
  status: DebugLogStatus;
  reason: string | null;
  sourceId: string | null;
  realpath: string | null;
  lines: string[];
  cursor: string | null;
  reset: boolean;
  resetReason: string | null;
  droppedBytes: number;
  truncatedHead: boolean;
  morePending: boolean;
  fileLoggingDisabled: boolean;
  sizeBytes: number;
}

export async function loadDebugLogSources(
  signal?: AbortSignal,
): Promise<DebugLogSources> {
  const response = await authFetch("/api/settings/debug/logs/sources", {
    signal,
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not list the log files."),
    );
  }
  const body = await response.json();
  return {
    sources: (body.sources ?? []).map((source: Record<string, unknown>) => ({
      id: String(source.id),
      family: String(source.family),
      label: String(source.label),
      realpath: String(source.realpath),
      sizeBytes: Number(source.size_bytes ?? 0),
      modifiedAt: Number(source.modified_at ?? 0),
      isCurrent: Boolean(source.is_current),
    })),
    defaultSourceId: body.default_source_id ?? null,
    fileLoggingDisabled: Boolean(body.file_logging_disabled),
  };
}

export async function loadDebugLog(
  options: {
    sourceId?: string | null;
    cursor?: string | null;
    signal?: AbortSignal;
  } = {},
): Promise<DebugLogPage> {
  const params = new URLSearchParams();
  if (options.sourceId) params.set("source", options.sourceId);
  if (options.cursor) params.set("cursor", options.cursor);
  const query = params.toString();
  const response = await authFetch(
    `/api/settings/debug/logs${query ? `?${query}` : ""}`,
    {
      signal: options.signal,
    },
  );
  if (!response.ok) {
    throw new DebugLogRequestError(
      await readFastApiError(response, "Could not read the log."),
      response.status,
    );
  }
  const body = await response.json();
  return {
    status: body.status,
    reason: body.reason ?? null,
    sourceId: body.source_id ?? null,
    realpath: body.realpath ?? null,
    lines: body.lines ?? [],
    cursor: body.cursor ?? null,
    reset: Boolean(body.reset),
    resetReason: body.reset_reason ?? null,
    droppedBytes: Number(body.dropped_bytes ?? 0),
    truncatedHead: Boolean(body.truncated_head),
    morePending: Boolean(body.more_pending),
    fileLoggingDisabled: Boolean(body.file_logging_disabled),
    sizeBytes: Number(body.size_bytes ?? 0),
  };
}

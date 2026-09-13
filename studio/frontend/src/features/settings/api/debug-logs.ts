// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl, isTauri } from "@/lib/api-base";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { browserDownload } from "@/lib/native-files";
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

export const LOG_EXPORT_ENDPOINT = "/api/settings/debug/logs/export";

/**
 * outdated  - no such route, so the backend predates the feature
 * forbidden - the route rejects this caller (API key, keyless)
 * failed    - anything else; `message` carries the detail
 */
export type LogExportFailure = "outdated" | "forbidden" | "failed";

export class LogExportError extends Error {
  readonly failure: LogExportFailure;

  constructor(failure: LogExportFailure, message: string) {
    super(message);
    this.name = "LogExportError";
    this.failure = failure;
  }
}

function failureForStatus(status: number): LogExportFailure {
  if (status === 404) return "outdated";
  if (status === 403) return "forbidden";
  return "failed";
}

// Rust hands back a string, so the status is read out of it; only
// `stream_url_to_path` produces this phrase. Anchored at the START because two
// other errors can CONTAIN it -- "Failed to save {path}: {error}" embeds the
// caller's filename, and desktop-auth embeds raw stderr -- which would turn a
// disk-full error into a "forbidden" toast. A start anchor suffices: every
// other reachable error begins with its own fixed prefix.
const DESKTOP_STATUS_PATTERN = /^Download failed with status (\d{3})\./;

// Desktop auth can answer "this account has to log in" instead of returning a
// session (per-account isolation, shared installs). No request is made, so
// there is no status: the command returns this exact sentence (`LOGIN_REQUIRED`
// in native_file_dialogs.rs). Keep the two in step.
const DESKTOP_LOGIN_REQUIRED = "Log export requires a signed-in Studio session.";

function desktopExportError(error: unknown): LogExportError {
  const message =
    typeof error === "string"
      ? error
      : ((error as Error | undefined)?.message ?? String(error));
  if (message === DESKTOP_LOGIN_REQUIRED) {
    return new LogExportError("forbidden", message);
  }
  const status = Number(message.match(DESKTOP_STATUS_PATTERN)?.[1] ?? 0);
  return new LogExportError(failureForStatus(status), message);
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

/** `unsloth-logs-<YYYYmmdd-HHMMSS>.zip`, local time so it sorts with the session. */
export function logArchiveFilename(now: Date = new Date()): string {
  const day = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}`;
  const time = `${pad2(now.getHours())}${pad2(now.getMinutes())}${pad2(now.getSeconds())}`;
  return `unsloth-logs-${day}-${time}.zip`;
}

/**
 * Pack every log the picker lists into one ZIP. Returns the absolute path on
 * desktop, where Rust streamed it; null in a browser, which alone knows where
 * its downloads land.
 */
export async function exportAllLogs(): Promise<string | null> {
  const filename = logArchiveFilename();

  if (isTauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    try {
      // `filename` is one word because Tauri maps camelCase onto snake_case
      // parameters. It is only a suggestion: the command picks the destination
      // and the realpath it returns is the answer.
      return await invoke<string>("download_logs_to_downloads", {
        url: apiUrl(LOG_EXPORT_ENDPOINT),
        filename,
      });
    } catch (error) {
      throw desktopExportError(error);
    }
  }

  const response = await authFetch(LOG_EXPORT_ENDPOINT);
  if (!response.ok) {
    throw new LogExportError(
      failureForStatus(response.status),
      await readFastApiError(response, "Could not export the logs."),
    );
  }
  browserDownload(await response.blob(), filename);
  return null;
}

/**
 * Reveal the Unsloth directory. Desktop only: in a browser the folder is on the
 * server, not the user's machine, and the button is not rendered.
 */
export async function openLogsFolder(): Promise<void> {
  if (!isTauri) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("open_logs_dir");   // resolves ~/.unsloth/studio itself
}

/**
 * Reveal the folder the archive was saved into. Not `openLogsFolder`, which
 * opens ~/.unsloth/studio where the logs came FROM: naming one path and opening
 * another is the bug this avoids. `open_models_dir` is the app's generic "open
 * this directory" command (`open_existing_dir`), misnamed rather than misused.
 */
export async function revealSavedArchive(savedPath: string): Promise<void> {
  if (!isTauri) return;
  const separator = Math.max(
    savedPath.lastIndexOf("/"),
    savedPath.lastIndexOf("\\"),
  );
  if (separator <= 0) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("open_models_dir", { path: savedPath.slice(0, separator) });
}

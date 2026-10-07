// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken, refreshSession } from "@/features/auth";
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
  matchedSourceId: string | null;
  fileLoggingDisabled: boolean;
  logRoot: string | null;
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
  diagnosticPath?: string | null,
): Promise<DebugLogSources> {
  const query = diagnosticPath
    ? `?diagnostic_path=${encodeURIComponent(diagnosticPath)}`
    : "";
  const response = await authFetch(`/api/settings/debug/logs/sources${query}`, {
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
    matchedSourceId: body.matched_source_id ?? null,
    fileLoggingDisabled: Boolean(body.file_logging_disabled),
    logRoot: body.log_root ?? null,
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

// Anchored at the start: other errors embed this phrase (a filename, raw stderr), which would turn
// a disk-full error into a "forbidden" toast.
const DESKTOP_STATUS_PATTERN = /^Download failed with status (\d{3})\./;

// Must match LOGIN_REQUIRED in native_file_dialogs.rs.
const DESKTOP_LOGIN_REQUIRED =
  "Log export requires a signed-in Unsloth session.";

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

function desktopExportStatus(error: unknown): number {
  const message =
    typeof error === "string"
      ? error
      : ((error as Error | undefined)?.message ?? String(error));
  return Number(message.match(DESKTOP_STATUS_PATTERN)?.[1] ?? 0);
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

export function logArchiveFilename(now: Date = new Date()): string {
  const day = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}`;
  const time = `${pad2(now.getHours())}${pad2(now.getMinutes())}${pad2(now.getSeconds())}`;
  return `unsloth-logs-${day}-${time}.zip`;
}

/** Returns the saved path on desktop; null in a browser, which alone knows its downloads. */
export async function exportAllLogs(): Promise<string | null> {
  const filename = logArchiveFilename();

  if (isTauri) {
    // Tauri maps camelCase to snake_case, so `filename` is one word. `uiToken` is the multi-account
    // fallback; the command pins host, port and path so it reaches nothing the tab could not.
    const { invoke } = await import("@tauri-apps/api/core");
    const run = (uiToken: string | null) =>
      invoke<string>("download_logs_to_downloads", {
        url: apiUrl(LOG_EXPORT_ENDPOINT),
        filename,
        uiToken,
      });

    try {
      return await run(getAuthToken());
    } catch (error) {
      // Rust cannot refresh on 401, so refresh only here: refreshSession clears tokens on any non-2xx,
      // and doing it speculatively would sign out a user over a transient 500.
      if (desktopExportStatus(error) !== 401) throw desktopExportError(error);
      try {
        if (!(await refreshSession())) throw error;
      } catch {
        throw desktopExportError(error);
      }
      try {
        return await run(getAuthToken());
      } catch (retryError) {
        throw desktopExportError(retryError);
      }
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
 * Reveal the backend's logRoot (it honors UNSLOTH_STUDIO_HOME); open_logs_dir hard-codes
 * ~/.unsloth/studio/logs and is only the fallback. Desktop only.
 */
export async function openLogsFolder(
  logRoot?: string | null,
  selectedRealpath?: string | null,
): Promise<void> {
  if (!isTauri) return;
  const { invoke } = await import("@tauri-apps/api/core");
  const directory = logRoot || (selectedRealpath && parentDirectory(selectedRealpath));
  if (directory) {
    await invoke("open_models_dir", { path: directory });
    return;
  }
  await invoke("open_logs_dir");
}

function parentDirectory(path: string): string | null {
  const separator = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  if (separator < 0) return null;
  const parent = path.slice(0, separator);
  // Keep the separator for "" (Unix root) and "C:" (current dir on drive C:, not the C:\ root).
  if (parent === "" || /^[A-Za-z]:$/.test(parent)) {
    return path.slice(0, separator + 1);
  }
  return parent;
}

/** Open the archive's folder via open_models_dir, the generic open_existing_dir command. */
export async function revealSavedArchive(savedPath: string): Promise<void> {
  if (!isTauri) return;
  const directory = parentDirectory(savedPath);
  if (!directory) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("open_models_dir", { path: directory });
}

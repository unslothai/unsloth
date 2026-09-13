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
 * Why an export attempt failed, so the caller can say something useful rather
 * than repeat a status code:
 *
 *   outdated  - the route is not there, so the backend predates the feature
 *   forbidden - the route is there but rejects this caller (API key, keyless)
 *   failed    - anything else; `message` carries the detail
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

// The desktop side streams in Rust and only ever hands back a string, so the
// status has to be read out of it. `stream_url_to_path` formats a non-2xx as
// exactly "Download failed with status 404." and nothing else in src-tauri
// produces that phrase.
//
// Anchored to the whole message on purpose. Unanchored, two other error strings
// can be made to contain the phrase: "Failed to save {path}: {error}" embeds a
// path ending in the caller-supplied filename, and the desktop-auth failure
// embeds raw subprocess stderr. Either could turn a disk-full error into a
// "forbidden" toast.
const DESKTOP_STATUS_PATTERN = /^Download failed with status (\d{3})\./;

function desktopExportError(error: unknown): LogExportError {
  const message =
    typeof error === "string"
      ? error
      : ((error as Error | undefined)?.message ?? String(error));
  const status = Number(message.match(DESKTOP_STATUS_PATTERN)?.[1] ?? 0);
  return new LogExportError(failureForStatus(status), message);
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

/**
 * `unsloth-logs-<YYYYmmdd-HHMMSS>.zip`, stamped in local time so it sorts next
 * to whatever the user was doing when they hit the problem.
 */
export function logArchiveFilename(now: Date = new Date()): string {
  const day = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}`;
  const time = `${pad2(now.getHours())}${pad2(now.getMinutes())}${pad2(now.getSeconds())}`;
  return `unsloth-logs-${day}-${time}.zip`;
}

/**
 * Pack every log the picker lists into one ZIP.
 *
 * Returns the absolute path on desktop, where Rust streams the response
 * straight to the Downloads folder and knows where it landed. Returns null in a
 * browser: the blob goes through the normal download path and only the browser
 * knows where that is.
 */
export async function exportAllLogs(): Promise<string | null> {
  const filename = logArchiveFilename();

  if (isTauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    try {
      // Mirrors `download_logs_to_downloads` in
      // studio/src-tauri/src/native_file_dialogs.rs. Tauri maps a camelCase key
      // onto the command's snake_case parameter, so a single-word `filename` is
      // passed through as-is; `lib/native-files.ts` spells
      // `save_native_file_from_url`'s `file_name` parameter `fileName` for the
      // same reason. The command picks the destination itself, so `filename` is
      // only a suggestion and the realpath it returns is the answer.
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
 * Reveal the Unsloth directory in the system file manager. Desktop only, and a
 * no-op elsewhere: the button that calls it is not rendered in a browser, and
 * the folder it names is on the server rather than on the user's machine.
 */
export async function openLogsFolder(): Promise<void> {
  if (!isTauri) return;
  const { invoke } = await import("@tauri-apps/api/core");
  // Takes no arguments: the command already resolves ~/.unsloth/studio itself.
  await invoke("open_logs_dir");
}

/**
 * Reveal the folder the archive was just saved into.
 *
 * Not `openLogsFolder`: that opens ~/.unsloth/studio, where the logs came FROM,
 * and the archive went to Downloads. Naming one path and opening the other is
 * the bug this exists to avoid.
 *
 * `open_models_dir` is misnamed rather than misused -- it is the app's generic
 * "open this directory" command (`open_existing_dir` in commands.rs), already
 * registered, and it refuses anything that is not an existing directory.
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

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the auth barrel's React login page.
import { authFetch } from "@/features/auth/api";
import { disposableTimeoutSignal } from "@/features/hub/lib/abort-signals";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";
import type { ApiMonitorEntry } from "../types/api";

export const CHAT_MONITOR_ID_RESPONSE_HEADER = "X-Unsloth-Monitor-Id";
export const CHAT_MONITOR_STATUS_RESPONSE_HEADER = "X-Unsloth-Monitor-Status";
const API_MONITOR_READ_TIMEOUT_MS = 10_000;

export class ApiMonitorEntryRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiMonitorEntryRequestError";
    this.status = status;
  }
}

export function isPermanentApiMonitorEntryError(error: unknown): boolean {
  return error instanceof ApiMonitorEntryRequestError && error.status === 404;
}

export async function getApiMonitorEntry(
  id: string,
  caller?: AbortSignal,
  timeoutMs = API_MONITOR_READ_TIMEOUT_MS,
): Promise<ApiMonitorEntry> {
  const timeout = disposableTimeoutSignal(timeoutMs);
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (caller?.aborted) {
    abort();
  }
  caller?.addEventListener("abort", abort);
  timeout.signal.addEventListener("abort", abort);
  try {
    const response = await authFetch(
      `/api/inference/monitor/${encodeURIComponent(id)}`,
      { signal: controller.signal },
    );
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new ApiMonitorEntryRequestError(
        formatApiErrorBody(body) ?? `API monitor request failed (${response.status})`,
        response.status,
      );
    }
    return body as ApiMonitorEntry;
  } finally {
    caller?.removeEventListener("abort", abort);
    timeout.dispose();
  }
}

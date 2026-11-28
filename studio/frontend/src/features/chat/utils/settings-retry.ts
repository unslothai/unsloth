// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /api/chat/settings is extra="forbid": one bad field rejects all, so drop it, never requeue.

export class ChatSettingsRequestError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ChatSettingsRequestError";
    this.status = status;
    this.detail = detail;
  }
}

export function isAbortOrTimeout(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || error.name === "TimeoutError")
  );
}

/** A body that broke off mid-read (TypeError) is a transport failure; invalid JSON (SyntaxError) is not. */
export function asBodyTransportFailure(error: unknown): unknown {
  if (!(error instanceof TypeError)) return error;
  return Object.assign(
    new Error("The connection closed before the server's reply finished.", {
      cause: error,
    }),
    { unslothTransportFailure: true },
  );
}

/** 4xx is terminal except 408 and 429; network errors, 5xx and timeouts are transient. */
export function isTerminalSettingsRejection(error: unknown): boolean {
  if (!(error instanceof ChatSettingsRequestError)) return false;
  const { status } = error;
  if (status === 408 || status === 429) return false;
  return status >= 400 && status < 500;
}

/** A 401 may be a transient token-refresh failure; sign-out discards the pin queue itself. */
export function isRetryablePinError(error: unknown): boolean {
  return (
    !isTerminalSettingsRejection(error) ||
    (error as ChatSettingsRequestError).status === 401
  );
}

/** Top-level fields from FastAPI detail[].loc[0]; empty means drop the whole patch. */
export function rejectedSettingKeys(detail: unknown): string[] {
  if (!Array.isArray(detail)) return [];
  const keys = new Set<string>();
  for (const entry of detail) {
    if (entry == null || typeof entry !== "object") continue;
    const loc = (entry as { loc?: unknown }).loc;
    if (!Array.isArray(loc) || loc.length === 0) continue;
    const field = loc[0];
    if (typeof field === "string" && field.length > 0) keys.add(field);
  }
  return [...keys];
}

/** On terminal rejection drop the named fields; `progressed` bounds the retry loop. */
export function retryablePatchAfterFailure<T extends object>(
  patch: T,
  error: unknown,
): { patch: Partial<T>; dropped: string[]; progressed: boolean } {
  if (!isTerminalSettingsRejection(error)) {
    return { patch, dropped: [], progressed: false };
  }
  const rejected = rejectedSettingKeys(
    (error as ChatSettingsRequestError).detail,
  ).filter((key) => key in (patch as Record<string, unknown>));
  if (rejected.length === 0) {
    return { patch: {}, dropped: Object.keys(patch), progressed: false };
  }
  const kept: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(patch)) {
    if (!rejected.includes(key)) kept[key] = value;
  }
  return {
    patch: kept as Partial<T>,
    dropped: rejected,
    progressed: Object.keys(kept).length > 0,
  };
}

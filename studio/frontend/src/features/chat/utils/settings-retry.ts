// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /api/chat/settings is extra="forbid" and rejects the whole body on one bad field, so a
// permanently rejected value must be dropped, not requeued forever.

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

/** 4xx is terminal except 408 and 429; network errors, 5xx and timeouts are transient. */
export function isTerminalSettingsRejection(error: unknown): boolean {
  if (!(error instanceof ChatSettingsRequestError)) return false;
  const { status } = error;
  if (status === 408 || status === 429) return false;
  return status >= 400 && status < 500;
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

// Fetch caps total in-flight keepalive bodies at 64 KiB; larger patches go without keepalive.
const KEEPALIVE_BODY_BUDGET_BYTES = 60 * 1024;

export function isUnderKeepaliveBudget(body: string): boolean {
  // JSON is at least one byte per UTF-16 unit, so short bodies skip encoding.
  if (body.length <= KEEPALIVE_BODY_BUDGET_BYTES / 3) return true;
  return new TextEncoder().encode(body).byteLength <= KEEPALIVE_BODY_BUDGET_BYTES;
}

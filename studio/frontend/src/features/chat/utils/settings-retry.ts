// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which failed settings writes are worth keeping, and which have to be let go. The settings
// queue coalesces every change into one patch and requeues it when the PUT fails, which is
// right for a server that is down and wrong for one that refuses this patch.
// /api/chat/settings is extra="forbid" and rejects the WHOLE body on one bad field, so a
// permanently-rejected value requeued forever takes every later save down with it: after one
// 400 no chat setting can persist again for the life of the tab. Reachable without a bug on
// either side whenever the two versions differ.
// A new bundle against a rolled-back backend has every mirrored field rejected as extra_forbidden,
// which is exactly the shape below.

/** A non-2xx answer from the settings endpoint, carrying what the server said. */
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

/** Whether retrying this failure could ever succeed. A network error, a 5xx or a timeout is
 *  transient, so the patch is kept. A 4xx is the server saying this body is wrong. 408 and 429
 *  are the two 4xx that explicitly mean "later", so they stay retryable. */
export function isTerminalSettingsRejection(error: unknown): boolean {
  if (!(error instanceof ChatSettingsRequestError)) return false;
  const { status } = error;
  if (status === 408 || status === 429) return false;
  return status >= 400 && status < 500;
}

/** The top-level setting names a validation error blames, e.g. `["ragTopK"]`. FastAPI answers a
 *  pydantic failure with `detail: [{loc: ["ragTopK"], ...}]`. Only the first element of `loc`
 *  is used: a nested failure means that whole setting is unsendable, and the queue works in
 *  whole settings. An empty result means the field could not be identified, so the caller
 *  drops the patch rather than guessing. */
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

/** The part of `patch` still worth sending after `error`. Returns the patch unchanged when the
 *  failure is transient. On a terminal rejection it drops the named fields and keeps the rest,
 *  so one bad value cannot take the other settings with it; when no field can be named,
 *  everything is dropped. `progressed` is only true when the patch got strictly smaller, which
 *  is what bounds the retry loop by the number of fields. */
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

// The Fetch standard caps the TOTAL in-flight keepalive body at 64 KiB, and a valid
// researchWebsitePolicy carries up to 2000 domains of 253 characters, so a settings patch can
// be an order of magnitude over it. Over the budget the request fails immediately in every
// engine, so sending without keepalive is a chance rather than a certain failure -- and on
// the visibilitychange flush, where the page is only hidden, it simply succeeds.
const KEEPALIVE_BODY_BUDGET_BYTES = 60 * 1024;

export function isUnderKeepaliveBudget(body: string): boolean {
  // A JSON body is at least one byte per UTF-16 unit, so this cheap check settles every ordinary
  // patch without encoding a copy of a large one.
  if (body.length <= KEEPALIVE_BODY_BUDGET_BYTES / 3) return true;
  return new TextEncoder().encode(body).byteLength <= KEEPALIVE_BODY_BUDGET_BYTES;
}

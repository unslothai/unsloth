// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

/**
 * Render a FastAPI error response body into a human-readable string.
 *
 * FastAPI emits 422s as `{detail: Array<{loc, msg, type, input?}>}`. The
 * naive `body.detail || body.message` stringifies the array as
 * `[object Object]` (the #5409 regression). Shared here across chat,
 * export, history, datasets, and recipe-studio.
 *
 * Falls back through: array detail -> string detail -> message -> null.
 */

export type FastApiValidationError = {
  loc?: unknown[];
  msg?: string;
};

export function formatFastApiDetail(detail: unknown): string | null {
  if (typeof detail === "string" && detail) return detail;
  if (!Array.isArray(detail)) return null;
  const parts = detail
    .map((entry) => {
      if (!entry || typeof entry !== "object") return "";
      const { loc, msg } = entry as FastApiValidationError;
      const path = Array.isArray(loc)
        ? loc.filter((segment) => segment !== "body").join(".")
        : "";
      const message = typeof msg === "string" ? msg : "";
      if (path && message) return `${path}: ${message}`;
      return path || message;
    })
    .filter(Boolean);
  return parts.length > 0 ? parts.join("; ") : null;
}

/**
 * Convert a (likely non-ok) Response into the best human-readable error
 * message. Used by *-api.ts wrappers so toasts read `field: msg` instead
 * of `[object Object]` or `Request failed (422)`.
 */
export async function readFastApiError(
  response: Response,
  fallbackPrefix: string = "Request failed",
): Promise<string> {
  try {
    const payload = (await response.json()) as {
      detail?: unknown;
      message?: string;
    };
    const formatted = formatFastApiDetail(payload.detail);
    if (formatted) return formatted;
    if (typeof payload.message === "string" && payload.message) {
      return payload.message;
    }
  } catch {
    // fall through
  }
  return `${fallbackPrefix} (${response.status})`;
}

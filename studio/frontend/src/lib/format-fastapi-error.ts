// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

/** 422 `detail` arrays would stringify as `[object Object]` (#5409). Falls back through: array
 * detail -> string detail -> message -> null. */

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

// Cap recursion for malformed responses.
const MAX_ERROR_BODY_DEPTH = 4;

function formatErrorBody(body: unknown, depth: number): string | null {
  if (!body || typeof body !== "object") return null;

  const payload = body as {
    detail?: unknown;
    message?: unknown;
    error?: unknown;
  };
  const formatted = formatFastApiDetail(payload.detail);
  if (formatted) return formatted;
  // The same route may return a flat envelope or nest it inside `detail`.
  if (
    depth < MAX_ERROR_BODY_DEPTH &&
    payload.detail &&
    typeof payload.detail === "object" &&
    !Array.isArray(payload.detail)
  ) {
    const nested = formatErrorBody(payload.detail, depth + 1);
    if (nested) return nested;
  }
  if (typeof payload.message === "string" && payload.message) {
    return payload.message;
  }
  if (payload.error && typeof payload.error === "object") {
    const message = (payload.error as { message?: unknown }).message;
    if (typeof message === "string" && message) {
      return message;
    }
  }
  return null;
}

export function formatApiErrorBody(body: unknown): string | null {
  return formatErrorBody(body, 0);
}

export async function readFastApiError(
  response: Response,
  fallbackPrefix: string = "Request failed",
): Promise<string> {
  try {
    const formatted = formatApiErrorBody(await response.json());
    if (formatted) return formatted;
  } catch {
    // fall through
  }
  return `${fallbackPrefix} (${response.status})`;
}

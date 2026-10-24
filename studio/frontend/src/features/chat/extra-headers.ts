// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const FORBIDDEN_EXTRA_HEADERS = new Set([
  "authorization",
  "proxy-authorization",
  "cookie",
  "set-cookie",
  "host",
  "content-length",
  "content-type",
  "connection",
  "transfer-encoding",
  "te",
  "trailer",
  "upgrade",
  "keep-alive",
  "api-key",
  "x-api-key",
  "x-goog-api-key",
]);
const HEADER_NAME_RE = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;
const MAX_EXTRA_HEADERS = 16;

export function normalizeProviderExtraHeaders(
  value: unknown,
): Record<string, string> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const headers: Record<string, string> = {};
  for (const [rawName, rawValue] of Object.entries(value as Record<string, unknown>)) {
    if (typeof rawName !== "string" || typeof rawValue !== "string") continue;
    const name = rawName.trim();
    if (!name || !HEADER_NAME_RE.test(name)) continue;
    if (FORBIDDEN_EXTRA_HEADERS.has(name.toLowerCase())) continue;
    headers[name] = rawValue;
  }
  const names = Object.keys(headers);
  if (names.length === 0) return undefined;
  return Object.fromEntries(names.slice(0, MAX_EXTRA_HEADERS).map((name) => [name, headers[name]]));
}

export function extraHeadersToDraft(
  headers: Record<string, string> | undefined,
): string {
  return Object.entries(headers ?? {})
    .map(([name, value]) => `${name}: ${value}`)
    .join("\n");
}

export function parseExtraHeadersDraft(draft: string): Record<string, string> {
  const trimmed = draft.trim();
  if (!trimmed) return {};
  if (trimmed.startsWith("{")) {
    const parsed: unknown = JSON.parse(trimmed);
    const headers = normalizeProviderExtraHeaders(parsed);
    if (!headers) {
      throw new Error("Extra headers JSON must be an object of header names to string values.");
    }
    return headers;
  }
  const headers: Record<string, string> = {};
  for (const line of trimmed.split("\n")) {
    const text = line.trim();
    if (!text || text.startsWith("#")) continue;
    const separator = text.indexOf(":");
    if (separator <= 0) {
      throw new Error(`Invalid extra header line: ${text}`);
    }
    const name = text.slice(0, separator).trim();
    const value = text.slice(separator + 1).trim();
    if (!HEADER_NAME_RE.test(name)) {
      throw new Error(`Invalid extra header name: ${name}`);
    }
    if (FORBIDDEN_EXTRA_HEADERS.has(name.toLowerCase())) {
      throw new Error(`Cannot set reserved header ${name}.`);
    }
    headers[name] = value;
  }
  if (Object.keys(headers).length > MAX_EXTRA_HEADERS) {
    throw new Error(`At most ${MAX_EXTRA_HEADERS} extra headers are allowed.`);
  }
  return headers;
}

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
const HEADER_VALUE_RE = /^[\t\x20-\x7e]*$/;
const MAX_EXTRA_HEADERS = 16;

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
  const entries: [string, unknown][] = [];
  if (trimmed.startsWith("{")) {
    const parsed: unknown = JSON.parse(trimmed);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("Extra headers JSON must be an object of header names to string values.");
    }
    entries.push(...Object.entries(parsed as Record<string, unknown>));
  } else {
    for (const line of trimmed.split("\n")) {
      const text = line.trim();
      if (!text || text.startsWith("#")) continue;
      const separator = text.indexOf(":");
      if (separator <= 0) {
        throw new Error(`Invalid extra header line: ${text}`);
      }
      entries.push([text.slice(0, separator), text.slice(separator + 1)]);
    }
  }
  const headers: Record<string, string> = {};
  for (const [rawName, rawValue] of entries) {
    const name = rawName.trim();
    if (typeof rawValue !== "string") {
      throw new Error(`Header ${name} must have a string value.`);
    }
    const value = rawValue.trim();
    if (!HEADER_NAME_RE.test(name)) {
      throw new Error(`Invalid extra header name: ${name}`);
    }
    if (FORBIDDEN_EXTRA_HEADERS.has(name.toLowerCase())) {
      throw new Error(`Cannot set reserved header ${name}.`);
    }
    if (!HEADER_VALUE_RE.test(value)) {
      throw new Error(`Header ${name} must be printable ASCII.`);
    }
    headers[name] = value;
  }
  if (Object.keys(headers).length > MAX_EXTRA_HEADERS) {
    throw new Error(`At most ${MAX_EXTRA_HEADERS} extra headers are allowed.`);
  }
  return headers;
}

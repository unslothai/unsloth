// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One file a tool call created in the chat's sandbox. */
export type SandboxFile = {
  name: string;
  size: number | null;
};

const FILES_MARKER = "\n__FILES__:";

function isSandboxFile(entry: unknown): entry is SandboxFile {
  if (typeof entry !== "object" || entry === null) return false;
  const { name, size } = entry as { name?: unknown; size?: unknown };
  return (
    typeof name === "string" &&
    name.length > 0 &&
    (size === null || size === undefined || typeof size === "number")
  );
}

/**
 * Split a tool result into its visible text and the files the call created.
 *
 * `__FILES__` sits ahead of `__IMAGES__` because older clients slice from that
 * marker to the end. An unparseable payload leaves the text untouched.
 */
export function extractCreatedFiles(raw: string): {
  text: string;
  files: SandboxFile[];
} {
  const start = raw.lastIndexOf(FILES_MARKER);
  if (start === -1) return { text: raw, files: [] };

  const payloadStart = start + FILES_MARKER.length;
  const nextMarker = raw.indexOf("\n__", payloadStart);
  const end = nextMarker === -1 ? raw.length : nextMarker;
  try {
    const parsed: unknown = JSON.parse(raw.slice(payloadStart, end));
    // Every entry, not just the array: a tool printing `__FILES__:[null]` would
    // otherwise have its output eaten and throw while rendering file.name.
    if (!Array.isArray(parsed) || !parsed.every(isSandboxFile)) {
      return { text: raw, files: [] };
    }
    return { text: raw.slice(0, start) + raw.slice(end), files: parsed };
  } catch {
    return { text: raw, files: [] };
  }
}

export function sandboxFilePath(sessionId: string, filename: string): string {
  // Segment by segment: a file written to outputs/report.csv keeps a real "/"
  // in the URL, which encodeURIComponent on the whole name would have escaped.
  const path = filename
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `/api/inference/sandbox/${encodeURIComponent(sessionId)}/${path}`;
}

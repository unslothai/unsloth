// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One file a tool call created in the chat's sandbox. */
export type SandboxFile = {
  name: string;
  size: number | null;
};

const FILES_MARKER = "\n__FILES__:";

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
    const parsed = JSON.parse(raw.slice(payloadStart, end)) as SandboxFile[];
    if (!Array.isArray(parsed)) return { text: raw, files: [] };
    return { text: raw.slice(0, start) + raw.slice(end), files: parsed };
  } catch {
    return { text: raw, files: [] };
  }
}

export function sandboxFilePath(sessionId: string, filename: string): string {
  return `/api/inference/sandbox/${encodeURIComponent(sessionId)}/${encodeURIComponent(filename)}`;
}

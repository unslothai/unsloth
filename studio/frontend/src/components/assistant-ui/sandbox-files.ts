// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One file a tool call created in the chat's sandbox. */
export type SandboxFile = {
  name: string;
  size: number | null;
};

const FILES_MARKER = "\n__FILES__:";

/** The tools that emit the file envelope. Nothing else's output is an envelope. */
export const SANDBOX_FILE_TOOLS = new Set(["python", "terminal"]);

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

/** ``files`` as the cards need it: absent, or entries with a usable name. */
export function isSandboxFileList(val: unknown): boolean {
  if (val === undefined || val === null) return true;
  if (!Array.isArray(val)) return false;
  return val.every(
    (entry) =>
      typeof entry === "object" &&
      entry !== null &&
      typeof (entry as { name?: unknown }).name === "string",
  );
}

/**
 * A python/terminal result carrying the chat's sandbox context alongside the
 * text the model actually saw.
 */
export function isSandboxToolResult(
  val: unknown,
): val is { text: string; sessionId: string } {
  if (typeof val !== "object" || val === null) return false;
  const v = val as {
    text?: unknown;
    sessionId?: unknown;
    images?: unknown;
    files?: unknown;
  };
  // images too: it is always in Studio's own wrapper, and a tool result that
  // merely has text and sessionId is someone else's, whose other fields would
  // be dropped on export.
  return (
    typeof v.text === "string" &&
    typeof v.sessionId === "string" &&
    Array.isArray(v.images) &&
    // Persisted content can carry anything: the cards map over this and read
    // name off each entry, so anything else takes the whole chat view down.
    isSandboxFileList(v.files)
  );
}

/** Ids a path segment can carry: ASGI decodes %2F before it matches a route. */
const PATH_SAFE_SESSION = /^[A-Za-z0-9_-]{1,64}$/;

/** Where this session's files live, as the routes expect to be asked. */
export function sandboxRoutePrefix(sessionId: string): {
  prefix: string;
  query: string;
} {
  if (PATH_SAFE_SESSION.test(sessionId)) {
    return {
      prefix: `/api/inference/sandbox/${encodeURIComponent(sessionId)}`,
      query: "",
    };
  }
  // An API client can pick anything; carry it where it survives the round trip.
  return {
    prefix: "/api/inference/sandbox/_",
    query: `?session=${encodeURIComponent(sessionId)}`,
  };
}

/**
 * The sandbox a chat's tool calls run in. Threads inside a project share the
 * project's workspace, so their files land in one place instead of one folder
 * per thread.
 */
export function sandboxSessionIdFor(
  threadId: string | undefined,
  projectId: string | null | undefined,
): string | undefined {
  return projectId ? `project-${projectId}` : threadId;
}

export function sandboxFilePath(sessionId: string, filename: string): string {
  // Segment by segment: a file written to outputs/report.csv keeps a real "/"
  // in the URL, which encodeURIComponent on the whole name would have escaped.
  const path = filename
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  return `${prefix}/${path}${query}`;
}

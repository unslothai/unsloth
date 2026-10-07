// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type SandboxFile = {
  name: string;
  size: number | null;
};

const FILES_MARKER = "\n__FILES__:";

export const SANDBOX_FILE_TOOLS = new Set(["python", "terminal"]);

/** Includes hosted code_execution: the backend keeps its `__IMAGES__` line for the model too. */
export const IMAGE_SENTINEL_TOOLS = new Set([...SANDBOX_FILE_TOOLS, "code_execution"]);

function isSandboxFile(entry: unknown): entry is SandboxFile {
  if (typeof entry !== "object" || entry === null) return false;
  const { name, size } = entry as { name?: unknown; size?: unknown };
  return (
    typeof name === "string" &&
    name.length > 0 &&
    (size === null || size === undefined || typeof size === "number")
  );
}

/** `__FILES__` sits ahead of `__IMAGES__` because older clients slice from that marker on. */
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
    // Check every entry: `__FILES__:[null]` would otherwise throw while rendering file.name.
    if (!Array.isArray(parsed) || !parsed.every(isSandboxFile)) {
      return { text: raw, files: [] };
    }
    return { text: raw.slice(0, start) + raw.slice(end), files: parsed };
  } catch {
    return { text: raw, files: [] };
  }
}

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
  // Requires Unsloth's own wrapper fields; a result with only text and sessionId is someone else's.
  return (
    typeof v.text === "string" &&
    typeof v.sessionId === "string" &&
    Array.isArray(v.images) &&
    // Persisted content can carry anything; a bad entry would take the whole chat view down.
    isSandboxFileList(v.files)
  );
}

export function hasCreatedFiles(toolName: unknown, result: unknown): boolean {
  if (typeof toolName !== "string" || !SANDBOX_FILE_TOOLS.has(toolName)) return false;
  if (!isSandboxToolResult(result)) return false;
  const { files } = result as { files?: unknown[] | null };
  return Array.isArray(files) && files.length > 0;
}

/** Ids a path segment can carry: ASGI decodes %2F before it matches a route. */
const PATH_SAFE_SESSION = /^[A-Za-z0-9_-]{1,64}$/;

/** Shared with model-written markdown links, which carry this prefix back. */
export const SANDBOX_ROUTE_PREFIX = "/api/inference/sandbox/";

export function sandboxRoutePrefix(sessionId: string): {
  prefix: string;
  query: string;
} {
  if (PATH_SAFE_SESSION.test(sessionId)) {
    return {
      prefix: `${SANDBOX_ROUTE_PREFIX}${encodeURIComponent(sessionId)}`,
      query: "",
    };
  }
  return {
    prefix: "/api/inference/sandbox/_",
    query: `?session=${encodeURIComponent(sessionId)}`,
  };
}

/** Threads inside a project share the project's workspace. */
export function sandboxSessionIdFor(
  threadId: string | undefined,
  projectId: string | null | undefined,
): string | undefined {
  return projectId ? `project-${projectId}` : threadId;
}

/**
 * Mirrors `_SANDBOX_MEDIA_TYPES` in `backend/routes/inference.py`. `.svg` is absent on purpose:
 * an inline model-named SVG would be same-origin script execution.
 */
export const SANDBOX_INLINE_IMAGE_EXTS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".bmp",
  ".avif",
]);

/** Somebody else's URL: `data:`/`blob:` render as they are, `http(s)` is blocked by the sanitizer. */
const HAS_SCHEME_RE = /^[a-zA-Z][a-zA-Z0-9+\-.]*:/;
const PROTOCOL_RELATIVE_RE = /^[/\\]{2}/;

/** A stray `%` must not throw on every render; also decodes download names from the route path. */
export function decodeSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

/**
 * The sandbox file a model-written `src` points at, or null. Bare relative paths count; one
 * carrying `..` stays raw so it cannot fetch another chat's file or route.
 */
export function sandboxFileForSrc(src: string): string | null {
  const file = sandboxPathForSrc(src);
  if (file === null) return null;
  const name = file.slice(file.lastIndexOf("/") + 1);
  const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
  return SANDBOX_INLINE_IMAGE_EXTS.has(ext) ? file : null;
}

// What a one-segment link must end in to be a file, since `example.tech` is a site.
const LINK_FILE_EXTS = new Set(
  ("csv tsv json jsonl txt md markdown html htm pdf png jpg jpeg gif webp svg bmp py ipynb js ts sh " +
    "yaml yml toml xml log xlsx xls docx doc pptx odt ods zip tar gz parquet wav mp3 mp4 webm mov").split(" "),
);
// A first segment shaped like a domain (`docs.museum/report.pdf`, `пример.рф`) or an IPv4 address, not a folder.
const HOST_SEGMENT_RE = /^(?:(?:[\p{L}\p{N}-]+\.)+(?:\p{L}{2,}|xn--[a-z\d-]+)|\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?$/iu;

/** Sandbox file a markdown link targets (`outputs/report.csv`); needs an extension, so `#intro` stays a link. */
export function sandboxFileForHref(href: string): string | null {
  const trimmed = href.trim();
  if (trimmed.startsWith("#") || /^www\./i.test(trimmed)) return null;
  const file = sandboxPathForSrc(href);
  const ext = file && /[^/]\.([A-Za-z0-9]{1,8})$/.exec(file)?.[1]?.toLowerCase();
  if (!file || !ext) return null;
  const [first, ...rest] = file.split("/");
  if (rest.length === 0) return LINK_FILE_EXTS.has(ext) ? file : null;
  return HOST_SEGMENT_RE.test(first ?? "") ? null : file;
}

function sandboxPathForSrc(src: string): string | null {
  const trimmed = src.trim();
  if (!trimmed || HAS_SCHEME_RE.test(trimmed) || PROTOCOL_RELATIVE_RE.test(trimmed)) {
    return null;
  }
  const path = trimmed.split("?")[0].split("#")[0];
  if (path.startsWith("/") && !path.startsWith(SANDBOX_ROUTE_PREFIX)) {
    return null;
  }
  const segments = path.startsWith(SANDBOX_ROUTE_PREFIX)
    ? // The recorded sid is not part of the file path; sandboxSessionInSrc reads it back out.
      path.slice(SANDBOX_ROUTE_PREFIX.length).split("/").slice(1)
    : path.split("/");
  // Decode FIRST, then judge: `%2e%2e` is `..`, and a dot segment escapes the caller's scope.
  const decoded = segments.map(decodeSegment);
  // Encoded separators must not become new path segments after this check.
  if (decoded.some((segment) => segment === ".." || /[/\\]/.test(segment))) {
    return null;
  }
  const parts = decoded.filter((segment) => segment !== ".");
  return parts.length > 0 ? parts.join("/") : null;
}

export function sandboxSessionInSrc(src: string): string | null {
  const trimmed = src.trim();
  if (!trimmed || HAS_SCHEME_RE.test(trimmed) || PROTOCOL_RELATIVE_RE.test(trimmed)) {
    return null;
  }
  const [path, ...query] = trimmed.split("#")[0].split("?");
  if (!path.startsWith(SANDBOX_ROUTE_PREFIX)) return null;
  const segment = path.slice(SANDBOX_ROUTE_PREFIX.length).split("/")[0] ?? "";
  // `sandboxRoutePrefix` carries a not-path-safe id in the query under `_`; mirror it back out.
  if (query.length > 0) {
    const session = new URLSearchParams(query.join("?")).get("session");
    if (session) return session;
  }
  return segment ? decodeSegment(segment) : null;
}

/**
 * The session the src RECORDS wins: files stay where they were written when a chat moves
 * between projects. Only a path that records nothing falls back to the chat's current scope.
 */
export function markdownSandboxImageSrc(
  src: string,
  ctx: { threadId: string | undefined; projectId: string | null | undefined },
): string | null {
  const file = sandboxFileForSrc(src);
  if (file === null) return null;
  const sessionId =
    sandboxSessionInSrc(src) ?? sandboxSessionIdFor(ctx.threadId, ctx.projectId);
  return sessionId ? sandboxFilePath(sessionId, file) : null;
}

export function sandboxFilePath(sessionId: string, filename: string): string {
  // Segment by segment so a real "/" survives; encodeURIComponent on the whole name escapes it.
  const path = filename
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  return `${prefix}/${path}${query}`;
}

/** Route URL for a link to a tool-written file (a bare relative path would be blocked); null otherwise. */
export function markdownSandboxLinkHref(
  href: string,
  ctx: { threadId: string | undefined; projectId: string | null | undefined },
): string | null {
  const file = sandboxFileForHref(href);
  if (file === null) return null;
  const sessionId = sandboxSessionInSrc(href) ?? sandboxSessionIdFor(ctx.threadId, ctx.projectId);
  return sessionId ? sandboxFilePath(sessionId, file) : null;
}

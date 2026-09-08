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
  // images too: it is always in Unsloth's own wrapper, and a tool result that
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

/**
 * The route both sides spell: `sandboxRoutePrefix` builds it for the cards, and a model that
 * writes `![plot](/api/inference/sandbox/<sid>/plot.png)` into its own markdown carries it back.
 */
export const SANDBOX_ROUTE_PREFIX = "/api/inference/sandbox/";

/** Where this session's files live, as the routes expect to be asked. */
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

/**
 * Extensions the route serves inline as an image, mirroring `_SANDBOX_MEDIA_TYPES` in
 * `backend/routes/inference.py`. `.svg` is absent on purpose: the filename is model-chosen, so an
 * inline SVG would be same-origin script execution and the route keeps serving it as a download.
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

/**
 * `100%.png` is a real filename; a stray `%` must not throw on every render. Exported because the
 * download name is cut from the ENCODED route path and must be decoded back before it names a file.
 */
export function decodeSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

/**
 * The FILE a model-written markdown `src` points at, or null when it does not point at a sandbox
 * file at all. A bare relative path (`outputs/plot.png`) counts: the sanitizer already dropped every
 * scheme-carrying src, so a scheme-less image path in an answer is this chat's file and nothing else
 * -- and one carrying `..` points somewhere that is NOT this chat's folder, so it stays raw and fails
 * honestly instead of silently fetching another chat's file (or another route).
 */
export function sandboxFileForSrc(src: string): string | null {
  const trimmed = src.trim();
  if (!trimmed || HAS_SCHEME_RE.test(trimmed) || PROTOCOL_RELATIVE_RE.test(trimmed)) {
    return null;
  }
  // The sid rides in `?session=` when it is not path-safe; either way the caller reads it back with
  // sandboxSessionInSrc, so it never reaches the file path.
  const path = trimmed.split("?")[0].split("#")[0];
  if (path.startsWith("/") && !path.startsWith(SANDBOX_ROUTE_PREFIX)) {
    return null; // some other app route (`/assets/...`), not a sandbox file
  }
  const segments = path.startsWith(SANDBOX_ROUTE_PREFIX)
    ? // The recorded sid is not part of the file path; sandboxSessionInSrc reads it back out.
      path.slice(SANDBOX_ROUTE_PREFIX.length).split("/").slice(1)
    : path.split("/");
  // Decode FIRST, then judge: `%2e%2e` IS `..`, and a dot segment pops the scope segment the caller
  // prepends -- one reads another chat's folder, two land on another route. A bare `.` is noise URL
  // parsing drops anyway; drop it here so callers see one canonical shape.
  const decoded = segments.map(decodeSegment);
  // Encoded separators must not become new path segments after this check.
  if (decoded.some((segment) => segment === ".." || /[/\\]/.test(segment))) {
    return null;
  }
  const parts = decoded.filter((segment) => segment !== ".");
  const name = parts[parts.length - 1] ?? "";
  const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
  // A `.csv` is a download card, not an `<img>`; leave it to the file cards.
  if (!SANDBOX_INLINE_IMAGE_EXTS.has(ext)) return null;
  return parts.join("/");
}

/**
 * The session a `src` records for itself: the segment after the route prefix (or the `?session=`
 * query when the id was not path-safe at write time). A bare relative path records nothing: null.
 */
export function sandboxSessionInSrc(src: string): string | null {
  const trimmed = src.trim();
  if (!trimmed || HAS_SCHEME_RE.test(trimmed) || PROTOCOL_RELATIVE_RE.test(trimmed)) {
    return null;
  }
  const [path, ...query] = trimmed.split("#")[0].split("?");
  // A bare relative path records nothing: the caller's fallback scope is all there is.
  if (!path.startsWith(SANDBOX_ROUTE_PREFIX)) return null;
  const segment = path.slice(SANDBOX_ROUTE_PREFIX.length).split("/")[0] ?? "";
  // `sandboxRoutePrefix` carries a not-path-safe id in the query under `_`; mirror it back out.
  if (query.length > 0) {
    const session = new URLSearchParams(query.join("?")).get("session");
    // URLSearchParams has already decoded the query value.
    if (session) return session;
  }
  return segment ? decodeSegment(segment) : null;
}

/**
 * The URL to fetch for a sandbox `src`. The session the src RECORDS wins when it names one: that is
 * the folder the files landed in when the message was WRITTEN -- where a chat moved between projects
 * still has its older files, and exactly what the tool card above the prose resolves from its own
 * persisted envelope. A model echoes real workdir paths out of the stdout it saw; discarding that echo
 * is what broke those answers' images after a move. Only a path that records nothing (a bare
 * `outputs/plot.png`) falls back to this chat's CURRENT scope: `project-<id>` else threadId.
 */
export function markdownSandboxImageSrc(
  src: string,
  ctx: { threadId: string | undefined; projectId: string | null | undefined },
): string | null {
  const file = sandboxFileForSrc(src);
  if (file === null) return null;
  const sessionId =
    sandboxSessionInSrc(src) ?? sandboxSessionIdFor(ctx.threadId, ctx.projectId);
  // No recorded session and no thread yet means no directory to read from; the raw src stays as it is.
  return sessionId ? sandboxFilePath(sessionId, file) : null;
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

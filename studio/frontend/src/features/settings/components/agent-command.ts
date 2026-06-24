// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Build the `unsloth start <agent>` command for the API-keys panel.
// `unsloth start` reads UNSLOTH_STUDIO_URL (default 127.0.0.1:8888) and only
// auto-mints a key for a loopback server, so the bare command is correct only for
// the default local server. For a non-default port or tunnel/remote base, emit the
// URL (plus a key for non-loopback) so the copy targets what the UI shows.

const DEFAULT_STUDIO_PORT = "8888";
const DEFAULT_AGENT = "claude";

function isLoopbackHost(host: string): boolean {
  return (
    host === "127.0.0.1" ||
    host === "localhost" ||
    host === "::1" ||
    host === "[::1]"
  );
}

export function buildAgentCommand(
  base: string | null | undefined,
  key: string | null | undefined,
  os: "unix" | "windows",
  agent: string = DEFAULT_AGENT,
): string {
  const bare = `unsloth start ${agent}`;

  let url: URL | null = null;
  try {
    if (base) url = new URL(base);
  } catch {
    url = null;
  }
  // Unknown base: fall back to the bare default-local command.
  if (!url) return bare;

  const host = url.hostname.toLowerCase();
  const loopback = isLoopbackHost(host);
  // Default local server (127.0.0.1/localhost:8888): bare command auto-discovers it.
  if (loopback && url.port === DEFAULT_STUDIO_PORT) return bare;

  // Non-default server: set the URL; non-loopback also needs an explicit key.
  let cmd = bare;
  if (!loopback && key) cmd += ` --api-key ${key}`;

  const studioUrl = url.origin;
  return os === "windows"
    ? `$env:UNSLOTH_STUDIO_URL="${studioUrl}"; ${cmd}`
    : `UNSLOTH_STUDIO_URL=${studioUrl} ${cmd}`;
}

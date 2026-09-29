// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const STUDIO_DECISIONS_URL =
  /^http:\/\/(\[[^\]]+\]|[^/:?#]+)(?::\d+)?\/mcp\/decisions\/?(?:[?#].*)?$/i;

function isStudioLoopbackHost(host: string): boolean {
  const lower = host.toLowerCase();
  if (lower === "localhost") return true;
  if (!lower.startsWith("[") || !lower.endsWith("]")) {
    const octets = lower.split(".");
    return (
      octets.length === 4 &&
      octets[0] === "127" &&
      octets.every(
        (octet) =>
          /^\d{1,3}$/.test(octet) &&
          Number(octet) <= 255 &&
          String(Number(octet)) === octet,
      )
    );
  }

  const literal = lower.slice(1, -1).replace(/%25/i, "%").split("%", 1)[0];
  try {
    const canonical = new URL(`http://[${literal}]/`).hostname.toLowerCase();
    return (
      canonical === "[::1]" ||
      /^\[::ffff:7f[0-9a-f]{2}:[0-9a-f]{1,4}\]$/.test(canonical)
    );
  } catch {
    return false;
  }
}

// mcp_servers has no UNIQUE(url); normalize Studio's in-process Decisions endpoint so a preset
// toggle reuses rows saved under older ports and every backend-accepted loopback spelling.
export function normalizeMcpUrl(url: string): string {
  const normalized = (url || "").trim().toLowerCase().replace(/\/+$/, "");
  const match = STUDIO_DECISIONS_URL.exec(normalized);
  return match && isStudioLoopbackHost(match[1])
    ? "studio:decisions"
    : normalized;
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const DEFAULT_PORTS: Record<string, string> = { "http:": "80", "https:": "443" };

/**
 * Comparable form of a Hub endpoint. A trailing slash, different case or an
 * explicit :443 all name the same deployment; a path does not, since a mirror
 * can be mounted on any host, the official one included.
 *
 * Kept import-free so it is unit-testable outside vite.
 */
export function endpointKey(raw: string): string {
  try {
    const u = new URL(raw);
    const port = u.port || DEFAULT_PORTS[u.protocol] || "";
    // Trailing slashes only: the rest of the path is case-sensitive upstream.
    const path = u.pathname.replace(/\/+$/, "");
    return `${u.protocol}//${u.hostname.toLowerCase()}:${port}${path}`;
  } catch {
    return raw;
  }
}

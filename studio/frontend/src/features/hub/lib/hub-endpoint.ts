// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DEFAULT_HUB_ENDPOINT, usePlatformStore } from "@/config/env";
import { setHubProxyFirst } from "./hub-transport";

/**
 * A mirror is unreachable from the browser: the page CSP lists only the official
 * Hub, and the desktop policy is fixed at build time. Route those through the
 * backend up front rather than burning a guaranteed failure to find out.
 */
export function hubProxyFirst(): boolean {
  try {
    const endpoint = usePlatformStore.getState().hubEndpoint;
    if (!endpoint) return false;
    // Normalised: a trailing slash, different case or an explicit :443 is still
    // the official Hub, and forcing those through the proxy would give up
    // direct browser access for nothing.
    return originKey(endpoint) !== originKey(DEFAULT_HUB_ENDPOINT);
  } catch {
    return false;
  }
}

const DEFAULT_PORTS: Record<string, string> = { "http:": "80", "https:": "443" };

function originKey(raw: string): string {
  try {
    const u = new URL(raw);
    const port = u.port || DEFAULT_PORTS[u.protocol] || "";
    return `${u.protocol}//${u.hostname.toLowerCase()}:${port}`;
  } catch {
    return raw;
  }
}

setHubProxyFirst(hubProxyFirst);

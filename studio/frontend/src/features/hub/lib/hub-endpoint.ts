// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DEFAULT_HUB_ENDPOINT, usePlatformStore } from "@/config/env";
import { setHubProxyFirst } from "./hub-transport";

/**
 * A mirror is unreachable from the browser: the CSP lists only the official Hub
 * and the desktop policy is fixed at build time. Proxy up front rather than
 * burning a guaranteed failure to find out.
 */
export function hubProxyFirst(): boolean {
  try {
    const endpoint = usePlatformStore.getState().hubEndpoint;
    if (!endpoint) return false;
    // Normalised: a trailing slash, different case or an explicit :443 is still
    // the official Hub, and proxying those gives up direct access for nothing.
    return originKey(endpoint) !== originKey(DEFAULT_HUB_ENDPOINT);
  } catch {
    return false;
  }
}

/**
 * The configured Hub endpoint, trailing slash removed. The path is kept: the
 * backend builds its URLs from the whole of HF_ENDPOINT, so a subpath-mounted
 * mirror (https://mirror.example/hf) would otherwise have every relative asset
 * resolved one directory too high.
 */
export function hubEndpointBase(): string {
  try {
    const endpoint = usePlatformStore.getState().hubEndpoint;
    if (!endpoint) return DEFAULT_HUB_ENDPOINT;
    return new URL(endpoint).href.replace(/\/+$/, "");
  } catch {
    return DEFAULT_HUB_ENDPOINT;
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

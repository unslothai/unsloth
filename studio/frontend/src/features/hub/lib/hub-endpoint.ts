// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DEFAULT_HUB_ENDPOINT, usePlatformStore } from "@/config/env";
import { endpointKey } from "./endpoint-key";
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
    // Normalised, path included: a trailing slash or :443 is still the official
    // Hub, but https://huggingface.co/hf is a mirror mounted on its host, and on
    // origin alone that was fetched direct and read the wrong repo.
    return endpointKey(endpoint) !== endpointKey(DEFAULT_HUB_ENDPOINT);
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

setHubProxyFirst(hubProxyFirst);

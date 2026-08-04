// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DEFAULT_HUB_ENDPOINT, usePlatformStore } from "@/config/env";

/**
 * A mirror is unreachable from the browser: the page CSP lists only the official
 * Hub, and the desktop policy is fixed at build time. Route those through the
 * backend up front rather than burning a guaranteed failure to find out.
 */
export function hubProxyFirst(): boolean {
  try {
    const endpoint = usePlatformStore.getState().hubEndpoint;
    return Boolean(endpoint) && endpoint !== DEFAULT_HUB_ENDPOINT;
  } catch {
    return false;
  }
}

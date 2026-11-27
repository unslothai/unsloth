// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// startViewTransition crashes WebKitGTK 2.50 to 2.54 in any embedder, not just the desktop app (#16028).
export function isWebKitGTK(platform: string, userAgent: string): boolean {
  return (
    /linux/i.test(`${platform} ${userAgent}`) &&
    /AppleWebKit/.test(userAgent) &&
    !/android|chrom|edg/i.test(userAgent)
  );
}

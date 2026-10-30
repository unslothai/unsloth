// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Without accelerated compositing a view transition crashes WebKitGTK 2.50 to 2.54 (#16028).
export function isLinuxDesktopWebKit(
  desktop: boolean,
  platform: string,
  userAgent: string,
): boolean {
  return desktop && /linux/i.test(`${platform} ${userAgent}`) && !/android/i.test(userAgent);
}

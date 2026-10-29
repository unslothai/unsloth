// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether this is the Linux desktop app, where the theme switch skips the view transition.
 * WebKitGTK 2.54 aborts the web process on a view transition while accelerated compositing is
 * off (NVIDIA X11 drivers before 610, NVIDIA Wayland, the WEBKIT_DISABLE_* variables), and 2.50
 * and 2.52 never finish one there (#16028).
 */
export function isLinuxDesktopWebKit(
  desktop: boolean,
  platform: string,
  userAgent: string,
): boolean {
  return desktop && /linux/i.test(`${platform} ${userAgent}`) && !/android/i.test(userAgent);
}

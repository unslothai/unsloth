// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether this is the Linux desktop app, where the theme switch skips the view transition.
 * With accelerated compositing off (NVIDIA X11 drivers before 610, NVIDIA Wayland, the
 * WEBKIT_DISABLE_* variables), a view transition aborts the web process on WebKitGTK 2.54 and
 * crashes the app's UI process on 2.50 and 2.52 (#16028).
 */
export function isLinuxDesktopWebKit(
  desktop: boolean,
  platform: string,
  userAgent: string,
): boolean {
  return desktop && /linux/i.test(`${platform} ${userAgent}`) && !/android/i.test(userAgent);
}

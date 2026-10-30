// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isLinuxDesktopWebKit } from "../src/components/ui/theme-view-transition.ts";

const LINUX_WEBKIT =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";
const MAC_WEBKIT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)";
const WINDOWS_EDGE =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36 Edg/130.0";
const ANDROID_CHROME =
  "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Mobile Safari/537.36";

test("only the Linux desktop app skips the theme view transition", () => {
  assert.equal(isLinuxDesktopWebKit(true, "Linux x86_64", LINUX_WEBKIT), true);
  assert.equal(isLinuxDesktopWebKit(true, "", LINUX_WEBKIT), true);
  assert.equal(isLinuxDesktopWebKit(true, "MacIntel", MAC_WEBKIT), false);
  assert.equal(isLinuxDesktopWebKit(true, "Win32", WINDOWS_EDGE), false);
  assert.equal(isLinuxDesktopWebKit(false, "Linux x86_64", LINUX_WEBKIT), false);
  assert.equal(isLinuxDesktopWebKit(false, "Linux armv8l", ANDROID_CHROME), false);
  assert.equal(isLinuxDesktopWebKit(true, "Linux armv8l", ANDROID_CHROME), false);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isWebKitGTK } from "../src/components/ui/theme-view-transition.ts";

// Read from WebKitGTK 2.50.4 (Ubuntu 24.04), the engine under the Linux desktop app.
const WEBKITGTK =
  "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/60.5 Safari/605.1.15";
const MAC_WEBKIT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)";
const WINDOWS_EDGE =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36 Edg/130.0";
const LINUX_CHROME =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36";
const LINUX_HEADLESS_CHROME =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/130.0 Safari/537.36";
const LINUX_FIREFOX = "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0";
const ANDROID_CHROME =
  "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Mobile Safari/537.36";

test("only WebKitGTK skips the theme view transition", () => {
  assert.equal(isWebKitGTK("Linux x86_64", WEBKITGTK), true);
  assert.equal(isWebKitGTK("", WEBKITGTK), true);
  assert.equal(isWebKitGTK("MacIntel", MAC_WEBKIT), false);
  assert.equal(isWebKitGTK("Win32", WINDOWS_EDGE), false);
  assert.equal(isWebKitGTK("Linux x86_64", LINUX_CHROME), false);
  assert.equal(isWebKitGTK("Linux x86_64", LINUX_HEADLESS_CHROME), false);
  assert.equal(isWebKitGTK("Linux x86_64", LINUX_FIREFOX), false);
  assert.equal(isWebKitGTK("Linux armv8l", ANDROID_CHROME), false);
});

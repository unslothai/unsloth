// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  dictationEntryMode,
  insecureDictationGuidance,
  recordingPickerPlatform,
} from "../src/features/chat/utils/dictation-entry.ts";

test("ordinary HTTP LAN uses the recording-file fallback", () => {
  assert.equal(
    dictationEntryMode({
      isSecureContext: false,
      protocol: "http:",
      hostname: "192.168.1.42",
    }),
    "recording-file",
  );
});

test("HTTPS and loopback stay on live dictation", () => {
  assert.equal(
    dictationEntryMode({
      isSecureContext: true,
      protocol: "https:",
      hostname: "studio.example.com",
    }),
    "live",
  );
  for (const hostname of [
    "localhost",
    "studio.localhost",
    "127.0.0.1",
    "::1",
    "[::1]",
  ]) {
    assert.equal(
      dictationEntryMode({
        isSecureContext: false,
        protocol: "http:",
        hostname,
      }),
      "live",
    );
  }
});

test("a non-HTTP insecure context keeps the existing live error path", () => {
  assert.equal(
    dictationEntryMode({
      isSecureContext: false,
      protocol: "https:",
      hostname: "studio.example.com",
    }),
    "live",
  );
  assert.equal(
    insecureDictationGuidance("live"),
    "Open Unsloth at http://127.0.0.1 (localhost) or over HTTPS to dictate.",
  );
  assert.doesNotMatch(insecureDictationGuidance("live"), /choose a recording/i);
});

test("only the HTTP fallback advertises the recording picker", () => {
  assert.match(
    insecureDictationGuidance("recording-file"),
    /Press Dictate to choose a recording on this connection/,
  );
});

test("only Android receives the recorder action", () => {
  assert.equal(
    recordingPickerPlatform({
      userAgent: "Mozilla/5.0 (Linux; Android 15; Pixel 9)",
    }),
    "android",
  );
  assert.equal(
    recordingPickerPlatform({
      userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)",
    }),
    "ios",
  );
  assert.equal(
    recordingPickerPlatform({
      userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
      platform: "MacIntel",
      maxTouchPoints: 5,
    }),
    "ios",
  );
  assert.equal(
    recordingPickerPlatform({ userAgent: "Mozilla/5.0 (Windows NT 10.0)" }),
    "other",
  );
});

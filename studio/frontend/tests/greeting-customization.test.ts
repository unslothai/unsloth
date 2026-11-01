// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_GREETING,
  GREETING_ICON_LIMITS,
  GREETING_IMAGE_LIMITS,
  resolveGreeting,
  sanitizeGreeting,
  validGreetingUpload,
} from "../src/features/profile/greeting.ts";
import { readText } from "./helpers/kit.ts";

const PNG = "data:image/png;base64,AAAA";

test("the default greeting is the sloth with the built-in lines", () => {
  assert.deepEqual(resolveGreeting(DEFAULT_GREETING), {
    kind: "text",
    icon: { kind: "sloth" },
    text: null,
  });
});

test("the old switch still turns the whole greeting off", () => {
  assert.deepEqual(
    resolveGreeting({
      ...DEFAULT_GREETING,
      showGreetingSloth: false,
      greetingStyle: "image",
      greetingImageDataUrl: PNG,
    }),
    { kind: "off" },
  );
});

test("each mode resolves to what the welcome screen draws", () => {
  assert.deepEqual(
    resolveGreeting({
      ...DEFAULT_GREETING,
      greetingIcon: "custom",
      greetingIconDataUrl: PNG,
      greetingText: "  Hi  ",
    }),
    { kind: "text", icon: { kind: "custom", src: PNG }, text: "Hi" },
  );
  assert.deepEqual(
    resolveGreeting({ ...DEFAULT_GREETING, greetingIcon: "none" }),
    {
      kind: "text",
      icon: { kind: "none" },
      text: null,
    },
  );
  assert.deepEqual(
    resolveGreeting({
      ...DEFAULT_GREETING,
      greetingStyle: "image",
      greetingImageDataUrl: PNG,
    }),
    { kind: "image", src: PNG },
  );
});

test("a style whose upload is missing falls back instead of drawing nothing", () => {
  assert.equal(
    resolveGreeting({ ...DEFAULT_GREETING, greetingStyle: "image" }).kind,
    "text",
  );
  assert.deepEqual(
    resolveGreeting({ ...DEFAULT_GREETING, greetingIcon: "custom" }),
    { kind: "text", icon: { kind: "sloth" }, text: null },
  );
});

test("stored values that are not raster data URLs never reach an img", () => {
  for (const bad of [
    "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
    "javascript:alert(1)",
    "https://example.com/a.png",
    "data:image/png;base64,AA AA",
  ]) {
    const s = sanitizeGreeting({
      greetingIconDataUrl: bad,
      greetingImageDataUrl: bad,
    });
    assert.equal(s.greetingIconDataUrl, null, bad);
    assert.equal(s.greetingImageDataUrl, null, bad);
  }
  const s = sanitizeGreeting({
    greetingStyle: "video",
    greetingIcon: 3,
    greetingText: "x".repeat(500),
  });
  assert.equal(s.greetingStyle, "text");
  assert.equal(s.greetingIcon, "sloth");
  assert.equal(s.greetingText.length, 100);
});

test("uploads are image types under the cap, never SVG", () => {
  assert.equal(validGreetingUpload({ type: "image/png", size: 1000 }), true);
  assert.equal(validGreetingUpload({ type: "image/gif", size: 1000 }), true);
  assert.equal(
    validGreetingUpload({ type: "image/svg+xml", size: 1000 }),
    false,
  );
  assert.equal(validGreetingUpload({ type: "text/html", size: 1000 }), false);
  assert.equal(validGreetingUpload({ type: "image/png", size: 0 }), false);
  assert.equal(
    validGreetingUpload({ type: "image/png", size: 11 * 1024 * 1024 }),
    false,
  );
});

// The encoded data URL is ~4/3 of maxBytes; the server must accept the largest one the client makes.
test("the client size limits fit inside the server caps", () => {
  const source = readText("../../backend/routes/settings.py");
  const icon = /MAX_GREETING_ICON_DATA_URL_LENGTH = (\d+) \* 1024/.exec(source);
  assert.ok(icon, "could not find MAX_GREETING_ICON_DATA_URL_LENGTH");
  assert.ok(GREETING_ICON_LIMITS.maxBytes * 1.35 < Number(icon[1]) * 1024);
  const avatar = readText("../../backend/utils/personalization_settings.py");
  const image = /MAX_AVATAR_DATA_URL_BYTES = (\d+) \* 1024/.exec(avatar);
  assert.ok(image, "could not find MAX_AVATAR_DATA_URL_BYTES");
  assert.match(
    source,
    /MAX_GREETING_IMAGE_DATA_URL_LENGTH = MAX_AVATAR_DATA_URL_BYTES/,
  );
  assert.ok(GREETING_IMAGE_LIMITS.maxBytes * 1.35 < Number(image[1]) * 1024);
});

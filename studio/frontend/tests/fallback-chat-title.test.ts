// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  FALLBACK_CHAT_TITLE_MAX_LENGTH,
  fallbackTitleFromUserText,
} from "../src/features/chat/utils/fallback-chat-title.ts";

test("fallback titles retain text for responsive CSS ellipsis", () => {
  const title =
    "Use Python to generate and plot a Mandelbrot set with a custom colour palette";

  assert.equal(fallbackTitleFromUserText(title), title);
  assert.equal(fallbackTitleFromUserText(`${title}\nSecond line`), title);
  assert.equal(
    fallbackTitleFromUserText("  words    keep   spacing  "),
    "words keep spacing",
  );
  assert.equal(fallbackTitleFromUserText(""), "New Chat");
});

test("fallback titles remain bounded without storing a literal ellipsis", () => {
  const title = fallbackTitleFromUserText(
    "x".repeat(FALLBACK_CHAT_TITLE_MAX_LENGTH + 20),
  );

  assert.equal(title, "x".repeat(FALLBACK_CHAT_TITLE_MAX_LENGTH));
  assert.equal(title.endsWith("..."), false);
});

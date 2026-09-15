// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// Source-reading: one variable must reach every surface bounding the column,
// including the composer outside the thread root.

test("every width consumer reads --studio-chat-width", async () => {
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");
  const chatPage = await readSrcAsync("features/chat/chat-page.tsx");
  const css = await readSrcAsync("index.css");

  // Anchored to the space available, not an absolute cap: an absolute one
  // renders the same at two different settings once the window is the limit.
  assert.match(css, /--studio-chat-fill:\s*0;/);
  assert.match(
    css,
    /--studio-chat-width:\s*calc\(48rem \+ \(100% - 48rem\) \* var\(--studio-chat-fill\)\);/,
  );

  // Messages read --thread-content-max-width, which derives from this.
  assert.match(
    thread,
    /\["--thread-max-width" as string\]: "var\(--studio-chat-width, 48rem\)"/,
  );
  assert.match(
    thread,
    /\["--thread-content-max-width" as string\]:\s*\n?\s*"calc\(var\(--thread-max-width\) - 1\.5rem\)"/,
  );

  // Composer inside the thread, held 2rem narrower than the column as before.
  assert.match(
    thread,
    /unsloth-composer-shell[^"]*max-w-\[calc\(var\(--studio-chat-width,48rem\)-2rem\)\]/,
  );

  // Sibling of the thread root, so an inline var there cannot reach it.
  assert.match(chatPage, /max-w-\[var\(--studio-chat-width,48rem\)\]/);
  assert.doesNotMatch(chatPage, /mx-auto w-full max-w-\[48rem\]/);
});

test("the stored percent defaults to the previous column and is clamped", async () => {
  const store = await readSrcAsync(
    "features/settings/stores/appearance-custom-store.ts",
  );

  // 100 is the historic 48rem column, so a fresh install is unchanged.
  assert.match(
    store,
    /CHAT_WIDTH_RANGE = \{ min: 100, max: 200, default: 100 \}/,
  );
  // Same sanitizer as the font sizes: clamps out of range, rejects non-finite.
  assert.match(
    store,
    /chatWidth: sanitizeSize\(source\.chatWidth, CHAT_WIDTH_RANGE\)/,
  );

  // At the default the var is cleared, or stock carries an inline override.
  const at = store.indexOf("--studio-chat-fill");
  assert.notEqual(at, -1);
  const region = store.slice(at - 300, at + 300);
  assert.match(region, /c\.chatWidth !== CHAT_WIDTH_RANGE\.default/);
  assert.match(
    region,
    /\(c\.chatWidth - CHAT_WIDTH_RANGE\.default\) \/ 100/,
  );
  assert.match(region, /setVar\("--studio-chat-fill", null\)/);
});

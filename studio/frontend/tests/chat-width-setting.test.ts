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

  // The default lives in :root, so a stock document carries no inline override.
  assert.match(css, /--studio-chat-width:\s*100rem;/);

  // Messages read --thread-content-max-width, which derives from this.
  assert.match(
    thread,
    /\["--thread-max-width" as string\]: "var\(--studio-chat-width, 100rem\)"/,
  );
  assert.match(
    thread,
    /\["--thread-content-max-width" as string\]:\s*\n?\s*"calc\(var\(--thread-max-width\) - 1\.5rem\)"/,
  );

  // Composer inside the thread, held 2rem narrower than the column as before.
  assert.match(
    thread,
    /unsloth-composer-shell[^"]*max-w-\[calc\(var\(--studio-chat-width,100rem\)-2rem\)\]/,
  );

  // Sibling of the thread root, so an inline var there cannot reach it.
  assert.match(
    chatPage,
    /max-w-\[var\(--studio-chat-width,100rem\)\]/,
  );
  assert.doesNotMatch(chatPage, /mx-auto w-full max-w-\[48rem\]/);
});

test("the stored width is clamped and defaults to the CSS value", async () => {
  const store = await readSrcAsync(
    "features/settings/stores/appearance-custom-store.ts",
  );

  assert.match(
    store,
    /CHAT_WIDTH_RANGE = \{ min: 640, max: 2400, default: 1600 \}/,
  );
  // Same sanitizer as the font sizes: clamps out of range, rejects non-finite.
  assert.match(store, /chatWidth: sanitizeSize\(source\.chatWidth, CHAT_WIDTH_RANGE\)/);

  // Cleared at the default, or stock carries an inline override for good.
  const at = store.indexOf("--studio-chat-width");
  assert.notEqual(at, -1);
  const region = store.slice(at - 200, at + 200);
  assert.match(region, /c\.chatWidth !== CHAT_WIDTH_RANGE\.default/);
  assert.match(region, /setVar\("--studio-chat-width", null\)/);
});

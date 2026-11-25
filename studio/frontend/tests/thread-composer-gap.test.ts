// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const thread = readFileSync(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);

test("the bottom spacer leaves a ChatGPT-sized gap and gives back the scroll button row", () => {
  assert.match(thread, /const COMPOSER_SCROLL_GAP_PX = 8;/);
  const desired =
    thread.match(/const desired =\s+composerHeight([\s\S]*?);/)?.[1] ?? "";
  assert.match(desired, /^\s*\+\s*COMPOSER_SCROLL_GAP_PX -/);
  assert.match(desired, /footerRow\.offsetHeight/);
  assert.match(
    thread,
    /childWithClass\(viewportEl, "aui-thread-viewport-footer"\)/,
  );
});

test("the scroll-to-bottom button keeps its place above the composer", () => {
  assert.match(thread, /const SCROLL_BUTTON_GAP_PX = 14;/);
  assert.match(thread, /: composerHeight \+ SCROLL_BUTTON_GAP_PX;/);
  // A zero-height sticky row counts its shifted box as scroll room and strands extra space below the reply.
  assert.doesNotMatch(thread, /aui-thread-viewport-footer[^"]*\bh-0\b/);
});

test("the spacer follows the composer's target height, not each eased frame", () => {
  assert.match(
    thread,
    /editor\?\.style\.getPropertyValue\("--composer-editor-height"\)/,
  );
  assert.match(thread, /el\.offsetHeight \+ target - editor\.offsetHeight/);
  // Expanded writing has no transition and may be taller than the target: its live height counts.
  assert.match(
    thread,
    /editor\?\.closest\('\[data-writing-expanded="true"\]'\) != null/,
  );
  assert.match(thread, /editor && !expanded && Number\.isFinite\(target\)/);
});

test("opening or switching a thread never leaves stabilizer padding under the reply", () => {
  const scroll = readFileSync(
    new URL(
      "../src/components/assistant-ui/use-intent-aware-autoscroll.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const stabilize =
    scroll.match(
      /const stabilize = \(\): number => \{([\s\S]*?)\n {6}\};/,
    )?.[1] ?? "";
  assert.match(
    stabilize,
    /const runSettling =\s*aui\.thread\(\)\.getState\(\)\.isRunning \|\|/,
  );
  assert.match(stabilize, /const followActive =\s*runSettling &&/);
});

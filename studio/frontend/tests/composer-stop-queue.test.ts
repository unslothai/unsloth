// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

function composerSource(): Promise<string> {
  return readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
}

test("the running composer shows one action: Queue with a draft, else Stop", async () => {
  const source = await composerSource();
  const start = source.indexOf(
    "<AuiIf condition={({ thread }) => thread.isRunning}>",
  );
  assert.ok(start >= 0, "the running composer controls must exist");

  const end = source.indexOf("</AuiIf>", start);
  assert.ok(end > start, "the running composer controls must be closed");
  const controls = source.slice(start, end);

  const guardIndex = controls.indexOf("{queueDisabled ? (");
  const stopIndex = controls.indexOf('aria-label="Stop generating"');
  const queueIndex = controls.indexOf("aria-label={followUpLabel}");
  assert.ok(guardIndex >= 0, "Stop and Queue must be one exclusive choice");
  assert.ok(
    guardIndex < stopIndex && stopIndex < queueIndex,
    "Stop shows with nothing to queue, Queue once the draft is queueable",
  );
  assert.doesNotMatch(controls, /\{!queueDisabled \?/);

  const stopButton = controls.match(
    /<Button[\s\S]*?aria-label="Stop generating"[\s\S]*?>/,
  );
  assert.ok(stopButton, "the running composer must render a stop button");
  assert.match(
    stopButton[0],
    /onClick=\{stop\}/,
    "stop must still end the prompt queue run, not only the active reply",
  );
});

test("the stop shortcut still works while the draft hides the Stop button", async () => {
  const source = await composerSource();
  assert.match(
    source,
    /useShortcut\(\s*"stopGeneration",[\s\S]*?enabled: chatActive && \(threadRunning \|\| isResearchActive\)/,
  );
});

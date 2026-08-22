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

test("stop stays available when the running composer can queue", async () => {
  const source = await composerSource();
  const start = source.indexOf(
    "<AuiIf condition={({ thread }) => thread.isRunning}>",
  );
  assert.ok(start >= 0, "the running composer controls must exist");

  const end = source.indexOf("</AuiIf>", start);
  assert.ok(end > start, "the running composer controls must be closed");
  const controls = source.slice(start, end);
  const stopLabel = 'aria-label="Stop generating"';
  const queueGuard = "{!queueDisabled ?";

  // Stop is an action on the current reply, so it must not be inside the
  // queue-availability branch that is enabled by typing in the composer.
  const stopIndex = controls.indexOf(stopLabel);
  const queueGuardIndex = controls.indexOf(queueGuard);
  assert.ok(stopIndex >= 0, "the running composer must expose Stop generating");
  assert.ok(
    queueGuardIndex > stopIndex,
    "queue availability must be evaluated after the independent stop action",
  );
  assert.doesNotMatch(
    controls,
    /\{queueDisabled \?/,
    "Stop and Queue must not share an exclusive queueDisabled ternary",
  );
  assert.match(
    controls.slice(queueGuardIndex),
    /aria-label="Queue message"/,
    "the queue action must remain available when the draft is queueable",
  );
});

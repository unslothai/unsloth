// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The finished-run actions open Export on that run, and a past run with its output on disk gets them too. */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const progress = readFileSync(
  new URL(
    "../src/features/studio/sections/progress-section.tsx",
    import.meta.url,
  ),
  "utf8",
);
const history = readFileSync(
  new URL(
    "../src/features/studio/historical-training-view.tsx",
    import.meta.url,
  ),
  "utf8",
);

function milestoneCalloutBody(): string {
  const start = progress.indexOf("function MilestoneCallout(");
  assert.ok(start >= 0, "no MilestoneCallout");
  return progress.slice(start, progress.indexOf("\nfunction ", start + 1));
}

test("Export Model deep-links the finished run into the Export page", () => {
  const body = milestoneCalloutBody();
  assert.match(
    body,
    /to="\/export"\s+search=\{exportRunName \? \{ run: exportRunName \} : \{\}\}/,
  );
  assert.match(
    progress,
    /exportRunName=\{hasFinishedOutput \? exportRunName : null\}/,
  );
});

test("a past run shows the finished-run actions only while its output exists", () => {
  assert.match(
    progress,
    /\{\(!isHistorical \|\| hasFinishedOutput\) && \(\s*<MilestoneCallout/,
  );
  assert.match(
    progress,
    /showHalfwayHint=\{!isHistorical && showHalfwayHint\}/,
  );
  assert.match(progress, /data\.artifactsAvailable !== false &&/);
  assert.match(history, /artifactsAvailable: run\.artifacts_available,/);
});

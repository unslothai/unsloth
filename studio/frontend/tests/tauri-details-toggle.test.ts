// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function source(path: string): Promise<string> {
  return readFile(
    new URL(`../src/components/tauri/${path}`, import.meta.url),
    "utf8",
  );
}

function summaryForLabel(sourceText: string, label: string): string {
  const labelOffset = sourceText.indexOf(label);
  assert.notEqual(labelOffset, -1, `${label} toggle is missing`);
  const start = sourceText.lastIndexOf("<summary", labelOffset);
  const end = sourceText.indexOf("</summary>", labelOffset);
  assert.notEqual(start, -1, `${label} is not inside a summary`);
  assert.notEqual(end, -1, `${label} summary is not closed`);
  return sourceText.slice(start, end);
}

test("the shared detail toggle puts a custom down chevron after its label", async () => {
  const summary = summaryForLabel(await source("log-details.tsx"), "Show {label}");

  assert.match(summary, /\blist-none\b/);
  assert.match(summary, /\[&::\-webkit-details-marker\]:hidden/);
  assert.match(summary, /\bflex\b/);
  assert.ok(
    summary.indexOf("<HugeiconsIcon") > summary.indexOf("Show {label}") &&
      summary.includes("icon={ChevronDownIcon}"),
    "the chevron must be on the right",
  );
});

test("every Tauri screen names its logs through the shared toggle", async () => {
  const startup = await source("startup-screen.tsx");
  const update = await source("update-screen.tsx");

  for (const [sourceText, label] of [
    [startup, "installation details"],
    [startup, "setup details"],
    [update, "update details"],
  ] as const) {
    assert.match(
      sourceText,
      new RegExp(`<LogDetails label="${label}" lines=`),
      `${label} must render through LogDetails`,
    );
  }

  // The copies these replaced drifted apart on spacing and wording; a hand-rolled
  // <details> back in a screen is the shape that let that happen.
  for (const sourceText of [startup, update]) {
    assert.doesNotMatch(sourceText, /<details/);
  }
});

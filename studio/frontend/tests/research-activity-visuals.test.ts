// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/features/chat/components/research-activity-panel.tsx",
    import.meta.url,
  ),
  "utf8",
);

function between(start: string, end: string): string {
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex);
  assert.ok(startIndex >= 0 && endIndex > startIndex, `missing ${start}`);
  return source.slice(startIndex, endIndex);
}

test("research activity reuses Unsloth's standard thought and web icons", () => {
  const icon = between("function ActivityIcon", "const ActivityRow");

  assert.match(icon, /<BulbIcon className=\{className\}/);
  assert.match(icon, /<GlobeIcon className=\{className\}/);
  assert.doesNotMatch(icon, /<(?:Brain|BookOpen|Search)\b/);
});

test("cancelled research uses the requested Hugeicons dashed circle", () => {
  const icon = between("function ActivityIcon", "const ActivityRow");

  assert.match(source, /DashedLineCircleIcon as CircleDashedIcon/);
  assert.match(icon, /icon=\{CircleDashedIcon\}/);
  assert.doesNotMatch(icon, /<Square\b/);
});

test("timeline labels, icons, times, and disclosure controls share one center", () => {
  const row = between("const ActivityRow", "function PlanReview");
  const trigger = row.slice(
    row.indexOf("<CollapsibleTrigger"),
    row.indexOf("</CollapsibleTrigger>"),
  );

  assert.match(trigger, /relative flex min-h-10 w-full items-center/);
  assert.match(
    trigger,
    /absolute -left-7 top-1\/2 flex size-\[15px\] -translate-y-1\/2/,
  );
  assert.doesNotMatch(trigger, /items-start|className="mt-0\.5 shrink-0/);
});

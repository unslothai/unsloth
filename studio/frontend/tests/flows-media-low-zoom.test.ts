// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const src = (p: string) => readFileSync(new URL(`../src/features/flows/editor/${p}`, import.meta.url), "utf8");

test("zoomed far out, a card hides its text but keeps its media, as a small copy", () => {
  assert.match(src("node-card.tsx"), /lowDetail && "\[&_\.flow-detail\]:invisible \[&_\.flow-media\]:visible"/);
  const media = src("media.tsx");
  assert.match(media, /className=\{`flow-media \$\{className/);
  assert.match(media, /src=\{\(low && small\) \|\| url\}/);
  assert.match(media, /controls=\{!low\}/);
  assert.match(src("flow-node.tsx"), /low=\{lowDetail\}/);
});

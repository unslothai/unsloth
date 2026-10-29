// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No wire is routed while a card is dragged: one pass runs on release.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const src = (path: string) =>
  readFileSync(new URL(`../src/features/flows/editor/${path}`, import.meta.url), "utf8");

test("the router controller sends nothing while paused and routes again once it resumes", () => {
  const routing = src("wire-routing.tsx");
  assert.match(routing, /if \(!edges \|\| paused\) return;/);
  assert.match(routing, /\[nodesKey, edges, scope, store, paused\]/);
});

test("the canvas pauses routing for the length of a drag", () => {
  assert.match(src("canvas.tsx"), /<WireRouterController[^>]*paused=\{drag !== null\}/);
});

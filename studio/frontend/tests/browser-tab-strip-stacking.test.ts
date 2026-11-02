// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("no query container wraps the browser tab strip, so it stays above the macOS titlebar drag strip", () => {
  // Before Safari 18.5 container-type adds layout containment, a stacking context that traps the strip's z-40 (#14314).
  const panel = readSrc("features/browser/browser-panel.tsx");
  const root = panel.indexOf("export const BrowserPanel = memo(");
  const section = panel.indexOf("<section", root);
  const strip = panel.indexOf("<TabStrip ", section);
  assert.ok(root >= 0 && section > root && strip > section);
  assert.doesNotMatch(panel.slice(section, strip), /@container/);
  assert.match(panel, /className="browser-chrome relative z-40 flex /);
  assert.match(
    panel.slice(strip),
    /^<TabStrip [^\n]*\n(?:[^\n]*\n){0,3}\s*<div className="browser-panel-body @container\/browser /,
  );
});

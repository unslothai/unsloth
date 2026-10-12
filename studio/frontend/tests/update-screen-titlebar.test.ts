// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Check source wiring because rendering TauriWrapper requires the desktop runtime.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PROVIDER = readSrc("app/provider.tsx");

function region(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start);
  assert.ok(start !== -1, `region start not found: ${from}`);
  assert.ok(end > start, `region end not found after start: ${to}`);
  return source.slice(start, end);
}

test("the update layer reports the update screen with the flag that shows it", () => {
  const layer = region(
    PROVIDER,
    "function TauriUpdateLayer(",
    "const HIDDEN_TITLEBAR_SIDEBAR_ROUTES",
  );
  assert.match(layer, /const content = isUpdating \? \(\s*<UpdateScreen/);
  assert.match(
    layer,
    /useLayoutEffect\(\(\) => \{\s*onUpdateScreenChange\(isUpdating\);/,
  );
  assert.match(layer, /onUpdateScreenChange\(false\)/);
});

test("the titlebar drops the sidebar surface while the update screen is shown", () => {
  assert.match(PROVIDER, /onUpdateScreenChange=\{setUpdateScreenShown\}/);
  const surface = region(
    PROVIDER,
    "const showSidebarSurface =",
    "<WindowTitlebar showSidebarSurface={showSidebarSurface} />",
  );
  assert.match(surface, /!updateScreenShown/);
});

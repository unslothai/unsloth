// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

// A one-line row draws as a pill; a taller row keeps soft corners.
const ROW_RADIUS = /\brounded-row\b/;
const OTHER_RADIUS = /\brounded-(?:full|xl|\[\d+px\])(?=[\s"])/;

function functionBody(source: string, name: string): string {
  const start = source.indexOf(`function ${name}(`);
  if (start === -1) throw new Error(`no ${name} component`);
  const next = source.indexOf("\nfunction ", start + 1);
  return source.slice(start, next === -1 ? undefined : next);
}

test("context-menu item hover matches the standard dropdown pill", async () => {
  const contextItem = functionBody(
    await readText("../src/components/ui/context-menu.tsx"),
    "ContextMenuItem",
  );
  const dropdownItem = functionBody(
    await readText("../src/components/ui/dropdown-menu.tsx"),
    "DropdownMenuItem",
  );

  assert.match(dropdownItem, ROW_RADIUS);
  assert.match(contextItem, ROW_RADIUS);
  assert.doesNotMatch(contextItem, OTHER_RADIUS);
});

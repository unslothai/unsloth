// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const RECTANGLE_RADIUS = /rounded-\[11px\]/;
const PILL_RADIUS = /\brounded-(?:full|xl)\b/;

function componentSource(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

function functionBody(source: string, name: string): string {
  const start = source.indexOf(`function ${name}(`);
  if (start === -1) throw new Error(`no ${name} component`);
  const next = source.indexOf("\nfunction ", start + 1);
  return source.slice(start, next === -1 ? undefined : next);
}

test("context-menu item hover matches the standard dropdown rectangle", async () => {
  const contextItem = functionBody(
    await componentSource("../src/components/ui/context-menu.tsx"),
    "ContextMenuItem",
  );
  const dropdownItem = functionBody(
    await componentSource("../src/components/ui/dropdown-menu.tsx"),
    "DropdownMenuItem",
  );

  assert.match(dropdownItem, RECTANGLE_RADIUS);
  assert.match(contextItem, RECTANGLE_RADIUS);
  assert.doesNotMatch(contextItem, PILL_RADIUS);
});

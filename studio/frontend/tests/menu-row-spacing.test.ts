// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("submenu chevrons sit as far from the right edge as leading icons from the left", () => {
  for (const file of ["dropdown-menu", "context-menu", "menubar"]) {
    const source = readSrc(`components/ui/${file}.tsx`);
    assert.match(
      source,
      /className="ml-auto -mr-\[calc\(3\.5px\*var\(--ui-space-scale,1\)\)\] size-\[calc\(12px\*var\(--ui-space-scale,1\)\)\]"\n\s*\/>\n\s*<\/\w+Primitive\.SubTrigger>/,
      `${file} chevron`,
    );
  }
});

test("menu rows keep a small gap between them", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /\[data-slot="context-menu-sub-content"\],[\s\S]*?\[role="menuitemradio"\]\n\s*\) \{\n\s*margin-block: 2px;/,
  );
});

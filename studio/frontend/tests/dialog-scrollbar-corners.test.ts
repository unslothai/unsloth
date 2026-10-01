// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const DIALOG = readSrc("components/ui/dialog.tsx");
const ALERT = readSrc("components/ui/alert-dialog.tsx");
const MCP = readSrc("features/chat/chat-mcp-servers-dialog.tsx");
const SURFACES =
  ':is([data-slot="dialog-content"], [data-slot="alert-dialog-content"])';

test("dialogs scroll their rounded surface", () => {
  // The fix below only matters while the surface itself is the scroller.
  for (const source of [DIALOG, ALERT]) {
    assert.match(source, /rounded-4xl/);
    assert.match(source, /overflow-y-auto/);
  }
});

test("a dialog's scrollbar keeps its rounded corners", () => {
  // WebKit's native scrollbar squares the corners; ::-webkit-scrollbar only applies once the
  // standard scrollbar properties are reset, including the .dark override.
  const reset = CSS.indexOf(`${SURFACES},\n\t.dark ${SURFACES} {`);
  assert.notEqual(
    reset,
    -1,
    "dialog surfaces no longer reset the standard scrollbar",
  );
  assert.match(
    CSS.slice(reset, reset + 300),
    /scrollbar-width: auto;\n\t\tscrollbar-color: auto;/,
  );
  assert.ok(
    CSS.includes(
      `${SURFACES}::-webkit-scrollbar-track {\n\tmargin-block: var(--radius-4xl);\n}`,
    ),
    "the track no longer clears the corner radius",
  );
});

test("the MCP servers dialog scrolls its body, not its rounded surface", () => {
  assert.match(
    MCP,
    /<DialogContent\s+\/\/[^\n]*\n\s*\/\/[^\n]*\n\s*className="max-w-2xl max-h-\[85dvh\] grid-rows-\[auto_minmax\(0,1fr\)\] overflow-hidden"/,
  );
  assert.match(
    MCP,
    /<\/DialogHeader>\n\s*\{\/\*[^*]*\*\/\}\n\s*<div className="-mx-7 min-h-0 overflow-y-auto px-7">/,
  );
  assert.match(MCP, /\n {8}<\/div>\n {6}<\/DialogContent>/);
});

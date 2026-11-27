// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const PAGE = 'html[data-glass][data-chat-wallpaper-workspace-ready] [data-wallpaper-route="page"] > [data-slot="sidebar-inset"]';

test("page glass lifts the leading icon of a frosted field above the field's backdrop-filter", () => {
  const frost = CSS.indexOf(`${PAGE} :where(.elevated-card`);
  assert.ok(frost > 0, "the page glass rule frosts resting fields");
  assert.match(CSS.slice(frost, CSS.indexOf("}", frost)), /\[data-slot="input"\], \.panel-field, \.field-soft[\s\S]*backdrop-filter: blur/);
  const lift = CSS.indexOf(`${PAGE} :is(svg, span:has(> svg)).absolute:has(~ :is([data-slot="input"], .panel-field, .field-soft)) {`);
  assert.ok(lift > frost, "a backdrop-filter field paints over an earlier absolute icon (issue #16962)");
  assert.match(CSS.slice(lift, CSS.indexOf("}", lift)), /z-index: 1;/);
});

for (const [file, icon] of [
  ["features/hub/catalog/models-toolbar.tsx", /<HugeiconsIcon\s+icon=\{Search01Icon\}[^>]*className="pointer-events-none absolute [^"]*"\s*\/>\s*<Input\b/],
  ["features/chat/projects-page.tsx", /<span className="pointer-events-none absolute [^"]*">\s*<HugeiconsIcon icon=\{Search01Icon\}[^>]*\/>\s*<\/span>\s*<Input\b/],
  ["features/scheduled-tasks/tasks-page.tsx", /<span className="pointer-events-none absolute [^"]*">\s*<HugeiconsIcon icon=\{Search01Icon\}[^>]*\/>\s*<\/span>\s*<Input\b/],
  ["features/api-monitor/api-monitor-page.tsx", /<HugeiconsIcon\s+icon=\{Search01Icon\}[^>]*className="pointer-events-none absolute [^"]*"[^>]*\/>\s*<Input\b/],
] as const) {
  test(`${file}: the search icon is the absolute sibling before its Input that the lift matches`, () => {
    assert.match(readSrc(file), icon);
  });
}

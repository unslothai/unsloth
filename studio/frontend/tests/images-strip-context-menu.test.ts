// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("a right-click on an Images strip tile opens the tile's actions menu", () => {
  const page = readSrc("features/images/images-page.tsx");
  const tile = page.slice(page.indexOf("data-image-id={image.id}"));
  const handler = /onContextMenu=\{\(event\) => \{([^}]*)\}\}/.exec(tile)?.[1] ?? "";
  // The desktop webview's own image menu cannot open or copy a blob: thumbnail, so it must not show there.
  // A browser's menu works on them, so browsers keep it.
  // React bubbles a portalled child's event (the New project dialog) through the tile: only the tile's own DOM counts.
  assert.match(
    handler,
    /^\s*if \(!isTauri \|\| !event\.currentTarget\.contains\(event\.target as Node\)\) return;\s*event\.preventDefault\(\)/,
  );
  assert.match(handler, /setTileMenuId\(image\.id\)/);
  assert.match(tile, /open=\{tileMenuId === image\.id\}/);
  assert.match(tile, /onOpenChange=\{\(open\) => setTileMenuId\(open \? image\.id : null\)\}/);
  // A tab switch or a deleted tile drops the request, so it cannot reopen later.
  assert.match(page, /if \(tileMenuId && \(!active \|\| !images\.some\(/);
});

test("the gallery item menu can be opened from outside and still closes off-tab", () => {
  const menu = readSrc("components/gallery-item-menu.tsx");
  assert.match(menu, /const open = openProp \?\? ownOpen;/);
  assert.match(menu, /onOpenChange\?\.\(next\)/);
  assert.match(menu, /open=\{active && open\}/);
});

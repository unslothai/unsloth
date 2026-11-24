// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the More menu's side gutter sits inside its scroll viewport", () => {
  const css = readSrc("index.css");
  assert.match(css, /\.sidebar-more-menu\[data-slot\] \{\s*padding-inline: 0 !important;\s*\}/);
  assert.match(
    css,
    /\.sidebar-more-menu > \[data-slot="dropdown-menu-viewport"\] \{\s*padding-inline: var\(--spacing\);\s*\}/,
  );
});

test("New tags keep More rows at the plain row height", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  assert.equal(sidebar.match(/\{badge && <span className=\{NAV_TAG\}>\{badge\}<\/span>\}/g)?.length, 2);
  assert.match(sidebar, /const NAV_TAG =\s*"[^"]*\bpy-1\b[^"]*\btext-ui-10\b[^"]*\bleading-none\b/);
});

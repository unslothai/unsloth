// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SIDEBAR = readSrc("components/app-sidebar.tsx");

function block(source: string, start: string, end: string): string {
  const from = source.indexOf(start);
  assert.notEqual(from, -1, `missing ${start}`);
  const to = source.indexOf(end, from + start.length);
  assert.notEqual(to, -1, `missing ${end} after ${start}`);
  return source.slice(from, to);
}

test("the collapsed rail opens the Images and Audio workflows from their icons", () => {
  const inline = block(SIDEBAR, "{inlineNavIds.map((id) => {", "{/* Unpinned destinations");
  const flyout = block(inline, "flyout={", "\n                  >");
  // Only the collapsed rail, where the workflow rows under the icon are not listed.
  assert.match(flyout, /sidebarRowsLabelled \|\|/);
  assert.match(flyout, /rowState\.disabled \|\|/);
  assert.match(
    flyout,
    /<ImagesRailFlyout\s+label=\{row\.label\}\s+active=\{row\.active\}\s+onPick=\{pickImagesWorkflow\}/,
  );
  assert.match(
    flyout,
    /<AudioRailFlyout\s+label=\{row\.label\}\s+active=\{row\.active\}\s+onPick=\{pickAudioWorkflow\}/,
  );
  // The collapsed list still renders nothing, so the flyout is the only switcher there.
  assert.match(
    SIDEBAR,
    /listed=\{!collapsed && \(active \? pageMode !== "train" : expanded\)\}/,
  );
});

test("the rail flyout lists every workflow and greys out what the model cannot do", () => {
  const images = block(SIDEBAR, "function ImagesRailFlyout(", "function AudioRailFlyout(");
  assert.match(images, /tabs=\{WORKFLOW_TABS\}/);
  assert.match(images, /enabled=\{\(id\) => isWorkflowEnabled\(id, supported\)\}/);
  assert.match(
    images,
    /const current = props\.active && pageMode === "create" \? workflow : null;/,
  );

  const menu = block(SIDEBAR, "function MediaRailFlyout<", "type RailFlyoutProps<");
  assert.match(menu, /useHoverFlyout\(NAV_FLYOUT_INTENT, contentRef\)/);
  assert.match(menu, /event\.key !== "ArrowRight"/);
  assert.match(menu, /\{tabs\.map\(\(tab\) => \(/);
  assert.match(menu, /disabled=\{!enabled\(tab\.id\)\}/);
  assert.match(menu, /onSelect=\{\(\) => onPick\(tab\.id\)\}/);
  // A press on the icon itself stays a page open, not a menu toggle.
  assert.match(menu, /className="pointer-events-none absolute inset-0"/);
  assert.match(menu, /if \(insideRow\(event\.target\)\) event\.preventDefault\(\);/);
  // Escape hands focus back to the icon only after a keyboard open, never after a press elsewhere.
  assert.match(menu, /returnFocus\.current = true;\s+setKeyboardOpen\(true\);/);
  assert.match(menu, /else returnFocus\.current = false;/);
  assert.match(menu, /if \(!returnFocus\.current\) return;/);
});

test("a row with a flyout drops its label tooltip, every other row keeps it", () => {
  const nav = block(SIDEBAR, "function NavItem({", "const WORKFLOW_UNAVAILABLE");
  assert.match(nav, /tooltip=\{flyout \? undefined : \(tooltip \?\? label\)\}/);
  assert.match(nav, /\{flyout \? flyout\(row\) : row\}/);
});

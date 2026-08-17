// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path: string) =>
  readFile(new URL(`../src/${path}`, import.meta.url), "utf8");

const Z_INDEX_PATTERN = /z-\[(\d+)\]|\bz-(\d+)\b/;
const DECORATION_PATTERN =
  /data-slot="window-titlebar-decoration"[\s\S]*?className="([^"]+)"/;
const TITLEBAR_PATTERN = /<header\s+className=\{cn\(\s*"([^"]+)"/;
const DIALOG_OVERLAY_PATTERN =
  /data-slot="dialog-overlay"[\s\S]*?"([^"]*\bz-50\b[^"]*)"/;
const ALERT_DIALOG_OVERLAY_PATTERN =
  /data-slot="alert-dialog-overlay"[\s\S]*?"([^"]*\bz-50\b[^"]*)"/;
const SETTINGS_OVERLAY_PATTERN = /overlayClassName="([^"]*\bz-\[60\][^"]*)"/;
const SETTINGS_SURFACE_PATTERN =
  /"([^"]*\bsettings-surface\b[^"]]*\bz-\[60\][^"]*)"/;
const TOP_FULL_PATTERN = /top-full/;
const CLOSED_DECORATION_PATTERN = /<\/div>\s*\)\}\s*$/;

function zIndex(block: string): number {
  const match = block.match(Z_INDEX_PATTERN);
  if (!match) throw new Error(`no z-index class in ${block}`);
  return Number(match[1] ?? match[2]);
}

test("titlebar decoration stays below modal backdrops and window controls", async () => {
  const [titlebar, dialog, alertDialog] = await Promise.all([
    source("components/tauri/window-titlebar.tsx"),
    source("components/ui/dialog.tsx"),
    source("components/ui/alert-dialog.tsx"),
  ]);

  const decoration = titlebar.match(DECORATION_PATTERN);
  const titlebarHeader = titlebar.match(TITLEBAR_PATTERN);
  const dialogOverlay = dialog.match(DIALOG_OVERLAY_PATTERN);
  const alertDialogOverlay = alertDialog.match(ALERT_DIALOG_OVERLAY_PATTERN);

  assert.ok(decoration);
  assert.ok(titlebarHeader);
  assert.ok(dialogOverlay);
  assert.ok(alertDialogOverlay);

  const decorationLayer = zIndex(decoration[1]);
  const titlebarLayer = zIndex(titlebarHeader[1]);
  for (const overlay of [dialogOverlay[1], alertDialogOverlay[1]]) {
    const overlayLayer = zIndex(overlay);
    assert.ok(decorationLayer < overlayLayer);
    assert.ok(overlayLayer < titlebarLayer);
  }
});

test("below-titlebar decoration is not trapped in the titlebar stacking context", async () => {
  const titlebar = await source("components/tauri/window-titlebar.tsx");
  const decorationIndex = titlebar.indexOf(
    'data-slot="window-titlebar-decoration"',
  );
  const headerIndex = titlebar.indexOf("<header", decorationIndex);

  assert.notEqual(decorationIndex, -1);
  assert.notEqual(headerIndex, -1);
  assert.ok(decorationIndex < headerIndex);
  assert.match(
    titlebar.slice(decorationIndex, headerIndex),
    CLOSED_DECORATION_PATTERN,
  );

  const headerEnd = titlebar.indexOf("</header>", headerIndex);
  assert.notEqual(headerEnd, -1);
  const header = titlebar.slice(headerIndex, headerEnd);
  assert.doesNotMatch(header, TOP_FULL_PATTERN);
});

test("settings stays above ordinary chat surfaces and below window controls", async () => {
  const [titlebar, dialog, settings] = await Promise.all([
    source("components/tauri/window-titlebar.tsx"),
    source("components/ui/dialog.tsx"),
    source("features/settings/settings-dialog.tsx"),
  ]);

  const titlebarHeader = titlebar.match(TITLEBAR_PATTERN);
  const dialogOverlay = dialog.match(DIALOG_OVERLAY_PATTERN);
  const settingsOverlay = settings.match(SETTINGS_OVERLAY_PATTERN);
  const settingsSurface = settings.match(SETTINGS_SURFACE_PATTERN);

  assert.ok(titlebarHeader);
  assert.ok(dialogOverlay);
  assert.ok(settingsOverlay);
  assert.ok(settingsSurface);

  const ordinaryModalLayer = zIndex(dialogOverlay[1]);
  const settingsOverlayLayer = zIndex(settingsOverlay[1]);
  const settingsSurfaceLayer = zIndex(settingsSurface[1]);
  const titlebarLayer = zIndex(titlebarHeader[1]);

  assert.equal(settingsOverlayLayer, settingsSurfaceLayer);
  assert.ok(ordinaryModalLayer < settingsOverlayLayer);
  assert.ok(settingsSurfaceLayer < titlebarLayer);
});

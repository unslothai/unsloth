// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const MENU = await readFile(
  new URL("../src/features/chat/permission-mode-select.tsx", import.meta.url),
  "utf8",
);
const SANDBOX_TAB = await readFile(
  new URL("../src/features/settings/tabs/sandbox-tab.tsx", import.meta.url),
  "utf8",
);
const OLD_GENERAL = await readFile(
  new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
  "utf8",
);

test("the banner's Learn more opens the Permissions section of Settings > Sandbox", () => {
  assert.match(MENU, /openSettings\("sandbox", \{ scrollTarget: "sandbox-permissions" \}\)/);
  assert.match(SANDBOX_TAB, /if \(scrollTarget !== "sandbox-permissions"\) return;/);
  assert.match(SANDBOX_TAB, /ref=\{permissionsRef\}/);
  assert.doesNotMatch(OLD_GENERAL, /permissionsRef|PermissionModeDropdown|general-permissions/);
});

test("Settings explains only the selected level and drops the menu's sandbox controls", () => {
  assert.match(SANDBOX_TAB, /label=\{t\(`settings\.general\.permissions\.names\.\$\{activePermission\.value\}`\)\}/);
  assert.match(SANDBOX_TAB, /t\(`settings\.general\.permissions\.details\.\$\{activePermission\.value\}`\)/);
  assert.match(SANDBOX_TAB, /<PermissionModeDropdown sandboxControls=\{false\} \/>/);
  assert.doesNotMatch(SANDBOX_TAB, /permissions\.bypass(Label|Description)/);
});

test("the menu heading carries the Sandbox switch in place of Learn more", () => {
  const label = MENU.slice(
    MENU.indexOf("export function PermissionMenuLabel"),
    MENU.indexOf("/** Under the rows"),
  );
  assert.doesNotMatch(label, /Learn more/);
  assert.match(label, /<DropdownMenuPrimitive\.CheckboxItem/);
  // The keyboard toggles it without closing the menu or picking a row.
  assert.match(label, /onSelect=\{\(event\) => event\.preventDefault\(\)\}/);
  assert.match(label, /<Switch\b/);
  assert.match(label, /settings\.sandbox\.levelLowShort/);
});

test("menu descriptions stay to one short line", () => {
  const options = MENU.slice(MENU.indexOf("export const PERMISSION_MODE_OPTIONS"), MENU.indexOf("] as const;"));
  const descriptions = [...options.matchAll(/description: "([^"]+)",/g)].map((m) => m[1]);
  assert.equal(descriptions.length, 4);
  for (const text of descriptions) assert.ok(text.length <= 60, text);
});

test("Full access looks like every other level and asks in a short, neutral confirmation", () => {
  for (const source of [MENU, SANDBOX_TAB]) assert.doesNotMatch(source, /text-bypass|data-variant=\{fullAccess/);
  assert.match(MENU, /icon: ShieldAlertGlyph,/);
  assert.match(MENU, /Turn on Full access\?/);
  assert.doesNotMatch(MENU, /text-sky-/);
  assert.match(MENU, /className="gap-5 p-7 ring-0 [^"]*"\s+onOverlayClick=\{onClose\}/);
  assert.match(MENU, /icon: InternetIcon,/);
  assert.match(MENU, /outside the sandbox, including:/);
  assert.match(MENU, /<AlertDialogCancel variant="muted">Cancel<\/AlertDialogCancel>/);
  assert.match(MENU, /onOpenAutoFocus=\{\(event\) => \{\s*event\.preventDefault\(\);/);
  assert.match(MENU, /<AlertDialogDescription className="text-pretty leading-relaxed">/);
  assert.equal([...MENU.matchAll(/title: "/g)].length, 3);
});

const DICTATION = await readFile(
  new URL("../src/features/chat/bypass-permissions-menu-item.tsx", import.meta.url),
  "utf8",
);
const EN = await readFile(new URL("../src/i18n/locales/en.ts", import.meta.url), "utf8");

test("the dictation submenu has the heading and the Sandbox switch too", () => {
  assert.match(DICTATION, /<PermissionMenuLabel sandboxControls \/>\s*<PermissionModeMenuItems/);
});

test("confirmation Learn more hands Settings the focus from before the dialog", () => {
  assert.match(MENU, /returnFocusRef\.current = lastFocusOutsideMenus\(\);/);
  // A closing menu resolves to its trigger, which outlives it.
  assert.match(MENU, /menu\.getAttribute\("aria-labelledby"\)/);
  assert.match(MENU, /openSettings\("sandbox", \{\s*scrollTarget: "sandbox-permissions",\s*opener: returnFocusRef\.current,/);
});

test("Ask says provider-hosted tools are not paused", () => {
  assert.match(EN, /Tools run by an external provider are not paused\./);
});

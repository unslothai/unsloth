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

test("the question mark opens the Permissions section of Settings > Sandbox", () => {
  assert.match(MENU, /openSettings\("sandbox", \{ scrollTarget: "sandbox-permissions" \}\)/);
  assert.match(SANDBOX_TAB, /if \(scrollTarget !== "sandbox-permissions"\) return;/);
  assert.match(SANDBOX_TAB, /ref=\{permissionsRef\}/);
  assert.doesNotMatch(OLD_GENERAL, /permissionsRef|PermissionModeDropdown|general-permissions/);
});

test("Settings explains only the selected level and drops the menu's sandbox controls", () => {
  // The selected level's full explanation stays on the page, not behind a hover.
  assert.match(SANDBOX_TAB, /label=\{t\("settings\.sandbox\.permissionLabel"\)\}/);
  // Named, then explained: "Approve for me: Runs routine tool calls…".
  assert.match(SANDBOX_TAB, /name=\{t\(`settings\.general\.permissions\.names\.\$\{activePermission\.value\}`\)\}\s*detail=\{t\(`settings\.general\.permissions\.details\.\$\{activePermission\.value\}`\)\}/);
  assert.match(SANDBOX_TAB, /settings\.sandbox\.levelHighDetail/);
  assert.match(SANDBOX_TAB, /settings\.sandbox\.levelLowDetail/);
  assert.match(SANDBOX_TAB, /settings\.sandbox\.levelFullAccessNote/);
  assert.match(SANDBOX_TAB, /<PermissionModeDropdown sandboxControls=\{false\} \/>/);
  assert.doesNotMatch(SANDBOX_TAB, /permissions\.bypass(Label|Description)/);
});

test("Settings picks the sandbox level with the Queue/Steer style Low | High toggle", () => {
  const section = SANDBOX_TAB.slice(
    SANDBOX_TAB.indexOf("function PermissionsSection"),
    SANDBOX_TAB.indexOf("export function SandboxTab"),
  );
  assert.match(section, /hub-tab-toggle inline-flex/);
  assert.match(section, /\(\["low", "high"\] as const\)/);
  // Full access adds a Disabled option and locks the others.
  assert.match(section, /\(\["off", "low", "high"\] as const\)/);
  assert.match(section, /disabled=\{disabled && !selected\}/);
  // Each description names the option it explains.
  assert.match(section, /<SelectedOptionDescription/);
  assert.match(section, /aria-pressed=\{selected\}/);
  assert.doesNotMatch(section, /<Switch\b/);
  assert.doesNotMatch(section, /hint=/);
});

test("the menu heading carries a Sandbox chip that opens the level picker, Learn more on top", () => {
  const label = MENU.slice(
    MENU.indexOf("export function PermissionMenuLabel"),
    MENU.indexOf("/** The option rows shared"),
  );
  assert.doesNotMatch(label, /Tool call permissions/);
  assert.match(label, /settings\.general\.permissions\.sectionTitle/);
  // "Sandbox High ›": a submenu trigger, not a switch.
  assert.match(label, /<DropdownMenuPrimitive\.SubTrigger/);
  assert.match(label, /className="sandbox-level-chip\b/);
  // Click to open: hover must not open the picker.
  assert.match(label, /onPointerMove=\{\(event\) => event\.preventDefault\(\)\}/);
  // Controlled: stays open until a click outside the picker, the chip, or ArrowLeft.
  assert.match(label, /<DropdownMenuPrimitive\.Sub\s+open=\{open\}/);
  assert.match(label, /addEventListener\("pointerdown", onPointerDown, true\)/);
  assert.match(label, /if \(next\) setOpen\(true\);/);
  // Offset for the side it lands on: left when only the left fits.
  assert.match(label, /!fitsRight && fitsLeft\s*\?\s*chipBox\.left - menuBox\.left \+ SANDBOX_PICKER_GAP/);
  assert.match(label, /icon=\{ChevronRightStandardIcon\}/);
  assert.doesNotMatch(label, /<Switch\b|CheckboxItem/);
  assert.match(label, /<DropdownMenuSubContent[\s\S]*?className="unsloth-plus-menu\b/);
  assert.match(label, /settings\.sandbox\.levelLowShort/);
  assert.match(label, /settings\.sandbox\.levelHighShort/);
  // Learn more sits in the picker's heading row, not under the levels.
  assert.match(label, /settings\.sandbox\.levelPickerTitle"\)\}<\/span>\s*<DropdownMenuPrimitive\.Item[\s\S]*?settings\.sandbox\.learnMore/);
  assert.doesNotMatch(label, /DropdownMenuSeparator/);
  // Measured offsets keep the picker clear of the menu.
  assert.match(label, /sideOffset=\{offsets\.side\}/);
});

test("the rows carry no sandbox hint and no setup banner", () => {
  const rows = MENU.slice(MENU.indexOf("export function PermissionModeMenuItems"));
  assert.doesNotMatch(rows, /sandboxSetup\.unavailable|levelLowRowHint|notSetUp|SandboxSetupBanner/);
  assert.doesNotMatch(MENU, /SandboxSetupBanner|sandboxBanner/);
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

test("the dictation submenu has the heading and the Sandbox picker too", () => {
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

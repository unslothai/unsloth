// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, readText, registerBundlerResolver } from "./helpers/kit.ts";

// The menu is native and the hook is React, so the contract between them is asserted on source.
const APP_MENU = readText("../../src-tauri/src/app_menu.rs");
const MAIN_RS = readText("../../src-tauri/src/main.rs");
const HOOK = readSrc("app/use-app-menu-actions.ts");
const CHORDS = readSrc("app/app-menu-chords.ts");
const ROOT = readSrc("app/routes/__root.tsx");

const rowsOf = (table: string) =>
  [...APP_MENU.match(new RegExp(`const ${table}: &\\[Row\\] = &\\[([\\s\\S]*?)\\];`))![1].matchAll(
    /Row::Action\(\s*"([a-z-]+)"/g,
  )].map((m) => m[1]);
const fileActions = rowsOf("FILE_ROWS");
const viewActions = rowsOf("VIEW_ROWS");
const rustActions = [...fileActions, ...viewActions];
const hookActions = [...CHORDS.match(/export type AppMenuAction =([^;]+);/)![1].matchAll(/"([a-z-]+)"/g)].map(
  (m) => m[1],
);

test("the menus and the renderer name the same actions", () => {
  assert.deepEqual(fileActions, ["new-chat", "new-temporary-chat", "open-folder"]);
  assert.deepEqual(viewActions, [
    "toggle-sidebar",
    "find",
    "previous-chat",
    "next-chat",
    "back",
    "forward",
    "zoom-in",
    "zoom-out",
    "actual-size",
  ]);
  assert.deepEqual([...hookActions].sort(), [...rustActions].sort());
});

test("every action is handled in the app shell", () => {
  for (const action of rustActions) {
    assert.ok(ROOT.includes(`"${action}":`), `__root.tsx handles ${action}`);
  }
});

test("both sides use the same event and command names", () => {
  assert.match(APP_MENU, /APP_MENU_ACTION_EVENT: &str = "app-menu-action"/);
  assert.match(HOOK, /listen<AppMenuAction>\("app-menu-action"/);
  assert.match(HOOK, /invoke\("set_app_menu_actions"/);
  assert.match(MAIN_RS, /app_menu::set_app_menu_actions,/);
});

test("items start disabled and the renderer disables them when it goes away", () => {
  assert.match(APP_MENU, /\.enabled\(false\)/);
  assert.match(HOOK, /return \(\) => sync\(\[\]\)/);
});

test("the shortcuts shown match the web shortcuts they stand for", () => {
  const shortcuts = readSrc("features/settings/lib/keyboard-shortcuts.ts");
  assert.match(APP_MENU, /"New Chat", "CmdOrCtrl\+N"/);
  assert.match(shortcuts, /def\("newChat", "Mod\+Shift\+KeyO", \{ defaultAlternateBinding: "Mod\+KeyN" \}\)/);
  assert.match(APP_MENU, /"New Temporary Chat",\s*"CmdOrCtrl\+Shift\+N"/);
  assert.match(shortcuts, /def\("newTemporaryChat", "Mod\+Shift\+KeyN"\)/);
  // No web shortcut owns Cmd+O, so the menu's is free.
  assert.doesNotMatch(shortcuts, /"Mod\+KeyO"/);
});

test("View items for web shortcuts show the chord the web shortcut uses", () => {
  const shortcuts = readSrc("features/settings/lib/keyboard-shortcuts.ts");
  const pairs: [string, string, string, string][] = [
    ["toggle-sidebar", "CmdOrCtrl+B", "toggleSidebar", "Mod+KeyB"],
    ["find", "CmdOrCtrl+F", "findInPage", "Mod+KeyF"],
    ["previous-chat", "CmdOrCtrl+Shift+[", "previousChat", "Mod+Shift+BracketLeft"],
    ["next-chat", "CmdOrCtrl+Shift+]", "nextChat", "Mod+Shift+BracketRight"],
  ];
  for (const [action, accelerator, id, binding] of pairs) {
    const row = new RegExp(`Row::Action\\(\\s*"${action}",\\s*"[^"]+",\\s*"([^"]+)"`);
    assert.equal(APP_MENU.match(row)?.[1], accelerator, action);
    assert.match(shortcuts, new RegExp(`def\\("${id}", "${binding.replace(/\+/g, "\\+")}"`));
    assert.ok(ROOT.includes(`viaShortcut("${id}"`), `${action} runs ${id}`);
  }
  // The rest have no web shortcut, so the menu's chords take no key from one.
  for (const binding of ["Mod+BracketLeft", "Mod+BracketRight", "Mod+Equal", "Mod+Minus", "Mod+Digit0"]) {
    assert.doesNotMatch(shortcuts, new RegExp(`"${binding.replace(/\+/g, "\\+")}"`));
  }
});

test("View keeps the native Enter Full Screen below the Unsloth items", () => {
  assert.match(APP_MENU, /for item in &native \{\s*submenu\.append\(item\)\?;/);
});

test("Close keeps the native close so Cmd+W still reaches the window's close handling", () => {
  assert.match(APP_MENU, /PredefinedMenuItem::close_window\(app, Some\("Close"\)\)/);
});

test("Open Folder lands on the new project's Sources, not its chats", () => {
  const openFolder = readSrc("features/chat/utils/open-folder-as-project.ts");
  const landing = readSrc("features/chat/chat-page.tsx");
  // Marked only once the folder is linked, before the caller navigates.
  assert.ok(
    openFolder.indexOf("markProjectSourcesPending(project.id)") >
      openFolder.indexOf("createLinkedFolder("),
  );
  assert.match(landing, /hasProjectSourcesPending\(projectId\) \? "sources" : "chats"/);
  // A failed link keeps the project: deleting it would delete chats that joined it meanwhile.
  assert.doesNotMatch(openFolder, /deleteChatProject/);
  assert.ok(openFolder.indexOf("markProjectSourcesPending(project.id)") > openFolder.indexOf("} finally {"));
  // Keyed by project, so a new project mounts fresh and reads the marker.
  assert.match(landing, /<ProjectLanding\s+key=\{baseView\.projectId\}/);
});

test("menu triggers reach the newest mounted handler that claims the action", async () => {
  registerBundlerResolver();
  const { registerShortcutTrigger, triggerShortcut } = await import(
    "../src/features/settings/hooks/use-shortcut.ts"
  );
  const calls: string[] = [];
  assert.equal(triggerShortcut("toggleSidebar"), false, "nothing mounted");
  const offOld = registerShortcutTrigger("toggleSidebar", {
    claims: () => true,
    run: () => calls.push("old"),
  });
  const offNew = registerShortcutTrigger("toggleSidebar", {
    claims: () => false,
    run: () => calls.push("declined"),
  });
  // The newest declines (its claims() said no), so the older one runs, and only once.
  assert.equal(triggerShortcut("toggleSidebar"), true);
  assert.deepEqual(calls, ["old"]);
  offOld();
  assert.equal(triggerShortcut("toggleSidebar"), false, "only a declining handler left");
  offNew();
  assert.equal(triggerShortcut("toggleSidebar"), false, "unregistered");
});

test("menu chords follow the user's bindings and never steal a web shortcut's chord", async () => {
  registerBundlerResolver();
  const { menuAccelerators } = await import("../src/app/app-menu-chords.ts");
  const defaults = menuAccelerators({});
  // The defaults match what the native menu is built with.
  const rustAccel = (action: string) =>
    APP_MENU.match(new RegExp(`Row::Action\\(\\s*"${action}",\\s*"[^"]+",\\s*"([^"]+)"`))?.[1];
  const native = (accel: string) =>
    accel
      .replace(/\+([A-Z])$/, "+Key$1")
      .replace(/\+(\d)$/, "+Digit$1")
      .replace("+[", "+BracketLeft")
      .replace("+]", "+BracketRight")
      .replace("+=", "+Equal")
      .replace(/\+-$/, "+Minus");
  for (const action of rustActions) {
    assert.equal(defaults[action as keyof typeof defaults], native(rustAccel(action)!), action);
  }
  // Rebound: the item shows the new chord.
  assert.equal(
    menuAccelerators({ toggleSidebar: { primary: "Mod+KeyJ" } } as never)["toggle-sidebar"],
    "CmdOrCtrl+KeyJ",
  );
  // Cleared: no chord, so the menu does not keep answering the old one.
  assert.equal(
    menuAccelerators({ findInPage: { primary: null } } as never)["find"],
    null,
  );
  // A fixed item whose chord a web shortcut now owns gives it up.
  assert.equal(
    menuAccelerators({ findInPage: { primary: "Mod+KeyO" } } as never)["open-folder"],
    null,
  );
  // A chord without Cmd or Ctrl never reaches the menu, which would take it from text fields.
  assert.equal(
    menuAccelerators({ toggleSidebar: { primary: "Alt+KeyB" } } as never)["toggle-sidebar"],
    null,
  );
});

test("menu items for web shortcuts follow a mounted handler, and honour claims", () => {
  const hook = readSrc("features/settings/hooks/use-shortcut.ts");
  // Registered on `enabled` alone, so a cleared chord still leaves the menu item working.
  assert.match(hook, /if \(!enabled\) return;\s*return registerShortcutTrigger\(id, \{\s*claims: \(\) => latestRef\.current\.claims\?\.\(\) !== false,/);
  // Availability asks claims(), and the root has it re-ask when a modal opens or closes.
  assert.match(hook, /\(triggers\.get\(id\) \?\? \[\]\)\.some\(\(t\) => t\.claims\(\)\)/);
  assert.match(hook, /attributeFilter: \["aria-hidden", "inert"\]/);
  for (const id of ["toggleSidebar", "findInPage", "previousChat", "nextChat"]) {
    assert.ok(ROOT.includes(`useShortcutAvailable("${id}", isTauri)`), `${id} enables its item`);
  }
});

test("Back and Forward follow page history, and zoom steps the interface scale", () => {
  assert.match(ROOT, /"back": routeShortcutEnabled \? \(\) => window\.history\.back\(\) : null/);
  assert.match(ROOT, /"forward": routeShortcutEnabled \? \(\) => window\.history\.forward\(\) : null/);
  assert.match(ROOT, /"zoom-in": zoomBy\(1\)/);
  assert.match(ROOT, /"zoom-out": zoomBy\(-1\)/);
  assert.match(ROOT, /"actual-size": \(\) => useInterfaceScaleStore\.getState\(\)\.reset\(\)/);
});

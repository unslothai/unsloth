// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

import { readSrc, readText, registerBundlerResolver } from "./helpers/kit.ts";

// The native menu exists only on macOS, so chords are resolved as a Mac would on any runner.
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {
    platform: "MacIntel",
    userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
  },
});

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
const goActions = rowsOf("GO_ROWS");
const settingsActions = rowsOf("SETTINGS_ROWS");
const helpActions = rowsOf("HELP_ROWS");
const rustActions = [...fileActions, ...viewActions, ...goActions, ...helpActions];
const HELP = readSrc("components/help-actions.ts");
const actionsOf = (source: string, type: string) =>
  [...source.match(new RegExp(`export type ${type} =([^;]+);`))![1].matchAll(/"([a-z-]+)"/g)].map(
    (m) => m[1],
  );
assert.match(CHORDS, /export type AppMenuAction =\s*\| HelpAction/);
const hookActions = [...actionsOf(CHORDS, "AppMenuAction"), ...actionsOf(HELP, "HelpAction")];

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
  assert.deepEqual(helpActions, [
    "help-documentation",
    "help-keyboard-shortcuts",
    "help-whats-new",
    "help-troubleshooting",
    "help-system-status",
    "help-send-feedback",
  ]);
  const helpGroups = [...HELP.match(/HELP_GROUPS[^=]*=([\s\S]*?);/)![1].matchAll(/"([a-z-]+)"/g)].map(
    (m) => m[1],
  );
  assert.deepEqual(helpGroups, helpActions, "the sidebar's Help lists the menu's items in order");
  assert.deepEqual(goActions, [
    "go-chat",
    "go-projects",
    "go-library",
    "go-hub",
    "go-train",
    "go-recipes",
    "go-images",
    "go-video",
    "go-audio",
    "go-export",
  ]);
  assert.match(APP_MENU, /Row::Submenu\("Settings", SETTINGS_ROWS\)/);
  assert.deepEqual([...hookActions].sort(), [...rustActions].sort());
});

test("Go > Settings lists every Settings page, in the dialog's order", () => {
  const dialog = readSrc("features/settings/settings-dialog.tsx");
  const tabs = [...dialog.slice(dialog.indexOf("const TABS")).matchAll(/id: "([a-z-]+)"/g)].map((m) => m[1]);
  assert.deepEqual(settingsActions, tabs.slice(0, settingsActions.length).map((tab) => `settings-${tab}`));
  const store = readSrc("features/settings/stores/settings-dialog-store.ts");
  const known = [...store.match(/SETTINGS_TABS = \[([\s\S]*?)\]/)![1].matchAll(/"([a-z-]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...settingsActions].sort(), known.map((tab) => `settings-${tab}`).sort());
  assert.match(CHORDS, /export type SettingsMenuAction = `settings-\$\{SettingsTab\}`;/);
  assert.match(ROOT, /`settings-\$\{tab\}`,\s*!isAuthFlowRoute && settingsTabVisible\(tab, isOwner\)/);
});

test("every action is handled in the app shell", () => {
  for (const action of rustActions) {
    assert.ok(ROOT.includes(`"${action}":`), `__root.tsx handles ${action}`);
  }
  assert.match(ROOT, /isAuthFlowRoute \|\| !helpActionAvailable\(action, isOwner\)/);
  assert.match(readSrc("components/app-sidebar.tsx"), /\.filter\(\(action\) => helpActionAvailable\(action, isOwner\)\)/);
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
  const marked = openFolder.indexOf("markProjectSourcesPending(project.id)");
  assert.ok(marked > openFolder.indexOf("await createChatProject("));
  assert.ok(marked < openFolder.indexOf("await createLinkedFolder("));
  assert.match(landing, /hasProjectSourcesPending\(projectId\) \? "sources" : "chats"/);
  assert.doesNotMatch(openFolder, /deleteChatProject/);
  assert.match(landing, /<ProjectLanding\s+key=\{baseView\.projectId\}/);
});

test("Open Folder still lands on Sources after a visit during the link", () => {
  const dropzone = readSrc("features/rag/components/project-source-dropzone.tsx");
  const openFolder = readSrc("features/chat/utils/open-folder-as-project.ts");
  const landing = readSrc("features/chat/chat-page.tsx");
  const m = new Function(
    `${stripTypeScriptTypes(dropzone.slice(dropzone.indexOf("const projectsWithPendingSources"), dropzone.indexOf("/** Upload staged files"))).replace(/^export /gm, "")}
    return { markProjectSourcesPending, hasProjectSourcesPending, consumeProjectSourcesPending, noteProjectLandingMounted, isProjectLandingMounted };`,
  )();
  const mount = (id: string) => (m.consumeProjectSourcesPending(id), m.noteProjectLandingMounted(id));
  const settle = (id: string) => {
    if (!m.isProjectLandingMounted(id)) m.markProjectSourcesPending(id);
  };
  assert.match(landing, /consumeProjectSourcesPending\(projectId\);\s*return noteProjectLandingMounted\(projectId\);/);
  assert.match(openFolder, /if \(!isProjectLandingMounted\(project\.id\)\) markProjectSourcesPending\(project\.id\);/);

  m.markProjectSourcesPending("a");
  const leave = mount("a");
  leave();
  settle("a");
  assert.equal(m.hasProjectSourcesPending("a"), true);

  m.markProjectSourcesPending("b");
  mount("b");
  settle("b");
  assert.equal(m.hasProjectSourcesPending("b"), false);

  m.markProjectSourcesPending("c");
  settle("c");
  assert.equal(m.hasProjectSourcesPending("c"), true);
});

test("every menu item stays disabled while the desktop app is not showing the app", () => {
  const hook = readSrc("app/use-app-menu-actions.ts");
  const provider = readSrc("app/provider.tsx");
  assert.match(provider, /setDesktopShellReady\(canMountApp && appShellReady\);\s*return \(\) => setDesktopShellReady\(false\);/);
  assert.match(provider, /const showApp = canMountApp && appShellReady;/);
  assert.match(ROOT, /\}, desktopShellReady\);/);
  assert.match(hook, /const enabled = ready\s*\?/);
  assert.match(hook, /latest\.current = ready \? handlers : \{\};/);
});

test("Open Folder is disabled until the open in progress finishes", () => {
  const openFolder = readSrc("features/chat/utils/open-folder-as-project.ts");
  assert.match(openFolder, /if \(opening\) return null;\s*setOpening\(true\);/);
  assert.match(openFolder, /\} finally \{\s*setOpening\(false\);/);
  assert.match(ROOT, /pathLeasesSupported && !ragUnavailable && !openingFolder/);
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
  const rustAccel = (action: string) =>
    APP_MENU.match(new RegExp(`Row::Action\\(\\s*"${action}",\\s*"[^"]+",\\s*"([^"]+)"`))?.[1];
  const native = (accel: string) =>
    accel
      .replace(/\+([A-Z])$/, "+Key$1")
      .replace(/\+(\d)$/, "+Digit$1")
      .replace("+[", "+BracketLeft")
      .replace("+]", "+BracketRight")
      .replace("+=", "+Equal")
      .replace(/\+-$/, "+Minus")
      .replace("+/", "+Slash");
  for (const action of rustActions) {
    const accel = rustAccel(action);
    assert.equal(defaults[action as keyof typeof defaults], accel ? native(accel) : null, action);
  }
  assert.equal(
    menuAccelerators({ toggleSidebar: { primary: "Mod+KeyJ" } } as never)["toggle-sidebar"],
    "CmdOrCtrl+KeyJ",
  );
  assert.equal(
    menuAccelerators({ findInPage: { primary: null } } as never)["find"],
    null,
  );
  assert.equal(
    menuAccelerators({ findInPage: { primary: "Mod+KeyO" } } as never)["open-folder"],
    null,
  );
  assert.equal(
    menuAccelerators({ toggleSidebar: { primary: "Mod+KeyW" } } as never)["toggle-sidebar"],
    null,
  );
  assert.equal(
    menuAccelerators({ findInPage: { primary: "Mod+KeyM", alternate: "Mod+KeyJ" } } as never)["find"],
    "CmdOrCtrl+KeyJ",
  );
  assert.match(MAIN_RS, /MenuItemBuilder::with_id\(APP_QUIT_MENU_ID, "Quit Unsloth"\)\s*\.accelerator\("CmdOrCtrl\+Q"\)/);
  const { MENU_CHORDS, NATIVE_MENU_CHORDS } = await import("../src/app/app-menu-chords.ts");
  for (const { chord } of Object.values(MENU_CHORDS)) {
    if (chord) assert.ok(!NATIVE_MENU_CHORDS.has(chord), `${chord} is not a native chord`);
  }
  assert.equal(
    menuAccelerators({ toggleSidebar: { primary: "Alt+KeyB" } } as never)["toggle-sidebar"],
    null,
  );
});

test("menu items for web shortcuts follow a mounted handler, and honour claims", () => {
  const hook = readSrc("features/settings/hooks/use-shortcut.ts");
  assert.match(hook, /if \(!enabled\) return;\s*return registerShortcutTrigger\(id, \{\s*claims: \(\) => latestRef\.current\.claims\?\.\(\) !== false,/);
  assert.match(hook, /\(triggers\.get\(id\) \?\? \[\]\)\.some\(\(t\) => t\.claims\(\)\)/);
  assert.match(hook, /attributeFilter: \["aria-hidden", "inert"\]/);
  assert.match(hook, /modalObserver\.observe\(document\.body, \{ childList: true \}\)/);
  for (const id of ["toggleSidebar", "findInPage", "previousChat", "nextChat"]) {
    assert.ok(ROOT.includes(`useShortcutAvailable("${id}", isTauri)`), `${id} enables its item`);
  }
});

test("Back and Forward follow page history, and zoom steps the interface scale", () => {
  assert.match(ROOT, /"back": routeShortcutEnabled \? \(\) => window\.history\.back\(\) : null/);
  assert.match(ROOT, /"forward": routeShortcutEnabled \? \(\) => window\.history\.forward\(\) : null/);
  assert.match(ROOT, /"zoom-in": \(\) => zoomInterfaceFromMenu\(1\)/);
  assert.match(ROOT, /"zoom-out": \(\) => zoomInterfaceFromMenu\(-1\)/);
  assert.match(ROOT, /"actual-size": \(\) => zoomInterfaceFromMenu\(0\)/);
});

test("Help items reuse the icon of the Settings tab they open", () => {
  const dialog = readSrc("features/settings/settings-dialog.tsx");
  const tabIcon = (id: string) =>
    dialog.match(new RegExp(`id: "${id}",\\s*labelKey: "[^"]+",\\s*icon: (\\w+)`))?.[1];
  const helpIcon = (action: string) => HELP.match(new RegExp(`"${action}": \\{[^}]*icon: (\\w+)`))?.[1];
  for (const [action, tab] of [
    ["help-keyboard-shortcuts", "keyboard-shortcuts"],
    ["help-troubleshooting", "debugging"],
    ["help-system-status", "resources"],
  ]) {
    const icon = tabIcon(tab);
    assert.ok(icon, `the ${tab} tab's icon`);
    assert.equal(helpIcon(action), icon, action);
  }
});

test("About Unsloth uses the info icon Studio uses everywhere else", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  const about = sidebar.slice(0, sidebar.indexOf('{t("shell.helpMenu.about")}'));
  assert.match(about.slice(about.lastIndexOf("<HugeiconsIcon")), /^<HugeiconsIcon icon=\{InformationCircleIcon\}/);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import type { SandboxCapability } from "../src/features/chat/api/sandbox-capability.ts";
import type { PersistedChatSettings } from "../src/features/chat/api/chat-settings-api.ts";
import {
  osSandboxMissing,
  sandboxSwitchState,
} from "../src/features/chat/sandbox-level.ts";
import { assignSanitizedMirroredSettings } from "../src/features/chat/utils/mirrored-chat-settings.ts";
import { settingsTabVisible } from "../src/features/settings/settings-tab-visibility.ts";
import {
  SETTINGS_SEARCH_INDEX,
  renderedSearchEntries,
} from "../src/features/settings/settings-search.ts";
import { useSettingsDialogStore } from "../src/features/settings/stores/settings-dialog-store.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);
const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore, loadSandboxLevel, CHAT_SANDBOX_LEVEL_KEY } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

const LEVEL_KEY = "unsloth_chat_sandbox_level";

test("the sandbox level defaults to High; only an explicit Low reads as Low", () => {
  assert.equal(CHAT_SANDBOX_LEVEL_KEY, LEVEL_KEY);
  // A profile from before the setting existed.
  localStorageFake.delete(LEVEL_KEY);
  assert.equal(loadSandboxLevel(), "high");
  assert.equal(useChatRuntimeStore.getState().sandboxLevel, "high");
  for (const garbled of ["", "LOW", "medium", "null"]) {
    localStorageFake.set(LEVEL_KEY, garbled);
    assert.equal(loadSandboxLevel(), "high", garbled);
  }
  localStorageFake.set(LEVEL_KEY, "low");
  assert.equal(loadSandboxLevel(), "low");
  localStorageFake.delete(LEVEL_KEY);
});

test("setting the level stores it, and a saved server value hydrates it", async () => {
  useChatRuntimeStore.getState().setSandboxLevel("low");
  assert.equal(useChatRuntimeStore.getState().sandboxLevel, "low");
  assert.equal(localStorageFake.get(LEVEL_KEY), "low");
  useChatRuntimeStore.getState().setSandboxLevel("high");
  assert.equal(localStorageFake.get(LEVEL_KEY), "high");

  settingsHttp.settings = { sandboxLevel: "low" };
  useChatRuntimeStore.setState({ settingsHydrated: false });
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.equal(useChatRuntimeStore.getState().sandboxLevel, "low");
  useChatRuntimeStore.getState().setSandboxLevel("high");
});

test("the mirrored patch keeps high and low and drops anything else", () => {
  for (const level of ["high", "low"]) {
    const out: PersistedChatSettings = {};
    assignSanitizedMirroredSettings({ sandboxLevel: level }, out);
    assert.equal(out.sandboxLevel, level);
  }
  const out: PersistedChatSettings = {};
  assignSanitizedMirroredSettings({ sandboxLevel: "medium" }, out);
  assert.equal("sandboxLevel" in out, false);
});

const ADAPTER = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

test("every request that carries permission_mode carries sandbox_level next to it", () => {
  const sites = [...ADAPTER.matchAll(/^(\s*)permission_mode: permissionMode,\n\s*(.*)$/gm)];
  // Token count extras, the hosted-provider body with local tools, the local chat body.
  assert.equal(sites.length, 3);
  for (const site of sites) {
    assert.match(site[2], /^sandbox_level: (runtime\.)?sandboxLevel,$/);
  }
  const countExtras = ADAPTER.slice(
    ADAPTER.indexOf("export async function buildLocalTokenCountExtras"),
    ADAPTER.indexOf("permission_mode: permissionMode,"),
  );
  assert.match(countExtras, /^\s*sandboxLevel,$/m);
});

function capability(overrides: Partial<SandboxCapability>): SandboxCapability {
  return {
    pythonOsIsolated: true,
    terminalOsIsolated: true,
    backend: "bubblewrap",
    platform: "linux",
    reason: "",
    setupAction: null,
    manualCommand: "",
    canRunSetup: false,
    setupBlocked: null,
    needsConsent: false,
    ...overrides,
  };
}

test("the OS sandbox counts as missing only on a settled answer without Python or Terminal isolation", () => {
  const neither = capability({ pythonOsIsolated: false, terminalOsIsolated: false, backend: "none" });
  assert.equal(osSandboxMissing(capability({ terminalOsIsolated: false, backend: "none" })), true);
  assert.equal(osSandboxMissing(capability({ pythonOsIsolated: false, backend: "none" })), true);
  assert.equal(osSandboxMissing(neither), true);
  assert.equal(osSandboxMissing(capability({})), false);
  // No answer yet, or an old server that cannot say.
  assert.equal(osSandboxMissing(null), false);
  assert.equal(osSandboxMissing({ ...neither, backend: "unknown" }), false);
});

test("without an OS sandbox the switch reads Low even with High saved", () => {
  const neither = capability({ pythonOsIsolated: false, terminalOsIsolated: false, backend: "none" });
  assert.deepEqual(sandboxSwitchState("high", "auto", neither), { checked: false, disabled: false });
  assert.deepEqual(sandboxSwitchState("high", "auto", capability({})), { checked: true, disabled: false });
  assert.deepEqual(sandboxSwitchState("high", "auto", null), { checked: true, disabled: false });
  assert.deepEqual(sandboxSwitchState("high", "full", neither), { checked: false, disabled: true });
});

test("the switch is on for High and disabled with the saved value under Full access", () => {
  assert.deepEqual(sandboxSwitchState("high", "auto"), { checked: true, disabled: false });
  assert.deepEqual(sandboxSwitchState("low", "ask"), { checked: false, disabled: false });
  assert.deepEqual(sandboxSwitchState("low", "full"), { checked: false, disabled: true });
  assert.deepEqual(sandboxSwitchState("high", "full"), { checked: true, disabled: true });
});

const SANDBOX_TAB = readFileSync(
  new URL("../src/features/settings/tabs/sandbox-tab.tsx", import.meta.url),
  "utf8",
);

test("a managed account sees the Sandbox tab, with Permissions only and no owner-only reads", () => {
  assert.equal(settingsTabVisible("sandbox", false), true);
  const tab = SANDBOX_TAB.slice(
    SANDBOX_TAB.indexOf("export function SandboxTab()"),
    SANDBOX_TAB.indexOf("function OsSandboxSections()"),
  );
  assert.match(tab, /<PermissionsSection \/>/);
  assert.match(tab, /\{isOwner \? \(\s*<OsSandboxSections \/>/);
  // The owner-only routes (/api/settings/sandbox and the setup) are read only below.
  const permissions = SANDBOX_TAB.slice(
    SANDBOX_TAB.indexOf("function PermissionsSection()"),
    SANDBOX_TAB.indexOf("function OsSandboxSections()"),
  );
  assert.doesNotMatch(
    permissions,
    /loadSandboxStatus|loadSandboxSetup|loadHostPreparation|startSandboxSetup|updateSandboxSettings/,
  );
  // Search does not offer the owner's rows to a managed account.
  const managed = renderedSearchEntries(SETTINGS_SEARCH_INDEX, "sandbox", "huggingface", false);
  assert.ok(managed.includes("settings.general.permissions.sectionTitle"));
  assert.ok(!managed.includes("settings.sandbox.toolsSection"));
  assert.ok(
    renderedSearchEntries(SETTINGS_SEARCH_INDEX, "sandbox", "huggingface", true).includes(
      "settings.sandbox.toolsSection",
    ),
  );
});

test("the old general-permissions link opens Permissions under Settings > Sandbox", () => {
  // An explicit opener skips reading document focus, which node does not have.
  const dialog = useSettingsDialogStore;
  dialog.getState().openDialog("general", { scrollTarget: "general-permissions", opener: null });
  assert.equal(dialog.getState().activeTab, "sandbox");
  assert.equal(dialog.getState().scrollTarget, "sandbox-permissions");
  dialog.getState().closeDialog();
  dialog.getState().openDialog("sandbox", { scrollTarget: "sandbox-permissions", opener: null });
  assert.equal(dialog.getState().activeTab, "sandbox");
  assert.equal(dialog.getState().scrollTarget, "sandbox-permissions");
  // Reselecting the tab keeps the pending jump; another tab drops it.
  dialog.getState().setActiveTab("sandbox");
  assert.equal(dialog.getState().scrollTarget, "sandbox-permissions");
  dialog.getState().setActiveTab("general");
  assert.equal(dialog.getState().scrollTarget, null);
  dialog.getState().closeDialog();
  dialog.getState().openDialog("general", { scrollTarget: "general-hub", opener: null });
  assert.equal(dialog.getState().activeTab, "general");
  assert.equal(dialog.getState().scrollTarget, "general-hub");
  dialog.getState().closeDialog();
});

test("a queued run keeps the level it was queued under, and a switch invalidates the queue", async () => {
  const { snapshotQueuedChatRunSettings } = await import(
    "../src/features/chat/utils/queued-chat-run-settings.ts"
  );
  useChatRuntimeStore.getState().setSandboxLevel("high");
  const queued = snapshotQueuedChatRunSettings(useChatRuntimeStore.getState());
  assert.equal(queued.sandboxLevel, "high");
  const epoch = useChatRuntimeStore.getState().queuedSettingsEpoch;
  useChatRuntimeStore.getState().setSandboxLevel("low");
  assert.equal(useChatRuntimeStore.getState().queuedSettingsEpoch, epoch + 1);
  assert.equal(queued.sandboxLevel, "high");
  useChatRuntimeStore.getState().setSandboxLevel("high");
});

test("every token count body carries sandbox_level, also when no tool is selected", () => {
  const start = ADAPTER.indexOf("export async function buildLocalTokenCountExtras");
  const countExtras = ADAPTER.slice(start, ADAPTER.indexOf("\n}\n", start));
  const bodies = [...countExtras.matchAll(/return \{\n([\s\S]*?)\n\s*\};/g)].map((m) => m[1]);
  assert.equal(bodies.length, 3);
  for (const body of bodies) assert.match(body, /sandbox_level: sandboxLevel,/);
});

const MENU = readFileSync(
  new URL("../src/features/chat/permission-mode-select.tsx", import.meta.url),
  "utf8",
);

test("Low reads no OS sandbox capability, in the menu and in Settings", () => {
  assert.match(MENU, /useSandboxCapability\(sandboxLevel === "high"\)/);
  assert.match(MENU, /if \(!enabled\) return;/);
  assert.match(SANDBOX_TAB, /useSandboxCapability\(sandboxLevel === "high"\)/);
});

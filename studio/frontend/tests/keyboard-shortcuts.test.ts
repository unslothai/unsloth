// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  SHORTCUT_DEFS,
  SHORTCUT_SLOTS,
  bindingFromEvent,
  defaultBindingFor,
  formatBindingLabel,
  formatBindingValue,
  activationBelongsToFocus,
  isAcceptableBinding,
  isBrowserReservedBinding,
  isShortcutId,
  matchesBinding,
  parseBinding,
} from "../src/features/settings/lib/keyboard-shortcuts.ts";
import {
  KEYBOARD_SHORTCUTS_STORAGE_KEY,
  findConflicts,
  isSlotOverridden,
  migrateStoredOverrides,
  resolveAllBindings,
  resolveBinding,
  resolveBindings,
  shortcutOwningBinding,
} from "../src/features/settings/stores/keyboard-shortcuts-store.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";

function keyEvent(
  code: string,
  mods: Partial<{
    metaKey: boolean;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    /** What a Windows or Linux layout reports for AltGr, alongside Ctrl+Alt. */
    altGraph: boolean;
  }> = {},
) {
  const { altGraph = false, ...rest } = mods;
  return {
    code,
    metaKey: false,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    ...rest,
    getModifierState: (key: string) => key === "AltGraph" && altGraph,
  };
}

test("a binding round-trips through serialize and parse", () => {
  const binding = {
    code: "KeyO",
    mod: true,
    ctrl: false,
    shift: true,
    alt: false,
  };
  const value = formatBindingValue(binding);
  assert.equal(value, "Mod+Shift+KeyO");
  assert.deepEqual(parseBinding(value), binding);
});

test("parse rejects junk, empties and modifier-only values", () => {
  assert.equal(parseBinding(null), null);
  assert.equal(parseBinding(""), null);
  assert.equal(parseBinding("Mod+ShiftLeft"), null);
  assert.equal(parseBinding("Hyper+KeyO"), null);
});

test("every shipped default parses, on either platform", () => {
  for (const def of SHORTCUT_DEFS) {
    for (const mac of [true, false]) {
      for (const slot of SHORTCUT_SLOTS) {
        const value = defaultBindingFor(def, slot, mac);
        if (value === null) continue;
        assert.ok(
          parseBinding(value),
          `${def.id}.${slot} has an unparseable default: ${value}`,
        );
      }
    }
  }
});

// formatBindingValue emits Mod, Ctrl, Alt, Shift, key. A default written in any
// other order would never equal a chord the recorder produces, so rebinding to
// the shipped chord would read as a change and the reset button would never
// appear to have worked.
test("every shipped default is already in canonical order", () => {
  for (const def of SHORTCUT_DEFS) {
    for (const mac of [true, false]) {
      for (const slot of SHORTCUT_SLOTS) {
        const value = defaultBindingFor(def, slot, mac);
        if (value === null) continue;
        const parsed = parseBinding(value);
        assert.ok(parsed);
        assert.equal(
          formatBindingValue(parsed),
          value,
          `${def.id}.${slot} is not canonical: ${value}`,
        );
      }
    }
  }
});

// Off macOS there is no ⌃-versus-⌘ distinction, so a Ctrl-only default would be
// both unreachable and a duplicate of the Mod row. Those actions must ship a
// different chord there.
test("no Ctrl-only default survives onto Windows and Linux", () => {
  for (const def of SHORTCUT_DEFS) {
    for (const slot of SHORTCUT_SLOTS) {
      const parsed = parseBinding(defaultBindingFor(def, slot, false));
      if (!parsed) continue;
      assert.ok(
        !parsed.ctrl,
        `${def.id}.${slot} keeps a Ctrl chord off macOS: ${formatBindingValue(parsed)}`,
      );
    }
  }
});

test("a focused control keeps its own Enter", () => {
  const enter = parseBinding("Enter");
  assert.ok(enter);
  const deny = { tagName: "BUTTON", getAttribute: () => null };
  // preventDefault on a window keydown cancels the click the browser would
  // have made, so approving here would overrule the button the user picked.
  assert.equal(activationBelongsToFocus(enter, deny), true);
  assert.equal(activationBelongsToFocus(enter, null), false);
  assert.equal(
    activationBelongsToFocus(enter, { tagName: "DIV", getAttribute: () => null }),
    false,
  );
  // A div acting as a button counts too.
  assert.equal(
    activationBelongsToFocus(enter, {
      tagName: "DIV",
      getAttribute: (name: string) => (name === "role" ? "button" : null),
    }),
    true,
  );
  // Escape activates nothing, so declining still works from any focus.
  const escape = parseBinding("Escape");
  assert.ok(escape);
  assert.equal(activationBelongsToFocus(escape, deny), false);
  // A chord with a modifier is nobody else's.
  const modEnter = parseBinding("Mod+Enter");
  assert.ok(modEnter);
  assert.equal(activationBelongsToFocus(modEnter, deny), false);
});

test("nothing that deletes chats ships on a chord", () => {
  const del = SHORTCUT_DEFS.find((def) => def.id === "deleteSelectedChats");
  assert.ok(del);
  for (const slot of SHORTCUT_SLOTS) {
    for (const mac of [true, false]) {
      assert.equal(defaultBindingFor(del, slot, mac), null);
    }
  }
});

test("an unassigned action ships both slots empty", () => {
  const forkChat = SHORTCUT_DEFS.find((def) => def.id === "forkChat");
  assert.ok(forkChat);
  assert.equal(defaultBindingFor(forkChat, "primary", true), null);
  assert.equal(defaultBindingFor(forkChat, "alternate", true), null);
});

test("matching is exact about modifiers", () => {
  const binding = parseBinding("Mod+Shift+KeyO");
  assert.ok(binding);
  // Non-mac, so Mod is Ctrl.
  assert.ok(
    matchesBinding(
      binding,
      keyEvent("KeyO", { ctrlKey: true, shiftKey: true }),
      false,
    ),
  );
  // Missing Shift, extra Alt, and wrong key must all miss.
  assert.equal(
    matchesBinding(binding, keyEvent("KeyO", { ctrlKey: true }), false),
    false,
  );
  assert.equal(
    matchesBinding(
      binding,
      keyEvent("KeyO", { ctrlKey: true, shiftKey: true, altKey: true }),
      false,
    ),
    false,
  );
  assert.equal(
    matchesBinding(
      binding,
      keyEvent("KeyP", { ctrlKey: true, shiftKey: true }),
      false,
    ),
    false,
  );
});

test("off-platform Meta does not satisfy a Mod binding", () => {
  const binding = parseBinding("Mod+KeyB");
  assert.ok(binding);
  // Windows key held instead of Ctrl, on a non-mac platform.
  assert.equal(
    matchesBinding(binding, keyEvent("KeyB", { metaKey: true }), false),
    false,
  );
});

test("on macOS Mod is Cmd, and a bare Ctrl is not a substitute", () => {
  const binding = parseBinding("Mod+KeyB");
  assert.ok(binding);
  assert.ok(matchesBinding(binding, keyEvent("KeyB", { metaKey: true }), true));
  assert.equal(
    matchesBinding(binding, keyEvent("KeyB", { ctrlKey: true }), true),
    false,
  );
  // Ctrl is bindable in its own right on macOS.
  const ctrlBinding = parseBinding("Ctrl+KeyB");
  assert.ok(ctrlBinding);
  assert.ok(
    matchesBinding(ctrlBinding, keyEvent("KeyB", { ctrlKey: true }), true),
  );
});

test("a Ctrl chord from a Mac cannot fire on Windows or Linux", () => {
  // Ctrl is Mod off macOS, so this value is unreachable there. What it must
  // not do is drop the modifier it cannot express and fire on the bare key.
  const binding = parseBinding("Ctrl+KeyB");
  assert.ok(binding);
  assert.equal(matchesBinding(binding, keyEvent("KeyB"), false), false);
  assert.equal(
    matchesBinding(binding, keyEvent("KeyB", { ctrlKey: true }), false),
    false,
  );
});

test("AltGr typing does not fire an Alt chord off macOS", () => {
  const binding = parseBinding("Mod+Alt+KeyC");
  assert.ok(binding);
  // A real Ctrl+Alt press on a US layout still works.
  assert.ok(
    matchesBinding(
      binding,
      keyEvent("KeyC", { ctrlKey: true, altKey: true }),
      false,
    ),
  );
  // AltGr+C, which types ć on a Polish layout, must not.
  assert.equal(
    matchesBinding(
      binding,
      keyEvent("KeyC", { ctrlKey: true, altKey: true, altGraph: true }),
      false,
    ),
    false,
  );
  assert.equal(
    bindingFromEvent(
      keyEvent("KeyC", { ctrlKey: true, altKey: true, altGraph: true }),
      false,
    ),
    null,
  );
});

test("macOS Option still fires an ⌥ chord when AltGraph is reported", () => {
  // WebKit and Chromium report AltGraph for Option, so the guard above has to
  // stay off macOS or every ⌥ chord in the list would stop working.
  const binding = parseBinding("Mod+Alt+KeyC");
  assert.ok(binding);
  assert.ok(
    matchesBinding(
      binding,
      keyEvent("KeyC", { metaKey: true, altKey: true, altGraph: true }),
      true,
    ),
  );
});

test("no default takes a chord the browser owns without a reason", () => {
  // The exceptions are the spec chords Studio keeps for the desktop build,
  // where they work; everything else has to be reachable on the web.
  const deliberate = new Set([
    "Mod+KeyN",
    "Mod+Shift+KeyN",
    "Mod+Tab",
    "Mod+Shift+Tab",
    "Ctrl+Tab",
    "Ctrl+Shift+Tab",
  ]);
  for (const def of SHORTCUT_DEFS) {
    for (const slot of SHORTCUT_SLOTS) {
      for (const mac of [true, false]) {
        const value = defaultBindingFor(def, slot, mac);
        if (!value || deliberate.has(value)) continue;
        assert.equal(
          isBrowserReservedBinding(value, mac),
          false,
          `${def.id}.${slot} defaults to ${value}, which the browser owns (mac=${mac})`,
        );
      }
    }
  }
});

test("recording a chord ignores a lone modifier", () => {
  assert.equal(
    bindingFromEvent(keyEvent("ShiftLeft", { shiftKey: true }), false),
    null,
  );
  const binding = bindingFromEvent(keyEvent("KeyK", { ctrlKey: true }), false);
  assert.ok(binding);
  assert.equal(formatBindingValue(binding), "Mod+KeyK");
  // The same physical chord on macOS is Cmd, not Ctrl.
  const macBinding = bindingFromEvent(
    keyEvent("KeyK", { metaKey: true }),
    true,
  );
  assert.ok(macBinding);
  assert.equal(formatBindingValue(macBinding), "Mod+KeyK");
});

test("a bare letter is refused but function keys stand alone", () => {
  assert.equal(
    isAcceptableBinding({
      code: "KeyK",
      mod: false,
      ctrl: false,
      shift: false,
      alt: false,
    }),
    false,
  );
  assert.ok(
    isAcceptableBinding({
      code: "F5",
      mod: false,
      ctrl: false,
      shift: false,
      alt: false,
    }),
  );
  // Enter is bare, and only allowed for the two actions that register solely
  // while an approval prompt is on screen.
  const enter = {
    code: "Enter",
    mod: false,
    ctrl: false,
    shift: false,
    alt: false,
  };
  assert.equal(isAcceptableBinding(enter), false);
  assert.ok(isAcceptableBinding(enter, true));
});

test("only prompt-gated actions ship a bare-key default", () => {
  for (const def of SHORTCUT_DEFS) {
    for (const slot of SHORTCUT_SLOTS) {
      for (const mac of [true, false]) {
        const parsed = parseBinding(defaultBindingFor(def, slot, mac));
        if (!parsed) continue;
        assert.ok(
          isAcceptableBinding(parsed, def.allowBareKey),
          `${def.id}.${slot} ships a chord the recorder would refuse`,
        );
      }
    }
  }
  const bareKeyIds = SHORTCUT_DEFS.filter((def) => def.allowBareKey).map(
    (def) => def.id,
  );
  assert.deepEqual(bareKeyIds.sort(), [
    "approveToolRequest",
    "declineToolRequest",
  ]);
});

test("labels use each platform's own modifier notation", () => {
  const binding = parseBinding("Mod+Shift+KeyO");
  assert.ok(binding);
  assert.equal(formatBindingLabel(binding, true), "⇧⌘O");
  assert.equal(formatBindingLabel(binding, false), "Ctrl+Shift+O");
  const comma = parseBinding("Mod+Comma");
  assert.ok(comma);
  assert.equal(formatBindingLabel(comma, true), "⌘,");
});

test("an override wins, and null means unassigned", () => {
  assert.equal(resolveBinding({}, "toggleSidebar"), "Mod+KeyB");
  assert.equal(
    resolveBinding(
      { toggleSidebar: { primary: "Mod+Alt+KeyB" } },
      "toggleSidebar",
    ),
    "Mod+Alt+KeyB",
  );
  // Present-but-null is a deliberate clear, not a fallback to the default.
  assert.equal(
    resolveBinding({ toggleSidebar: { primary: null } }, "toggleSidebar"),
    null,
  );
  // Overriding one slot leaves the other on its shipped chord.
  assert.equal(
    resolveBinding({ nextChat: { primary: null } }, "nextChat", "alternate"),
    "Mod+Alt+ArrowRight",
  );
});

test("both slots resolve together", () => {
  assert.deepEqual(resolveBindings({}, "nextChat"), {
    primary: "Mod+Shift+BracketRight",
    alternate: "Mod+Alt+ArrowRight",
  });
  assert.deepEqual(resolveBindings({}, "archiveChat"), {
    primary: "Mod+Shift+KeyA",
    alternate: null,
  });
});

test("a slot reports whether it carries an edit", () => {
  assert.equal(isSlotOverridden({}, "nextChat", "primary"), false);
  const overrides = { nextChat: { alternate: null } };
  assert.equal(isSlotOverridden(overrides, "nextChat", "primary"), false);
  assert.ok(isSlotOverridden(overrides, "nextChat", "alternate"));
});

// Builds before alternates existed stored `id -> string | null`. Read as-is,
// that shape would resolve to nothing and silently revert every customization
// made since the shortcuts tab shipped.
test("the pre-alternate override shape migrates to the primary slot", () => {
  const legacy = JSON.stringify({
    toggleSidebar: "Mod+Alt+KeyB",
    searchChats: null,
    someRemovedAction: "Mod+KeyZ",
  });
  const migrated = migrateStoredOverrides(JSON.parse(legacy));
  assert.deepEqual(migrated, {
    toggleSidebar: { primary: "Mod+Alt+KeyB" },
    searchChats: { primary: null },
  });
  assert.equal(resolveBinding(migrated, "toggleSidebar"), "Mod+Alt+KeyB");
  assert.equal(resolveBinding(migrated, "searchChats"), null);
  // The alternate is untouched, so a default added later still reaches them.
  assert.equal(
    resolveBinding(migrated, "searchChats", "alternate"),
    "Mod+Shift+KeyP",
  );
});

test("the current override shape round-trips unchanged", () => {
  const stored = { nextChat: { primary: "Mod+KeyG", alternate: null } };
  assert.deepEqual(migrateStoredOverrides(stored), stored);
});

test("defaults ship without conflicts on either platform", () => {
  // findConflicts resolves against this process's platform, so check the other
  // one by hand: ⌘1-9 and ⌃1-9 collapse onto each other off macOS, and the
  // registry has to have moved one of them.
  assert.equal(findConflicts({}).size, 0);
  for (const mac of [true, false]) {
    const seen = new Map<string, string>();
    for (const def of SHORTCUT_DEFS) {
      for (const slot of SHORTCUT_SLOTS) {
        const value = defaultBindingFor(def, slot, mac);
        if (value === null) continue;
        const owner = seen.get(value);
        assert.ok(
          owner === undefined || owner === def.id,
          `${value} is claimed by both ${owner} and ${def.id} (mac=${mac})`,
        );
        seen.set(value, def.id);
      }
    }
  }
  const all = resolveAllBindings({});
  assert.equal(all.newChat.primary, "Mod+Shift+KeyO");
  assert.equal(all.newChat.alternate, "Mod+KeyN");
});

test("two actions on one chord are both flagged", () => {
  const conflicts = findConflicts({ toggleSidebar: { primary: "Mod+KeyK" } });
  assert.deepEqual(
    [...conflicts].sort(),
    ["searchChats", "toggleSidebar"].sort(),
  );
});

// A clash across slots is just as real as one between two primaries: both
// chords fire the same listener path, so the tab has to say so.
test("an alternate clashing with another action's primary is flagged", () => {
  const conflicts = findConflicts({
    archiveChat: { alternate: "Mod+KeyB" },
  });
  assert.deepEqual(
    [...conflicts].sort(),
    ["archiveChat", "toggleSidebar"].sort(),
  );
});

test("an action's own two slots never conflict with each other", () => {
  const conflicts = findConflicts({
    archiveChat: { alternate: "Mod+Shift+KeyA" },
  });
  assert.equal(conflicts.size, 0);
});

test("cleared actions never count as conflicting", () => {
  const conflicts = findConflicts({
    toggleSidebar: { primary: null },
    searchChats: { primary: null, alternate: null },
  });
  assert.equal(conflicts.size, 0);
});

test("ids from an older build are rejected", () => {
  assert.ok(isShortcutId("newChat"));
  assert.equal(isShortcutId("someRemovedAction"), false);
});

/** Position in SHORTCUT_DEFS, which is what ownership is decided by. */
const registryIndex = (id: string) =>
  SHORTCUT_DEFS.findIndex((def) => def.id === id);

test("a contested chord is owned by the earlier action in registry order", () => {
  // Derived, not hard-coded: the list is deliberately ordered for the UI, so a
  // reorder must keep this rule rather than trip a name-shaped assertion.
  const overrides = { toggleSidebar: { primary: "Mod+KeyK" } };
  const owner = shortcutOwningBinding(overrides, "Mod+KeyK");
  assert.ok(owner);
  const claimants = ["searchChats", "toggleSidebar"];
  assert.deepEqual(
    owner,
    claimants.sort((a, b) => registryIndex(a) - registryIndex(b))[0],
  );
  // The loser keeps its own binding elsewhere, so ownership is per chord.
  assert.equal(shortcutOwningBinding(overrides, "Mod+Comma"), "openSettings");
});

test("exactly one owner exists per contested chord", () => {
  const overrides = {
    toggleSidebar: { primary: "Mod+KeyK" },
    newChat: { primary: "Mod+KeyK" },
  };
  const contested = [...findConflicts(overrides)];
  assert.equal(contested.length, 3);
  const owners = new Set(
    contested.map(() => shortcutOwningBinding(overrides, "Mod+KeyK")),
  );
  assert.equal(owners.size, 1);
  assert.equal(
    shortcutOwningBinding(overrides, "Mod+KeyK"),
    contested.sort((a, b) => registryIndex(a) - registryIndex(b))[0],
  );
});

test("an unbound or unclaimed chord has no owner", () => {
  assert.equal(shortcutOwningBinding({}, null), null);
  assert.equal(shortcutOwningBinding({}, "Mod+Alt+KeyZ"), null);
  // A cleared action does not own the chord it used to have.
  assert.equal(
    shortcutOwningBinding(
      { searchChats: { primary: null } },
      "Mod+KeyK",
    ),
    null,
  );
});

// Reset-all is the documented escape hatch. A chord bound to something the
// browser eats leaves the Shortcuts tab itself hard to reach, so the General
// reset has to cover this key or the user is stuck with it.
test("Reset all local preferences clears the rebound chords", async () => {
  const source = await readFile(
    new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
    "utf8",
  );
  const keys = source.slice(
    source.indexOf("const PREFS_KEYS"),
    source.indexOf("];", source.indexOf("const PREFS_KEYS")),
  );
  assert.ok(keys, "PREFS_KEYS moved; this contract needs updating");
  assert.ok(
    keys.includes("KEYBOARD_SHORTCUTS_STORAGE_KEY") ||
      keys.includes(`"${KEYBOARD_SHORTCUTS_STORAGE_KEY}"`),
    `${KEYBOARD_SHORTCUTS_STORAGE_KEY} missing from PREFS_KEYS`,
  );
});

// Every overlay carries the tab label and every row, or a non-English install
// shows an English word in the middle of a translated settings dialog.
test("every locale overlay carries the shortcut strings", async () => {
  const locales = [
    "ar", "de", "es", "fr", "hi", "it", "ja", "ko", "pt-br", "ru", "zh-CN",
  ];
  for (const locale of locales) {
    const source = await readFile(
      new URL(`../src/i18n/locales/${locale}.ts`, import.meta.url),
      "utf8",
    );
    const at = source.indexOf("keyboardShortcuts: {");
    assert.notEqual(
      at,
      -1,
      `${locale} is missing the settings.keyboardShortcuts subtree`,
    );
    const subtree = source.slice(at, source.indexOf("\n    },", at));
    for (const key of ["title", "resetAll", "conflictShadowed", "unassigned"]) {
      assert.ok(
        subtree.includes(`${key}:`),
        `${locale} is missing settings.keyboardShortcuts.${key}`,
      );
    }
    for (const def of SHORTCUT_DEFS) {
      assert.ok(
        subtree.includes(`${def.id}: {`),
        `${locale} is missing settings.keyboardShortcuts.actions.${def.id}`,
      );
    }
  }
});

test("a cleared alternate keeps its row, so it can be restored", async () => {
  const source = await readFile(
    new URL("../src/features/settings/tabs/keyboard-shortcuts-tab.tsx", import.meta.url),
    "utf8",
  );
  // Clearing a slot stores null, so a row that keys off the resolved value
  // alone would hide the slot's own restore control along with the chord.
  assert.match(
    source,
    /hasAlternate =\s*\n\s*defaultBindingFor\(def, "alternate", mac\) !== null/,
  );
});

test("the chords that need one target do not fire where there are two", async () => {
  // No React renderer here, so this asserts on source, like its siblings.
  const read = async (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");
  const chatPage = await read("../src/features/chat/chat-page.tsx");
  const thread = await read("../src/components/assistant-ui/thread.tsx");
  const toolCard = await read(
    "../src/components/assistant-ui/tool-confirmation-controls.tsx",
  );

  // Compare drops the header pickers and gives each pane its own, so the
  // header chord would toggle state nothing renders.
  assert.match(
    chatPage,
    /const headerPickersShown = active && view\.mode !== "compare";/,
  );
  assert.match(chatPage, /enabled: headerPickersShown \}/);
  // Both panes mount the last message's fork button, and the chord would go
  // to whichever registered its listener first.
  assert.match(thread, /enabled: chatActive && !inComparePane && isLast/);
  // Same for a second parked tool request: neither card claims the keys.
  assert.match(toolCard, /soleRequest &&\n\s*showControls &&/);
});

test("the composer chords outlive the recording bar", async () => {
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  // Dictation swaps ComposerRightControls out for the recording bar, so a
  // chord registered in there could start dictation and never stop it.
  const controls = thread.indexOf("const ComposerRightControls:");
  assert.ok(controls !== -1, "the controls component moved");
  for (const id of ["startDictation", "sendMessage"]) {
    const at = thread.indexOf(`useShortcut(\n    "${id}"`);
    const inline = thread.indexOf(`useShortcut("${id}"`);
    const found = at === -1 ? inline : at;
    assert.ok(found !== -1, `${id} lost its call site`);
    assert.ok(found < controls, `${id} registers inside the recording swap`);
  }
  // Send goes through the form, which runs the parking, queueing and refusing
  // that the runtime's own send knows nothing about, and through the recording
  // bar's own path while dictation is running.
  assert.match(thread, /formRef\.current\?\.requestSubmit\(\);/);
  assert.match(
    thread,
    /if \(isDictating\) \{\n\s*if \(!dictationBlocked\) sendAfterDictation\(\);/,
  );
});

test("a collapsed sidebar section is not published for the chords", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  // Navigation and Select all walk what is on screen, so a section the user
  // closed counts as gone, the same as a closed project folder.
  assert.match(
    sidebar,
    /visiblePinnedItems = useMemo\(\s*\(\) => \(pinnedOpen \? sortedPinnedChatItems : \[\]\)/,
  );
  assert.match(
    sidebar,
    /visibleRecentItems = useMemo\(\s*\(\) => \(chatOpen \? sortedRecentChatItems : \[\]\)/,
  );
  assert.match(sidebar, /organizeBy !== "project" \|\| !projectsOpen/);
  // And the published lists are the filtered ones.
  assert.match(sidebar, /pinnedItems: visiblePinnedItems,/);
  assert.match(sidebar, /recentItems: visibleRecentItems,/);
});

test("the MCP chord does not live behind the MCP pill", async () => {
  const button = await readFile(
    new URL("../src/features/chat/mcp-composer-button.tsx", import.meta.url),
    "utf8",
  );
  const page = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  // MCP ships off for a chat and the pill only renders once it is on, so a
  // chord registered inside the pill would do nothing until it was found by
  // hand. The dialog and its chord mount for the chat instead.
  const pill = button.indexOf("export function McpComposerButton");
  const mount = button.indexOf("export function McpServersDialogMount");
  assert.ok(pill !== -1 && mount > pill, "the mount moved");
  assert.ok(
    button.indexOf('useShortcut("openMcpServers"') > mount,
    "the chord is back inside the pill",
  );
  // Mounted through the route change, not gated on `active`: the flag lives in
  // a store, so a dialog left open has to be closed on the way out rather than
  // unmounted with it still set.
  assert.match(page, /\n\s*<McpServersDialogMount \/>/);
  assert.match(button, /if \(!chatActive && open\) setOpen\(false\);/);
  assert.match(button, /open=\{chatActive && open\}/);
});

test("the copy chords keep their gesture across the read", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const clipboard = await readFile(
    new URL("../src/lib/copy-to-clipboard.ts", import.meta.url),
    "utf8",
  );
  // Both copies read storage first, and a strict engine drops the gesture
  // across that await, leaving writeText and its execCommand fallback with
  // nothing to run inside. The write starts with a promised payload instead.
  assert.match(clipboard, /"text\/plain": payload\.then\(/);
  for (const fn of ["copyChatItemAsMarkdown", "copyChatSessionId"]) {
    const at = sidebar.indexOf(`async function ${fn}(`);
    assert.ok(at !== -1, `${fn} moved`);
    const body = sidebar.slice(at, sidebar.indexOf("\n  }", at));
    assert.ok(
      body.includes("copyToClipboardFrom(async () =>"),
      `${fn} awaits its read before starting the write`,
    );
  }
});

test("every settings tab survives a reload", () => {
  // The persisted-tab check reads this same list, so a tab added to the union
  // alone can no longer be rejected back to General.
  assert.ok(SETTINGS_TABS.includes("keyboard-shortcuts"));
  assert.equal(new Set(SETTINGS_TABS).size, SETTINGS_TABS.length);
});

test("a Super chord off macOS records nothing rather than a different chord", () => {
  // matchesBinding rejects a non-mac event carrying Meta, so there is nowhere to
  // put it. Persisting Alt+K for a user who pressed Super+Alt+K would assign an
  // action to a chord they did not choose, and fire it on Alt+K alone.
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true, altKey: true }), false),
    null,
  );
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true }), false),
    null,
  );
  // Ctrl+Meta is likewise not a Mod chord with the Meta quietly dropped.
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true, ctrlKey: true }), false),
    null,
  );
  // The same chords on macOS are ordinary Cmd bindings and still record.
  const mac = bindingFromEvent(
    keyEvent("KeyK", { metaKey: true, altKey: true }),
    true,
  );
  assert.ok(mac);
  assert.equal(formatBindingValue(mac), "Mod+Alt+KeyK");
});

// The sidebar search tooltip and the Settings menu row are the two hints a
// user sees outside the shortcuts tab. Hard-coded, they keep advertising the
// shipped chord after a rebind, and a dead one after a clear.
test("the sidebar hints render the bound chord, not the shipped default", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  for (const literal of ['"⌘K"', '"Ctrl+K"', "<DropdownMenuShortcut>⌘,"]) {
    assert.ok(
      !source.includes(literal),
      `app-sidebar still hard-codes ${literal}`,
    );
  }
  assert.ok(source.includes('useShortcutLabel("searchChats")'));
  assert.ok(source.includes('useShortcutLabel("openSettings")'));
  // Unassigned actions must drop the hint rather than render an empty key cap.
  assert.ok(source.includes("{searchShortcutLabel && ("));
  assert.ok(source.includes("{settingsShortcutLabel && ("));
});

test("a hint label follows the override and disappears when cleared", () => {
  const label = (
    overrides: Parameters<typeof resolveBinding>[0],
    id: "searchChats" | "openSettings",
  ) => {
    const binding = parseBinding(resolveBinding(overrides, id));
    return binding ? formatBindingLabel(binding, false) : null;
  };
  assert.equal(label({}, "searchChats"), "Ctrl+K");
  assert.equal(
    label({ searchChats: { primary: "Mod+Shift+KeyF" } }, "searchChats"),
    "Ctrl+Shift+F",
  );
  assert.equal(label({ searchChats: { primary: null } }, "searchChats"), null);
  assert.equal(label({}, "openSettings"), "Ctrl+,");
});

// A row with no i18n keys renders a raw key path, and one missing from the
// search index cannot be found from the settings search box.
test("every action is translated and indexed for settings search", async () => {
  const en = await readFile(
    new URL("../src/i18n/locales/en.ts", import.meta.url),
    "utf8",
  );
  const at = en.indexOf("    keyboardShortcuts: {");
  assert.notEqual(at, -1);
  const subtree = en.slice(at, en.indexOf("\n    },", at));
  for (const def of SHORTCUT_DEFS) {
    assert.ok(
      subtree.includes(`${def.id}: {`),
      `en.ts is missing settings.keyboardShortcuts.actions.${def.id}`,
    );
    assert.equal(
      def.labelKey,
      `settings.keyboardShortcuts.actions.${def.id}.label`,
    );
    assert.equal(
      def.descriptionKey,
      `settings.keyboardShortcuts.actions.${def.id}.description`,
    );
  }

  const index = await readFile(
    new URL("../src/features/settings/settings-search.ts", import.meta.url),
    "utf8",
  );
  for (const def of SHORTCUT_DEFS) {
    assert.ok(
      index.includes(`"${def.labelKey}"`),
      `${def.id} is missing from the settings search index`,
    );
  }
});

// Every action needs a place to run, or the row is a control that does nothing.
test("every action has a useShortcut call site", async () => {
  const files = [
    "../src/app/routes/__root.tsx",
    "../src/components/app-sidebar.tsx",
    "../src/components/ui/sidebar.tsx",
    "../src/components/assistant-ui/thread.tsx",
    "../src/components/assistant-ui/tool-confirmation-controls.tsx",
    "../src/features/chat/chat-page.tsx",
    "../src/features/chat/shared-composer.tsx",
    "../src/features/chat/mcp-composer-button.tsx",
    "../src/features/chat/components/chat-search-dialog.tsx",
    "../src/features/api-monitor/api-monitor-overlay.tsx",
  ];
  const sources = await Promise.all(
    files.map((file) => readFile(new URL(file, import.meta.url), "utf8")),
  );
  const joined = sources.join("\n");
  for (const def of SHORTCUT_DEFS) {
    // Biome wraps longer calls, so the id can land on its own line.
    const called = new RegExp(`useShortcut\\(\\s*"${def.id}"`).test(joined);
    // The numbered slots register through <Shortcut> elements built from a
    // template id, because a loop of hooks breaks the rules of hooks.
    const slot = /^(goToChat|goToRecentChat)(\d)$/.exec(def.id);
    const rendered =
      slot !== null &&
      joined.includes(`id={\`${slot[1]}\${slot}\` as ShortcutId}`);
    assert.ok(called || rendered, `${def.id} has no useShortcut call site`);
  }
});

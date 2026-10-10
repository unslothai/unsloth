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
  keystrokeMatchesBinding,
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
  shortcutMatchingEvent,
  shortcutOwningBinding,
} from "../src/features/settings/stores/keyboard-shortcuts-store.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";

import { readSrc, readSrcAsync } from "./helpers/kit.ts";

const SRC__ROOT = readSrc("app/routes/__root.tsx");
const APP_SIDEBAR = readSrc("components/app-sidebar.tsx");
const THREAD = readSrc("components/assistant-ui/thread.tsx");
const CHAT_PAGE = readSrc("features/chat/chat-page.tsx");
const KEYBOARD_SHORTCUTS_TAB = readSrc("features/settings/tabs/keyboard-shortcuts-tab.tsx");
const EN = readSrc("i18n/locales/en.ts");

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

// Defaults must be in formatBindingValue order (Mod, Ctrl, Alt, Shift, key) to equal recorded chords.
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

// Off macOS Ctrl is Mod, so a Ctrl-only default would be unreachable and duplicate the Mod row.
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
  // preventDefault on a window keydown cancels the browser's click on the focused button.
  assert.equal(activationBelongsToFocus(enter, deny), true);
  assert.equal(activationBelongsToFocus(enter, null), false);
  assert.equal(
    activationBelongsToFocus(enter, { tagName: "DIV", getAttribute: () => null }),
    false,
  );
  assert.equal(
    activationBelongsToFocus(enter, {
      tagName: "DIV",
      getAttribute: (name: string) => (name === "role" ? "button" : null),
    }),
    true,
  );
  const escape = parseBinding("Escape");
  assert.ok(escape);
  assert.equal(activationBelongsToFocus(escape, deny), false);
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
  assert.ok(
    matchesBinding(
      binding,
      keyEvent("KeyO", { ctrlKey: true, shiftKey: true }),
      false,
    ),
  );
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
  const ctrlBinding = parseBinding("Ctrl+KeyB");
  assert.ok(ctrlBinding);
  assert.ok(
    matchesBinding(ctrlBinding, keyEvent("KeyB", { ctrlKey: true }), true),
  );
});

test("a Ctrl chord from a Mac cannot fire on Windows or Linux", () => {
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
  // WebKit and Chromium report AltGraph for Option, so the AltGr guard must stay off macOS.
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

test("the tab-search chord counts as browser-owned on both platforms", () => {
  assert.ok(isBrowserReservedBinding("Mod+Shift+KeyA", true));
  assert.ok(isBrowserReservedBinding("Mod+Shift+KeyA", false));
});

// The walk ends on a key coming up, so a window losing focus can strand it.
test("the recent walk ends on losing the window, not just on keyup", async () => {
  const at = APP_SIDEBAR.indexOf("const end = () =>");
  assert.ok(at !== -1, "the traversal listener moved");
  const body = APP_SIDEBAR.slice(at, APP_SIDEBAR.indexOf("}, []);", at));
  assert.match(body, /window\.addEventListener\("keyup", onKeyUp\);/);
  assert.match(body, /window\.addEventListener\("blur", end\);/);
  assert.match(body, /window\.removeEventListener\("blur", end\);/);
  assert.match(
    body,
    /if \(event\.ctrlKey \|\| event\.metaKey \|\| event\.altKey \|\| event\.shiftKey\)/,
  );
});

test("the private-window chord is reserved and carries no default", () => {
  for (const mac of [true, false]) {
    assert.ok(isBrowserReservedBinding("Mod+Shift+KeyP", mac));
    assert.ok(isBrowserReservedBinding("Mod+Shift+KeyN", mac));
  }
  const search = SHORTCUT_DEFS.find((d) => d.id === "searchChats");
  assert.ok(search);
  for (const mac of [true, false]) {
    assert.equal(defaultBindingFor(search, "primary", mac), "Mod+KeyK");
    assert.equal(defaultBindingFor(search, "alternate", mac), null);
  }
});

test("the workspace chords land where the guard lets them", async () => {
  assert.match(
    SRC__ROOT,
    /const chatOnlyMeasured = usePlatformStore\(\n\s*\(s\) => s\.isChatOnly\(\) && !s\.capabilitiesUnknown\(\),/,
  );
  assert.match(
    SRC__ROOT,
    /const routeShortcutEnabled = !isAuthFlowRoute && !settingsDialogOpen;/,
  );
  assert.match(
    SRC__ROOT,
    /useShortcut\("switchToTrain", goTo\("\/studio"\), \{\n\s*enabled: routeShortcutEnabled && !chatOnlyMeasured,/,
  );
  // /video checks only auth, so the chord would land on the unsupported-hardware gate.
  assert.match(
    SRC__ROOT,
    /const videoDisabled =\n\s*videoNavHint\(chatOnlyMeasured, chatOnlyReason\) !== undefined;/,
  );
  assert.match(
    SRC__ROOT,
    /useShortcut\("switchToVideo", goTo\("\/video"\), \{\n\s*enabled: routeShortcutEnabled && !videoDisabled,/,
  );
  assert.match(
    APP_SIDEBAR,
    /const videoDisabledHint = videoNavHint\(chatOnlyMeasured, chatOnlyReason\);/,
  );
  assert.match(APP_SIDEBAR, /disabled: videoDisabled,/);

  for (const [id, path] of [
    ["switchToProjects", "/projects"],
    ["switchToHub", "/hub"],
    ["switchToExport", "/export"],
  ] as const) {
    const at = SRC__ROOT.indexOf(`useShortcut("${id}"`);
    assert.ok(at !== -1, `${id} lost its call site`);
    assert.ok(
      !SRC__ROOT.slice(at, SRC__ROOT.indexOf(");", at)).includes("chatOnlyMeasured"),
      `${id} gated on a verdict its route does not answer to`,
    );
    assert.match(SRC__ROOT, new RegExp(`goTo\\("${path.replace("/", "\\/")}"\\)`));
  }
});

// Safari's Develop-menu chords are opt-in, so they are not treated as browser-reserved.
test("an opt-in developer chord is not treated as taken", () => {
  for (const value of ["Mod+Alt+KeyE", "Mod+Alt+KeyR"]) {
    for (const mac of [true, false]) {
      assert.equal(
        isBrowserReservedBinding(value, mac),
        false,
        `${value} warns about a menu most users never turn on`,
      );
    }
  }
  for (const [id, value] of [
    ["archiveChat", "Mod+Alt+KeyE"],
    ["renameChat", "Mod+Alt+KeyR"],
  ] as const) {
    const def = SHORTCUT_DEFS.find((d) => d.id === id);
    assert.ok(def);
    assert.equal(defaultBindingFor(def, "primary", true), value);
  }
  assert.ok(isBrowserReservedBinding("Mod+Alt+KeyU", true));
});

test("the browsers' own run on macOS is reserved there and only there", () => {
  // Chrome owns the ⌥⌘ run on macOS; off macOS these read as Ctrl+Alt, which none claim.
  const macOwned = [
    "Mod+Alt+KeyU",
    "Mod+Alt+KeyP",
    "Mod+Alt+KeyI",
    "Mod+Alt+KeyJ",
    "Mod+Alt+KeyB",
    "Mod+Alt+KeyN",
    "Mod+Alt+KeyF",
    "Mod+Alt+KeyC",
    "Mod+Alt+KeyK",
    "Mod+Alt+ArrowLeft",
    "Mod+Alt+ArrowRight",
    "Mod+Alt+ArrowUp",
    "Mod+Alt+ArrowDown",
  ];
  for (const value of macOwned) {
    assert.ok(isBrowserReservedBinding(value, true), `${value} unflagged`);
    assert.equal(
      isBrowserReservedBinding(value, false),
      false,
      `${value} warns off macOS for nothing`,
    );
  }
  for (const value of [
    "Mod+Alt+KeyE",
    "Mod+Alt+KeyO",
    "Mod+Alt+KeyS",
    "Mod+Alt+KeyA",
    "Mod+Alt+KeyR",
    "Mod+Alt+Digit1",
  ]) {
    assert.equal(isBrowserReservedBinding(value, true), false, value);
  }
});

test("view source, the element picker and Page Setup carry no Unsloth action", () => {
  for (const [id, mac, other] of [
    ["toggleApiMonitor", "Ctrl+Shift+KeyU", "Mod+Alt+Shift+KeyM"],
    ["copySessionId", "Ctrl+Shift+KeyC", "Mod+Alt+KeyC"],
    ["togglePinChat", "Ctrl+Shift+KeyP", "Mod+Alt+KeyP"],
  ] as const) {
    const def = SHORTCUT_DEFS.find((d) => d.id === id);
    assert.ok(def);
    assert.equal(defaultBindingFor(def, "primary", true), mac);
    assert.equal(defaultBindingFor(def, "primary", false), other);
    assert.equal(isBrowserReservedBinding(mac, true), false);
    assert.equal(isBrowserReservedBinding(other, false), false);
  }
  for (const [id, value] of [
    ["markChatUnread", "Mod+Alt+KeyU"],
    ["clearAllUnreads", "Mod+Alt+Shift+KeyU"],
  ] as const) {
    const def = SHORTCUT_DEFS.find((d) => d.id === id);
    assert.ok(def);
    assert.equal(defaultBindingFor(def, "primary", false), value);
  }
});

// GTK and IBus bind Ctrl+Shift+U to hex entry, so off macOS it belongs to text composition.
test("no default sits on Linux's own text-composition prefix", () => {
  for (const def of SHORTCUT_DEFS) {
    for (const slot of SHORTCUT_SLOTS) {
      assert.notEqual(
        defaultBindingFor(def, slot, false),
        "Mod+Shift+KeyU",
        `${def.id}.${slot} ships GTK's hex-entry prefix off macOS`,
      );
    }
  }
  const unread = SHORTCUT_DEFS.find((d) => d.id === "markChatUnread");
  assert.ok(unread);
  assert.equal(defaultBindingFor(unread, "primary", true), "Mod+Shift+KeyU");
});

test("no default takes a chord the browser owns without a reason", () => {
  // Exceptions are spec chords kept for the desktop build, where they work.
  const deliberate = new Set([
    "Mod+KeyN",
    "Mod+Shift+KeyN",
    "Mod+Tab",
    "Mod+Shift+Tab",
    "Ctrl+Tab",
    "Ctrl+Shift+Tab",
    // Safari and Chrome own the bracket pair on macOS only.
    "Mod+Shift+BracketLeft",
    "Mod+Shift+BracketRight",
    // Find in page replaces the browser's own find, so it takes that chord on the web too.
    "Mod+KeyF",
    "Mod+KeyP",
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
  // Being on the list must MEAN the chord is flagged, or a dropped reserved value goes unnoticed.
  for (const value of deliberate) {
    assert.ok(
      isBrowserReservedBinding(value, true) ||
        isBrowserReservedBinding(value, false),
      `${value} is listed as a deliberate exception but nothing flags it`,
    );
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
  // Bare Enter only for the two actions registered solely while an approval prompt is shown.
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

test("bare Escape is the recorder's own exit, so only a prompt-gated row takes it", async () => {
  const bare = {
    code: "Escape",
    mod: false,
    ctrl: false,
    shift: false,
    alt: false,
  };
  assert.equal(isAcceptableBinding(bare), false);
  assert.ok(isAcceptableBinding(bare, true));
  assert.ok(isAcceptableBinding({ ...bare, shift: true }));

  // The recorder swallows every keydown, so bare Escape has to stay its way out.
  assert.match(
    KEYBOARD_SHORTCUTS_TAB,
    /event\.code === "Escape" &&\n(?:\s*![a-zA-Z.]+ &&\n)+\s*!def\?\.allowBareKey\n\s*\) \{\n\s*setRecording\(null\);/,
  );

  for (const def of SHORTCUT_DEFS) {
    for (const slot of SHORTCUT_SLOTS) {
      for (const mac of [true, false]) {
        if (defaultBindingFor(def, slot, mac) !== "Escape") continue;
        assert.ok(
          def.allowBareKey,
          `${def.id}.${slot} ships bare Escape without allowBareKey`,
        );
      }
    }
  }
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
  // newChat, not nextChat: resolveBinding reads the host platform and ⌥⌘→ is macOS only.
  assert.equal(
    resolveBinding({ newChat: { primary: null } }, "newChat", "alternate"),
    "Mod+KeyN",
  );
});

test("both slots resolve together", () => {
  // Platform-independent chords only: resolveBindings reads the host's.
  assert.deepEqual(resolveBindings({}, "newChat"), {
    primary: "Mod+Shift+KeyO",
    alternate: "Mod+KeyN",
  });
  assert.deepEqual(resolveBindings({}, "archiveChat"), {
    primary: "Mod+Alt+KeyE",
    alternate: null,
  });
});

test("the chat walk ships no arrow alternate on either platform", () => {
  // ⌥⌘→ is Chrome's next tab and Ctrl+Alt+→ switches desktops on GNOME/KDE.
  for (const id of ["nextChat", "previousChat"] as const) {
    const def = SHORTCUT_DEFS.find((d) => d.id === id);
    assert.ok(def);
    for (const mac of [true, false]) {
      assert.equal(defaultBindingFor(def, "alternate", mac), null);
    }
    assert.match(
      String(defaultBindingFor(def, "primary", true)),
      /^Mod\+Shift\+Bracket(Left|Right)$/,
    );
  }
  assert.ok(isBrowserReservedBinding("Mod+Alt+ArrowRight", true));
  assert.ok(isBrowserReservedBinding("Mod+Alt+ArrowLeft", true));
});

test("a slot reports whether it carries an edit", () => {
  assert.equal(isSlotOverridden({}, "nextChat", "primary"), false);
  const overrides = { nextChat: { alternate: null } };
  assert.equal(isSlotOverridden(overrides, "nextChat", "primary"), false);
  assert.ok(isSlotOverridden(overrides, "nextChat", "alternate"));
});

// Older builds stored `id -> string | null`; read as-is it would revert every customization.
test("the pre-alternate override shape migrates to the primary slot", () => {
  const legacy = JSON.stringify({
    toggleSidebar: "Mod+Alt+KeyB",
    searchChats: null,
    newChat: "Mod+Alt+KeyJ",
    someRemovedAction: "Mod+KeyZ",
  });
  const migrated = migrateStoredOverrides(JSON.parse(legacy));
  assert.deepEqual(migrated, {
    toggleSidebar: { primary: "Mod+Alt+KeyB" },
    searchChats: { primary: null, alternate: null },
    newChat: { primary: "Mod+Alt+KeyJ" },
  });
  assert.equal(resolveBinding(migrated, "toggleSidebar"), "Mod+Alt+KeyB");
  assert.equal(resolveBinding(migrated, "searchChats"), null);
  assert.equal(resolveBinding(migrated, "newChat", "alternate"), "Mod+KeyN");
});

// A stored null once meant the action was off; clearing only the primary would re-enable it.
test("an action cleared before alternates existed stays cleared", () => {
  const migrated = migrateStoredOverrides(JSON.parse('{"newChat":null}'));
  assert.deepEqual(migrated, { newChat: { primary: null, alternate: null } });
  assert.deepEqual(resolveBindings(migrated, "newChat"), {
    primary: null,
    alternate: null,
  });
  const shipped = SHORTCUT_DEFS.find((d) => d.id === "newChat");
  assert.ok(shipped);
  assert.equal(defaultBindingFor(shipped, "alternate", true), "Mod+KeyN");
  for (const slot of SHORTCUT_SLOTS) {
    assert.ok(isSlotOverridden(migrated, "newChat", slot), slot);
  }
});

test("the current override shape round-trips unchanged", () => {
  const stored = { nextChat: { primary: "Mod+KeyG", alternate: null } };
  assert.deepEqual(migrateStoredOverrides(stored), stored);
});

test("defaults ship without conflicts on either platform", () => {
  // findConflicts uses this process's platform, so check the other: ⌘1-9 and ⌃1-9 collapse off macOS.
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
    archiveChat: { alternate: "Mod+Alt+KeyE" },
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
  // Derived, not hard-coded: the list is ordered for the UI and may be reordered.
  const overrides = { toggleSidebar: { primary: "Mod+KeyK" } };
  const owner = shortcutOwningBinding(overrides, "Mod+KeyK");
  assert.ok(owner);
  const claimants = ["searchChats", "toggleSidebar"];
  assert.deepEqual(
    owner,
    claimants.sort((a, b) => registryIndex(a) - registryIndex(b))[0],
  );
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
  assert.equal(
    shortcutOwningBinding(
      { searchChats: { primary: null } },
      "Mod+KeyK",
    ),
    null,
  );
});

// A chord bound to something the browser eats can make the Shortcuts tab hard to reach.
test("Reset all local preferences clears the rebound chords", async () => {
  const source = await readSrcAsync("features/settings/tabs/general-tab.tsx");
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

test("every locale overlay carries the shortcut strings", async () => {
  const locales = [
    "ar", "de", "es", "fr", "hi", "it", "ja", "ko", "pt-br", "ru", "sv",
    "zh-CN",
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
      const row = new RegExp(
        `\\n        ${def.id}: \\{\\n          label: "[^"]+",\\n          description: "[^"]+",\\n        \\},`,
      );
      assert.match(
        subtree,
        row,
        `${locale} is missing a label or description for settings.keyboardShortcuts.actions.${def.id}`,
      );
    }
  }
});

test("a cleared alternate keeps its row, so it can be restored", async () => {
  // Clearing a slot stores null, so keying off the resolved value would hide its restore control.
  assert.match(
    KEYBOARD_SHORTCUTS_TAB,
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

  // Compare drops the header pickers, so the header chord would toggle state nothing renders.
  assert.match(
    chatPage,
    /const headerPickersShown = active && view\.mode !== "compare";/,
  );
  assert.match(chatPage, /enabled: headerPickersShown[,\s]*\}/);
  // Both panes mount the last message; the first registered listener would win.
  assert.match(thread, /enabled: chatActive && !inComparePane && !forkDisabled/);
  assert.match(
    toolCard,
    /soleRequest &&\n\s*!selectionActive &&\n\s*showControls &&/,
  );
});

// These surfaces' open flags outlive what they open, so leaving without closing reopens them.
test("a chord's surface does not come back open on the next visit", async () => {
  const read = async (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");
  const chatPage = await read("../src/features/chat/chat-page.tsx");
  const mcp = await read("../src/features/chat/mcp-composer-button.tsx");

  assert.match(
    chatPage,
    /const projectSwitcherShown = headerPickersShown && Boolean\(currentProjectId\);/,
  );
  assert.match(chatPage, /useShortcut\(\n\s*"openProjectPicker",[\s\S]*?enabled: projectSwitcherShown/);
  assert.match(
    chatPage,
    /if \(!projectSwitcherShown && projectPickerOpen\) \{\n\s*setProjectPickerOpen\(false\);/,
  );
  assert.match(
    chatPage,
    /\{view\.mode !== "compare" && currentProjectId && \(/,
    "the reset no longer matches what the switcher renders by",
  );

  // The MCP dialog's flag lives in a store, so an unmount alone leaves it armed.
  assert.match(mcp, /if \(!chatActive && open\) setOpen\(false\);/);
  assert.match(
    mcp,
    /useEffect\(\(\) => \{\n\s*return \(\) => useMcpServersDialogStore\.getState\(\)\.setOpen\(false\);\n\s*\}, \[\]\);/,
  );
});

// The sidebar clears selection on Escape outside the registry, so modified Escape must not reach it.
test("only bare Escape drops the selection", async () => {
  const at = APP_SIDEBAR.indexOf('if (event.key !== "Escape"');
  assert.notEqual(at, -1, "the selection listener moved");
  const block = APP_SIDEBAR.slice(at, APP_SIDEBAR.indexOf("clearSelection();", at));
  for (const modifier of ["metaKey", "ctrlKey", "altKey", "shiftKey"]) {
    assert.ok(
      block.includes(`event.${modifier}`),
      `${modifier} still reaches the selection listener`,
    );
  }
  assert.ok(block.includes("event.defaultPrevented"));
  const clearAll = SHORTCUT_DEFS.find((d) => d.id === "clearAllUnreads");
  assert.ok(clearAll);
  assert.equal(defaultBindingFor(clearAll, "primary", true), "Shift+Escape");
});

// The mobile drawer is owned by SidebarProvider outside the Outlet, so it survives navigation.
test("the workspace chords do not leave the mobile drawer over the workspace", async () => {
  const read = async (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");
  const root = await read("../src/app/routes/__root.tsx");
  const sidebar = await read("../src/components/app-sidebar.tsx");

  assert.match(root, /<SidebarProvider/);
  assert.match(root, /useShortcut\("switchToProjects", goTo\("\/projects"\)/);

  // href, not pathname: a new chat from /chat only moves the search.
  assert.match(
    sidebar,
    /useEffect\(\(\) => \{\n\s*if \(isMobile\) setOpenMobile\(false\);\n\s*\}, \[href, isMobile, setOpenMobile\]\);/,
  );
  assert.match(sidebar, /href: s\.location\.href,/);

  assert.match(
    sidebar,
    /const closeMobileIfOpen = \(\) => \{\n\s*if \(isMobile\) setOpenMobile\(false\);\n\s*\};/,
  );
});

// ActionBarRoot returns null when hidden, so a chord registered in an action bar is often gone.
test("the fork chord is registered where it mounts, not from an action bar", async () => {

  const start = THREAD.indexOf("const ForkChatShortcut: FC = () => {");
  assert.notEqual(start, -1, "the fork registration moved");
  const block = THREAD.slice(start, THREAD.indexOf("\n};", start));
  assert.match(block, /useShortcut\(\n\s*"forkChat",/);
  assert.match(block, /return null;/);

  const buttonAt = THREAD.indexOf("const ForkMessageButton: FC = () => {");
  const button = THREAD.slice(buttonAt, THREAD.indexOf("\n};", buttonAt));
  assert.ok(
    !button.includes("useShortcut"),
    "an autohidden bar cannot hold the registration",
  );

  // Chord, button and sidebar menu share one in-flight flag module, or two forks could post.
  const FORK_STORE = await readSrcAsync("features/chat/utils/fork-in-flight.ts");
  assert.match(
    FORK_STORE,
    /export const useForkInFlight = create<\{\n\s*forking: boolean;/,
  );
  assert.ok(!THREAD.includes("const useForkInFlight = create<"));
  assert.match(THREAD, /const pending = useForkInFlight\(\(s\) => s\.forking\);/);
  assert.match(
    THREAD,
    /if \(useForkInFlight\.getState\(\)\.forking\) return;/,
  );
  assert.ok(
    !THREAD.includes("const [pending, setPending] = useState(false);"),
    "the per-instance flag is what let two forks run",
  );

  const mounts = THREAD.match(
    /<MessagePrimitive\.If last=\{true\}>\n\s*<ForkChatShortcut \/>\n\s*<\/MessagePrimitive\.If>/g,
  );
  assert.equal(mounts?.length, 2);
  for (const role of ["const AssistantMessage", "const UserMessage: FC"]) {
    const at = THREAD.indexOf(role);
    assert.notEqual(at, -1, `${role} moved`);
    assert.ok(
      THREAD.slice(at, THREAD.indexOf("\n};", at)).includes("<ForkChatShortcut />"),
      `${role} does not mount the fork chord`,
    );
  }
});

// The tour's opener pins the picker and an effect unpins it when no tour runs.
test("the model picker chord opens without the tour's pin", async () => {

  assert.match(
    CHAT_PAGE,
    /const openModelSelector = useCallback\(\(\) => \{\n\s*setModelSelectorLocked\(true\);/,
  );
  assert.match(
    CHAT_PAGE,
    /if \(tour\.open\) return;\n\s*if \(!modelSelectorLocked\) return;[\s\S]{0,200}?setModelSelectorOpen\(false\);/,
  );

  assert.match(
    CHAT_PAGE,
    /const toggleModelSelector = useCallback\(\(\) => \{\n(?:\s*\/\/[^\n]*\n)*\s*if \(modelSelectorLocked\) return;\n\s*setModelSelectorOpen\(\(open\) => !open\);/,
  );
  assert.match(CHAT_PAGE, /useShortcut\(\n\s*"openModelPicker",[\s\S]*?toggleModelSelector\(\);/);

  // Three mentions, all the tour's: declaration, step builder argument, memo dependency.
  assert.equal((CHAT_PAGE.match(/openModelSelector/g) ?? []).length, 3);
});

// The rename pill needs its row on screen, which a chord cannot guarantee.
test("the rename chord does not land in a surface only a row can show", async () => {
  assert.match(
    APP_SIDEBAR,
    /useShortcut\("renameChat", \(\) => \{[\s\S]*?withActiveChat\(\(item\) => openRenameChat\(item, false\)\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /onSelect=\{\(\) => openRenameChat\(item, true, list\?\.scope\)\}/,
  );
  assert.match(
    APP_SIDEBAR,
    /function openRenameChat\(item: SidebarItem, inline = true, rowScope\?: string\)/,
  );

  assert.match(
    APP_SIDEBAR,
    /const isRenamingThis =\n\s*renamingTarget\?\.kind === "chat" &&\n\s*renamingTarget\.inline &&/,
  );
  assert.match(
    APP_SIDEBAR,
    /\(renamingTarget\.kind !== "chat" \|\| !renamingTarget\.inline\)/,
  );
  assert.ok(APP_SIDEBAR.includes('t("shell.dialog.renameChat.title")'));
});

// The sidebar answers bare Escape without consuming it, so one press could also deny a tool call.
test("one Escape does not both drop a selection and deny a tool call", async () => {
  const read = async (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");
  const sidebar = await read("../src/components/app-sidebar.tsx");
  const toolCard = await read(
    "../src/components/assistant-ui/tool-confirmation-controls.tsx",
  );
  const store = await read(
    "../src/features/chat/stores/chat-navigation-store.ts",
  );

  // The tool card stands down: a dropped selection is recoverable, a denied call is not.
  assert.match(toolCard, /const selectionActive = useChatNavigationStore\(/);
  assert.match(toolCard, /!selectionActive &&/);
  for (const id of ["approveToolRequest", "declineToolRequest"]) {
    const at = toolCard.indexOf(`"${id}",`);
    assert.ok(at !== -1, `${id} lost its call site`);
    assert.match(
      toolCard.slice(at, toolCard.indexOf("\n  );", at)),
      /enabled: keyboardReady/,
    );
  }

  // Cleared on unmount so an unmounted sidebar cannot leave the card mute.
  assert.match(
    sidebar,
    /const selectionActive = selectionCount > 0 \|\| projectSelectionCount > 0;/,
  );
  assert.match(
    sidebar,
    /setSelectionActive\(selectionActive\);\n\s*return \(\) => setSelectionActive\(false\);/,
  );
  assert.match(store, /selectionActive: boolean;/);
  assert.match(store, /selectionActive: false,/);

  // Dictation reads defaultPrevented before cancelling, so the sidebar must not consume Escape.
  const at = sidebar.indexOf("Escape is the way out of a selection");
  const body = sidebar.slice(at, sidebar.indexOf("}, [selectionActive", at));
  assert.ok(!body.includes("preventDefault"), "the listener consumes Escape");
  assert.ok(!body.includes(", true)"), "the listener moved to capture");
});

test("the composer chords outlive the recording bar", async () => {
  // Dictation swaps ComposerRightControls out, so a chord registered there could never stop it.
  const controls = THREAD.indexOf("const ComposerRightControls:");
  assert.ok(controls !== -1, "the controls component moved");
  for (const id of ["startDictation", "sendMessage"]) {
    const at = THREAD.indexOf(`useShortcut(\n    "${id}"`);
    const inline = THREAD.indexOf(`useShortcut("${id}"`);
    const found = at === -1 ? inline : at;
    assert.ok(found !== -1, `${id} lost its call site`);
    assert.ok(found < controls, `${id} registers inside the recording swap`);
  }
  // Send goes through the form, which runs parking, queueing and refusing the runtime skips.
  assert.match(THREAD, /formRef\.current\?\.requestSubmit\(\);/);
  assert.match(
    THREAD,
    /if \(isDictating\) \{\n\s*if \(!dictationBlocked\) sendAfterDictation\(\);/,
  );
});

test("a collapsed sidebar section is not published for the chords", async () => {
  assert.match(
    APP_SIDEBAR,
    /chatListsOnScreen && pinnedOpen\n\s*\? pinnedRows\.flatMap\(/,
  );
  assert.match(
    APP_SIDEBAR,
    /visibleCustomSections\.flatMap\(\(section\) =>\n\s*collapsedSectionIds\.has\(section\.id\)\n\s*\? \[\]/,
  );
  assert.match(
    APP_SIDEBAR,
    /chatListsOnScreen && chatOpen \? sortedRecentChatItems/,
  );
  assert.match(
    APP_SIDEBAR,
    /folderChatItems\(true, \[row\.project\]\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /folderChatItems\(projectsSectionRendered && projectsOpen, visibleProjectRecords\)/,
  );
  // In one list every project chat is a Recents row, so a folder must not list it again.
  assert.match(APP_SIDEBAR, /if \(!chatListsOnScreen \|\| organizeBy !== "project" \|\| !open\)/);
  assert.match(APP_SIDEBAR, /pinnedItems: upToPinnedChatItems,/);
  assert.match(APP_SIDEBAR, /recentItems: visibleRecentItems,/);
});

test("the MCP chord does not live behind the MCP pill", async () => {
  const button = await readSrcAsync("features/chat/mcp-composer-button.tsx");
  // MCP ships off and the pill only renders once on, so the chord mounts with the chat instead.
  const pill = button.indexOf("export function McpComposerButton");
  const mount = button.indexOf("export function McpServersDialogMount");
  assert.ok(pill !== -1 && mount > pill, "the mount moved");
  assert.ok(
    button.indexOf('useShortcut("openMcpServers"') > mount,
    "the chord is back inside the pill",
  );
  // The flag lives in a store, so an open dialog has to be closed on the way out.
  assert.match(CHAT_PAGE, /\n\s*<McpServersDialogMount \/>/);
  assert.match(button, /if \(!chatActive && open\) setOpen\(false\);/);
  assert.match(button, /open=\{chatActive && open\}/);
});

test("the copy chords keep their gesture across the read", async () => {
  const clipboard = await readSrcAsync("lib/copy-to-clipboard.ts");
  // A strict engine drops the gesture across the storage await, so pass a promised payload.
  assert.match(clipboard, /"text\/plain": payload\.then\(/);
  for (const fn of ["copyChatItemAsMarkdown", "copyChatSessionId"]) {
    const at = APP_SIDEBAR.indexOf(`async function ${fn}(`);
    assert.ok(at !== -1, `${fn} moved`);
    const body = APP_SIDEBAR.slice(at, APP_SIDEBAR.indexOf("\n  }", at));
    assert.ok(
      body.includes("copyToClipboardFrom(async () =>"),
      `${fn} awaits its read before starting the write`,
    );
  }
});

test("the project picker chord is described by what it does", async () => {
  assert.match(CHAT_PAGE, /onSelectProject=\{openProjectLanding\}/);
  const at = EN.indexOf("openProjectPicker: {");
  const entry = EN.slice(at, EN.indexOf("},", at));
  assert.ok(!/move/i.test(entry), "no move-to-project promise");
  assert.ok(entry.includes("project"));
});

test("the new-chat chords stay out of the auth flow", async () => {
  // /login has no shell and requireAuth bounces /chat straight back.
  for (const id of ["newChat", "newTemporaryChat", "newStandaloneChat"]) {
    // The id, not the whole call: two of the three wrap onto their own line.
    const at = SRC__ROOT.indexOf(`"${id}"`);
    assert.ok(at !== -1, `${id} is registered`);
    const call = SRC__ROOT.slice(at, SRC__ROOT.indexOf("\n  );", at) + 5);
    assert.match(call, /enabled: routeShortcutEnabled/, `${id} is gated`);
  }
});

test("switching back to Chat lands on the view the user left", async () => {
  const at = SRC__ROOT.indexOf('"switchToChat"');
  assert.ok(at !== -1, "switchToChat is registered");
  const call = SRC__ROOT.slice(at, SRC__ROOT.indexOf("\n  );", at) + 5);
  assert.match(call, /navigate\(\{ to: "\/chat", search: chatSearch \}\)/);
  assert.match(call, /enabled: routeShortcutEnabled/);
  assert.match(SRC__ROOT, /useShortcut\("switchToImages", goTo\("\/images"\)/);
  // location.search is the raw URL's, not the matched route's.
  assert.match(SRC__ROOT, /useState<ChatSearch>\(\{\}\)/);
});

test("opening a chat by chord drops the selection, as clicking a row does", async () => {
  // Archive, pin and mark-unread prefer the selection, so a stale one hits off-screen rows.
  const at = APP_SIDEBAR.indexOf("function openChatItem(");
  const body = APP_SIDEBAR.slice(at, APP_SIDEBAR.indexOf("\n  }", at));
  assert.ok(body.includes("clearSelection()"), "the shared opener clears it");
  assert.ok(
    !APP_SIDEBAR.includes("clearSelection();\n                openChatItem(item);"),
    "the row no longer clears it separately",
  );
});

test("effort chords only run for a model whose effort is read", async () => {
  const at = CHAT_PAGE.indexOf("const shiftReasoningEffort");
  const body = CHAT_PAGE.slice(at, CHAT_PAGE.indexOf("useShortcut(\"cycleReasoningEffort\"", at));
  // enable_thinking models still list levels, but the request drops the effort.
  assert.match(body, /state\.reasoningStyle === "reasoning_effort"/);
  assert.match(body, /state\.reasoningStyle === "enable_thinking_effort"/);
  assert.match(body, /!state\.supportsReasoning \|\| !isEffort/);
});

test("New chat inherits the project on screen, inferred or not", async () => {
  assert.match(SRC__ROOT, /isChatRoute \? chatRuntime\.activeProjectId : null/);
  assert.match(CHAT_PAGE, /const projectId = thread\?\.projectId \?\? null;/);
  assert.match(CHAT_PAGE, /const projectId = threads\[0\]\?\.projectId \?\? null;/);
  assert.match(CHAT_PAGE, /setCurrentProjectId\(projectId\);\n\s*useChatRuntimeStore\.getState\(\)\.setActiveProjectId\(projectId\);/);
  assert.match(CHAT_PAGE, /runtime\.setActiveProjectId\(currentProjectId\);/);
  // Off Chat the page is hidden rather than unmounted, so its project stays excluded.
  assert.match(SRC__ROOT, /isChatRoute \? chatRuntime\.activeProjectId : null/);
  assert.match(
    SRC__ROOT,
    /useShortcut\("newStandaloneChat", \(\) => startNewChat\(\{ standalone: true \}\)/,
  );
  assert.match(SRC__ROOT, /const projectId = options\?\.standalone \? null : openProjectId;/);
});

test("every settings tab survives a reload", () => {
  // The persisted-tab check reads this same list.
  assert.ok(SETTINGS_TABS.includes("keyboard-shortcuts"));
  assert.equal(new Set(SETTINGS_TABS).size, SETTINGS_TABS.length);
});

test("a Super chord off macOS records nothing rather than a different chord", () => {
  // matchesBinding rejects a non-mac event with Meta, so Super+Alt+K must not record as Alt+K.
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true, altKey: true }), false),
    null,
  );
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true }), false),
    null,
  );
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true, ctrlKey: true }), false),
    null,
  );
  const mac = bindingFromEvent(
    keyEvent("KeyK", { metaKey: true, altKey: true }),
    true,
  );
  assert.ok(mac);
  assert.equal(formatBindingValue(mac), "Mod+Alt+KeyK");
});

// Hints outside the shortcuts tab must follow rebinds and clears.
test("the sidebar hints render the bound chord, not the shipped default", async () => {
  for (const literal of ['"⌘K"', '"Ctrl+K"', "<DropdownMenuShortcut>⌘,"]) {
    assert.ok(
      !APP_SIDEBAR.includes(literal),
      `app-sidebar still hard-codes ${literal}`,
    );
  }
  assert.ok(APP_SIDEBAR.includes('useShortcutLabel("searchChats")'));
  assert.ok(APP_SIDEBAR.includes('useShortcutLabel("openSettings")'));
  assert.ok(APP_SIDEBAR.includes("{searchShortcutLabel && ("));
  assert.ok(APP_SIDEBAR.includes("{settingsShortcutLabel && ("));
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

test("every action is translated and indexed for settings search", async () => {
  const at = EN.indexOf("    keyboardShortcuts: {");
  assert.notEqual(at, -1);
  const subtree = EN.slice(at, EN.indexOf("\n    },", at));
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

  const index = await readSrcAsync("features/settings/settings-search.ts");
  for (const def of SHORTCUT_DEFS) {
    assert.ok(
      index.includes(`"${def.labelKey}"`),
      `${def.id} is missing from the settings search index`,
    );
  }
});

test("every action has a useShortcut call site", async () => {
  const files = [
    "../src/app/routes/__root.tsx",
    "../src/components/app-sidebar.tsx",
    "../src/components/command-palette.tsx",
    "../src/components/ui/sidebar.tsx",
    "../src/components/assistant-ui/thread.tsx",
    "../src/components/assistant-ui/tool-confirmation-controls.tsx",
    "../src/features/chat/chat-page.tsx",
    "../src/features/chat/shared-composer.tsx",
    "../src/features/chat/mcp-composer-button.tsx",
    "../src/features/chat/components/chat-search-dialog.tsx",
    "../src/features/api-monitor/api-monitor-overlay.tsx",
    "../src/features/find-in-page/components/find-in-page.tsx",
    "../src/features/browser/browser-panel.tsx",
    "../src/features/browser/browser-toggle.tsx",
  ];
  const sources = await Promise.all(
    files.map((file) => readFile(new URL(file, import.meta.url), "utf8")),
  );
  const joined = sources.join("\n");
  for (const def of SHORTCUT_DEFS) {
    // Biome wraps longer calls, so the id can land on its own line.
    const called = new RegExp(`useShortcut\\(\\s*"${def.id}"`).test(joined);
    // Numbered slots register through <Shortcut> elements: a loop of hooks breaks the rules of hooks.
    const slot = /^(goToChat|goToRecentChat)(\d)$/.exec(def.id);
    const rendered =
      slot !== null &&
      joined.includes(`id={\`${slot[1]}\${slot}\` as ShortcutId}`);
    assert.ok(called || rendered, `${def.id} has no useShortcut call site`);
  }
});

// The call-site test accepts numbered slots on the template alone, so tie the loop to the registry.
test("the Recents loop registers every numbered slot the registry declares", () => {
  const slots = /const RECENT_SLOT_NUMBERS = \[([\d, ]+)\] as const;/.exec(APP_SIDEBAR);
  assert.ok(slots, "RECENT_SLOT_NUMBERS is no longer a literal list");
  assert.deepEqual(
    slots[1].split(",").map((n) => `goToRecentChat${n.trim()}`),
    SHORTCUT_DEFS.map((def) => def.id).filter((id) =>
      id.startsWith("goToRecentChat"),
    ),
  );
});

// Held chords auto-repeat past the OS delay; toggles and archives must not.
test("auto-repeat only reaches the actions that walk a list", async () => {
  const read = async (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");
  const hook = await read("../src/features/settings/hooks/use-shortcut.ts");
  const at = hook.indexOf("event.preventDefault();");
  assert.ok(at !== -1, "the hook stopped consuming the chord");
  assert.match(
    hook.slice(at),
    /event\.preventDefault\(\);\n(?:\s*\/\/[^\n]*\n)*\s*if \(event\.repeat && !repeats\) return;/,
  );
  assert.match(hook, /repeats = false,\n(?:\s*\w+,\n)*\s*\} = options;/);
  assert.match(
    hook,
    /\[bindings, enabled, skipInTextFields, textFieldException, repeats\]/,
  );

  const sidebar = await read("../src/components/app-sidebar.tsx");
  const walkers = [
    "nextChat",
    "previousChat",
    "nextRecentlyViewedChat",
    "previousRecentlyViewedChat",
  ];
  for (const id of walkers) {
    const call = sidebar.indexOf(`useShortcut("${id}"`);
    assert.ok(call !== -1, `${id} lost its call site`);
    assert.match(
      sidebar.slice(call, sidebar.indexOf(");", call)),
      /repeats: true/,
      `${id} should step while held`,
    );
  }
  const optedIn = sidebar.match(/repeats: true/g) ?? [];
  assert.equal(optedIn.length, walkers.length);
});

test("the published chat lists stop where the sidebar stops", async () => {
  // Plus the icon rail, which hides the groups in CSS rather than dropping them.
  assert.match(
    APP_SIDEBAR,
    /const chatListsOnScreen =\n\s*!isStudioRoute &&\n\s*!showTrainingRecents &&\n\s*\(isMobile \|\| sidebarState !== "collapsed"\);/,
  );
  for (const group of [
    /if \(!chatListsOnScreen \|\| organizeBy !== "project" \|\| !open\) return \[\];/,
    /chatListsOnScreen && pinnedOpen\n\s*\? pinnedRows\.flatMap\([\s\S]*?: \[\],/,
    /const customSectionChatItems = useMemo\(\(\) => \{\n\s*const bySection = new Map<string, SidebarItem\[\]>\(\);\n\s*if \(!chatListsOnScreen\) return bySection;/,
    /\(chatListsOnScreen && chatOpen \? sortedRecentChatItems : \[\]\)/,
  ]) {
    assert.match(APP_SIDEBAR, group);
  }
  const selectAll = APP_SIDEBAR.indexOf("const selectAllChats = useCallback(");
  assert.ok(selectAll !== -1, "selectAllChats moved");
  assert.match(
    APP_SIDEBAR.slice(selectAll, APP_SIDEBAR.indexOf("\n  }, [", selectAll)),
    /const ids = renderedChatItems\.map\(\(item\) => item\.id\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /const renderedChatItems = useMemo\(\n\s*\(\) => \[\.\.\.upToPinnedChatItems, \.\.\.belowPinnedChatItems, \.\.\.visibleRecentItems\],/,
  );
  const rendered = APP_SIDEBAR.slice(APP_SIDEBAR.indexOf("return (", selectAll));
  for (const name of [
    "pinnedSectionChatItems",
    "customSectionChatItems",
    "visibleRecentItems",
    "renderedChatItems",
    "sectionProjectChatItems",
  ]) {
    assert.ok(!rendered.includes(name), `${name} is read by the JSX too`);
  }
});

// Bulk chords prefer the selection, so one carried off screen would be invisible and live.
test("a selection does not outlive the rows it was made on", async () => {
  assert.match(
    APP_SIDEBAR,
    /if \(!chatRowsOnScreen\) \{\n\s*clearSelection\(\);\n\s*return;\n\s*\}/,
  );
  // Stricter: a closed mobile sheet unmounts rows while the lists still exist.
  assert.match(
    APP_SIDEBAR,
    /const chatRowsOnScreen = chatListsOnScreen && \(!isMobile \|\| openMobile\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /const selectAllChats = useCallback\(\(\) => \{\n\s*if \(!chatRowsOnScreen\) return;/,
  );
  assert.match(
    APP_SIDEBAR,
    /\(chatListsOnScreen && chatOpen \? sortedRecentChatItems : \[\]\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /for \(const id of prev\) \{\n\s*if \(renderedChatIds\.has\(id\)\) kept\.add\(id\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /return kept\.size === prev\.size \? prev : kept;/,
  );
  assert.match(
    APP_SIDEBAR,
    /\}, \[chatRowsOnScreen, clearSelection, renderedChatIds, renderedProjectIds\]\);/,
  );
  // selectionActive counts folder rows too, so stale ones would keep the tool card's Escape aside.
  assert.match(
    APP_SIDEBAR,
    /for \(const id of prev\) \{\n\s*if \(renderedProjectIds\.has\(id\)\) kept\.add\(id\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /if \(projectAnchor && !renderedProjectIds\.has\(projectAnchor\)\) \{\n\s*projectAnchorRef\.current = null;/,
  );
  assert.match(
    APP_SIDEBAR,
    /const renderedProjectIds = useMemo\(\(\) => \{\n\s*if \(!chatListsOnScreen\) return new Set<string>\(\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /if \(pinnedOpen\) \{\n\s*for \(const project of pinnedProjectRecords\) ids\.add\(project\.id\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /if \(projectsOpen && projectsSectionConfigured\) \{\n\s*for \(const project of visibleProjectRecords\) ids\.add\(project\.id\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /const selectionActive =\n?\s*selectionCount > 0 \|\| projectSelectionCount > 0;/,
  );
  assert.match(
    APP_SIDEBAR,
    /const renderedChatIds = useMemo\(\n\s*\(\) => new Set\(renderedChatItems\.map\(\(item\) => item\.id\)\),/,
  );
  for (const id of [
    "archiveChat",
    "markChatUnread",
    "togglePinChat",
    "deleteSelectedChats",
  ]) {
    const at = APP_SIDEBAR.indexOf(`useShortcut("${id}"`);
    assert.ok(at !== -1, `${id} lost its call site`);
    assert.match(
      APP_SIDEBAR.slice(at, APP_SIDEBAR.indexOf("\n  });", at)),
      /selectionCount > 0/,
      `${id} no longer prefers the selection`,
    );
  }
  assert.match(
    APP_SIDEBAR,
    /if \(anchor && !renderedChatIds\.has\(anchor\.id\)\) \{\n\s*selectionAnchorRef\.current = null;/,
  );
});

// Acting clears the selection, so a repeated chord would otherwise hit the open chat.
test("a selection chord does not fall through to the open chat", async () => {
  assert.match(APP_SIDEBAR, /const SELECTION_ACTION_GRACE_MS = \d+;/);
  for (const id of ["archiveChat", "markChatUnread", "togglePinChat"]) {
    const body = APP_SIDEBAR.slice(
      APP_SIDEBAR.indexOf(`useShortcut("${id}"`),
      APP_SIDEBAR.indexOf("\n  });", APP_SIDEBAR.indexOf(`useShortcut("${id}"`)),
    );
    // Per-action latches: a shared one would block Archive after Pin took the selection.
    assert.match(
      body,
      new RegExp(`actOnSelection\\("${id}",`),
      `${id} does not stamp the latch under its own name`,
    );
    assert.match(
      body,
      new RegExp(
        `if \\(followsSelectionAction\\("${id}"\\)\\) return;[\\s\\S]*withActiveChat\\(`,
      ),
      `${id} reaches the open chat without checking its own latch`,
    );
  }
  const del = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf('useShortcut("deleteSelectedChats"'),
    APP_SIDEBAR.indexOf("\n  });", APP_SIDEBAR.indexOf('useShortcut("deleteSelectedChats"')),
  );
  assert.doesNotMatch(del, /withActiveChat\(/);
});

test("clearing every unread says what it cleared", async () => {
  const body = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf('useShortcut("clearAllUnreads"'),
    APP_SIDEBAR.indexOf("\n  });", APP_SIDEBAR.indexOf('useShortcut("clearAllUnreads"')),
  );
  // Counted before the wipe; rows, not threads, since a Compare row is backed by two.
  assert.match(body, /const cleared = countUnreadRows\(state\);[\s\S]*state\.clearAllUnreads\(\)/);
  assert.match(body, /if \(state\.unreadThreadIds\.size === 0\) \{\n\s*toast\.info\(/);
  assert.match(body, /toast\.success\(/);
});

// The backend reuses "call_0" per response, so it cannot be the store key.
test("a parked tool request is keyed by its own approval, not call_0", async () => {
  const adapter = await readSrcAsync("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /const toolConfirmationScopeId = resolvedThreadId\n\s*\? `\$\{sandboxSessionId \|\| "_default"\}:\$\{resolvedThreadId\}`/,
  );
  assert.match(adapter, /\? `\$\{toolConfirmationScopeId\}:\$\{approvalId\}`/);
  assert.match(
    adapter,
    /\(\) => `\$\{backendToolCallId\}:\$\{crypto\.randomUUID\(\)\}`/,
  );

  const { resolveToolCallPartId } = await import(
    "../src/features/chat/tool-call-id.ts"
  );
  let minted = 0;
  const mint = () => `call_0:${(minted += 1)}`;
  const paneA = resolveToolCallPartId(new Map(), "call_0", undefined, "", mint);
  const paneB = resolveToolCallPartId(new Map(), "call_0", undefined, "", mint);
  assert.notEqual(paneA, paneB);
  const ids = new Map<string, string>();
  const first = resolveToolCallPartId(ids, "call_0", undefined, "", mint);
  assert.equal(
    resolveToolCallPartId(ids, "call_0", undefined, "", mint),
    first,
  );
  assert.equal(
    resolveToolCallPartId(ids, "call_0", "sess:thread:tok", "", mint),
    "sess:thread:tok",
  );
});

// A dependency rebuilt each render re-runs the guard effect forever (React error #185).
test("the rows the selection guard reads keep their identity", async () => {
  for (const name of [
    "visibleProjectRecords",
    "pinnedRows",
    "visibleRecentItems",
    "sectionProjectChatItems",
    "pinnedSectionChatItems",
    "renderedChatItems",
    "renderedChatIds",
  ]) {
    const at = APP_SIDEBAR.indexOf(`const ${name} = `);
    assert.notEqual(at, -1, `${name} is gone`);
    assert.match(
      APP_SIDEBAR.slice(at, at + name.length + 40),
      /= useMemo\(/,
      `${name} is rebuilt every render and feeds a selection effect`,
    );
  }
  const builder = APP_SIDEBAR.indexOf("const folderChatItems = ");
  assert.notEqual(builder, -1, "folderChatItems is gone");
  assert.match(APP_SIDEBAR.slice(builder, builder + 64), /= useCallback\(/);
});

// groupThreads returns a fresh array, so calling it during render breaks list identity.
test("the sidebar item lists are built once per change, not per render", async () => {
  const hook = await readSrcAsync("features/chat/hooks/use-chat-sidebar-items.ts");
  for (const name of ["items", "archivedItems"]) {
    const at = hook.indexOf(`const ${name} = `);
    assert.notEqual(at, -1, `${name} is gone`);
    assert.match(
      hook.slice(at, at + name.length + 30),
      /= useMemo\(/,
      `${name} is rebuilt every render and feeds a selection effect`,
    );
  }
});

// Ctrl/Cmd-clicking a project selects no chat; that must not read as "no selection".
test("the chat-only chords stand aside for a project selection", async () => {
  for (const id of ["archiveChat", "markChatUnread", "togglePinChat"]) {
    const at = APP_SIDEBAR.indexOf(`useShortcut("${id}", () => {`);
    assert.notEqual(at, -1, `${id} is gone`);
    const body = APP_SIDEBAR.slice(at, APP_SIDEBAR.indexOf("\n  });", at));
    const stand = body.indexOf("projectsOnlySelected()");
    assert.notEqual(stand, -1, `${id} acts on the open chat under a project selection`);
    assert.ok(
      stand < body.indexOf("withActiveChat("),
      `${id} checks the project selection too late`,
    );
  }
  assert.match(
    APP_SIDEBAR,
    /const projectsOnlySelected = \(\) =>\n\s*selectionCount === 0 && projectSelectionCount > 0;/,
  );
});

// Safari and Chrome use Shift-Command-[ / ] for tab switching on macOS.
test("the chat walk's bracket pair is reserved on macOS only", () => {
  for (const value of ["Mod+Shift+BracketLeft", "Mod+Shift+BracketRight"]) {
    assert.ok(
      isBrowserReservedBinding(value, true),
      `${value} warns nobody on the platform that takes it`,
    );
    assert.equal(
      isBrowserReservedBinding(value, false),
      false,
      `${value} is not taken off macOS`,
    );
  }
  const walk = SHORTCUT_DEFS.find((def) => def.id === "nextChat");
  assert.equal(defaultBindingFor(walk!, "primary", true), "Mod+Shift+BracketRight");
});

// The desktop signs out through the OS account menu, so the chord can never fire there.
test("the logout row is not offered on the desktop build", async () => {
  const logout = SHORTCUT_DEFS.find((def) => def.id === "logOut");
  assert.equal(logout?.webOnly, true);
  assert.deepEqual(
    SHORTCUT_DEFS.filter((def) => def.webOnly).map((def) => def.id),
    ["logOut"],
  );
  assert.match(KEYBOARD_SHORTCUTS_TAB, /!\(isTauri && def\.webOnly\)/);
});

// The composer's keydown runs first and preventDefaults, so it must look follow-up chords up itself.
const FOLLOW_UP_IDS = ["queueMessage", "steerMessage"] as const;

const modEnterOn = (mac: boolean) => ({
  code: "Enter",
  metaKey: mac,
  ctrlKey: !mac,
  shiftKey: false,
  altKey: false,
});

test("an Enter chord bound to queue or steer is recognised in the composer", () => {
  for (const mac of [true, false]) {
    const modEnter = modEnterOn(mac);
    assert.equal(shortcutMatchingEvent({}, FOLLOW_UP_IDS, modEnter, mac), null);
    assert.equal(
      shortcutMatchingEvent(
        { queueMessage: { primary: "Mod+Enter" } },
        FOLLOW_UP_IDS,
        modEnter,
        mac,
      ),
      "queueMessage",
    );
    assert.equal(
      shortcutMatchingEvent(
        { queueMessage: { primary: "Mod+Enter" } },
        FOLLOW_UP_IDS,
        { ...modEnter, shiftKey: true },
        mac,
      ),
      null,
    );
    // Shadowed by a higher-ranked action, the same rule useShortcut applies.
    assert.equal(
      shortcutMatchingEvent(
        {
          newChat: { primary: "Mod+Enter" },
          queueMessage: { primary: "Mod+Enter" },
        },
        FOLLOW_UP_IDS,
        modEnter,
        mac,
      ),
      null,
    );
  }
});

test("the composer submits an Enter-bound chord with the named behavior", async () => {
  const { composerFollowUpBehavior, followUpSubmitIntent } = await import(
    "../src/features/chat/utils/composer-preferences.ts"
  );
  for (const preference of ["queue", "steer"] as const) {
    for (const [id, behavior] of [
      ["queueMessage", "queue"],
      ["steerMessage", "steer"],
    ] as const) {
      const named = shortcutMatchingEvent(
        { [id]: { primary: "Mod+Enter" } },
        FOLLOW_UP_IDS,
        modEnterOn(true),
        true,
      );
      assert.equal(named, id);
      assert.equal(
        composerFollowUpBehavior(
          preference,
          followUpSubmitIntent(preference, behavior),
        ),
        behavior,
        `${preference} preference, ${id} chord`,
      );
    }
  }
  const submitOnKey = THREAD.slice(THREAD.indexOf("const submitOnKey = useCallback("));
  const body = submitOnKey.slice(0, submitOnKey.indexOf("\n  );"));
  assert.match(body, /const named = followUpShortcutBehavior\(event\);/);
  assert.match(body, /submitIntentRef\.current = named\n?\s*\?/);
});

// Compare hides the thread composer for SharedComposer, so chords must register in both.
test("the follow-up chords are registered in both composers", async () => {
  const shared = await readSrcAsync("features/chat/shared-composer.tsx");
  for (const id of FOLLOW_UP_IDS) {
    assert.match(
      THREAD,
      new RegExp(`useShortcut\\(\\s*\n?\\s*"${id}"`),
      `thread.tsx does not register ${id}`,
    );
    assert.match(
      shared,
      new RegExp(`useShortcut\\(\\s*\n?\\s*"${id}"`),
      `shared-composer.tsx does not register ${id}`,
    );
  }
  assert.match(CHAT_PAGE, /<Thread hideComposer=\{true\}/);
});

function chord(value: string) {
  const parsed = parseBinding(value);
  assert.ok(parsed, `unparsable test binding ${value}`);
  return parsed;
}

test("a keystroke search matches the chord it names", () => {
  assert.equal(
    keystrokeMatchesBinding(chord("Mod+Shift+KeyO"), chord("Mod+Shift+KeyO")),
    true,
  );
  assert.equal(
    keystrokeMatchesBinding(chord("Mod+Shift+KeyO"), chord("Mod+Shift+KeyN")),
    false,
    "a different key is a different chord",
  );
});

test("a keystroke search widens as modifiers come off", () => {
  for (const bound of ["Mod+Shift+KeyO", "Mod+Alt+KeyO", "Mod+Alt+Shift+KeyO"]) {
    assert.equal(
      keystrokeMatchesBinding(chord("KeyO"), chord(bound)),
      true,
      bound,
    );
  }
  assert.equal(
    keystrokeMatchesBinding(chord("Mod+Shift+KeyO"), chord("Mod+Alt+Shift+KeyO")),
    true,
  );
  assert.equal(
    keystrokeMatchesBinding(chord("Mod+Shift+KeyO"), chord("Mod+Alt+KeyO")),
    false,
  );
});

test("a keystroke search never matches a chord missing a modifier it holds", () => {
  for (const pressed of ["Mod+KeyB", "Ctrl+KeyB", "Alt+KeyB", "Shift+KeyB"]) {
    assert.equal(
      keystrokeMatchesBinding(chord(pressed), chord("KeyB")),
      false,
      pressed,
    );
  }
});

test("the matcher finds every shipped default by its own chord and bare key", async () => {
  for (const def of SHORTCUT_DEFS) {
    for (const slot of SHORTCUT_SLOTS) {
      for (const mac of [true, false]) {
        const value = defaultBindingFor(def, slot, mac);
        if (!value) continue;
        const bound = chord(value);
        assert.equal(
          keystrokeMatchesBinding(bound, bound),
          true,
          `${def.id}.${slot} does not find itself`,
        );
        assert.equal(
          keystrokeMatchesBinding({ ...bound, mod: false, ctrl: false, shift: false, alt: false }, bound),
          true,
          `${def.id}.${slot} is not found by its bare key`,
        );
      }
    }
  }
});

test("the shortcuts tab arms the keystroke search ahead of Radix and the registry", async () => {
  const src = await readFile(
    new URL(
      "../src/features/settings/tabs/keyboard-shortcuts-tab.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Capture phase, or the dialog's Escape closes it before the box sees the key.
  assert.match(
    src,
    /window\.addEventListener\("keydown", onKeyDown, \{ capture: true \}\)/,
  );
  // bindingFromEvent carries the fallback for engines reporting no event.code.
  const listener = src.slice(
    src.indexOf("if (!byKeystroke || recording) return;"),
  );
  assert.doesNotMatch(listener, /event\.code\s*===/);
  assert.match(listener, /const binding = bindingFromEvent\(event\);/);
  assert.match(listener, /binding\.code === "Escape"/);
  assert.match(src, /if \(!byKeystroke \|\| recording\) return;/);
});

/** Bare Escape backs out of the chord box, so it is not taken as a query; the name search finds it. */
test("the chord box keeps bare Escape, and the name search still finds that row", () => {
  const decline = SHORTCUT_DEFS.find((def) => def.id === "declineToolRequest");
  assert.ok(decline);
  assert.equal(defaultBindingFor(decline, "primary", true), "Escape");
  const escape = parseBinding("Escape");
  assert.ok(escape);
  assert.equal(formatBindingLabel(escape, true), "Esc");
  assert.ok(formatBindingLabel(escape, true).toLowerCase().includes("esc"));
  assert.match(
    KEYBOARD_SHORTCUTS_TAB,
    /binding\.code === "Escape" && bare && !binding\.shift/,
  );
  const shiftEscape = parseBinding("Shift+Escape");
  assert.ok(shiftEscape);
  assert.equal(shiftEscape.shift, true);
});

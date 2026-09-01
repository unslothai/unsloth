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

test("the tab-search chord counts as browser-owned on both platforms", () => {
  // Chrome's tab search on both, and Firefox's add-ons manager off macOS, so
  // the tab warns about it wherever a user picks it.
  assert.ok(isBrowserReservedBinding("Mod+Shift+KeyA", true));
  assert.ok(isBrowserReservedBinding("Mod+Shift+KeyA", false));
});

// The walk is the one chord whose end is a key coming up rather than going
// down, so it is the one that a window losing focus can strand.
test("the recent walk ends on losing the window, not just on keyup", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const at = sidebar.indexOf("const end = () =>");
  assert.ok(at !== -1, "the traversal listener moved");
  const body = sidebar.slice(at, sidebar.indexOf("}, []);", at));
  // Both signals reach the same end, and both are torn down again.
  assert.match(body, /window\.addEventListener\("keyup", onKeyUp\);/);
  assert.match(body, /window\.addEventListener\("blur", end\);/);
  assert.match(body, /window\.removeEventListener\("blur", end\);/);
  // Still only once every modifier is up: a walk is held, so a Tab release
  // with Ctrl still down is mid-walk, not the end of it.
  assert.match(
    body,
    /if \(event\.ctrlKey \|\| event\.metaKey \|\| event\.altKey \|\| event\.shiftKey\)/,
  );
});

// Firefox's new private window, which no page can cancel. Chrome's incognito
// on ⇧⌘N was already reserved; this is the other half of the same pair.
test("the private-window chord is reserved and carries no default", () => {
  for (const mac of [true, false]) {
    assert.ok(isBrowserReservedBinding("Mod+Shift+KeyP", mac));
    assert.ok(isBrowserReservedBinding("Mod+Shift+KeyN", mac));
  }
  // Search keeps ⌘K alone rather than moving the alternate somewhere else.
  const search = SHORTCUT_DEFS.find((d) => d.id === "searchChats");
  assert.ok(search);
  for (const mac of [true, false]) {
    assert.equal(defaultBindingFor(search, "primary", mac), "Mod+KeyK");
    assert.equal(defaultBindingFor(search, "alternate", mac), null);
  }
});

// Train and Video are the two rows the sidebar grays out on a measured
// verdict, so they are the two workspace chords that can put a gate where the
// user's workspace was.
test("the workspace chords land where the guard lets them", async () => {
  const root = await readFile(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    root,
    /const chatOnlyMeasured = usePlatformStore\(\n\s*\(s\) => s\.isChatOnly\(\) && !s\.capabilitiesUnknown\(\),/,
  );
  assert.match(
    root,
    /const routeShortcutEnabled = !isAuthFlowRoute && !settingsDialogOpen;/,
  );
  assert.match(
    root,
    /useShortcut\("switchToTrain", goTo\("\/studio"\), \{\n\s*enabled: routeShortcutEnabled && !chatOnlyMeasured,/,
  );
  // Video has its own predicate rather than the chat-only one: /video checks
  // auth and nothing else, so the chord would land on the unsupported-hardware
  // gate. Read through the same helper the disabled row reads.
  assert.match(
    root,
    /const videoDisabled =\n\s*videoNavHint\(chatOnlyMeasured, chatOnlyReason\) !== undefined;/,
  );
  assert.match(
    root,
    /useShortcut\("switchToVideo", goTo\("\/video"\), \{\n\s*enabled: routeShortcutEnabled && !videoDisabled,/,
  );
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    sidebar,
    /const videoDisabledHint = videoNavHint\(chatOnlyMeasured, chatOnlyReason\);/,
  );
  assert.match(sidebar, /disabled: videoDisabled,/);

  // The rest are on the allowlist, so they stay reachable and ungated: gating
  // them would take away a page the guard is happy to serve.
  for (const [id, path] of [
    ["switchToProjects", "/projects"],
    ["switchToHub", "/hub"],
    ["switchToExport", "/export"],
  ] as const) {
    const at = root.indexOf(`useShortcut("${id}"`);
    assert.ok(at !== -1, `${id} lost its call site`);
    assert.ok(
      !root.slice(at, root.indexOf(");", at)).includes("chatOnlyMeasured"),
      `${id} gated on a verdict its route does not answer to`,
    );
    assert.match(root, new RegExp(`goTo\\("${path.replace("/", "\\/")}"\\)`));
  }
});

// A chord behind a hidden developer menu is not one the browser took from the
// user, and these sets drive a warning the user reads. Safari has both of
// these, but only once the Develop menu is switched on, so reserving them
// would tell every macOS user something untrue about their own keyboard.
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
  // So they stay usable as defaults, which is where the chat chords sit.
  for (const [id, value] of [
    ["archiveChat", "Mod+Alt+KeyE"],
    ["renameChat", "Mod+Alt+KeyR"],
  ] as const) {
    const def = SHORTCUT_DEFS.find((d) => d.id === id);
    assert.ok(def);
    assert.equal(defaultBindingFor(def, "primary", true), value);
  }
  // The bar is what the browser takes out of the box, which ⌥⌘U is: Chrome
  // ships view source on it with nothing to enable.
  assert.ok(isBrowserReservedBinding("Mod+Alt+KeyU", true));
});

test("the browsers' own run on macOS is reserved there and only there", () => {
  // ⌥⌘ is Chrome's run: view source, dev tools, console, bookmarks, split
  // view, web search, Page Setup, tab switching; Firefox adds its element
  // picker and console. Off macOS these read as Ctrl+Alt, which none claim.
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
  // The letters left on that run stay usable, or the chat chords lose the
  // family they are built on.
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

// Three actions whose macOS chord Chrome owns: view source, the element picker
// and Page Setup. Each keeps its letter on the run the composer pair uses, so
// the mnemonic survives the platform swap.
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
  // The API monitor is the one that gives its letter up rather than swapping
  // runs: off macOS U carries the two unread actions and has nothing left.
  for (const [id, value] of [
    ["markChatUnread", "Mod+Alt+KeyU"],
    ["clearAllUnreads", "Mod+Alt+Shift+KeyU"],
  ] as const) {
    const def = SHORTCUT_DEFS.find((d) => d.id === id);
    assert.ok(def);
    assert.equal(defaultBindingFor(def, "primary", false), value);
  }
});

// GTK binds hex entry to Ctrl+Shift+U in GtkIMContextSimple, and IBus binds it
// again, so off macOS that chord belongs to text composition. A composer with
// focus is where Mark unread is most likely to be pressed, so it cannot be a
// chord the input method is also listening for.
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
  // Still fine on macOS, which has no such prefix.
  const unread = SHORTCUT_DEFS.find((d) => d.id === "markChatUnread");
  assert.ok(unread);
  assert.equal(defaultBindingFor(unread, "primary", true), "Mod+Shift+KeyU");
});

test("no default takes a chord the browser owns without a reason", () => {
  // The exceptions are the spec chords Unsloth keeps for the desktop build,
  // where they work; everything else has to be reachable on the web.
  const deliberate = new Set([
    "Mod+KeyN",
    "Mod+Shift+KeyN",
    "Mod+Tab",
    "Mod+Shift+Tab",
    "Ctrl+Tab",
    "Ctrl+Shift+Tab",
    // The chat walk. Safari and Chrome own the bracket pair on macOS only, and
    // the desktop build is where these are pressed.
    "Mod+Shift+BracketLeft",
    "Mod+Shift+BracketRight",
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
  // Shift keeps it clear of the exit, which is where clearAllUnreads ships.
  assert.ok(isAcceptableBinding({ ...bare, shift: true }));

  // The recorder swallows every keydown, so bare Escape has to stay its way
  // out, except on the rows whose own chord it is.
  const tab = await readFile(
    new URL(
      "../src/features/settings/tabs/keyboard-shortcuts-tab.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    tab,
    /event\.code === "Escape" &&\n(?:\s*![a-zA-Z.]+ &&\n)+\s*!def\?\.allowBareKey\n\s*\) \{\n\s*setRecording\(null\);/,
  );

  // Which leaves no shipped bare Escape that its own tab could not record.
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
  // Overriding one slot leaves the other on its shipped chord. newChat, not
  // nextChat: the ⌥⌘→ pair is macOS only, and resolveBinding reads the host's
  // platform, so asserting it here would pass on a Mac and fail on CI.
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
  // ⌥⌘→ is Chrome's own next tab, and the same chord off macOS is Ctrl+Alt+→,
  // desktop switching on GNOME and KDE. Taken everywhere, so the bracket pair
  // carries these alone.
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

// Builds before alternates existed stored `id -> string | null`. Read as-is,
// that shape would resolve to nothing and silently revert every customization
// made since the shortcuts tab shipped.
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
    // A clear reaches both slots, a rebind only the one the user set.
    searchChats: { primary: null, alternate: null },
    newChat: { primary: "Mod+Alt+KeyJ" },
  });
  assert.equal(resolveBinding(migrated, "toggleSidebar"), "Mod+Alt+KeyB");
  assert.equal(resolveBinding(migrated, "searchChats"), null);
  // The alternate is untouched, so the shipped one still reaches a user who
  // rebound the primary before alternates existed.
  assert.equal(resolveBinding(migrated, "newChat", "alternate"), "Mod+KeyN");
});

// Back then an action had one chord, so a stored null meant the action was
// off. Clearing only the primary would hand it whatever alternate has shipped
// since and switch it back on, which is what newChat carries on ⌘N.
test("an action cleared before alternates existed stays cleared", () => {
  const migrated = migrateStoredOverrides(JSON.parse('{"newChat":null}'));
  assert.deepEqual(migrated, { newChat: { primary: null, alternate: null } });
  assert.deepEqual(resolveBindings(migrated, "newChat"), {
    primary: null,
    alternate: null,
  });
  // The shipped alternate is real, so this is the slot that would have come
  // back had the clear stopped at the primary.
  const shipped = SHORTCUT_DEFS.find((d) => d.id === "newChat");
  assert.ok(shipped);
  assert.equal(defaultBindingFor(shipped, "alternate", true), "Mod+KeyN");
  // Both slots read as edits, so the tab offers to reset them.
  for (const slot of SHORTCUT_SLOTS) {
    assert.ok(isSlotOverridden(migrated, "newChat", slot), slot);
  }
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
      // Both halves: a row with a label and no description renders an empty
      // second line rather than falling back to English.
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
  assert.match(chatPage, /enabled: headerPickersShown[,\s]*\}/);
  // Both panes mount the last message, and the chord would go to whichever
  // registered its listener first. `If last` carries the other half.
  assert.match(thread, /enabled: chatActive && !inComparePane && !forkDisabled/);
  // Same for a second parked tool request: neither card claims the keys.
  assert.match(
    toolCard,
    /soleRequest &&\n\s*!selectionActive &&\n\s*showControls &&/,
  );
});

// Both of these open a surface whose "is it open" flag outlives the thing it
// opens, so leaving without closing brings it back on the next visit.
test("a chord's surface does not come back open on the next visit", async () => {
  const read = async (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");
  const chatPage = await read("../src/features/chat/chat-page.tsx");
  const mcp = await read("../src/features/chat/mcp-composer-button.tsx");

  // The switcher renders on Chat, outside Compare, with a project. Reading the
  // whole condition means Compare and a standalone chat reset it too, not just
  // going off-route.
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

  // The MCP dialog's flag lives in a store, so an unmount with no
  // chatActive=false render in front of it leaves the dialog armed.
  assert.match(mcp, /if \(!chatActive && open\) setOpen\(false\);/);
  assert.match(
    mcp,
    /useEffect\(\(\) => \{\n\s*return \(\) => useMcpServersDialogStore\.getState\(\)\.setOpen\(false\);\n\s*\}, \[\]\);/,
  );
});

// Escape clears the selection from the sidebar's own listener, not the
// registry, so nothing stops another chord built on Escape from reaching it.
// Clear all unreads ships one on macOS.
test("only bare Escape drops the selection", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const at = sidebar.indexOf('if (event.key !== "Escape"');
  assert.notEqual(at, -1, "the selection listener moved");
  const block = sidebar.slice(at, sidebar.indexOf("clearSelection();", at));
  // Every modifier, so ⇧Esc goes to Clear all unreads alone.
  for (const modifier of ["metaKey", "ctrlKey", "altKey", "shiftKey"]) {
    assert.ok(
      block.includes(`event.${modifier}`),
      `${modifier} still reaches the selection listener`,
    );
  }
  assert.ok(block.includes("event.defaultPrevented"));
  // The chord it has to stay clear of.
  const clearAll = SHORTCUT_DEFS.find((d) => d.id === "clearAllUnreads");
  assert.ok(clearAll);
  assert.equal(defaultBindingFor(clearAll, "primary", true), "Shift+Escape");
});

// The mobile drawer is a Sheet owned by SidebarProvider, which __root mounts
// outside the Outlet, so it survives a navigation. Every sidebar row closes it
// by hand afterwards; the chords are registered above that provider and cannot.
test("the workspace chords do not leave the mobile drawer over the workspace", async () => {
  const read = async (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");
  const root = await read("../src/app/routes/__root.tsx");
  const sidebar = await read("../src/components/app-sidebar.tsx");

  // The provider outlives the route, which is what makes this necessary.
  assert.match(root, /<SidebarProvider/);
  assert.match(root, /useShortcut\("switchToProjects", goTo\("\/projects"\)/);

  // Closed on the location instead, which the chords change and the rows do
  // too. href, not pathname: a new chat from /chat only moves the search.
  assert.match(
    sidebar,
    /useEffect\(\(\) => \{\n\s*if \(isMobile\) setOpenMobile\(false\);\n\s*\}, \[href, isMobile, setOpenMobile\]\);/,
  );
  assert.match(sidebar, /href: s\.location\.href,/);

  // The rows keep their own call: opening a chat moves neither, and the drawer
  // still has to go.
  assert.match(
    sidebar,
    /const closeMobileIfOpen = \(\) => \{\n\s*if \(isMobile\) setOpenMobile\(false\);\n\s*\};/,
  );
});

// An action bar is the wrong place to register a chord: ActionBarRoot returns
// null when hidden and the user bar is autohide="always", so its children are
// gone unless the message is hovered. The assistant bar never mounted the fork
// button, so any thread ending in a reply had no listener at all.
test("the fork chord is registered where it mounts, not from an action bar", async () => {
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );

  // The registration is its own component, rendering nothing.
  const start = thread.indexOf("const ForkChatShortcut: FC = () => {");
  assert.notEqual(start, -1, "the fork registration moved");
  const block = thread.slice(start, thread.indexOf("\n};", start));
  assert.match(block, /useShortcut\(\n\s*"forkChat",/);
  assert.match(block, /return null;/);

  // The button keeps the click and takes no part in the chord.
  const buttonAt = thread.indexOf("const ForkMessageButton: FC = () => {");
  const button = thread.slice(buttonAt, thread.indexOf("\n};", buttonAt));
  assert.ok(
    !button.includes("useShortcut"),
    "an autohidden bar cannot hold the registration",
  );

  // Two instances of the action now exist on the last message, the chord's and
  // the button's, so the in-flight flag cannot be either one's own state: the
  // chord followed by a click would post two forks with two thread ids.
  assert.match(
    thread,
    /const useForkInFlight = create<\{\n\s*forking: boolean;/,
  );
  assert.match(thread, /const pending = useForkInFlight\(\(s\) => s\.forking\);/);
  assert.match(
    thread,
    /if \(useForkInFlight\.getState\(\)\.forking\) return;/,
  );
  assert.ok(
    !thread.includes("const [pending, setPending] = useState(false);"),
    "the per-instance flag is what let two forks run",
  );

  // Mounted from both roles, since either can be the last message, and only
  // for the last one, which is the message a fork may be taken from.
  const mounts = thread.match(
    /<MessagePrimitive\.If last=\{true\}>\n\s*<ForkChatShortcut \/>\n\s*<\/MessagePrimitive\.If>/g,
  );
  assert.equal(mounts?.length, 2);
  for (const role of ["const AssistantMessage", "const UserMessage: FC"]) {
    const at = thread.indexOf(role);
    assert.notEqual(at, -1, `${role} moved`);
    assert.ok(
      thread.slice(at, thread.indexOf("\n};", at)).includes("<ForkChatShortcut />"),
      `${role} does not mount the fork chord`,
    );
  }
});

// The tour's opener pins the picker open, and an effect shuts anything left
// pinned while no tour is running. A chord routed through it would open the
// picker and lose it on the next tick.
test("the model picker chord opens without the tour's pin", async () => {
  const chatPage = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );

  // The pin, and the effect that keys on it.
  assert.match(
    chatPage,
    /const openModelSelector = useCallback\(\(\) => \{\n\s*setModelSelectorLocked\(true\);/,
  );
  assert.match(
    chatPage,
    /if \(tour\.open\) return;\n\s*if \(!modelSelectorLocked\) return;[\s\S]{0,200}?setModelSelectorOpen\(false\);/,
  );

  // So the chord takes its own door, and stands aside while a step is on it.
  assert.match(
    chatPage,
    /const toggleModelSelector = useCallback\(\(\) => \{\n(?:\s*\/\/[^\n]*\n)*\s*if \(modelSelectorLocked\) return;\n\s*setModelSelectorOpen\(\(open\) => !open\);/,
  );
  assert.match(chatPage, /useShortcut\(\n\s*"openModelPicker",[\s\S]*?toggleModelSelector\(\);/);

  // Three mentions left, all the tour's: the declaration, the step builder's
  // argument, and that memo's dependency. Nothing else may pin it open.
  assert.equal((chatPage.match(/openModelSelector/g) ?? []).length, 3);
});

// The inline rename pill is rendered by the row, so it needs the row to be on
// screen. A chord has no row under the cursor, and the open chat may be behind
// a collapsed section, past a folder's limit, or on a route with no chat list.
test("the rename chord does not land in a surface only a row can show", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  // The chord asks for the dialog; the context menu, which has a row under the
  // cursor by definition, keeps the pill.
  assert.match(
    sidebar,
    /useShortcut\("renameChat", \(\) => \{[\s\S]*?withActiveChat\(\(item\) => openRenameChat\(item, false\)\);/,
  );
  assert.match(sidebar, /onSelect=\{\(\) => openRenameChat\(item\)\}/);
  assert.match(
    sidebar,
    /function openRenameChat\(item: SidebarItem, inline = true\)/,
  );

  // The pill is gated on it, so a dialog rename cannot also arm a row that is
  // off screen and surprise the user when it comes back.
  assert.match(
    sidebar,
    /const isRenamingThis =\n\s*renamingTarget\?\.kind === "chat" &&\n\s*renamingTarget\.inline &&/,
  );
  // And the dialog takes chats now, which is what its chat strings were always
  // written for.
  assert.match(
    sidebar,
    /\(renamingTarget\.kind !== "chat" \|\| !renamingTarget\.inline\)/,
  );
  assert.ok(sidebar.includes('t("shell.dialog.renameChat.title")'));
});

// Bare Escape has more than one owner. The sidebar answers it while rows are
// selected and does not consume it, so without a guard one press would clear
// the selection and deny a waiting tool call at the same time.
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

  // The tool card is the one that stands down: dropping a selection is undone
  // by selecting again, and denying a call is not.
  assert.match(toolCard, /const selectionActive = useChatNavigationStore\(/);
  assert.match(toolCard, /!selectionActive &&/);
  // Both keys, not just Escape: the buttons stay either way.
  for (const id of ["approveToolRequest", "declineToolRequest"]) {
    const at = toolCard.indexOf(`"${id}",`);
    assert.ok(at !== -1, `${id} lost its call site`);
    assert.match(
      toolCard.slice(at, toolCard.indexOf("\n  );", at)),
      /enabled: keyboardReady/,
    );
  }

  // Published from the sidebar, cleared on unmount so an unmounted sidebar
  // cannot leave the card mute.
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

  // And the sidebar listener stays passive. Dictation reads defaultPrevented
  // before cancelling, so consuming Escape here would let a stale selection
  // outrank a live recording.
  const at = sidebar.indexOf("Escape is the way out of a selection");
  const body = sidebar.slice(at, sidebar.indexOf("}, [selectionActive", at));
  assert.ok(!body.includes("preventDefault"), "the listener consumes Escape");
  assert.ok(!body.includes(", true)"), "the listener moved to capture");
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
    /chatListsOnScreen && pinnedOpen \? sortedPinnedChatItems/,
  );
  assert.match(
    sidebar,
    /chatListsOnScreen && chatOpen \? sortedRecentChatItems/,
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

test("the project picker chord is described by what it does", async () => {
  const page = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  // The header switcher navigates to the chosen project's landing; moving a
  // chat between projects is a different flow, on the chat row's own menu.
  assert.match(page, /onSelectProject=\{openProjectLanding\}/);
  const strings = await readFile(
    new URL("../src/i18n/locales/en.ts", import.meta.url),
    "utf8",
  );
  const at = strings.indexOf("openProjectPicker: {");
  const entry = strings.slice(at, strings.indexOf("},", at));
  assert.ok(!/move/i.test(entry), "no move-to-project promise");
  assert.ok(entry.includes("project"));
});

test("the new-chat chords stay out of the auth flow", async () => {
  const root = await readFile(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  // /login has no shell, and requireAuth bounces /chat straight back, so these
  // are gated like the workspace chords beside them.
  for (const id of ["newChat", "newTemporaryChat", "newStandaloneChat"]) {
    // The id, not the whole call: two of the three wrap onto their own line.
    const at = root.indexOf(`"${id}"`);
    assert.ok(at !== -1, `${id} is registered`);
    const call = root.slice(at, root.indexOf("\n  );", at) + 5);
    assert.match(call, /enabled: routeShortcutEnabled/, `${id} is gated`);
  }
});

test("switching back to Chat lands on the view the user left", async () => {
  const root = await readFile(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  // ChatPage renders the frozen search while off-route, and a bare /chat is a
  // fresh chat, so the chord has to hand that search back to the router.
  const at = root.indexOf('"switchToChat"');
  assert.ok(at !== -1, "switchToChat is registered");
  const call = root.slice(at, root.indexOf("\n  );", at) + 5);
  assert.match(call, /navigate\(\{ to: "\/chat", search: chatSearch \}\)/);
  assert.match(call, /enabled: routeShortcutEnabled/);
  // The other workspaces keep the bare helper; only chat carries a search.
  assert.match(root, /useShortcut\("switchToImages", goTo\("\/images"\)/);
  // location.search is the raw URL's, not the matched route's, so a seeded
  // freeze would hand /images?project=x to the chord as a chat never opened.
  assert.match(root, /useState<ChatSearch>\(\{\}\)/);
});

test("opening a chat by chord drops the selection, as clicking a row does", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  // Archive, pin and mark-unread prefer the selection when there is one, so a
  // stale one sends them to rows that are no longer on screen.
  const at = sidebar.indexOf("function openChatItem(");
  const body = sidebar.slice(at, sidebar.indexOf("\n  }", at));
  assert.ok(body.includes("clearSelection()"), "the shared opener clears it");
  // One path only: the row reaches it through the same function.
  assert.ok(
    !sidebar.includes("clearSelection();\n                openChatItem(item);"),
    "the row no longer clears it separately",
  );
});

test("effort chords only run for a model whose effort is read", async () => {
  const page = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  const at = page.indexOf("const shiftReasoningEffort");
  const body = page.slice(at, page.indexOf("useShortcut(\"cycleReasoningEffort\"", at));
  // enable_thinking models still list levels, but the request drops the effort.
  assert.match(body, /state\.reasoningStyle === "reasoning_effort"/);
  assert.match(body, /state\.reasoningStyle === "enable_thinking_effort"/);
  assert.match(body, /!state\.supportsReasoning \|\| !isEffort/);
});

test("New chat inherits the project on screen, inferred or not", async () => {
  const root = await readFile(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  const page = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  // On Chat the runtime's project is the visible one, inferred ones included:
  // the page resolves it from the thread or the compare pair when the URL
  // carries no ?project=, so a chat in a project stays in it.
  assert.match(root, /isChatRoute \? chatRuntime\.activeProjectId : null/);
  assert.match(page, /const projectId = thread\?\.projectId \?\? null;/);
  assert.match(page, /const projectId = threads\[0\]\?\.projectId \?\? null;/);
  assert.match(page, /setCurrentProjectId\(projectId\);\n\s*useChatRuntimeStore\.getState\(\)\.setActiveProjectId\(projectId\);/);
  // The page's own New chat button starts from the same value, so the chord
  // and the button cannot disagree about which project a new chat is in.
  assert.match(
    page,
    /const navigationProjectId = search\.project \?\? currentProjectId;/,
  );
  assert.match(page, /runtime\.setActiveProjectId\(navigationProjectId\);/);
  // Off Chat the page is hidden rather than unmounted, so the runtime still
  // names a project the user is not looking at. That one stays excluded.
  assert.match(root, /isChatRoute \? chatRuntime\.activeProjectId : null/);
  // Leaving the project is its own action, so this one must not also do it.
  assert.match(
    root,
    /useShortcut\("newStandaloneChat", \(\) => startNewChat\(\{ standalone: true \}\)/,
  );
  assert.match(root, /const projectId = options\?\.standalone \? null : openProjectId;/);
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

// Holding a chord past the OS repeat delay resends it. A toggle would land
// wherever the user let go, and an archive would run once per repeat.
test("auto-repeat only reaches the actions that walk a list", async () => {
  const read = async (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");
  const hook = await read("../src/features/settings/hooks/use-shortcut.ts");
  // Suppressed after preventDefault: the chord is ours either way, so the
  // browser must not act on the repeats we drop.
  const at = hook.indexOf("event.preventDefault();");
  assert.ok(at !== -1, "the hook stopped consuming the chord");
  assert.match(
    hook.slice(at),
    /event\.preventDefault\(\);\n(?:\s*\/\/[^\n]*\n)*\s*if \(event\.repeat && !repeats\) return;/,
  );
  // Off by default, so a new call site is one-shot until it says otherwise.
  assert.match(hook, /repeats = false,\n\s*\} = options;/);
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
  // Everything else is one-shot. A held toggle or archive is not a gesture.
  const optedIn = sidebar.match(/repeats: true/g) ?? [];
  assert.equal(optedIn.length, walkers.length);
});

// The chords read the published lists, so those have to end where the screen
// does: whole-sidebar gates included, not just each section's disclosure.
test("the published chat lists stop where the sidebar stops", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  // The same two conditions the three chat groups render behind, plus the
  // icon rail, which hides them in CSS rather than dropping them.
  assert.match(
    sidebar,
    /const chatListsOnScreen =\n\s*!isStudioRoute &&\n\s*!showTrainingRecents &&\n\s*\(isMobile \|\| sidebarState !== "collapsed"\);/,
  );
  for (const group of [
    /if \(!chatListsOnScreen \|\| organizeBy !== "project" \|\| !projectsOpen\)/,
    /\(chatListsOnScreen && pinnedOpen \? sortedPinnedChatItems : \[\]\)/,
    /\(chatListsOnScreen && chatOpen \? sortedRecentChatItems : \[\]\)/,
  ]) {
    assert.match(sidebar, group);
  }
  // Select All reads the same three arrays, so it cannot reach further than
  // the walk does.
  const selectAll = sidebar.indexOf("const selectAllChats = useCallback(");
  assert.ok(selectAll !== -1, "selectAllChats moved");
  assert.match(
    sidebar.slice(selectAll, sidebar.indexOf("\n  }, [", selectAll)),
    /\.\.\.visiblePinnedItems,\n\s*\.\.\.renderedProjectChatItems,\n\s*\.\.\.visibleRecentItems,/,
  );
  // Gating the arrays is enough because nothing renders from them.
  const rendered = sidebar.slice(sidebar.indexOf("return (", selectAll));
  for (const name of [
    "visiblePinnedItems",
    "visibleRecentItems",
    "renderedProjectChatItems",
  ]) {
    assert.ok(!rendered.includes(name), `${name} is read by the JSX too`);
  }
});

// The bulk chords take the selection over the open chat, and a selection has
// no presence outside the rows, so one carried off screen is invisible and
// still live.
test("a selection does not outlive the rows it was made on", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  // The whole sidebar going takes the whole selection with it.
  assert.match(
    sidebar,
    /if \(!chatRowsOnScreen\) \{\n\s*clearSelection\(\);\n\s*return;\n\s*\}/,
  );
  // Which is the stricter of the two: the lists can exist while their rows do
  // not, because a closed mobile sheet unmounts them the way the rail does.
  assert.match(
    sidebar,
    /const chatRowsOnScreen = chatListsOnScreen && \(!isMobile \|\| openMobile\);/,
  );
  // Select All is the one that builds a selection out of nothing, so it takes
  // the same gate rather than waiting for the cleanup to undo it.
  assert.match(
    sidebar,
    /const selectAllChats = useCallback\(\(\) => \{\n\s*if \(!chatRowsOnScreen\) return;/,
  );
  // Navigation is deliberately left on the looser one.
  assert.match(
    sidebar,
    /\(chatListsOnScreen && chatOpen \? sortedRecentChatItems : \[\]\)/,
  );
  // A single section closing takes only its own rows: the rest are still on
  // screen, so the selection is held to what is rendered rather than dropped.
  assert.match(
    sidebar,
    /for \(const id of prev\) \{\n\s*if \(renderedChatIds\.has\(id\)\) kept\.add\(id\);/,
  );
  assert.match(
    sidebar,
    /return kept\.size === prev\.size \? prev : kept;/,
  );
  assert.match(
    sidebar,
    /\}, \[chatRowsOnScreen, clearSelection, renderedChatIds, renderedProjectIds\]\);/,
  );
  // Folder rows are selectable too, and selectionActive counts them, so one
  // left behind by a closed section keeps the tool card's Escape standing
  // aside for a selection with nothing on screen to show for it.
  assert.match(
    sidebar,
    /for \(const id of prev\) \{\n\s*if \(renderedProjectIds\.has\(id\)\) kept\.add\(id\);/,
  );
  assert.match(
    sidebar,
    /if \(projectAnchor && !renderedProjectIds\.has\(projectAnchor\)\) \{\n\s*projectAnchorRef\.current = null;/,
  );
  // The three ways a folder row leaves without the sidebar going with it.
  assert.match(
    sidebar,
    /const renderedProjectIds = useMemo\(\(\) => \{\n\s*if \(!chatListsOnScreen \|\| organizeBy !== "project" \|\| !projectsOpen\) \{\n\s*return new Set<string>\(\);\n\s*\}\n\s*return new Set\(visibleProjectRecords\.map\(\(project\) => project\.id\)\);/,
  );
  // Both counts feed the flag, which is why both have to be pruned.
  assert.match(
    sidebar,
    /const selectionActive =\n?\s*selectionCount > 0 \|\| projectSelectionCount > 0;/,
  );
  // Built from the three arrays that already carry every disclosure state, so
  // a collapse or a "show less" needs nothing restated here.
  assert.match(
    sidebar,
    /const renderedChatIds = useMemo\(\(\) => \{[\s\S]*?visiblePinnedItems[\s\S]*?renderedProjectChatItems[\s\S]*?visibleRecentItems/,
  );
  // Which is what makes the four bulk branches safe to leave as they are.
  for (const id of [
    "archiveChat",
    "markChatUnread",
    "togglePinChat",
    "deleteSelectedChats",
  ]) {
    const at = sidebar.indexOf(`useShortcut("${id}"`);
    assert.ok(at !== -1, `${id} lost its call site`);
    assert.match(
      sidebar.slice(at, sidebar.indexOf("\n  });", at)),
      /selectionCount > 0/,
      `${id} no longer prefers the selection`,
    );
  }
  // The anchor goes with its row, so a later shift-click cannot reach back to
  // one that is no longer there.
  assert.match(
    sidebar,
    /if \(anchor && !renderedChatIds\.has\(anchor\.id\)\) \{\n\s*selectionAnchorRef\.current = null;/,
  );
});

// Acting on a selection clears it, so the same chord pressed again reads
// selectionCount as 0. Without a latch it archives the open chat, which was
// never selected, and says nothing about it.
test("a selection chord does not fall through to the open chat", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  assert.match(sidebar, /const SELECTION_ACTION_GRACE_MS = \d+;/);
  for (const id of ["archiveChat", "markChatUnread", "togglePinChat"]) {
    const body = sidebar.slice(
      sidebar.indexOf(`useShortcut("${id}"`),
      sidebar.indexOf("\n  });", sidebar.indexOf(`useShortcut("${id}"`)),
    );
    // Both halves name the action. A shared latch would hold back Archive
    // after Pin took the selection, and that is a different command the user
    // chose, not the repeat this guard exists for.
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
  // deleteSelectedChats needs none of this: it has no open-chat branch.
  const del = sidebar.slice(
    sidebar.indexOf('useShortcut("deleteSelectedChats"'),
    sidebar.indexOf("\n  });", sidebar.indexOf('useShortcut("deleteSelectedChats"')),
  );
  assert.doesNotMatch(del, /withActiveChat\(/);
});

// The only action with no menu item anywhere and no undo, so a silent wipe
// leaves the user nothing to tell it apart from a dead key.
test("clearing every unread says what it cleared", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const body = sidebar.slice(
    sidebar.indexOf('useShortcut("clearAllUnreads"'),
    sidebar.indexOf("\n  });", sidebar.indexOf('useShortcut("clearAllUnreads"')),
  );
  // Counted before the wipe, or the toast reports zero every time. Rows, not
  // threads: a Compare row is backed by two and would be cleared as two chats.
  assert.match(body, /const cleared = countUnreadRows\(state\);[\s\S]*state\.clearAllUnreads\(\)/);
  assert.match(body, /if \(state\.unreadThreadIds\.size === 0\) \{\n\s*toast\.info\(/);
  assert.match(body, /toast\.success\(/);
});

// Two parked requests must count as two, or both cards claim Enter and mount
// order picks the winner. Compare panes make that easy to get wrong: the
// backend reuses "call_0" per response, so the store key cannot be it.
test("a parked tool request is keyed by its own approval, not call_0", async () => {
  const adapter = await readFile(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  // Scope first, so two panes differ even before the approval token does.
  assert.match(
    adapter,
    /const toolConfirmationScopeId = resolvedThreadId\n\s*\? `\$\{sandboxSessionId \|\| "_default"\}:\$\{resolvedThreadId\}`/,
  );
  assert.match(adapter, /\? `\$\{toolConfirmationScopeId\}:\$\{approvalId\}`/);
  // The other branch mints a unique part id rather than reusing the backend's.
  assert.match(
    adapter,
    /\(\) => `\$\{backendToolCallId\}:\$\{crypto\.randomUUID\(\)\}`/,
  );

  const { resolveToolCallPartId } = await import(
    "../src/features/chat/tool-call-id.ts"
  );
  // One run's map cannot hand another run's the same id for "call_0".
  let minted = 0;
  const mint = () => `call_0:${(minted += 1)}`;
  const paneA = resolveToolCallPartId(new Map(), "call_0", undefined, "", mint);
  const paneB = resolveToolCallPartId(new Map(), "call_0", undefined, "", mint);
  assert.notEqual(paneA, paneB);
  // Within a run the same backend id keeps resolving to the one card.
  const ids = new Map<string, string>();
  const first = resolveToolCallPartId(ids, "call_0", undefined, "", mint);
  assert.equal(
    resolveToolCallPartId(ids, "call_0", undefined, "", mint),
    first,
  );
  // A confirmation id wins outright: that is the scoped key above.
  assert.equal(
    resolveToolCallPartId(ids, "call_0", "sess:thread:tok", "", mint),
    "sess:thread:tok",
  );
});

// The selection guard's effect depends on the set of rendered rows, and its
// setState bails out on an unchanged selection. React still re-renders once to
// discover that, so a dependency rebuilt during that render schedules the
// effect again, without end. React error #185, which took down the whole chat
// route rather than just the sidebar.
test("the rows the selection guard reads keep their identity", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  for (const name of [
    "visibleProjectRecords",
    "visiblePinnedItems",
    "visibleRecentItems",
    "renderedProjectChatItems",
    "renderedChatIds",
  ]) {
    const at = sidebar.indexOf(`const ${name} = `);
    assert.notEqual(at, -1, `${name} is gone`);
    assert.match(
      sidebar.slice(at, at + name.length + 40),
      /= useMemo\(/,
      `${name} is rebuilt every render and feeds a selection effect`,
    );
  }
});

// The root of the chain the test above pins. groupThreads returns a fresh
// array, so calling it during render gives every derived sidebar list a new
// identity on every render, which is what made the selection guard's effect
// re-run without end.
test("the sidebar item lists are built once per change, not per render", async () => {
  const hook = await readFile(
    new URL("../src/features/chat/hooks/use-chat-sidebar-items.ts", import.meta.url),
    "utf8",
  );
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

// Ctrl/Cmd-clicking a project row selects it without selecting any chat. The
// chat-only chords used to read that as "no selection" and act on the open
// chat, so Archive archived a chat the user had not pointed at.
test("the chat-only chords stand aside for a project selection", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  for (const id of ["archiveChat", "markChatUnread", "togglePinChat"]) {
    const at = sidebar.indexOf(`useShortcut("${id}", () => {`);
    assert.notEqual(at, -1, `${id} is gone`);
    const body = sidebar.slice(at, sidebar.indexOf("\n  });", at));
    const stand = body.indexOf("projectsOnlySelected()");
    assert.notEqual(stand, -1, `${id} acts on the open chat under a project selection`);
    assert.ok(
      stand < body.indexOf("withActiveChat("),
      `${id} checks the project selection too late`,
    );
  }
  // Only when no chat is selected: a mixed selection still has chats to act on.
  assert.match(
    sidebar,
    /const projectsOnlySelected = \(\) =>\n\s*selectionCount === 0 && projectSelectionCount > 0;/,
  );
});

// Apple lists Shift-Command-] and Shift-Command-[ as Safari's Show Next Tab
// and Show Previous Tab, and Chrome carries the same pair on macOS. The chat
// walk ships on them for the desktop build, so Settings has to say so rather
// than let a web user rebind into a chord the browser takes first.
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
  // Still the shipped default, which is why it is on the deliberate list.
  const walk = SHORTCUT_DEFS.find((def) => def.id === "nextChat");
  assert.equal(defaultBindingFor(walk!, "primary", true), "Mod+Shift+BracketRight");
});

// The desktop signs out through the OS account menu, so the chord's handler
// returns there and the sidebar hides its own logout item. Offering the row in
// Settings let a desktop user bind a key that can never fire.
test("the logout row is not offered on the desktop build", async () => {
  const logout = SHORTCUT_DEFS.find((def) => def.id === "logOut");
  assert.equal(logout?.webOnly, true);
  // Nothing else claims it, or the flag would hide a working action.
  assert.deepEqual(
    SHORTCUT_DEFS.filter((def) => def.webOnly).map((def) => def.id),
    ["logOut"],
  );
  const tab = await readFile(
    new URL(
      "../src/features/settings/tabs/keyboard-shortcuts-tab.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(tab, /!\(isTauri && def\.webOnly\)/);
});

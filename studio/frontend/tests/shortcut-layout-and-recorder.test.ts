// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  zoomChordTaken,
  zoomDirectionForKey,
  zoomTypedOffZoomKey,
} from "../src/features/interface-zoom/lib/zoom-chords.ts";
import {
  SHORTCUT_DEFS,
  SHORTCUT_SLOTS,
  bindingFromEvent,
  bindingMatchesQuery,
  defaultBindingFor,
  formatBindingValueLabel,
  formatCode,
  isAcceptableBinding,
  isBrowserReservedBinding,
  isBrowserUnreachableBinding,
  isModEnter,
  modEnterLabel,
  parseBinding,
  unrecordableModifier,
} from "../src/features/settings/lib/keyboard-shortcuts.ts";
import { readSrc } from "./helpers/kit.ts";

function press(
  code: string,
  key: string,
  mods: Partial<{
    metaKey: boolean;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    altGraph: boolean;
  }> = {},
) {
  const { altGraph = false, ...rest } = mods;
  return {
    code,
    key,
    metaKey: false,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    ...rest,
    getModifierState: (name: string) => name === "AltGraph" && altGraph,
  };
}

const defOf = (id: string) => {
  const def = SHORTCUT_DEFS.find((d) => d.id === id);
  assert.ok(def, `${id} is not registered`);
  return def;
};

// Much of Europe types "-" on the US Slash key, so Mod+"-" reads as Mod+Slash.
test('Mod and "-" on the Slash key zooms out, not the shortcuts list', () => {
  const ownsSlash = (value: string) => value === "Mod+Slash";
  for (const [mac, mods] of [
    [false, { ctrlKey: true }],
    [true, { metaKey: true }],
  ] as const) {
    const event = press("Slash", "-", mods);
    assert.equal(zoomDirectionForKey(event, mac), -1);
    assert.equal(zoomChordTaken(event, -1, mac, ownsSlash), false);
    assert.equal(zoomTypedOffZoomKey(event, "Slash", mac), true);
    assert.equal(
      zoomTypedOffZoomKey(press("Slash", "/", mods), "Slash", mac),
      false,
    );
    assert.equal(
      zoomTypedOffZoomKey(press("Minus", "-", mods), "Minus", mac),
      false,
    );
    assert.equal(
      zoomChordTaken(
        press("Minus", "-", mods),
        -1,
        mac,
        (v) => v === "Mod+Minus",
      ),
      true,
    );
  }
  assert.match(
    readSrc("features/settings/hooks/use-shortcut.ts"),
    /if \(zoomTypedOffZoomKey\(event, hit\.code, isMacPlatform\(\)\)\) return;/,
  );
});

// Swiss Ctrl+Shift+1 and Turkish Ctrl+Shift+4 type "+" and are workspace chords off macOS.
test("a digit chord that types + keeps its shortcut", () => {
  for (const [code, key] of [
    ["Digit1", "+"],
    ["Digit4", "+"],
  ] as const) {
    const event = press(code, key, { ctrlKey: true, shiftKey: true });
    assert.notEqual(zoomDirectionForKey(event, false), null);
    assert.equal(zoomTypedOffZoomKey(event, code, false), false, code);
    const value = `Mod+Shift+${code}`;
    assert.equal(
      zoomChordTaken(event, 1, false, (v) => v === value),
      true,
      code,
    );
  }
  assert.equal(
    defaultBindingFor(defOf("switchToChat"), "primary", false),
    "Mod+Shift+Digit1",
  );
});

test("stopGeneration ships on Mod+. and clashes with nothing", () => {
  const stop = defOf("stopGeneration");
  for (const mac of [true, false]) {
    const value = defaultBindingFor(stop, "primary", mac);
    assert.equal(value, "Mod+Period");
    assert.equal(isBrowserReservedBinding(value, mac), false);
    const parsed = parseBinding(value);
    assert.ok(parsed && isAcceptableBinding(parsed));
    for (const other of SHORTCUT_DEFS) {
      if (other.id === "stopGeneration") continue;
      for (const slot of SHORTCUT_SLOTS) {
        assert.notEqual(
          defaultBindingFor(other, slot, mac),
          value,
          `${other.id}.${slot} also ships on ${value} (mac=${mac})`,
        );
      }
    }
  }
  assert.equal(formatBindingValueLabel("Mod+Period", true), "⌘.");
  assert.equal(formatBindingValueLabel("Mod+Period", false), "Ctrl+.");
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const at = thread.indexOf('useShortcut(\n    "stopGeneration",');
  assert.notEqual(at, -1, "stopGeneration has no handler");
  const body = thread.slice(at, thread.indexOf("\n  );", at));
  assert.match(body, /stop\(\);/);
  assert.match(body, /composer\.cancel\(\)/);
  assert.doesNotMatch(body, /skipInTextFields/);
  assert.match(
    readSrc("features/chat/shared-composer.tsx"),
    /useShortcut\("stopGeneration", stop, \{\n\s*enabled: chatActive && busy,/,
  );
});

test("Firefox's Reader View warns on both platforms, and renameChat keeps it", () => {
  for (const mac of [true, false]) {
    assert.ok(isBrowserReservedBinding("Mod+Alt+KeyR", mac));
    assert.equal(
      defaultBindingFor(defOf("renameChat"), "primary", mac),
      "Mod+Alt+KeyR",
    );
  }
});

test("tab and window chords are flagged desktop-only, per platform", () => {
  for (const mac of [true, false]) {
    for (const id of [
      "newTemporaryChat",
      "nextRecentlyViewedChat",
      "previousRecentlyViewedChat",
    ]) {
      assert.ok(
        isBrowserUnreachableBinding(
          defaultBindingFor(defOf(id), "primary", mac),
          mac,
        ),
        `${id} (mac=${mac})`,
      );
    }
    assert.equal(isBrowserUnreachableBinding("Mod+Shift+KeyO", mac), false);
    assert.ok(isBrowserUnreachableBinding("Mod+KeyN", mac));
  }
  for (const id of ["nextChat", "previousChat"]) {
    const value = defaultBindingFor(defOf(id), "primary", true);
    assert.ok(isBrowserUnreachableBinding(value, true), id);
    assert.equal(isBrowserUnreachableBinding(value, false), false, id);
  }
});

test("the recorder names the AltGr or Windows key it cannot bind", () => {
  const winK = press("KeyK", "k", { metaKey: true });
  assert.equal(bindingFromEvent(winK, false), null);
  assert.equal(unrecordableModifier(winK, false), "meta");
  const altGrQ = press("KeyQ", "@", {
    ctrlKey: true,
    altKey: true,
    altGraph: true,
  });
  assert.equal(bindingFromEvent(altGrQ, false), null);
  assert.equal(unrecordableModifier(altGrQ, false), "altGraph");
  assert.equal(
    unrecordableModifier(press("MetaLeft", "Meta", { metaKey: true }), false),
    null,
  );
  assert.equal(
    unrecordableModifier(
      press("AltRight", "AltGraph", { altGraph: true }),
      false,
    ),
    null,
  );
  assert.equal(unrecordableModifier(winK, true), null);
  assert.equal(
    unrecordableModifier(press("KeyK", "k", { ctrlKey: true }), false),
    null,
  );
  const tab = readSrc("features/settings/tabs/keyboard-shortcuts-tab.tsx");
  assert.match(tab, /"settings\.keyboardShortcuts\.metaUnsupported"/);
  assert.match(tab, /"settings\.keyboardShortcuts\.altGraphUnsupported"/);
});

test("formatCode names the ISO, JIS, keypad and navigation keys", () => {
  assert.equal(formatCode("IntlBackslash", false), "ISO \\");
  assert.equal(formatCode("IntlRo", false), "Ro");
  assert.equal(formatCode("IntlYen", false), "¥");
  assert.equal(formatCode("Numpad1", false), "Num 1");
  assert.equal(formatCode("NumpadAdd", false), "Num +");
  assert.equal(formatCode("NumpadSubtract", true), "Num -");
  assert.equal(formatCode("NumpadEnter", false), "Num Enter");
  assert.equal(formatCode("PageUp", false), "Page Up");
  assert.equal(formatCode("PageDown", false), "Page Down");
  assert.equal(formatCode("Home", false), "Home");
  assert.equal(formatCode("End", false), "End");
  assert.equal(formatCode("Insert", false), "Ins");
  assert.equal(formatCode("Delete", false), "Del");
  assert.equal(formatCode("Backspace", true), "⌫");
  assert.equal(formatCode("Backspace", false), "Backspace");
  assert.equal(
    formatBindingValueLabel("Mod+Backspace", false),
    "Ctrl+Backspace",
  );
  assert.equal(formatBindingValueLabel("Mod+Backspace", true), "⌘⌫");
});

test("searching by modifier words finds the chord", () => {
  const cmdShiftO = parseBinding("Mod+Shift+KeyO");
  const optCmdR = parseBinding("Mod+Alt+KeyR");
  const ctrlShiftP = parseBinding("Ctrl+Shift+KeyP");
  assert.ok(cmdShiftO && optCmdR && ctrlShiftP);
  for (const q of [
    "cmd",
    "command",
    "⌘",
    "shift o",
    "⇧",
    "cmd+shift+o",
    "⇧⌘o",
  ]) {
    assert.ok(bindingMatchesQuery(cmdShiftO, q, true), q);
  }
  for (const q of ["opt", "option", "alt", "⌥", "⌥⌘r", "opt cmd r"]) {
    assert.ok(bindingMatchesQuery(optCmdR, q, true), q);
  }
  for (const q of ["ctrl", "control", "⌃", "ctrl shift p"]) {
    assert.ok(bindingMatchesQuery(ctrlShiftP, q, true), q);
  }
  assert.ok(bindingMatchesQuery(cmdShiftO, "ctrl shift o", false));
  assert.ok(bindingMatchesQuery(cmdShiftO, "control", false));
  assert.equal(bindingMatchesQuery(cmdShiftO, "cmd", false), false);
  assert.equal(bindingMatchesQuery(cmdShiftO, "alt", true), false);
  assert.equal(bindingMatchesQuery(optCmdR, "cmd o", true), false);
  assert.equal(bindingMatchesQuery(cmdShiftO, "c", true), false);
  assert.match(
    readSrc("features/settings/tabs/keyboard-shortcuts-tab.tsx"),
    /return parsed \? bindingMatchesQuery\(parsed, q, mac\) : false;/,
  );
});

test("Mod+Enter is the platform's own modifier, and never mid-composition", () => {
  const enter = (
    mods: Partial<{ metaKey: boolean; ctrlKey: boolean }>,
    extra = {},
  ) => ({
    key: "Enter",
    metaKey: false,
    ctrlKey: false,
    ...mods,
    ...extra,
  });
  assert.ok(isModEnter(enter({ metaKey: true }), true));
  assert.equal(isModEnter(enter({ ctrlKey: true }), true), false);
  assert.ok(isModEnter(enter({ ctrlKey: true }), false));
  assert.equal(isModEnter(enter({ metaKey: true }), false), false);
  assert.equal(
    isModEnter(enter({ metaKey: true, ctrlKey: true }), true),
    false,
  );
  assert.equal(isModEnter(enter({}), false), false);
  assert.equal(
    isModEnter({ ...enter({ ctrlKey: true }), key: "a" }, false),
    false,
  );
  assert.equal(
    isModEnter(enter({ ctrlKey: true }, { isComposing: true }), false),
    false,
  );
  assert.equal(
    isModEnter(
      enter({ ctrlKey: true }, { nativeEvent: { isComposing: true } }),
      false,
    ),
    false,
  );
  assert.ok(isModEnter(enter({ metaKey: true }, { keyCode: 229 }), true));
  assert.equal(modEnterLabel(true), "⌘↵");
  assert.equal(modEnterLabel(false), "Ctrl+Enter");
  for (const path of [
    "features/chat/components/edit-project-dialog.tsx",
    "features/images/paste-parameters-dialog.tsx",
    "features/chat/components/chat-skills-dialog.tsx",
    "features/settings/components/decision-try-dialog.tsx",
  ]) {
    const source = readSrc(path);
    assert.match(source, /isModEnter\(/, path);
    assert.doesNotMatch(source, /\.metaKey \|\| [a-z]+\.ctrlKey/, path);
  }
  assert.doesNotMatch(
    readSrc("components/assistant-ui/thread.tsx"),
    /e\.key === 'Enter' && \(e\.ctrlKey \|\| e\.metaKey\)/,
  );
});

// A request parked in a background chat has no card here; gating on the store left Escape dead.
test("Escape in the composer is left to a mounted, keyboard-ready decline", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /const declineMounted = useShortcutAvailable\("declineToolRequest", true\);/,
  );
  assert.match(
    thread,
    /cancelOnEscape=\{!\(declineMounted && declineOwnsEscape\)\}/,
  );
  assert.doesNotMatch(thread, /Object\.values\(s\.toolConfirmations\)/);
  const controls = readSrc(
    "components/assistant-ui/tool-confirmation-controls.tsx",
  );
  assert.match(
    controls,
    /useShortcut\("declineToolRequest", [^\n]+\n\s*enabled: keyboardReady,/,
  );
  assert.match(
    controls,
    /const keyboardReady =\n\s*!disclosure &&\n\s*chatActive &&\n\s*!fullAccessOpen &&\n\s*soleRequest &&/,
  );
});

test("search chats fires from the composer, and toggleSidebar not behind a modal", () => {
  const search = readSrc("features/chat/components/chat-search-dialog.tsx");
  assert.match(
    search,
    /useShortcut\("searchChats", [^\n]+\n\s*skipInTextFields: true,\n\s*textFieldException: COMPOSER_INPUT_SELECTOR,/,
  );
  assert.match(
    readSrc("components/ui/sidebar.tsx"),
    /useShortcut\("toggleSidebar", toggleSidebar, \{\n\s*claims: \(\) =>\n\s*\(isMobile && openMobile\) \|\| document\.querySelector\(OPEN_DIALOG_SELECTOR\) === null,/,
  );
});

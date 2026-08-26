// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  isImeComposing,
  isSurfaceBackgrounded,
  isSurfaceInForeground,
  typesInTextField,
} = await import(
  "../src/features/settings/hooks/use-shortcut.ts"
);
const { isAcceptableBinding, parseBinding } = await import(
  "../src/features/settings/lib/keyboard-shortcuts.ts"
);

/** A keydown the dispatcher would see, with only the fields it reads. */
function keydown(init: { isComposing?: boolean; keyCode?: number }) {
  return {
    isComposing: init.isComposing ?? false,
    keyCode: init.keyCode ?? 27,
  } as unknown as KeyboardEvent;
}

// Escape cancels an IME candidate and Enter commits one, and both ship bare
// here, so without this a CJK user dismissing a candidate answers the tool
// call. Two signals because neither is reliable alone: isComposing on WebKit,
// the legacy 229 on Chromium.
test("a keydown mid-IME-composition is not a chord", () => {
  assert.equal(isImeComposing(keydown({ isComposing: true })), true);
  assert.equal(isImeComposing(keydown({ keyCode: 229 })), true);
  assert.equal(isImeComposing(keydown({ isComposing: true, keyCode: 229 })), true);
  assert.equal(isImeComposing(keydown({})), false);
});

test("the dispatcher checks composition before it matches anything", async () => {
  const source = await readFile(
    new URL("../src/features/settings/hooks/use-shortcut.ts", import.meta.url),
    "utf8",
  );
  // Before the match, so no chord is found, and before preventDefault, so the
  // candidate window keeps its key.
  assert.match(
    source,
    /if \(isImeComposing\(event\)\) return;\n\s*const hit = bindings\.find/,
  );
});

// Tab moves focus and a chord consumes what it answers. Bound bare to a tool
// decision it makes that card's own buttons unreachable by keyboard, which is
// worse than the chord not existing.
test("bare Tab is refused even for a prompt-gated action", () => {
  assert.equal(isAcceptableBinding(parseBinding("Tab")!, true), false);
  assert.equal(isAcceptableBinding(parseBinding("Shift+Tab")!, true), false);
  assert.equal(isAcceptableBinding(parseBinding("Tab")!, false), false);
});

test("Tab held with a modifier is still a chord", () => {
  // The recently-viewed walk ships on exactly these.
  assert.equal(isAcceptableBinding(parseBinding("Ctrl+Tab")!, false), true);
  assert.equal(isAcceptableBinding(parseBinding("Mod+Tab")!, false), true);
  assert.equal(isAcceptableBinding(parseBinding("Mod+Shift+Tab")!, false), true);
});

test("refusing Tab does not disturb the other bare-key rules", () => {
  assert.equal(isAcceptableBinding(parseBinding("Enter")!, true), true);
  assert.equal(isAcceptableBinding(parseBinding("Escape")!, true), true);
  assert.equal(isAcceptableBinding(parseBinding("KeyG")!, false), false);
  assert.equal(isAcceptableBinding(parseBinding("F5")!, false), true);
  assert.equal(isAcceptableBinding(parseBinding("Shift+Escape")!, false), true);
});

/** Install a document whose querySelectorAll answers with `els`. */
function withElements(...els: { closest: (selector: string) => unknown }[]) {
  (globalThis as { document?: unknown }).document = {
    querySelectorAll: () => els,
  };
}
const under = { closest: () => ({}) };
const clear = { closest: () => null };

// A dialog leaves the route mounted, so a chord gated only on the route still
// fires behind it. Radix marks the rest of the page aria-hidden for the life of
// a modal, which is the general signal rather than a per-dialog store.
test("a surface under a modal is not in the foreground", () => {
  withElements(under);
  assert.equal(isSurfaceInForeground(".aui-composer-input"), false);
});

test("a surface with nothing over it is in the foreground", () => {
  withElements(clear);
  assert.equal(isSurfaceInForeground(".aui-composer-input"), true);
});

test("a surface that is not rendered at all is not in the foreground", () => {
  withElements();
  assert.equal(isSurfaceInForeground(".aui-composer-input"), false);
});

// Entering Compare leaves the base view mounted and inert behind the panes, so
// the first composer in the document is the hidden one. Asking it alone called
// Compare backgrounded and killed its dictation chord outright.
test("a hidden earlier match does not mask a visible later one", () => {
  withElements(under, clear);
  assert.equal(isSurfaceInForeground(".aui-composer-input"), true);
});

test("every match under a modal is still not the foreground", () => {
  withElements(under, under);
  assert.equal(isSurfaceInForeground(".aui-composer-input"), false);
});

test("dictation asks at press time, not through enabled", async () => {
  const source = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const at = source.indexOf('useShortcut(\n    "startDictation"');
  assert.notEqual(at, -1, "the dictation chord lost its call site");
  const body = source.slice(at, source.indexOf("\n  );", at));
  // Inside the handler: `enabled` is read at render, and a dialog opening need
  // not re-render this component.
  assert.match(
    body,
    /\(\) => \{\n\s*\/\/[\s\S]*?if \(!isSurfaceInForeground\(COMPOSER_INPUT_SELECTOR\)\) return;/,
  );
});

// A write that fails with a good payload used to report nothing at all, so the
// chord was indistinguishable from a dead key.
test("both copy chords report a failed write", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /} else if \(empty\.value\) \{[\s\S]*?\} else \{[\s\S]*?toast\.error\("Could not copy this chat\."\)/,
  );
  assert.match(source, /toast\.error\("Could not copy the session id\."\)/);
});

// Current membership says nothing about where a chat's older files went: it can
// join a project, record that session, and move back out.
test("the sandbox probe does not skip a chat that is out of a project", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const at = source.indexOf("async function sandboxSessionIdsHolding");
  assert.notEqual(at, -1);
  const body = source.slice(at, source.indexOf("\n  }", at));
  assert.doesNotMatch(body, /if \(!item\.projectId\) return recorded;/);
  assert.match(
    body,
    /if \(await sandboxHasFiles\(candidate\)\) held\.push\(candidate\);/,
  );
});

// Both composers register the dictation chord, and only one is on screen at a
// time, so the foreground check has to be on both or Compare keeps the hole.
test("both composers gate dictation on the foreground", async () => {
  for (const path of [
    "../src/components/assistant-ui/thread.tsx",
    "../src/features/chat/shared-composer.tsx",
  ]) {
    const source = await readFile(new URL(path, import.meta.url), "utf8");
    const at = source.indexOf('useShortcut(\n    "startDictation"');
    assert.notEqual(at, -1, `${path} lost its dictation chord`);
    const body = source.slice(at, source.indexOf("\n  );", at));
    assert.match(
      body,
      /if \(!isSurfaceInForeground\(COMPOSER_INPUT_SELECTOR\)\) return;/,
      `${path} starts the microphone behind a modal`,
    );
  }
});

// The sidebar used to hold the unread set in component state, which died with
// it. A module store does not, so the next account inherits it.
test("signing out drops the previous account's navigation state", async () => {
  const store = await readFile(
    new URL("../src/features/chat/stores/chat-navigation-store.ts", import.meta.url),
    "utf8",
  );
  assert.match(store, /resetAccountState: \(\) =>/);
  // A fresh Set, or every account after the first shares one.
  assert.match(store, /set\(\{ \.\.\.ACCOUNT_STATE, unreadThreadIds: new Set\(\), unreadRowIds: \{\} \}\)/);
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  // On unmount, which is what the auth routes do to the sidebar.
  assert.match(
    sidebar,
    /useEffect\(\n\s*\(\) => \(\) => useChatNavigationStore\.getState\(\)\.resetAccountState\(\),\n\s*\[\],\n\s*\);/,
  );
});

// The latch holds back a repeat of the action that took the selection, not a
// different command issued straight after it.
test("the selection latch is keyed by action", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  assert.match(sidebar, /selectionActedRef = useRef<\{ id: ShortcutId; at: number \} \| null>/);
  assert.match(sidebar, /last\?\.id === id &&/);
  assert.doesNotMatch(sidebar, /followsSelectionAction\(\)/);
});

// One selector has to mean "the composer" whichever of the two is on screen, or
// Escape stops declining in Compare and the dictation gate reads the wrong one.
test("both composers answer to the shared selector", async () => {
  const shared = await readFile(
    new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    "utf8",
  );
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const { COMPOSER_INPUT_SELECTOR } = await import(
    "../src/features/settings/hooks/use-shortcut.ts"
  );
  const className = COMPOSER_INPUT_SELECTOR.replace(/^\./, "");
  for (const [name, source] of [["shared", shared], ["thread", thread]]) {
    assert.match(
      source,
      new RegExp(`className="[^"]*\\b${className}\\b[^"]*"`),
      `the ${name} composer does not carry ${className}`,
    );
  }
});

// The recording bar replaces the composer input while dictation runs, so the
// foreground gate asks about an element that is gone for exactly as long as
// there is a recording to stop. Stopping has to come first.
test("stopping dictation is reachable once the input is gone", async () => {
  for (const path of [
    "../src/components/assistant-ui/thread.tsx",
    "../src/features/chat/shared-composer.tsx",
  ]) {
    const source = await readFile(new URL(path, import.meta.url), "utf8");
    const at = source.indexOf('useShortcut(\n    "startDictation"');
    const body = source.slice(at, source.indexOf("\n  );", at));
    const stop = body.search(/stopDictation\(\)/);
    const gate = body.search(/isSurfaceInForeground\(/);
    assert.notEqual(stop, -1, `${path} lost its stop branch`);
    assert.notEqual(gate, -1, `${path} lost its foreground gate`);
    assert.ok(
      stop < gate,
      `${path} gates the stop branch on a surface dictation removes`,
    );
  }
});

// Sending is not undoable, and a dialog over Chat leaves the chord registered
// with the draft behind it still submittable.
test("both composers refuse to send from behind a modal", async () => {
  for (const path of [
    "../src/components/assistant-ui/thread.tsx",
    "../src/features/chat/shared-composer.tsx",
  ]) {
    const source = await readFile(new URL(path, import.meta.url), "utf8");
    const at = source.indexOf('useShortcut(\n    "sendMessage"');
    assert.notEqual(at, -1, `${path} lost its send chord`);
    const body = source.slice(at, source.indexOf("\n  );", at));
    assert.match(
      body,
      /if \(!isSurfaceInForeground\(COMPOSER_INPUT_SELECTOR\)\) return;/,
      `${path} sends the hidden draft from behind a dialog`,
    );
  }
});

// A surface that is not rendered is not covered. The mobile sidebar lives in a
// drawer and is unmounted while it is closed, so reading "no match" as covered
// would leave every sidebar chord dead on mobile.
test("an absent surface is not a covered one", () => {
  const doc = globalThis.document;
  try {
    (globalThis as { document?: unknown }).document = {
      querySelectorAll: () => [],
    };
    assert.equal(isSurfaceBackgrounded(".gone"), false);
    assert.equal(isSurfaceInForeground(".gone"), false);
    const covered = { closest: () => ({}) };
    const open = { closest: () => null };
    (globalThis as { document?: unknown }).document = {
      querySelectorAll: () => [covered],
    };
    assert.equal(isSurfaceBackgrounded(".x"), true);
    (globalThis as { document?: unknown }).document = {
      querySelectorAll: () => [covered, open],
    };
    // One live match is enough: the base view stays mounted behind Compare.
    assert.equal(isSurfaceBackgrounded(".x"), false);
  } finally {
    (globalThis as { document?: unknown }).document = doc;
  }
});

// Window-level chords stay registered under a dialog, so the destructive ones
// have to ask at press time whether the sidebar is still the foreground.
test("the sidebar's mutating chords refuse to fire under a dialog", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  for (const id of [
    "archiveChat",
    "markChatUnread",
    "togglePinChat",
    "deleteSelectedChats",
    "renameChat",
    "clearAllUnreads",
  ]) {
    const at = sidebar.indexOf(`useShortcut("${id}", () => {`);
    assert.notEqual(at, -1, `${id} is gone`);
    const body = sidebar.slice(at, sidebar.indexOf("\n  });", at));
    assert.match(
      body,
      /if \(sidebarCovered\(\)\) return;/,
      `${id} acts on the chat behind an open dialog`,
    );
  }
  // Covered, not "not in the foreground", or the mobile drawer kills them all.
  assert.match(sidebar, /isSurfaceBackgrounded\(SIDEBAR_SELECTOR\)/);
  // And with the drawer closed the sidebar is unmounted, so the app root is
  // what carries the modal signal there.
  assert.match(
    sidebar,
    /document\.querySelector\(SIDEBAR_SELECTOR\) === null &&\n\s*isSurfaceBackgrounded\("#root"\)/,
  );
});

// A chat that ran tools before and after joining a project has files in the
// thread folder and in the project one. Probing only the thread folder
// answered for one and hid the other.
test("the sandbox probe leaves the shared project folder alone", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const at = sidebar.indexOf("async function sandboxSessionIdsHolding(");
  assert.notEqual(at, -1);
  const body = sidebar.slice(at, sidebar.indexOf("\n  }", at));
  // The shared project workspace is not probed: every chat in the project
  // writes there, so its files are no evidence about this one, and counting
  // them reported a second folder for any chat that joined a used project.
  assert.doesNotMatch(body, /sandboxSessionIdFor\(/);
  assert.doesNotMatch(body, /candidates\.add\(/);
  assert.equal(sidebar.split("sandboxSessionIdsHolding(ids)").length - 1, 2);
});

// Loading a model that drops the level in force leaves the effort set to one
// the model does not list, and indexOf then returns -1.
test("an unlisted reasoning effort steps to the first supported level", async () => {
  const page = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  const at = page.indexOf("const current = levels.indexOf(state.reasoningEffort);");
  assert.notEqual(at, -1);
  const body = page.slice(at, at + 700);
  assert.match(
    body,
    /if \(current === -1\) \{\n\s*state\.setReasoningEffort\(levels\[0\]\);/,
    "an unlisted effort still counts a step off an index that is not in the list",
  );
});

// The composer exception exists for a chord that types nothing there. Decline
// ships on Escape, but the row takes bare keys, so it can be rebound to one
// that does type, and the pass would then deny the request mid-sentence.
test("only a chord that types nothing keeps the composer exception", () => {
  const bare = (code: string) => ({
    code,
    mod: false,
    ctrl: false,
    shift: false,
    alt: false,
  });
  assert.equal(typesInTextField(bare("Escape")), false);
  assert.equal(typesInTextField(bare("F5")), false);
  assert.equal(typesInTextField(bare("Enter")), true);
  assert.equal(typesInTextField(bare("KeyA")), true);
  assert.equal(typesInTextField(bare("Backspace")), true);
  // A caret key inserts nothing, but a chord on one still has an edit to
  // stand aside for, so it does not get the pass either.
  assert.equal(typesInTextField(bare("ArrowUp")), true);
  // Held with anything but Shift it types nothing, whatever the key is.
  assert.equal(typesInTextField({ ...bare("KeyA"), mod: true }), false);
  assert.equal(typesInTextField({ ...bare("KeyA"), shift: true }), true);
});

test("the dispatcher drops the exception for a typing chord", async () => {
  const source = await readFile(
    new URL("../src/features/settings/hooks/use-shortcut.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /const exception = typesInTextField\(hit\) \? undefined : textFieldException;/,
  );
  assert.match(source, /isTextEntryFocused\(exception\)/);
});

// Every keydown is swallowed while recording, and on a bare-key row Escape is
// a chord rather than a cancel, so a keyboard-only user had no way out.
test("recording can be left from the keyboard", async () => {
  const tab = await readFile(
    new URL(
      "../src/features/settings/tabs/keyboard-shortcuts-tab.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const at = tab.indexOf("const onKeyDown = (event: KeyboardEvent) => {");
  assert.notEqual(at, -1);
  const body = tab.slice(at, at + 900);
  const exit = body.indexOf('event.code === "Tab"');
  const swallow = body.indexOf("event.preventDefault();");
  assert.notEqual(exit, -1, "recording has no keyboard exit");
  assert.ok(exit < swallow, "the exit is swallowed before it is read");
});

// The page stays mounted under a dialog, so `enabled` still says yes.
test("the header pickers do not open behind a dialog", async () => {
  const page = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  for (const id of ["openModelPicker", "openProjectPicker"]) {
    const at = page.indexOf(`"${id}",`);
    assert.notEqual(at, -1, `${id} is gone`);
    assert.match(
      page.slice(at, at + 260),
      /if \(chatCovered\(\)\) return;/,
      `${id} opens a control on the covered surface`,
    );
  }
  assert.match(
    page,
    /isSurfaceBackgrounded\(COMPOSER_INPUT_SELECTOR\)/,
  );
});

// The OS file chooser is the least dismissable thing a chord can raise.
test("both composers refuse to attach from behind a modal", async () => {
  for (const path of [
    "../src/components/assistant-ui/thread.tsx",
    "../src/features/chat/shared-composer.tsx",
  ]) {
    const source = await readFile(new URL(path, import.meta.url), "utf8");
    const at = source.indexOf('useShortcut(\n    "attachFiles"');
    assert.notEqual(at, -1, `${path} lost its attach chord`);
    const body = source.slice(at, source.indexOf("\n  );", at));
    assert.match(
      body,
      /if \(!isSurfaceInForeground\(COMPOSER_INPUT_SELECTOR\)\) return;/,
      `${path} opens the file chooser behind a dialog`,
    );
  }
});

// Answering a tool call the user cannot see is the one decision here that must
// not be reachable by accident, and the Chat route stays mounted under a
// dialog, so `keyboardReady` alone still says yes.
test("a tool call cannot be answered from behind a dialog", async () => {
  const source = await readFile(
    new URL(
      "../src/components/assistant-ui/tool-confirmation-controls.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  for (const id of ["approveToolRequest", "declineToolRequest"]) {
    const at = source.indexOf(`"${id}",`);
    assert.notEqual(at, -1, `${id} is gone`);
    const body = source.slice(at, at + 200);
    const guard = body.indexOf("if (chatCovered()) return;");
    const call = body.indexOf("resolve(");
    assert.notEqual(guard, -1, `${id} answers from behind a dialog`);
    assert.ok(guard < call, `${id} resolves before it checks`);
  }
  assert.match(source, /isSurfaceBackgrounded\(COMPOSER_INPUT_SELECTOR\)/);
});

// A selection made behind a dialog is invisible and still what the mutating
// chords act on afterwards, and the clipboard is outside the app entirely.
test("selection and clipboard chords stop at a covered sidebar", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  for (const id of ["selectAllChats", "copyChatAsMarkdown", "copySessionId"]) {
    const at = sidebar.indexOf(`useShortcut("${id}", () => {`);
    assert.notEqual(at, -1, `${id} is gone`);
    const body = sidebar.slice(at, sidebar.indexOf("\n  });", at));
    assert.match(
      body,
      /if \(sidebarCovered\(\)\) return;/,
      `${id} acts behind an open dialog`,
    );
  }
});

// The rows these chords stand in for stay enabled while the hardware verdict is
// out: resolveNavRowState returns disabled false for a pending row on purpose,
// so the click lands on a page that shows its own loading state. Gating the
// chords on the unknown verdict would make them disagree with their own rows.
test("the workspace chords wait on the same verdict their rows do", async () => {
  const root = await readFile(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  assert.match(root, /enabled: !isAuthFlowRoute && !chatOnlyMeasured,/);
  assert.match(root, /enabled: !isAuthFlowRoute && !videoDisabled,/);
  const rowState = await readFile(
    new URL("../src/components/nav-row-state.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    rowState,
    /if \(row\.pending\) \{\n\s*return \{\n\s*disabled: false,/,
    "a pending row now blocks the click the chord is allowed to make",
  );
});

// The reasoning, Fast mode and fork chords drive controls on the page behind a
// dialog just as the pickers do.
test("the remaining chat-page chords stop at a covered surface", async () => {
  const page = await readFile(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  for (const id of [
    "cycleReasoningEffort",
    "increaseReasoningEffort",
    "decreaseReasoningEffort",
    "toggleFastMode",
  ]) {
    const at = page.indexOf(`"${id}",`);
    assert.notEqual(at, -1, `${id} is gone`);
    assert.match(
      page.slice(at, at + 220),
      /if \(chatCovered\(\)\) return;/,
      `${id} acts on the covered surface`,
    );
  }
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const at = thread.indexOf('"forkChat",');
  assert.notEqual(at, -1);
  assert.match(
    thread.slice(at, at + 320),
    /if \(!isSurfaceInForeground\(COMPOSER_INPUT_SELECTOR\)\) return;/,
  );
});

// A non-modal popover leaves the composer the foreground, so the press-time
// check alone does not keep the send chord out of the picker's search box.
test("the send chords stand aside in any text field but the composer", async () => {
  for (const path of [
    "../src/components/assistant-ui/thread.tsx",
    "../src/features/chat/shared-composer.tsx",
  ]) {
    const source = await readFile(new URL(path, import.meta.url), "utf8");
    const at = source.indexOf('useShortcut(\n    "sendMessage"');
    assert.notEqual(at, -1, `${path} lost its send chord`);
    const body = source.slice(at, source.indexOf("\n  );", at));
    assert.match(body, /skipInTextFields: true,/, `${path} sends from any field`);
    assert.match(
      body,
      /textFieldException: COMPOSER_INPUT_SELECTOR,/,
      `${path} can no longer send from the composer itself`,
    );
  }
});

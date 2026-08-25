// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isImeComposing, isSurfaceBackgrounded, isSurfaceInForeground } = await import(
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

// Escape cancels an IME candidate and Enter commits one. Both are chords here
// (decline and approve ship bare), and declining takes the composer exception,
// so without this a CJK user dismissing a candidate answers the tool call.
// Two signals because neither is reliable alone: WebKit sets isComposing,
// Chromium reports the legacy 229, which is the pair the composer and the
// resource picker already read.
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
  assert.match(store, /set\(\{ \.\.\.ACCOUNT_STATE, unreadThreadIds: new Set\(\) \}\)/);
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
  assert.match(
    sidebar,
    /isSurfaceBackgrounded\('\[data-slot="sidebar"\]'\)/,
  );
});

// A chat that ran tools before and after joining a project has files in the
// thread folder and in the project one. Probing only the thread folder
// answered for one and hid the other.
test("the sandbox probe covers the current project folder too", async () => {
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const at = sidebar.indexOf("async function sandboxSessionIdsHolding(");
  assert.notEqual(at, -1);
  const body = sidebar.slice(at, sidebar.indexOf("\n  }", at));
  assert.match(body, /projectId: string \| null \| undefined/);
  assert.match(body, /sandboxSessionIdFor\(ids\[0\], projectId\)/);
  // Derived and then actually probed: naming it is not adding it.
  assert.match(body, /if \(projectSandbox\) candidates\.add\(projectSandbox\);/);
  // Both callers have to pass it, or the union is the old one.
  assert.equal(
    sidebar.split("sandboxSessionIdsHolding(ids, item.projectId)").length - 1,
    2,
  );
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

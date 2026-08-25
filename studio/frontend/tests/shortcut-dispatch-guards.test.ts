// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isImeComposing, isSurfaceInForeground } = await import(
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

/** Install a document whose querySelector answers with `el`. */
function withElement(el: { closest: (selector: string) => unknown } | null) {
  (globalThis as { document?: unknown }).document = {
    querySelector: () => el,
  };
}

// A dialog leaves the route mounted, so a chord gated only on the route still
// fires behind it. Radix marks the rest of the page aria-hidden for the life of
// a modal, which is the general signal rather than a per-dialog store.
test("a surface under a modal is not in the foreground", () => {
  withElement({ closest: () => ({}) });
  assert.equal(isSurfaceInForeground(".aui-composer-input"), false);
});

test("a surface with nothing over it is in the foreground", () => {
  withElement({ closest: () => null });
  assert.equal(isSurfaceInForeground(".aui-composer-input"), true);
});

test("a surface that is not rendered at all is not in the foreground", () => {
  withElement(null);
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
  assert.match(body, /if \(await sandboxHasFiles\(threadId\)\) held\.push\(threadId\);/);
});

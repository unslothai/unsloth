// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import ts from "typescript";

import {
  readSrc,
  readSrcAsync,
  registerBundlerResolver,
} from "./helpers/kit.ts";

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

const SRC__ROOT = readSrc("app/routes/__root.tsx");
const APP_SIDEBAR = readSrc("components/app-sidebar.tsx");
const CHAT_ROW_MENU = readSrc("features/chat/components/chat-row-menu.ts");
const OPEN_CHAT_FOLDER = readSrc("features/chat/components/open-chat-folder-item.tsx");
const THREAD = readSrc("components/assistant-ui/thread.tsx");
const CHAT_PAGE = readSrc("features/chat/chat-page.tsx");
const USE_SHORTCUT = readSrc("features/settings/hooks/use-shortcut.ts");

function keydown(init: { isComposing?: boolean; keyCode?: number }) {
  return {
    isComposing: init.isComposing ?? false,
    keyCode: init.keyCode ?? 27,
  } as unknown as KeyboardEvent;
}

// Escape/Enter drive IME candidates; isComposing covers WebKit, keyCode 229 Chromium.
test("a keydown mid-IME-composition is not a chord", () => {
  assert.equal(isImeComposing(keydown({ isComposing: true })), true);
  assert.equal(isImeComposing(keydown({ keyCode: 229 })), true);
  assert.equal(isImeComposing(keydown({ isComposing: true, keyCode: 229 })), true);
  assert.equal(isImeComposing(keydown({})), false);
});

test("the dispatcher checks composition before it matches anything", async () => {
  // Before the match and preventDefault, so the candidate window keeps its key.
  assert.match(
    USE_SHORTCUT,
    /if \(isImeComposing\(event\)\) return;\n\s*const hit = bindings\.find/,
  );
});

// Bare Tab moves focus; binding it would make the card's own buttons unreachable.
test("bare Tab is refused even for a prompt-gated action", () => {
  assert.equal(isAcceptableBinding(parseBinding("Tab")!, true), false);
  assert.equal(isAcceptableBinding(parseBinding("Shift+Tab")!, true), false);
  assert.equal(isAcceptableBinding(parseBinding("Tab")!, false), false);
});

test("Tab held with a modifier is still a chord", () => {
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

function withElements(...els: { closest: (selector: string) => unknown }[]) {
  (globalThis as { document?: unknown }).document = {
    querySelectorAll: () => els,
  };
}
const under = { closest: () => ({}) };
const clear = { closest: () => null };

// Radix marks the rest of the page aria-hidden under a modal; that is the general signal.
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

// Compare keeps the base view mounted and inert, so the first composer may be hidden.
test("a hidden earlier match does not mask a visible later one", () => {
  withElements(under, clear);
  assert.equal(isSurfaceInForeground(".aui-composer-input"), true);
});

test("every match under a modal is still not the foreground", () => {
  withElements(under, under);
  assert.equal(isSurfaceInForeground(".aui-composer-input"), false);
});

test("dictation asks at press time, not through enabled", async () => {
  const at = THREAD.indexOf('useShortcut(\n    "startDictation"');
  assert.notEqual(at, -1, "the dictation chord lost its call site");
  const body = THREAD.slice(at, THREAD.indexOf("\n  );", at));
  // Inside the handler: a dialog opening need not re-render this component.
  assert.match(
    body,
    /\(\) => \{\n\s*\/\/[\s\S]*?if \(!isSurfaceInForeground\(COMPOSER_INPUT_SELECTOR\)\) return;/,
  );
});

test("both copy chords report a failed write", async () => {
  assert.match(
    APP_SIDEBAR,
    /} else if \(empty\.value\) \{[\s\S]*?\} else \{[\s\S]*?toast\.error\("Could not copy this chat\."\)/,
  );
  assert.match(APP_SIDEBAR, /toast\.error\("Could not copy the session id\."\)/);
});

test("the sandbox probe does not skip a chat that is out of a project", async () => {
  const at = CHAT_ROW_MENU.indexOf("async function sandboxSessionIdsHolding");
  assert.notEqual(at, -1);
  const body = CHAT_ROW_MENU.slice(at, CHAT_ROW_MENU.indexOf("\n}", at));
  assert.doesNotMatch(body, /if \(!item\.projectId\) return recorded;/);
  assert.match(
    body,
    /if \(await sandboxHasFiles\(candidate\)\) held\.push\(candidate\);/,
  );
});

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

test("route shortcuts stay idle while Settings is open", async () => {
  const source = ts.createSourceFile(
    "__root.tsx",
    SRC__ROOT,
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
  const enabledById = new Map<string, string>();
  const visit = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText() === "useShortcut" &&
      ts.isStringLiteral(node.arguments[0]) &&
      ts.isObjectLiteralExpression(node.arguments[2])
    ) {
      const enabled = node.arguments[2].properties.find(
        (property): property is ts.PropertyAssignment =>
          ts.isPropertyAssignment(property) && property.name.getText() === "enabled",
      );
      if (enabled) {
        enabledById.set(node.arguments[0].text, enabled.initializer.getText());
      }
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);

  for (const id of [
    "newChat",
    "newTemporaryChat",
    "newStandaloneChat",
    "switchToChat",
    "switchToProjects",
    "switchToHub",
    "switchToRecipes",
    "switchToImages",
    "switchToAudio",
    "switchToExport",
  ]) {
    assert.equal(enabledById.get(id), "routeShortcutEnabled", id);
  }
  assert.equal(
    enabledById.get("switchToTrain"),
    "routeShortcutEnabled && !chatOnlyMeasured",
  );
  assert.equal(
    enabledById.get("switchToVideo"),
    "routeShortcutEnabled && !videoDisabled",
  );
  assert.match(
    SRC__ROOT,
    /const routeShortcutEnabled = !isAuthFlowRoute && !settingsDialogOpen;/,
  );
});

// A module store outlives the sidebar, so sign-out must reset it.
test("signing out drops the previous account's navigation state", async () => {
  const store = await readSrcAsync("features/chat/stores/chat-navigation-store.ts");
  assert.match(store, /resetAccountState: \(\) =>/);
  // A fresh Set, or every account after the first shares one.
  assert.match(store, /set\(\{ \.\.\.ACCOUNT_STATE, unreadThreadIds: new Set\(\), unreadRowIds: \{\} \}\)/);
  assert.match(
    APP_SIDEBAR,
    /useEffect\(\n\s*\(\) => \(\) => useChatNavigationStore\.getState\(\)\.resetAccountState\(\),\n\s*\[\],\n\s*\);/,
  );
});

test("the selection latch is keyed by action", async () => {
  assert.match(APP_SIDEBAR, /selectionActedRef = useRef<\{ id: ShortcutId; at: number \} \| null>/);
  assert.match(APP_SIDEBAR, /last\?\.id === id &&/);
  assert.doesNotMatch(APP_SIDEBAR, /followsSelectionAction\(\)/);
});

test("both composers answer to the shared selector", async () => {
  const shared = await readSrcAsync("features/chat/shared-composer.tsx");
  const { COMPOSER_INPUT_SELECTOR } = await import(
    "../src/features/settings/hooks/use-shortcut.ts"
  );
  const className = COMPOSER_INPUT_SELECTOR.replace(/^\./, "");
  for (const [name, source] of [["shared", shared], ["thread", THREAD]]) {
    assert.match(
      source,
      new RegExp(`className="[^"]*\\b${className}\\b[^"]*"`),
      `the ${name} composer does not carry ${className}`,
    );
  }
});

// The recording bar replaces the input, so stopping must precede the foreground gate.
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

// The mobile sidebar is unmounted when closed, so no match must not mean covered.
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
    assert.equal(isSurfaceBackgrounded(".x"), false);
  } finally {
    (globalThis as { document?: unknown }).document = doc;
  }
});

// Window-level chords stay registered under a dialog, so check foreground at press time.
test("the sidebar's mutating chords refuse to fire under a dialog", async () => {
  for (const id of [
    "archiveChat",
    "markChatUnread",
    "togglePinChat",
    "deleteSelectedChats",
    "renameChat",
    "clearAllUnreads",
  ]) {
    const at = APP_SIDEBAR.indexOf(`useShortcut("${id}", () => {`);
    assert.notEqual(at, -1, `${id} is gone`);
    const body = APP_SIDEBAR.slice(at, APP_SIDEBAR.indexOf("\n  });", at));
    assert.match(
      body,
      /if \(sidebarCovered\(\)\) return;/,
      `${id} acts on the chat behind an open dialog`,
    );
  }
  assert.match(APP_SIDEBAR, /isSurfaceBackgrounded\(SIDEBAR_SELECTOR\)/);
  assert.match(
    APP_SIDEBAR,
    /document\.querySelector\(SIDEBAR_SELECTOR\) === null &&\n\s*isSurfaceBackgrounded\("#root"\)/,
  );
});

test("the sandbox probe leaves the shared project folder alone", async () => {
  const at = CHAT_ROW_MENU.indexOf("async function sandboxSessionIdsHolding(");
  assert.notEqual(at, -1);
  const body = CHAT_ROW_MENU.slice(at, CHAT_ROW_MENU.indexOf("\n}", at));
  // The shared project workspace is written by every chat, so it is not evidence for this one.
  assert.doesNotMatch(body, /sandboxSessionIdFor\(/);
  assert.doesNotMatch(body, /candidates\.add\(/);
  assert.equal(APP_SIDEBAR.split("sandboxSessionIdsHolding(ids)").length - 1, 1);
  assert.equal(OPEN_CHAT_FOLDER.split("sandboxSessionIdsHolding(ids)").length - 1, 1);
});

// An effort the model does not list makes indexOf return -1.
test("an unlisted reasoning effort steps to the first supported level", async () => {
  const at = CHAT_PAGE.indexOf("const current = levels.indexOf(state.reasoningEffort);");
  assert.notEqual(at, -1);
  const body = CHAT_PAGE.slice(at, at + 700);
  assert.match(
    body,
    /if \(current === -1\) \{\n\s*state\.setReasoningEffort\(levels\[0\]\);/,
    "an unlisted effort still counts a step off an index that is not in the list",
  );
});

// Decline can be rebound to a typing key, so only non-typing chords keep the composer pass.
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
  assert.equal(typesInTextField(bare("ArrowUp")), true);
  assert.equal(typesInTextField({ ...bare("KeyA"), mod: true }), false);
  assert.equal(typesInTextField({ ...bare("KeyA"), shift: true }), true);
});

test("the dispatcher drops the exception for a typing chord", async () => {
  assert.match(
    USE_SHORTCUT,
    /const exception = typesInTextField\(hit\) \? undefined : textFieldException;/,
  );
  assert.match(USE_SHORTCUT, /isTextEntryFocused\(exception\)/);
});

// Keydowns are swallowed while recording and Escape may be a chord, so provide an exit.
test("recording can be left from the keyboard", async () => {
  const tab = await readSrcAsync("features/settings/tabs/keyboard-shortcuts-tab.tsx");
  const at = tab.indexOf("const onKeyDown = (event: KeyboardEvent) => {");
  assert.notEqual(at, -1);
  const body = tab.slice(at, at + 900);
  const exit = body.indexOf('event.code === "Tab"');
  const swallow = body.indexOf("event.preventDefault();");
  assert.notEqual(exit, -1, "recording has no keyboard exit");
  assert.ok(exit < swallow, "the exit is swallowed before it is read");
});

test("the header pickers do not open behind a dialog", async () => {
  for (const id of ["openModelPicker", "openProjectPicker"]) {
    const at = CHAT_PAGE.indexOf(`"${id}",`);
    assert.notEqual(at, -1, `${id} is gone`);
    assert.match(
      CHAT_PAGE.slice(at, at + 260),
      /if \(chatCovered\(\)\) return;/,
      `${id} opens a control on the covered surface`,
    );
  }
  assert.match(
    CHAT_PAGE,
    /isSurfaceBackgrounded\(COMPOSER_INPUT_SELECTOR\)/,
  );
});

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

// The Chat route stays mounted under a dialog, so keyboardReady alone still says yes.
test("a tool call cannot be answered from behind a dialog", async () => {
  const source = await readSrcAsync("components/assistant-ui/tool-confirmation-controls.tsx");
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

test("selection and clipboard chords stop at a covered sidebar", async () => {
  for (const id of ["selectAllChats", "copyChatAsMarkdown", "copySessionId"]) {
    const at = APP_SIDEBAR.indexOf(`useShortcut("${id}", () => {`);
    assert.notEqual(at, -1, `${id} is gone`);
    const body = APP_SIDEBAR.slice(at, APP_SIDEBAR.indexOf("\n  });", at));
    assert.match(
      body,
      /if \(sidebarCovered\(\)\) return;/,
      `${id} acts behind an open dialog`,
    );
  }
});

// resolveNavRowState leaves pending rows enabled, so chords must match their rows.
test("the workspace chords wait on the same verdict their rows do", async () => {
  assert.match(SRC__ROOT, /enabled: routeShortcutEnabled && !chatOnlyMeasured,/);
  assert.match(SRC__ROOT, /enabled: routeShortcutEnabled && !videoDisabled,/);
  const rowState = await readSrcAsync("components/nav-row-state.ts");
  assert.match(
    rowState,
    /if \(row\.pending\) \{\n\s*return \{\n\s*disabled: false,/,
    "a pending row now blocks the click the chord is allowed to make",
  );
});

test("the remaining chat-page chords stop at a covered surface", async () => {
  for (const id of [
    "cycleReasoningEffort",
    "increaseReasoningEffort",
    "decreaseReasoningEffort",
    "toggleFastMode",
  ]) {
    const at = CHAT_PAGE.indexOf(`"${id}",`);
    assert.notEqual(at, -1, `${id} is gone`);
    assert.match(
      CHAT_PAGE.slice(at, at + 220),
      /if \(chatCovered\(\)\) return;/,
      `${id} acts on the covered surface`,
    );
  }
  const at = THREAD.indexOf('"forkChat",');
  assert.notEqual(at, -1);
  assert.match(
    THREAD.slice(at, at + 320),
    /if \(!isSurfaceInForeground\(COMPOSER_INPUT_SELECTOR\)\) return;/,
  );
});

// A non-modal popover leaves the composer in the foreground.
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

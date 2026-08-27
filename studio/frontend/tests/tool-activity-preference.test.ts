// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The transition table for the collapse-tool-activity preference, the store's
// compatibility promise, and the JSX wiring that carries both.
//
// On the shape of the assertions here: the two state machines live in a plain
// .ts module precisely so they can be called, and the store is importable, so
// those claims are made by running the code. The cards are .tsx, which this
// runner cannot execute -- no jsdom, no JSX loader, and
// --experimental-strip-types does not strip JSX -- so their claims are made
// against the TypeScript AST rather than against a substring of their source.
// The difference matters: a substring assertion passes for any file that
// contains the characters, so it survives the feature being broken and dies on
// a reformat, which is the wrong way round. Rendered behaviour is covered by
// tests/studio/playwright_tool_activity.py.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import ts from "typescript";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const PREFERENCES_KEY = "unsloth_chat_preferences";

// Preferences exactly as a Studio from before this setting existed wrote them:
// every key the store had at the time, and no collapseToolActivityByDefault.
// Staged before the import, because persist hydrates at store creation.
const LEGACY_STATE = {
  confirmDeleteChats: false,
  alwaysDeleteChatFiles: true,
  showModelDisclaimer: true,
  showResponseModel: true,
  collapseThinkingByDefault: true,
  pastedTextMinChars: 8000,
};

store.set(
  PREFERENCES_KEY,
  JSON.stringify({ state: LEGACY_STATE, version: 0 }),
);

const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);
const { resolveToolActivityOpen, syncToolActivityPreference } = await import(
  "../src/components/assistant-ui/tool-activity-open-state.ts"
);

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");

/** Write `state` as a persisted record and hydrate the live store from it. */
async function rehydrateFrom(state: unknown): Promise<void> {
  store.set(PREFERENCES_KEY, JSON.stringify({ state, version: 0 }));
  await useChatPreferencesStore.persist.rehydrate();
}

// ---------------------------------------------------------------------------
// The store's compatibility promise, made against a real hydrate.
// ---------------------------------------------------------------------------

test("tool activity is collapsed by default", () => {
  assert.equal(
    useChatPreferencesStore.getInitialState().collapseToolActivityByDefault,
    true,
  );
});

test("a record written before the setting existed inherits the collapsed default", () => {
  // This is not an opt-in: `?? true` in merge() means an install that never
  // saw this setting starts collapsing tool activity the moment it upgrades.
  // Asserted against a real hydrate of a real legacy record so that the day
  // someone reconsiders that call, the test says so out loud instead of
  // quietly agreeing.
  assert.equal(
    useChatPreferencesStore.getState().collapseToolActivityByDefault,
    true,
  );
});

test("hydrating the new key leaves the older preferences alone", () => {
  // merge() is a hand-maintained allowlist, so adding a field to it is exactly
  // when one of the others goes missing.
  const state = useChatPreferencesStore.getState();
  assert.equal(state.confirmDeleteChats, false);
  assert.equal(state.alwaysDeleteChatFiles, true);
  assert.equal(state.showModelDisclaimer, true);
  assert.equal(state.showResponseModel, true);
  assert.equal(state.collapseThinkingByDefault, true);
  assert.equal(state.pastedTextMinChars, 8000);
  // The setters have to survive too: a merge returning only the saved fields
  // would leave a store with no way to write to it.
  assert.equal(typeof state.setCollapseToolActivityByDefault, "function");
});

test("turning the preference off round-trips through storage", async () => {
  // The direction that matters now that the default is on: a user who wants
  // their tool output back has to be able to keep it across a reload.
  await rehydrateFrom({
    ...LEGACY_STATE,
    collapseToolActivityByDefault: false,
  });
  assert.equal(
    useChatPreferencesStore.getState().collapseToolActivityByDefault,
    false,
    "an explicit off is being overridden by the new default",
  );
  useChatPreferencesStore.getState().setCollapseToolActivityByDefault(true);
  assert.equal(
    JSON.parse(store.get(PREFERENCES_KEY) ?? "{}").state
      .collapseToolActivityByDefault,
    true,
    "the preference is not persisted, so it would not survive a reload",
  );
  await rehydrateFrom(LEGACY_STATE);
});

test("an unreadable record leaves every default in place", async () => {
  store.set(PREFERENCES_KEY, "{not json");
  await assert.doesNotReject(async () => {
    await useChatPreferencesStore.persist.rehydrate();
  });
  const state = useChatPreferencesStore.getState();
  assert.equal(state.collapseToolActivityByDefault, true);
  assert.equal(typeof state.setCollapseToolActivityByDefault, "function");
  await rehydrateFrom(LEGACY_STATE);
});

// ---------------------------------------------------------------------------
// The transition table.
// ---------------------------------------------------------------------------

test("manual expansion survives updates while activity is collapsed", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      collapseByDefault: true,
      previousCollapseByDefault: true,
      isRunning: false,
      hasText: true,
    }),
    true,
  );
});

test("enabling collapsed activity closes an already open card", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      collapseByDefault: true,
      previousCollapseByDefault: false,
      isRunning: true,
      hasText: false,
    }),
    false,
  );
});

test("disabling collapsed activity restores automatic visibility", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: false,
      collapseByDefault: false,
      previousCollapseByDefault: true,
      isRunning: true,
      hasText: false,
    }),
    true,
  );
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      collapseByDefault: false,
      previousCollapseByDefault: false,
      isRunning: false,
      hasText: true,
    }),
    false,
  );
});

test("turning the preference off hands the card back to the automatic rules", () => {
  // Documented rather than accidental: a preference change is an explicit user
  // action, so it resets to whatever the automatic policy says rather than
  // preserving a manual expansion made under the old preference. The reverse
  // direction is the one that preserves manual state (first test above).
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      collapseByDefault: false,
      previousCollapseByDefault: true,
      isRunning: false,
      hasText: true,
    }),
    false,
  );
});

test("fallback cards react to live preference changes", () => {
  const manuallyOpen = {
    collapseByDefault: false,
    open: true,
  };
  const collapsed = syncToolActivityPreference(manuallyOpen, true, true);
  assert.deepEqual(collapsed, {
    collapseByDefault: true,
    open: false,
  });
  assert.deepEqual(syncToolActivityPreference(collapsed, false, true), {
    collapseByDefault: false,
    open: true,
  });
});

test("fallback cards preserve manual state until the preference changes", () => {
  const manuallyOpen = {
    collapseByDefault: true,
    open: true,
  };
  // Reference identity, not deep equality: the render-phase `if (synced !==
  // state) setState(...)` in ToolFallbackRoot and ToolGroupRoot terminates only
  // because an unchanged preference returns the very same object.
  assert.equal(
    syncToolActivityPreference(manuallyOpen, true, true),
    manuallyOpen,
  );
});

test("disabling collapsed activity respects a closed fallback default", () => {
  assert.deepEqual(
    syncToolActivityPreference(
      { collapseByDefault: true, open: false },
      false,
      false,
    ),
    { collapseByDefault: false, open: false },
  );
});

// ---------------------------------------------------------------------------
// AST helpers for the .tsx claims.
// ---------------------------------------------------------------------------

const sourceOf = async (path: string): Promise<ts.SourceFile> =>
  ts.createSourceFile(
    path,
    await read(path),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );

function walk(node: ts.Node, visit: (node: ts.Node) => void): void {
  visit(node);
  node.forEachChild((child) => walk(child, visit));
}

function find(root: ts.Node, match: (node: ts.Node) => boolean): ts.Node[] {
  const hits: ts.Node[] = [];
  walk(root, (node) => {
    if (match(node)) hits.push(node);
  });
  return hits;
}

/** The initializer of `const <name> = ...`. */
function initializerOf(root: ts.SourceFile, name: string): ts.Expression {
  const declaration = find(
    root,
    (node) =>
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name,
  )[0] as ts.VariableDeclaration | undefined;
  assert.ok(declaration?.initializer, `${root.fileName} has no const ${name}`);
  return declaration.initializer;
}

/** Every identifier read anywhere under `node`. */
function identifiersIn(node: ts.Node): Set<string> {
  const names = new Set<string>();
  walk(node, (child) => {
    if (ts.isIdentifier(child)) names.add(child.text);
  });
  return names;
}

/**
 * The local name a file binds the preference to, resolved through the store
 * selector rather than assumed. Hard-coding "collapseByDefault" would make
 * these assertions fail on a rename that changes nothing.
 */
function preferenceBinding(root: ts.SourceFile): string {
  const declaration = find(root, (node) => {
    if (!ts.isVariableDeclaration(node) || !node.initializer) return false;
    const init = node.initializer;
    return (
      ts.isCallExpression(init) &&
      ts.isIdentifier(init.expression) &&
      init.expression.text === "useChatPreferencesStore" &&
      identifiersIn(init).has("collapseToolActivityByDefault")
    );
  })[0] as ts.VariableDeclaration | undefined;
  assert.ok(
    declaration && ts.isIdentifier(declaration.name),
    `${root.fileName} never reads collapseToolActivityByDefault off the store`,
  );
  return declaration.name.text;
}

/** The JSX element named `name` nested anywhere under `node`. */
function jsxElement(node: ts.Node, name: string): ts.JsxElement {
  const hit = find(
    node,
    (child) =>
      ts.isJsxElement(child) &&
      ts.isIdentifier(child.openingElement.tagName) &&
      child.openingElement.tagName.text === name,
  )[0] as ts.JsxElement | undefined;
  assert.ok(hit, `no <${name}> element found`);
  return hit;
}

/** The JSX attribute `name` on the opening tag of `element`. */
function jsxAttribute(
  element: ts.JsxElement,
  name: string,
): ts.JsxAttribute | undefined {
  return element.openingElement.attributes.properties.find(
    (property): property is ts.JsxAttribute =>
      ts.isJsxAttribute(property) && property.name.getText() === name,
  );
}

// ---------------------------------------------------------------------------
// The wiring.
// ---------------------------------------------------------------------------

test("the shared hook resolves through the preference and the shared policy", async () => {
  const source = await sourceOf(
    "../src/components/assistant-ui/use-tool-activity-open.ts",
  );
  const selector = find(
    source,
    (node) =>
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "useChatPreferencesStore",
  )[0];
  assert.ok(selector, "the hook does not subscribe to the preference store");
  assert.ok(
    identifiersIn(selector).has("collapseToolActivityByDefault"),
    "the hook subscribes to the store but not to this preference",
  );
  const resolve = find(
    source,
    (node) =>
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "resolveToolActivityOpen",
  )[0] as ts.CallExpression | undefined;
  assert.ok(resolve, "the hook does not call the shared policy");
  const passed = identifiersIn(resolve.arguments[0] ?? resolve);
  for (const field of [
    "currentOpen",
    "collapseByDefault",
    "previousCollapseByDefault",
    "isRunning",
    "hasText",
  ]) {
    assert.ok(passed.has(field), `the policy is called without ${field}`);
  }
});

test("every tool card that opens itself routes through the shared hook", async () => {
  for (const file of [
    "../src/components/assistant-ui/tool-ui-code-execution.tsx",
    "../src/components/assistant-ui/tool-ui-knowledge-base.tsx",
    "../src/components/assistant-ui/tool-ui-web-search.tsx",
  ]) {
    const source = await sourceOf(file);
    const call = find(
      source,
      (node) =>
        ts.isCallExpression(node) &&
        ts.isIdentifier(node.expression) &&
        node.expression.text === "useToolActivityOpen",
    )[0] as ts.CallExpression | undefined;
    assert.ok(call, `${file} bypasses the shared automatic visibility policy`);
    // Both signals live, not pinned: useToolActivityOpen(true, hasText) would
    // still route through the hook while re-opening every card unconditionally.
    assert.equal(
      call.arguments.length,
      2,
      `${file} calls the shared hook with the wrong number of signals`,
    );
    for (const argument of call.arguments) {
      assert.ok(
        ts.isIdentifier(argument) || ts.isPropertyAccessExpression(argument),
        `${file} pins a shared-hook signal to ${argument.getText()}`,
      );
    }
  }
});

test("an uncontrolled fallback card takes its open state from the preference", async () => {
  // The claim a substring cannot make: pinning `isOpen` to true for an
  // uncontrolled card leaves every mention of the preference in this file
  // intact, so a source-text assertion would still pass while the setting did
  // nothing at all.
  const source = await sourceOf(
    "../src/components/assistant-ui/tool-fallback.tsx",
  );
  const preference = preferenceBinding(source);
  const synced = find(
    source,
    (node) =>
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "syncToolActivityPreference",
  )[0] as ts.CallExpression | undefined;
  assert.ok(synced, "the fallback card no longer uses the shared policy");
  assert.ok(
    ts.isVariableDeclaration(synced.parent),
    "the synced state is not bound to a name isOpen could read",
  );
  const syncedName = synced.parent.name.getText();
  assert.ok(
    identifiersIn(synced).has(preference),
    "the fallback card syncs against something other than the preference",
  );
  assert.ok(
    identifiersIn(synced).has("defaultOpen"),
    "the fallback card drops its own default when the preference flips",
  );

  const isOpen = initializerOf(source, "isOpen");
  assert.ok(
    identifiersIn(isOpen).has(syncedName),
    "an uncontrolled card opens without consulting the preference",
  );
  assert.ok(
    identifiersIn(isOpen).has("controlledOpen") &&
      identifiersIn(isOpen).has("isControlled"),
    "isOpen is no longer a controlled/uncontrolled choice",
  );
});

test("a card awaiting approval opens above the preference", async () => {
  // A parked call renders its command or script inside ToolFallbackContent
  // while Allow/Always allow/Deny render outside the card, so a collapsed card
  // asks for a decision about something the user cannot read. Radix does not
  // mount closed content, so it is absent rather than merely hidden.
  const source = await sourceOf(
    "../src/components/assistant-ui/tool-fallback.tsx",
  );
  const isOpen = initializerOf(source, "isOpen");
  assert.ok(
    ts.isBinaryExpression(isOpen) &&
      isOpen.operatorToken.kind === ts.SyntaxKind.BarBarToken,
    "isOpen no longer starts with an unconditional override",
  );
  assert.ok(
    identifiersIn(isOpen.left).has("awaitingApproval"),
    "awaiting approval is not the unguarded arm of isOpen",
  );
  assert.equal(
    identifiersIn(isOpen.left).has(preferenceBinding(source)),
    false,
    "the collapse preference can suppress an approval prompt's context",
  );

  // Every card that (a) is wrapped in withToolConfirmation and (b) can be closed by
  // the preference. Ask permission mode gates all of these, and each renders the
  // thing being approved -- command, script, query, code -- inside the collapsible.
  for (const file of [
    "../src/components/assistant-ui/tool-ui-terminal.tsx",
    "../src/components/assistant-ui/tool-ui-python.tsx",
    "../src/components/assistant-ui/tool-ui-web-search.tsx",
    "../src/components/assistant-ui/tool-ui-knowledge-base.tsx",
    "../src/components/assistant-ui/tool-ui-code-execution.tsx",
  ]) {
    const card = await sourceOf(file);
    const attribute = jsxAttribute(
      jsxElement(card, "ToolFallbackRoot"),
      "awaitingApproval",
    );
    assert.ok(
      attribute,
      `${file} can hide the command or script it is asking approval for`,
    );
    assert.ok(
      attribute.initializer &&
        ts.isJsxExpression(attribute.initializer) &&
        attribute.initializer.expression &&
        identifiersIn(attribute.initializer.expression).has(
          "awaitingApproval",
        ),
      `${file} pins awaitingApproval instead of passing the live value`,
    );
  }
});

test("a pending approval forces a group open regardless of the preference", async () => {
  const source = await sourceOf(
    "../src/components/assistant-ui/tool-group.tsx",
  );
  const preference = preferenceBinding(source);
  const forceOpen = initializerOf(source, "forceOpen");
  assert.ok(
    ts.isBinaryExpression(forceOpen) &&
      forceOpen.operatorToken.kind === ts.SyntaxKind.BarBarToken,
    "forceOpen is no longer a disjunction",
  );
  assert.ok(
    identifiersIn(forceOpen.left).has("hasPendingConfirmation"),
    "the approval signal is not the unguarded arm of forceOpen",
  );
  assert.equal(
    identifiersIn(forceOpen.left).has(preference),
    false,
    "a collapsed group can now hide a blocking approval prompt",
  );
  assert.ok(
    identifiersIn(forceOpen.right).has(preference),
    "the non-approval arm ignores the preference and forces groups open",
  );
  // And the element still consumes it, opting out of control when it is false:
  // a group pinned to `false` would be unopenable rather than merely closed.
  const open = jsxAttribute(jsxElement(source, "ToolGroupRoot"), "open");
  assert.ok(open?.initializer, "forceOpen is computed but not applied");
  const expression = ts.isJsxExpression(open.initializer)
    ? open.initializer.expression
    : undefined;
  assert.ok(
    expression && ts.isConditionalExpression(expression),
    "the group's open prop is no longer a controlled/uncontrolled choice",
  );
  assert.equal(expression.condition.getText(), "forceOpen");
  assert.equal(expression.whenTrue.getText(), "true");
  assert.equal(
    expression.whenFalse.getText(),
    "undefined",
    "a group that is not force-opened must fall back to its own state",
  );
});

test("a mounted group follows the preference like a mounted card", async () => {
  // ToolGroupImpl passes `undefined` whenever it is not forcing the group open,
  // so the group's own uncontrolled state is what is on screen for most of its
  // life. Without this it is the one disclosure the preference cannot reach.
  const source = await sourceOf(
    "../src/components/assistant-ui/tool-group.tsx",
  );
  const preference = preferenceBinding(source);
  const synced = find(
    source,
    (node) =>
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "syncToolActivityPreference",
  )[0] as ts.CallExpression | undefined;
  assert.ok(synced, "ToolGroupRoot does not apply live preference changes");
  assert.ok(
    identifiersIn(synced).has(preference),
    "the group syncs against something other than the preference",
  );
  assert.ok(
    ts.isVariableDeclaration(synced.parent),
    "the group's synced state is not bound to a name isOpen could read",
  );
  assert.ok(
    identifiersIn(initializerOf(source, "isOpen")).has(
      synced.parent.name.getText(),
    ),
    "the group computes a synced state and then ignores it",
  );
});

test("the Python script cell moves inside the collapsible when collapsing is on", async () => {
  // Two renders of one cell, each guarded by the opposite value: outside the
  // collapsible so a reopened chat still shows the script (#7165), inside it
  // when the user asked for quiet. Two copies of the same guard would render
  // the script twice.
  const source = await sourceOf(
    "../src/components/assistant-ui/tool-ui-python.tsx",
  );
  const preference = preferenceBinding(source);
  const root = jsxElement(source, "ToolFallbackRoot");
  const content = jsxElement(root, "ToolFallbackContent");

  /** Every `{<guard> && scriptCell}` under `scope`, as (negated, node) pairs. */
  const guardsFor = (scope: ts.Node) =>
    find(
      scope,
      (node) =>
        ts.isJsxExpression(node) &&
        !!node.expression &&
        ts.isBinaryExpression(node.expression) &&
        node.expression.operatorToken.kind ===
          ts.SyntaxKind.AmpersandAmpersandToken &&
        node.expression.right.getText() === "scriptCell",
    ).map((node) => {
      const guard = (
        (node as ts.JsxExpression).expression as ts.BinaryExpression
      ).left;
      assert.ok(
        identifiersIn(guard).has(preference),
        "the script cell is rendered without consulting the preference",
      );
      return {
        negated:
          ts.isPrefixUnaryExpression(guard) &&
          guard.operator === ts.SyntaxKind.ExclamationToken,
        node,
      };
    });

  const inside = guardsFor(content);
  const outside = guardsFor(root).filter(
    (hit) => !inside.some((other) => other.node === hit.node),
  );

  assert.deepEqual(
    inside.map((hit) => hit.negated),
    [false],
    "a collapsed Python card does not carry the script inside the collapsible",
  );
  assert.deepEqual(
    outside.map((hit) => hit.negated),
    [true],
    "the always-visible script cell is no longer guarded by the preference",
  );
});

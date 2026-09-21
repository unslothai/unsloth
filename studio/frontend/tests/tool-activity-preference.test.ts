// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The transition table for the tool-call visibility setting, the store's
// compatibility promise, and the JSX wiring that carries both.
//
// The reducers and the store are plain .ts, so those claims run the code. The
// cards are .tsx and this runner cannot execute JSX (no jsdom, no JSX loader),
// so their claims go through the TypeScript AST, not a substring of the source:
// a substring passes on broken code and fails on a reformat, which is backwards.
// Rendered behaviour lives in tests/studio/playwright_tool_activity.py.

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import {
  installLocalStorageFake,
  readText,
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
const {
  resolveToolActivityOpen,
  startsNewToolRound,
  syncToolActivityPreference,
  toolActivityOpen,
} =
  await import("../src/components/assistant-ui/tool-activity-open-state.ts");
const { foldIsActive } = await import(
  "../src/features/chat/utils/display-visibility.ts"
);

/** Write `state` as a persisted record and hydrate the live store from it. */
async function rehydrateFrom(state: unknown): Promise<void> {
  store.set(PREFERENCES_KEY, JSON.stringify({ state, version: 0 }));
  await useChatPreferencesStore.persist.rehydrate();
}

// ---------------------------------------------------------------------------
// The store's compatibility promise, made against a real hydrate.
// ---------------------------------------------------------------------------

test("tool calls are collapsed by default and thinking follows its stream", () => {
  const initial = useChatPreferencesStore.getInitialState();
  assert.equal(initial.toolVisibility, "collapsed");
  assert.equal(initial.thinkingVisibility, "auto");
});

test("a record written before the setting existed inherits the collapsed default", () => {
  // Not an opt-in: an install that never saw this setting starts collapsing tool activity the
  // moment it upgrades. Asserted against a real hydrate so reconsidering that call is loud.
  assert.equal(useChatPreferencesStore.getState().toolVisibility, "collapsed");
});

test("the old pair of switches carries over to the three-state settings", () => {
  // LEGACY_STATE has collapseThinkingByDefault: true and no tool key, so this covers both
  // halves: an explicit collapse survives the rename, an absent key lands on the new default.
  const state = useChatPreferencesStore.getState();
  assert.equal(state.thinkingVisibility, "collapsed");
  assert.equal(state.toolVisibility, "collapsed");
});

test("a legacy `off` becomes auto rather than always expanded", async () => {
  // The old `false` meant "open while running, close after", which is auto. Reading it as
  // expanded would leave every upgrading user with permanently open tool cards.
  await rehydrateFrom({
    ...LEGACY_STATE,
    collapseThinkingByDefault: false,
    collapseToolActivityByDefault: false,
  });
  const state = useChatPreferencesStore.getState();
  assert.equal(state.thinkingVisibility, "auto");
  assert.equal(state.toolVisibility, "auto");
  await rehydrateFrom(LEGACY_STATE);
});

test("a stored visibility wins over the legacy boolean beside it", async () => {
  // Both keys can coexist for one upgrade, and the new one is the one the user last set.
  await rehydrateFrom({
    ...LEGACY_STATE,
    collapseToolActivityByDefault: true,
    toolVisibility: "expanded",
  });
  assert.equal(useChatPreferencesStore.getState().toolVisibility, "expanded");
  await rehydrateFrom(LEGACY_STATE);
});

test("a value from a newer build falls back instead of blanking the row", async () => {
  await rehydrateFrom({ ...LEGACY_STATE, toolVisibility: "peek" });
  assert.equal(useChatPreferencesStore.getState().toolVisibility, "collapsed");
  await rehydrateFrom(LEGACY_STATE);
});

test("hydrating the new keys leaves the older preferences alone", () => {
  // merge() is a hand-maintained allowlist, so adding a field to it is exactly
  // when one of the others goes missing.
  const state = useChatPreferencesStore.getState();
  assert.equal(state.confirmDeleteChats, false);
  assert.equal(state.alwaysDeleteChatFiles, true);
  assert.equal(state.showModelDisclaimer, true);
  assert.equal(state.showResponseModel, true);
  assert.equal(state.pastedTextMinChars, 8000);
  // The setters have to survive too: a merge returning only the saved fields
  // would leave a store with no way to write to it.
  assert.equal(typeof state.setToolVisibility, "function");
  assert.equal(typeof state.setThinkingVisibility, "function");
});

test("each visibility round-trips through storage", async () => {
  for (const visibility of ["auto", "expanded", "collapsed"] as const) {
    useChatPreferencesStore.getState().setToolVisibility(visibility);
    useChatPreferencesStore.getState().setThinkingVisibility(visibility);
    const saved = JSON.parse(store.get(PREFERENCES_KEY) ?? "{}").state;
    assert.equal(
      saved.toolVisibility,
      visibility,
      "the tool setting is not persisted, so it would not survive a reload",
    );
    assert.equal(
      saved.thinkingVisibility,
      visibility,
      "the thinking setting is not persisted, so it would not survive a reload",
    );
    await useChatPreferencesStore.persist.rehydrate();
    assert.equal(useChatPreferencesStore.getState().toolVisibility, visibility);
    assert.equal(
      useChatPreferencesStore.getState().thinkingVisibility,
      visibility,
    );
  }
  await rehydrateFrom(LEGACY_STATE);
});

test("an unreadable record leaves every default in place", async () => {
  store.set(PREFERENCES_KEY, "{not json");
  await assert.doesNotReject(async () => {
    await useChatPreferencesStore.persist.rehydrate();
  });
  const state = useChatPreferencesStore.getState();
  assert.equal(state.toolVisibility, "collapsed");
  assert.equal(typeof state.setToolVisibility, "function");
  assert.equal(typeof state.setThinkingVisibility, "function");
  await rehydrateFrom(LEGACY_STATE);
});

// ---------------------------------------------------------------------------
// Fold, and the one setting it cannot coexist with.
// ---------------------------------------------------------------------------

test("folding tool calls into Thinking gives way to always expanded", () => {
  // Honouring both would pin the calls open inside something closed, so the fold stands down.
  assert.equal(foldIsActive(true, "collapsed"), true);
  assert.equal(foldIsActive(true, "auto"), true);
  assert.equal(foldIsActive(true, "expanded"), false);
  assert.equal(foldIsActive(false, "collapsed"), false);
  assert.equal(foldIsActive(false, "expanded"), false);
});

test("the fold preference is only suspended, not cleared", async () => {
  // Settings disables the row while it cannot apply, but the stored value has to survive.
  await rehydrateFrom({
    ...LEGACY_STATE,
    foldToolActivityIntoThinking: true,
    toolVisibility: "expanded",
  });
  const state = useChatPreferencesStore.getState();
  assert.equal(state.foldToolActivityIntoThinking, true);
  assert.equal(foldIsActive(state.foldToolActivityIntoThinking, "expanded"), false);
  state.setToolVisibility("collapsed");
  assert.equal(
    foldIsActive(
      useChatPreferencesStore.getState().foldToolActivityIntoThinking,
      "collapsed",
    ),
    true,
  );
  await rehydrateFrom(LEGACY_STATE);
});

// ---------------------------------------------------------------------------
// The transition table.
// ---------------------------------------------------------------------------

test("manual expansion survives updates while activity is collapsed", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      visibility: "collapsed",
      previousVisibility: "collapsed",
      isRunning: false,
      hasText: true,
    }),
    true,
  );
});

test("always expanded keeps a card open through running, finished and answered", () => {
  for (const [isRunning, hasText] of [
    [true, false],
    [false, false],
    [false, true],
  ] as const) {
    assert.equal(
      resolveToolActivityOpen({
        currentOpen: true,
        visibility: "expanded",
        previousVisibility: "expanded",
        isRunning,
        hasText,
      }),
      true,
      `closed itself at running=${isRunning} hasText=${hasText}`,
    );
  }
});

test("a card closed by hand stays closed under always expanded", () => {
  // The setting says where a card starts, not where it stays.
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: false,
      visibility: "expanded",
      previousVisibility: "expanded",
      isRunning: false,
      hasText: true,
    }),
    false,
  );
});

test("switching to collapsed closes an already open card", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      visibility: "collapsed",
      previousVisibility: "auto",
      isRunning: true,
      hasText: false,
    }),
    false,
  );
});

test("switching to auto restores automatic visibility", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: false,
      visibility: "auto",
      previousVisibility: "collapsed",
      isRunning: true,
      hasText: false,
    }),
    true,
  );
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      visibility: "auto",
      previousVisibility: "auto",
      isRunning: false,
      hasText: true,
    }),
    false,
  );
});

test("switching to always expanded opens a card that was collapsed and finished", () => {
  // A call that already ran, on a message already on screen, opening because the setting did.
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: false,
      visibility: "expanded",
      previousVisibility: "collapsed",
      isRunning: false,
      hasText: true,
    }),
    true,
  );
});

test("changing the setting hands the card back to the automatic rules", () => {
  // Deliberate: a setting change is an explicit action, so it resets rather than preserving a
  // manual expansion made under the old setting. Between changes manual state is preserved.
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      visibility: "auto",
      previousVisibility: "collapsed",
      isRunning: false,
      hasText: true,
    }),
    false,
  );
});

test("switching to auto opens a running card on a message that already has text", () => {
  // The call still running belongs to a message whose prose arrived first, so hasText is
  // already true when the setting changes. Expand while running is about the call, not the text.
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: false,
      visibility: "auto",
      previousVisibility: "collapsed",
      isRunning: true,
      hasText: true,
    }),
    true,
    "a running card stayed closed when the setting changed to Expand while running",
  );
  // A finished call on the same message still opens nothing.
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: false,
      visibility: "auto",
      previousVisibility: "collapsed",
      isRunning: false,
      hasText: true,
    }),
    false,
  );
});


test("a hand-opened controlled card keeps its open when the answer starts", () => {
  const transition = (hasText: boolean, override?: boolean | null) =>
    resolveToolActivityOpen({
      currentOpen: true,
      visibility: "auto",
      previousVisibility: "auto",
      isRunning: false,
      hasText,
      override,
    });
  assert.equal(transition(false, true), true);
  assert.equal(
    transition(true, true),
    true,
    "the answer arriving closed a controlled card the user opened",
  );
  assert.equal(
    transition(true),
    false,
    "an untouched card must still close when the answer starts",
  );
});


test("the round boundary is the call starting again", () => {
  // The reasoning block's rule, applied here: a round starts when the activity resumes.
  assert.equal(startsNewToolRound(true, false), true);
  assert.equal(startsNewToolRound(true, true), false);
  assert.equal(startsNewToolRound(false, true), false);
  assert.equal(startsNewToolRound(false, false), false);
});

test("a new round hands a controlled card back to the setting", () => {
  // Regenerate reuses the card, so the previous round's hand-set state must not survive into
  // the new one: the setting decides where the card starts again.
  const rerun = (
    visibility: "collapsed" | "auto" | "expanded",
    override: boolean | null,
  ) =>
    resolveToolActivityOpen({
      currentOpen: true,
      visibility,
      previousVisibility: visibility,
      isRunning: true,
      hasText: true,
      override,
      startedNewRound: true,
    });
  // The answer text arrived in the old round, so hasText is already true for this one too.
  assert.equal(
    rerun("auto", false),
    true,
    "a card closed in the old round stayed closed for the new run",
  );
  assert.equal(
    rerun("collapsed", true),
    false,
    "a card opened in the old round stayed pinned open for the new run",
  );
  assert.equal(rerun("expanded", false), true);
});


test("fallback cards react to live preference changes", () => {
  const manuallyOpen = {
    visibility: "auto" as const,
    active: true,
    override: true,
  };
  const collapsed = syncToolActivityPreference(manuallyOpen, "collapsed", true);
  // The setting change drops the manual open, so the card follows the new setting.
  assert.equal(collapsed.override, null);
  assert.equal(toolActivityOpen(collapsed), false);
  assert.equal(toolActivityOpen(syncToolActivityPreference(collapsed, "auto", true)), true);
  // Always expanded ignores the card's own activity and opens it regardless.
  assert.equal(
    toolActivityOpen(syncToolActivityPreference(collapsed, "expanded", false)),
    true,
  );
});

test("fallback cards preserve manual state until the preference changes", () => {
  const manuallyOpen = {
    visibility: "collapsed" as const,
    active: true,
    override: true,
  };
  // Reference identity, not deep equality: the render-phase `if (synced !==
  // state) setState(...)` in ToolFallbackRoot and ToolGroupRoot terminates only
  // because an unchanged preference and activity return the very same object.
  assert.equal(
    syncToolActivityPreference(manuallyOpen, "collapsed", true),
    manuallyOpen,
  );
  const manuallyClosed = {
    visibility: "expanded" as const,
    active: true,
    override: false,
  };
  assert.equal(
    syncToolActivityPreference(manuallyClosed, "expanded", true),
    manuallyClosed,
  );
});

test("switching to auto respects a card whose call has finished", () => {
  assert.equal(
    toolActivityOpen(
      syncToolActivityPreference(
        { visibility: "collapsed", active: false, override: null },
        "auto",
        false,
      ),
    ),
    false,
  );
});

test("an auto card closes itself when its call stops running", () => {
  // The gap this covers: an uncontrolled card mounted with defaultOpen={isRunning} used to open
  // and never close again, because an unchanged setting returned the state untouched.
  const running = { visibility: "auto" as const, active: true, override: null };
  assert.equal(toolActivityOpen(running), true);
  const finished = syncToolActivityPreference(running, "auto", false);
  assert.equal(toolActivityOpen(finished), false);
});

test("a hand-opened auto card survives its call finishing", () => {
  // Activity moving on its own must not discard a manual open, only the setting may.
  const opened = { visibility: "auto" as const, active: true, override: true };
  const finished = syncToolActivityPreference(opened, "auto", false);
  assert.equal(finished.override, true);
  assert.equal(toolActivityOpen(finished), true);
});

test("a hand-closed running card stays closed while it runs", () => {
  const closed = { visibility: "auto" as const, active: true, override: false };
  assert.equal(toolActivityOpen(closed), false);
  assert.equal(toolActivityOpen(syncToolActivityPreference(closed, "auto", true)), false);
});

test("always expanded opens a card that mounted with nothing running", () => {
  // Generic and MCP cards render with whatever their status says; expanded ignores it.
  const idle = { visibility: "expanded" as const, active: false, override: null };
  assert.equal(toolActivityOpen(idle), true);
});

// ---------------------------------------------------------------------------
// AST helpers for the .tsx claims.
// ---------------------------------------------------------------------------

const sourceOf = async (path: string): Promise<ts.SourceFile> =>
  ts.createSourceFile(
    path,
    await readText(path),
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
 * The local name a file binds the setting to, resolved through the store
 * selector rather than assumed. Hard-coding "visibility" would make these
 * assertions fail on a rename that changes nothing.
 */
function preferenceBinding(
  root: ts.SourceFile,
  scope: ts.Node = root,
): string {
  const declaration = find(scope, (node) => {
    if (!ts.isVariableDeclaration(node) || !node.initializer) return false;
    const init = node.initializer;
    return (
      ts.isCallExpression(init) &&
      ts.isIdentifier(init.expression) &&
      init.expression.text === "useChatPreferencesStore" &&
      identifiersIn(init).has("toolVisibility")
    );
  })[0] as ts.VariableDeclaration | undefined;
  assert.ok(
    declaration && ts.isIdentifier(declaration.name),
    `${root.fileName} never reads toolVisibility off the store`,
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
    identifiersIn(selector).has("toolVisibility"),
    "the hook subscribes to the store but not to this setting",
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
    "visibility",
    "previousVisibility",
    "isRunning",
    "hasText",
    // Without this the hook cannot clear the old round's manual state when a call re-runs.
    "startedNewRound",
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
    "the visibility setting can suppress an approval prompt's context",
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
  // Scoped to ToolGroupImpl: ToolGroupRoot reads the same setting above, for its own
  // uncontrolled state, and resolving to that binding would test the wrong component.
  const preference = preferenceBinding(
    source,
    initializerOf(source, "ToolGroupImpl"),
  );
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
  const root = find(
    source,
    (node) =>
      ts.isFunctionDeclaration(node) && node.name?.text === "ToolGroupRoot",
  )[0];
  assert.ok(root, "ToolGroupRoot is gone");
  const preference = preferenceBinding(source, root);
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

test("the generic fallback card tells its root whether the call is running", async () => {
  // Without this the root's defaultOpen stays false, so "Expand while running" could never
  // reach an unknown or MCP tool call.
  const source = await sourceOf(
    "../src/components/assistant-ui/tool-fallback.tsx",
  );
  const impl = initializerOf(source, "ToolFallbackImpl");
  const root = jsxElement(impl, "ToolFallbackRoot");
  const attribute = jsxAttribute(root, "defaultOpen");
  assert.ok(attribute, "the generic card mounts collapsed even while it runs");
  assert.ok(
    attribute.initializer &&
      ts.isJsxExpression(attribute.initializer) &&
      attribute.initializer.expression &&
      identifiersIn(attribute.initializer.expression).has("isToolCallRunning"),
    "the generic card pins defaultOpen instead of reading its live status",
  );
});

test("a tool group tells its root whether its own calls are running", async () => {
  const source = await sourceOf(
    "../src/components/assistant-ui/tool-group.tsx",
  );
  const impl = initializerOf(source, "ToolGroupImpl");
  const attribute = jsxAttribute(jsxElement(impl, "ToolGroupRoot"), "defaultOpen");
  assert.ok(attribute, "a running group mounts collapsed under auto");
  assert.ok(
    attribute.initializer &&
      ts.isJsxExpression(attribute.initializer) &&
      attribute.initializer.expression &&
      identifiersIn(attribute.initializer.expression).has("groupRunning"),
    "the group pins defaultOpen instead of reading its live activity",
  );
  // Scoped to the group's own calls, not the whole message, so it goes quiet once they finish.
  const running = initializerOf(source, "groupRunning");
  const names = identifiersIn(running);
  assert.ok(
    names.has("startIndex") && names.has("endIndex"),
    "the group's activity signal ignores which calls belong to it",
  );
  assert.ok(
    names.has("result"),
    "the group stays open for the whole turn instead of until its calls have results",
  );
});

test("the Python script cell moves inside the collapsible when tool calls are collapsed", async () => {
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

test("created files stay outside the collapsible on Python and Terminal cards", async () => {
  for (const file of [
    "../src/components/assistant-ui/tool-ui-python.tsx",
    "../src/components/assistant-ui/tool-ui-terminal.tsx",
  ]) {
    const source = await sourceOf(file);
    const root = jsxElement(source, "ToolFallbackRoot");
    const content = jsxElement(root, "ToolFallbackContent");
    const files = find(
      root,
      (node) =>
        (ts.isJsxSelfClosingElement(node) || ts.isJsxOpeningElement(node)) &&
        ts.isIdentifier(node.tagName) &&
        node.tagName.text === "SandboxFiles",
    );
    assert.equal(
      files.length,
      1,
      `${file} must render SandboxFiles once under ToolFallbackRoot`,
    );
    const insideContent = find(content, (node) => node === files[0]);
    assert.equal(
      insideContent.length,
      0,
      `${file} hid SandboxFiles inside ToolFallbackContent`,
    );
    // No wrapper element: an empty one would sit in the DOM of every card that created nothing.
    assert.equal(files[0].parent, root, `${file} wraps SandboxFiles`);
    const cls = (files[0] as ts.JsxSelfClosingElement).attributes.properties.find(
      (property): property is ts.JsxAttribute =>
        ts.isJsxAttribute(property) && property.name.getText() === "className",
    );
    assert.match(
      cls?.initializer?.getText() ?? "",
      /ml-5/,
      `${file} is missing ml-5, so the file row sits flush with the trigger`,
    );
  }
});

test("a call that created files keeps its group from collapsing", async () => {
  const { hasCreatedFiles } = await import(
    "../src/components/assistant-ui/sandbox-files.ts"
  );
  const wrapped = (files: unknown) => ({
    text: "",
    images: [],
    sessionId: "s",
    files,
  });
  assert.equal(hasCreatedFiles("terminal", wrapped([{ name: "a.txt", size: 3 }])), true);
  assert.equal(hasCreatedFiles("python", wrapped([{ name: "a.txt", size: null }])), true);
  assert.equal(hasCreatedFiles("terminal", wrapped([])), false);
  assert.equal(hasCreatedFiles("terminal", wrapped(undefined)), false);
  assert.equal(hasCreatedFiles("terminal", "plain output"), false);
  assert.equal(
    hasCreatedFiles("mcp__fs__write", wrapped([{ name: "a.txt", size: 3 }])),
    false,
  );

  // The rule lives in tool-fold-exemptions.ts; the group asks it per part.
  const group = await sourceOf(
    "../src/components/assistant-ui/tool-group.tsx",
  );
  assert.match(
    initializerOf(group, "containsUngroupedTool").getText(),
    /\.some\(holdsOwnOutput\)/,
  );
  const source = await sourceOf(
    "../src/components/assistant-ui/tool-fold-exemptions.ts",
  );
  const call = find(
    source,
    (node) =>
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "hasCreatedFiles",
  )[0] as ts.CallExpression | undefined;
  assert.ok(call, "a grouped terminal call hides its created files again");
  assert.deepEqual(
    call.arguments.map((argument) => argument.getText()),
    ["part.toolName", "part.result"],
  );
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc, installLocalStorageFake } from "./helpers/kit.ts";

const page = ts.createSourceFile(
  "chat-page.tsx",
  readSrc("features/chat/chat-page.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const composer = ts.createSourceFile(
  "shared-composer.tsx",
  readSrc("features/chat/shared-composer.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
function matches(root: ts.Node, predicate: (n: ts.Node) => boolean) {
  const found: ts.Node[] = [];
  function visit(n: ts.Node) {
    if (predicate(n)) found.push(n);
    ts.forEachChild(n, visit);
  }
  visit(root);
  return found;
}
function variable(root: ts.Node, name: string, file: ts.SourceFile) {
  const found = matches(
    root,
    (n) => ts.isVariableDeclaration(n) && n.name.getText(file).includes(name),
  );
  assert.ok(found.length <= 1, name);
  return found[0] as ts.VariableDeclaration | undefined;
}
function run(code: string, deps: Record<string, unknown>) {
  const { outputText } = ts.transpileModule(code, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  return new Function(...Object.keys(deps), outputText)(...Object.values(deps));
}
const submitted = variable(composer, "sentPairId", composer);
const unsent = variable(composer, "unsentPair", composer)!;
const key = variable(composer, "draftKey", composer)!;
const clear = variable(composer, "clearSubmittedDraft", composer)!;

for (const name of ["LoraCompareContent", "GeneralCompareContent"]) {
  for (const initialSaved of [false, undefined]) {
    test(`${name} (saved=${initialSaved}) keeps the submitted pair scope when its active-only composer remounts`, async () => {
      installLocalStorageFake();
      const drafts = await import(
        `../src/features/chat/utils/composer-draft.ts?remount=${name}`
      );
      const owner = matches(
        page,
        (n) => ts.isFunctionExpression(n) && n.name?.text === name,
      )[0];
      assert.ok(owner);
      const marker = variable(owner, "submittedPairId", page);
      const prop = (name: string) => {
        const node = matches(
          owner,
          (n) => ts.isJsxAttribute(n) && n.name.getText(page) === name,
        )[0] as ts.JsxAttribute | undefined;
        return node
          ? (node.initializer as ts.JsxExpression).expression!.getText(page)
          : "undefined";
      };
      let parentState: unknown = null;
      let childState: unknown = null;
      let pairId = "new-pair";
      let pairSaved = initialSaved;
      const render = () => {
        // The content stays mounted; only its active-only composer gets a fresh state slot.
        const props = run(
          `${marker ? `const ${marker.getText(page)};` : ""}
        return {pairSubmitted: ${prop("pairSubmitted")}, onPairSubmitted: ${prop("onPairSubmitted")}};`,
          {
            pairId,
            useState: () => [
              parentState,
              (value: unknown) => {
                parentState = value;
              },
            ],
          },
        );
        return run(
          `${submitted ? `const ${submitted.getText(composer)};` : ""}
        const ${unsent.getText(composer)};
        const ${key.getText(composer)};
        const ${clear.getText(composer)};
        return {draftKey, clearSubmittedDraft};`,
          {
            ...drafts,
            ...props,
            draftId: pairId,
            pairSaved,
            shareDraft: false,
            useState: () => [
              childState,
              (value: unknown) => {
                childState = value;
              },
            ],
            unsavedDraftRef: { current: null },
            setCurrentText: () => {},
            setIsWritingExpanded: () => {},
            setPendingImages: () => {},
            setPendingAudio: () => {},
            clearPendingAudioStore: () => {},
            textareaRef: { current: null },
          },
        );
      };
      let state = render();
      assert.equal(
        state.draftKey,
        drafts.composerDraftKey(
          initialSaved === false ? drafts.NEW_COMPARE_DRAFT_ID : pairId,
        ),
      );
      drafts.writeComposerDraft(state.draftKey, "first prompt");
      state.clearSubmittedDraft();
      pairSaved = false; // A history read can finish after the first submission.
      state = render();
      assert.equal(state.draftKey, drafts.composerDraftKey(pairId));
      drafts.writeComposerDraft(
        state.draftKey,
        "follow-up during the first run",
      );
      childState = null; // Leaving another page unmounts SharedComposer, not the content owner.
      state = render();
      assert.equal(state.draftKey, drafts.composerDraftKey(pairId));
      assert.equal(
        drafts.latestDraft(state.draftKey),
        "follow-up during the first run",
      );
      pairId = "different-new-pair";
      childState = null;
      assert.equal(
        render().draftKey,
        drafts.composerDraftKey(drafts.NEW_COMPARE_DRAFT_ID),
      );
    });
  }
}

const restoreEffect = matches(
  composer,
  (n) => ts.isCallExpression(n) && n.expression.getText(composer) === "useEffect" &&
    n.arguments[0]?.getText(composer).includes("draftScopeBeforeRef.current"),
)[0] as ts.CallExpression;
assert.ok(restoreEffect);

for (const olderSharedDraft of ["", "older shared draft"]) {
  test(`a reloaded Compare draft is adopted when sharing is enabled (older=${Boolean(olderSharedDraft)})`, async () => {
    const { store } = installLocalStorageFake();
    const drafts = await import(`../src/features/chat/utils/composer-draft.ts?adoption=${Math.random()}`);
    store.set(drafts.composerDraftKey(drafts.NEW_COMPARE_DRAFT_ID), "restored Compare prompt");
    if (olderSharedDraft) store.set(drafts.composerDraftKey("", true), olderSharedDraft);
    let pairSaved: boolean | undefined;
    let shareDraft = false;
    let text = "";
    const remoteDraftRef = { current: null };
    const unsavedDraftRef = { current: null };
    const draftScopeBeforeRef = { current: drafts.composerDraftKey("pair") };
    const draftOwnerBeforeRef = { current: { draftId: "pair", shareDraft: false, draftKey: draftScopeBeforeRef.current } };
    const render = () => {
      // The second commit is the state update caused by restoring the selected slot.
      for (let pass = 0; pass < 5; pass++) {
        const beforeText = text;
        run(`const ${unsent.getText(composer)};
          const ${key.getText(composer)};
          const draftScope = draftKey;
          return (${restoreEffect.arguments[0].getText(composer)})();`, {
          ...drafts, draftId: "pair", pairSaved, pairSubmitted: false, shareDraft,
          incognito: false, text, remoteDraftRef, unsavedDraftRef,
          draftScopeBeforeRef, draftOwnerBeforeRef, draftSource: Symbol("compare"),
          setText: (next: string) => { text = next; },
          setTimeout: () => 0, clearTimeout: () => {},
        });
        if (text === beforeText) return;
      }
      assert.fail("Compare draft restore did not settle");
    };
    render();
    pairSaved = false;
    render();
    assert.equal(text, "restored Compare prompt");
    assert.equal(drafts.adoptCurrentDraft(), true);
    shareDraft = true;
    render();
    assert.equal(text, "restored Compare prompt");
    assert.equal(drafts.latestDraft(), "restored Compare prompt");
    assert.equal(drafts.latestDraft(drafts.composerDraftKey(drafts.NEW_COMPARE_DRAFT_ID)), "");
  });
}

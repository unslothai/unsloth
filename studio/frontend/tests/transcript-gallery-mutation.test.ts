// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

test("pending transcript mutations use the current view and deletion callback", async () => {
  const source = readSrc("features/audio/transcript-gallery.tsx");
  const tree = ts.createSourceFile(
    "transcript-gallery.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const component = tree.statements.find(
    (node): node is ts.FunctionDeclaration =>
      ts.isFunctionDeclaration(node) && node.name?.text === "TranscriptGallery",
  );
  assert.ok(component?.body);
  const statements = [];
  for (const statement of component.body.statements) {
    if (ts.isReturnStatement(statement)) break;
    statements.push(statement.getText(tree));
  }
  const { outputText } = ts.transpileModule(
    statements.join("\n") + "\nreturn { mutate, refresh, setArchived };",
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  for (const { initialView, nextView, removed } of [
    { initialView: false, nextView: true, removed: ["deleted"] },
    { initialView: true, nextView: false, removed: ["deleted"] },
    { initialView: false, nextView: false, removed: ["deleted"] },
    { initialView: false, nextView: true, removed: null },
  ]) {
    const state: unknown[] = [];
    const refs: { current: unknown }[] = [];
    const requests: boolean[] = [];
    const deletions: unknown[] = [];
    let stateIndex = 0;
    let refIndex = 0;
    const scope = {
      active: true,
      currentId: "deleted",
      latest: null,
      autoSelect: false,
      onSelect: () => {},
      onDelete: (ids: unknown) => deletions.push(["stale", ids]),
      useState: (initial: unknown) => {
        const index = stateIndex++;
        if (!(index in state)) state[index] = initial;
        return [
          state[index],
          (value: unknown) => {
            state[index] = value;
          },
        ];
      },
      useRef: (initial: unknown) => refs[refIndex++] ??= { current: initial },
      useCallback: (callback: unknown) => callback,
      useLayoutEffect: (effect: () => void) => effect(),
      useEffect: () => {},
      listTranscripts: async (archived: boolean) => {
        requests.push(archived);
        return { transcripts: [{ id: String(archived) }], next_cursor: null };
      },
      toast: { error: (message: string) => assert.fail(message) },
    };
    const render = () => {
      stateIndex = 0;
      refIndex = 0;
      return new Function(...Object.keys(scope), outputText)(
        ...Object.values(scope),
      );
    };
    render().setArchived(initialView);
    const before = render();
    let finish!: () => void;
    const request = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const mutation = before.mutate(() => request, removed);
    before.setArchived(nextView);
    scope.onDelete = (ids: unknown) => deletions.push(["current", ids]);
    if (removed === null) scope.currentId = "retained-archived";
    const after = render();
    await after.refresh();
    finish();
    await mutation;
    assert.deepEqual(requests, [nextView, nextView]);
    assert.deepEqual(state[0], [{ id: String(nextView) }]);
    assert.deepEqual(deletions, [["current", removed]]);
  }
});

test("a failed view switch does not leave the other view's rows on screen", async () => {
  // `records`/`cursor` are written only by refresh's success path, so a rejected refresh
  // used to leave the previous view's rows under the new heading, with the wrong cursor.
  const source = readSrc("features/audio/transcript-gallery.tsx");
  const tree = ts.createSourceFile(
    "transcript-gallery.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const component = tree.statements.find(
    (node): node is ts.FunctionDeclaration =>
      ts.isFunctionDeclaration(node) && node.name?.text === "TranscriptGallery",
  );
  assert.ok(component?.body);
  const statements = [];
  for (const statement of component.body.statements) {
    if (ts.isReturnStatement(statement)) break;
    statements.push(statement.getText(tree));
  }
  const { outputText } = ts.transpileModule(
    statements.join("\n") + "\nreturn { refresh, setArchived };",
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );

  const state: unknown[] = [];
  const refs: { current: unknown }[] = [];
  let stateIndex = 0;
  let refIndex = 0;
  let failNext = false;
  const errors: string[] = [];
  const scope = {
    active: true,
    currentId: null,
    latest: null,
    autoSelect: false,
    onSelect: () => {},
    onDelete: () => {},
    useState: (initial: unknown) => {
      const index = stateIndex++;
      if (!(index in state)) state[index] = initial;
      return [
        state[index],
        (value: unknown) => {
          state[index] = value;
        },
      ];
    },
    useRef: (initial: unknown) => refs[refIndex++] ??= { current: initial },
    useCallback: (callback: unknown) => callback,
    useLayoutEffect: (effect: () => void) => effect(),
    useEffect: () => {},
    listTranscripts: async (archived: boolean) => {
      if (failNext) throw new Error("Could not load transcripts.");
      return {
        transcripts: [{ id: String(archived) }],
        next_cursor: `cursor-${archived}`,
      };
    },
    toast: { error: (message: string) => errors.push(message) },
  };
  const render = () => {
    stateIndex = 0;
    refIndex = 0;
    return new Function(...Object.keys(scope), outputText)(
      ...Object.values(scope),
    );
  };

  // History loads normally.
  render().setArchived(false);
  await render().refresh();
  assert.deepEqual(state[0], [{ id: "false" }], "History rows should be loaded");
  assert.equal(state[2], "cursor-false");

  // Switch to Archived; that refresh fails.
  failNext = true;
  render().setArchived(true);
  await render().refresh();

  assert.deepEqual(errors, ["Could not load transcripts."]);
  assert.deepEqual(
    state[0],
    [],
    "the Archived view must not render the History rows it failed to replace",
  );
  assert.equal(
    state[2],
    null,
    "the History cursor must not survive into the Archived view, or Load more mixes them",
  );
});

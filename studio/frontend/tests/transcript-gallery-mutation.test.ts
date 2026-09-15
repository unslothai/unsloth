// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

test("pending transcript mutations refresh the current archive view", async () => {
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
  for (const [initialView, nextView] of [
    [false, true],
    [true, false],
    [false, false],
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
      onDelete: (ids: unknown) => deletions.push(ids),
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
    const mutation = before.mutate(() => request, ["deleted"]);
    before.setArchived(nextView);
    const after = render();
    await after.refresh();
    finish();
    await mutation;
    assert.deepEqual(requests, [nextView, nextView]);
    assert.deepEqual(state[0], [{ id: String(nextView) }]);
    assert.deepEqual(deletions, [["deleted"]]);
  }
});

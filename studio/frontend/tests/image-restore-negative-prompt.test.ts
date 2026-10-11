// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

function restoreSettingsWith(negativeCapable: boolean) {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile(
    "images-page.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let declaration = "";
  const visit = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(tree) === "restoreSettings"
    )
      declaration = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declaration);
  const { outputText } = ts.transpileModule(
    `${declaration}\nreturn restoreSettings;`,
    {
      compilerOptions: { target: ts.ScriptTarget.ES2022 },
    },
  );
  const state: Record<string, unknown> = {};
  const known: Record<string, unknown> = {
    useCallback: (fn: unknown) => fn,
    negativeCapable,
    sizeLimits: { maxSide: 2048 },
    MIN_DIM: 256,
    restorableSize: (width: number, height: number) => ({ width, height }),
    matchAspect: () => ({ key: "1:1", portrait: false }),
    restoreInputsNote: () => null,
    CONDITIONED_WORKFLOW_INPUTS: {},
    toast: { success: () => {} },
  };
  const scope = new Proxy(known, {
    has: (target, key) =>
      typeof key === "string" && (key in target || /^set[A-Z]/.test(key)),
    get: (target, key) =>
      typeof key === "string" && key in target
        ? target[key]
        : (...args: unknown[]) => {
            state[key as string] = args.length > 1 ? args : args[0];
          },
  });
  const restore = new Function("scope", `with (scope) { ${outputText} }`)(
    scope,
  );
  return { restore, state };
}

test("restoring an image keeps its negative prompt while a model that ignores one is loaded", () => {
  const { restore, state } = restoreSettingsWith(false);
  restore({
    prompt: "a lighthouse",
    negative_prompt: "blurry, text",
    guidance: 5,
    steps: 30,
    seed: 7,
    width: 1024,
    height: 1024,
    workflow: "create",
  });
  assert.equal(state.setNegativePrompt, "blurry, text");
});

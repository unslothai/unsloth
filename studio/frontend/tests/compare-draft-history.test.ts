// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";
import {
  composerDraftKey,
  NEW_COMPARE_DRAFT_ID,
} from "../src/features/chat/utils/composer-draft.ts";

const source = ts.createSourceFile(
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

function find(root: ts.Node, predicate: (n: ts.Node) => boolean): ts.Node {
  const nodes: ts.Node[] = [];
  function visit(node: ts.Node) {
    if (predicate(node)) nodes.push(node);
    ts.forEachChild(node, visit);
  }
  visit(root);
  assert.equal(nodes.length, 1);
  return nodes[0];
}

function evaluate(code: string, deps: Record<string, unknown>) {
  const js = ts.transpileModule(code, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  return new Function(...Object.keys(deps), js)(...Object.values(deps));
}

const unsentPair = find(
  composer,
  (n) =>
    ts.isVariableDeclaration(n) && n.name.getText(composer) === "unsentPair",
) as ts.VariableDeclaration;
const draftKey = find(
  composer,
  (n) => ts.isVariableDeclaration(n) && n.name.getText(composer) === "draftKey",
) as ts.VariableDeclaration;

type Thread = { id: string; modelType: string; modelId?: string };
for (const name of ["LoraCompareContent", "GeneralCompareContent"]) {
  const component = find(
    source,
    (n) => ts.isFunctionExpression(n) && n.name?.text === name,
  );
  const effect = find(
    component,
    (n) =>
      ts.isCallExpression(n) &&
      n.expression.getText(source) === "useEffect" &&
      n.arguments[0]
        ?.getText(source)
        .includes("listStoredChatThreads({ pairId })"),
  ) as ts.CallExpression;
  const prop = find(
    component,
    (n) => ts.isJsxAttribute(n) && n.name.getText(source) === "pairSaved",
  ) as ts.JsxAttribute;
  const expression = (prop.initializer as ts.JsxExpression).expression!;

  for (const scenario of [
    "failed",
    "empty",
    "saved",
    "failed-known-empty",
    "failed-known-saved",
  ] as const) {
    test(`${name}: ${scenario} history selects the correct draft scope`, async () => {
      let resolve!: (threads: Thread[]) => void;
      let reject!: (reason: Error) => void;
      const read = new Promise<Thread[]>((yes, no) => {
        resolve = yes;
        reject = no;
      });
      const state: Record<string, unknown> = {
        threadsSettled: false,
        pairSaved:
          scenario === "failed-known-empty"
            ? false
            : scenario === "failed-known-saved"
              ? true
              : undefined,
        baseThreadId: undefined,
        loraThreadId: undefined,
        model1ThreadId: undefined,
        model2ThreadId: undefined,
      };
      const cleanup = evaluate(
        `return (${effect.arguments[0].getText(source)})();`,
        {
          pairId: "saved-pair",
          anyRunning: false,
          comparing: false,
          listedPairRef: {
            current: scenario.startsWith("failed-known") ? "saved-pair" : null,
          },
          listStoredChatThreads: () => read,
          isExpectedBackgroundChatStorageError: () => true,
          resolveComparePaneThreadIds: (threads: Thread[]) => ({
            first: threads.find((t) => t.modelType === "model1")?.id,
            second: threads.find((t) => t.modelType === "model2")?.id,
          }),
          setThreadsSettled: (value: boolean) => {
            state.threadsSettled = value;
          },
          setPairSaved: (value: boolean | undefined) => {
            state.pairSaved = value;
          },
          setBaseThreadId: (value?: string) => {
            state.baseThreadId = value;
          },
          setLoraThreadId: (value?: string) => {
            state.loraThreadId = value;
          },
          setModel1ThreadId: (value?: string) => {
            state.model1ThreadId = value;
          },
          setModel2ThreadId: (value?: string) => {
            state.model2ThreadId = value;
          },
          setPairLoraModelId: () => {},
        },
      );
      if (scenario.startsWith("failed"))
        reject(new Error("history unavailable"));
      else
        resolve(
          scenario === "empty"
            ? []
            : [
                {
                  id: "pane",
                  modelType: name === "LoraCompareContent" ? "base" : "model1",
                },
              ],
        );
      await read.catch(() => {});
      await Promise.resolve();
      await Promise.resolve();
      const pairSaved = evaluate(
        `return (${expression.getText(source)});`,
        state,
      );
      assert.equal(
        pairSaved,
        scenario === "failed"
          ? undefined
          : scenario === "saved" || scenario === "failed-known-saved",
      );
      const key = evaluate(
        `const unsentPair = ${unsentPair.initializer!.getText(composer)}; return ${draftKey.initializer!.getText(composer)};`,
        {
          pairSaved,
          draftId: "saved-pair",
          sentPairId: null,
          pairSubmitted: false,
          shareDraft: false,
          composerDraftKey,
          NEW_COMPARE_DRAFT_ID,
        },
      );
      assert.equal(
        key,
        composerDraftKey(
          scenario === "empty" || scenario === "failed-known-empty"
            ? NEW_COMPARE_DRAFT_ID
            : "saved-pair",
        ),
      );
      assert.equal(
        state.threadsSettled,
        true,
        "failed history still releases shell readiness",
      );
      cleanup();
    });
  }
}

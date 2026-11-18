// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";

const file = ts.createSourceFile(
  "shared-composer.tsx",
  readSrc("features/chat/shared-composer.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function find(root: ts.Node, predicate: (n: ts.Node) => boolean): ts.Node[] {
  const found: ts.Node[] = [];
  const visit = (n: ts.Node) => {
    if (predicate(n)) found.push(n);
    ts.forEachChild(n, visit);
  };
  visit(root);
  return found;
}

const composer = find(
  file,
  (n) => ts.isFunctionDeclaration(n) && n.name?.text === "SharedComposer",
)[0]!;
const declarations = find(
  composer,
  (n) =>
    ts.isVariableStatement(n) &&
    n.declarationList.declarations.some(
      (d) => ts.isIdentifier(d.name) && ["activeThreadId", "exportThreadIds"].includes(d.name.text),
    ),
).map((n) => n.getText(file));
const exportChat = find(
  composer,
  (n) => ts.isPropertyAssignment(n) && n.name.getText(file) === "exportChat",
)[0]!;
const onSelect = find(
  exportChat,
  (n) => ts.isJsxAttribute(n) && n.name.getText(file) === "onSelect",
)[0] as ts.JsxAttribute;
const handler = (onSelect.initializer as ts.JsxExpression).expression!.getText(file);

async function exportFromCompare(activeThreadId: string | null) {
  const exported: string[] = [];
  const code = `${declarations.join("\n")}\nreturn (${handler});`;
  const { outputText } = ts.transpileModule(code, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  const deps = {
    model1ThreadId: "pane-1",
    model2ThreadId: "pane-2",
    useChatRuntimeStore: (select: (s: { activeThreadId: string | null }) => unknown) =>
      select({ activeThreadId }),
    composerExportFormats: () => [
      { label: "Markdown", fn: async (id: string) => void exported.push(id) },
    ],
    toast: { error: () => {} },
    isDownloadCancelled: () => false,
  };
  const select = new Function(...Object.keys(deps), outputText)(...Object.values(deps));
  select("Markdown");
  await new Promise((resolve) => setTimeout(resolve, 0));
  return exported;
}

test("exporting a reopened Compare downloads only its two panes, not the chat open before it", async () => {
  assert.deepEqual(await exportFromCompare("earlier-private-chat"), ["pane-1", "pane-2"]);
});

test("exporting a Compare with no other chat open downloads both panes", async () => {
  assert.deepEqual(await exportFromCompare(null), ["pane-1", "pane-2"]);
});

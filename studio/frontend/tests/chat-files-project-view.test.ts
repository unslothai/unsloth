// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";

import { stubJsxRuntime, type StubElement } from "./helpers/module-stubs.ts";

type View = { mode: "project" | "single" | "compare"; projectId?: string; threadId?: string };

const source = ts.createSourceFile(
  "chat-page.tsx",
  readFileSync(new URL("../src/features/chat/chat-page.tsx", import.meta.url), "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function findNode(predicate: (node: ts.Node) => boolean): ts.Node {
  let result: ts.Node | undefined;
  function visit(node: ts.Node) {
    if (!result && predicate(node)) result = node;
    if (!result) ts.forEachChild(node, visit);
  }
  visit(source);
  assert.ok(result, "expected the chat Files wiring in chat-page.tsx");
  return result;
}

function evaluate<T>(body: string, bindings: Record<string, unknown>): T {
  const { outputText } = ts.transpileModule(body, {
    fileName: "chat-files-fragment.tsx",
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
    },
  });
  return new Function("require", "exports", ...Object.keys(bindings), outputText)(
    () => stubJsxRuntime(),
    {},
    ...Object.values(bindings),
  ) as T;
}

function count(value: unknown, type: string): number {
  if (Array.isArray(value)) return value.reduce((sum, child) => sum + count(child, type), 0);
  if (!value || typeof value !== "object" || !("props" in value)) return 0;
  const node = value as StubElement;
  return (node.type === type ? 1 : 0) + count(node.props.children, type);
}

function headerMenu(view: View, projectChatShownId: string | null): unknown {
  const menu = findNode(
    (node) =>
      ts.isJsxExpression(node) &&
      ts.isJsxElement(node.parent) &&
      node.getText(source).includes("<ChatFilesMenu />"),
  ) as ts.JsxExpression;
  assert.ok(menu.expression);
  return evaluate(`return (${menu.expression.getText(source)});`, {
    view,
    projectChatShownId,
    ChatFilesMenu: "files-menu",
  });
}

function bridges(view: View): number {
  const runtime = findNode(
    (node) =>
      ts.isJsxElement(node) &&
      node.openingElement.tagName.getText(source) === "ChatRuntimeProvider" &&
      node.openingElement.attributes.properties.some(
        (prop) =>
          ts.isJsxAttribute(prop) &&
          prop.name.getText(source) === "modelType" &&
          prop.initializer?.getText(source) === '"base"',
      ),
  );
  const rendered = evaluate<StubElement>(`return (${runtime.getText(source)});`, {
    ChatRuntimeProvider: "runtime",
    baseView: view,
    projectNewThreadNonce: "draft",
    baseBackgrounded: false,
    markProjectRuntimeReady: () => {},
    handleCompactionCountChange: () => {},
    NativeAttachmentTargetContext: { Provider: "target" },
    NativeAttachmentOwnerScopeContext: { Provider: "owner" },
    baseAttachmentOwnerScope: null,
    NativeAttachmentDragContext: { Provider: "drag" },
    baseAttachmentTargetKey: "key",
    nativeModelDropState: { status: "idle" },
    ProjectLanding: "project",
    SingleContent: "single",
    TemporaryChatSaveBridge: "temporary-save",
    ChatFilesBridge: "chat-files",
    currentProject: { name: "Project" },
    currentProjectItems: [],
    rotateProjectNewThreadNonce: () => {},
    currentProjectItemsLoaded: true,
    projectsLoading: false,
    projectRuntimeReady: true,
  });
  return count(rendered, "chat-files");
}

test("a chat started on a project page collects its files", () => {
  assert.equal(bridges({ mode: "project", projectId: "p1" }), 1);
  assert.equal(bridges({ mode: "single", threadId: "t1" }), 1);
});

test("the Files menu shows for a project chat, not for the empty project page or Compare", () => {
  assert.ok(headerMenu({ mode: "project", projectId: "p1" }, "t1"));
  // An outgoing chat's id can linger in the store over an empty project page.
  assert.equal(headerMenu({ mode: "project", projectId: "p1" }, null), null);
  assert.ok(headerMenu({ mode: "single", threadId: "t1" }, "t1"));
  assert.ok(headerMenu({ mode: "single" }, null));
  assert.equal(headerMenu({ mode: "compare" }, "t1"), null);
});

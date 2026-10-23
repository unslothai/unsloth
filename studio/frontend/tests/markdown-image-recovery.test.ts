// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { runInNewContext } from "node:vm";
import ts from "@typescript/typescript6";

import { readSrc } from "./helpers/kit.ts";

function renderer(
  backendPathUrl = (url: string) => url,
  desktopSignedMediaPath = (_url: string): string | null => null,
  sandboxState: Record<string, unknown> = { status: "idle" },
  protectedBackendPath = (_url: string): string | null => null,
) {
  const source = readSrc("components/assistant-ui/markdown-text.tsx");
  const parsed = ts.createSourceFile(
    "markdown-text.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const statement = parsed.statements.find(
    (node) =>
      ts.isVariableStatement(node) &&
      node.declarationList.declarations.some(
        (decl) => decl.name.getText(parsed) === "MarkdownImage",
      ),
  );
  assert.ok(statement, "production MarkdownImage declaration exists");
  const code = ts.transpileModule(
    statement.getText(parsed) + "\nexports.render = MarkdownImage;",
    {
      compilerOptions: {
        jsx: ts.JsxEmit.React,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  const slots: unknown[] = [];
  let cursor = 0;
  const context = {
    exports: {} as { render: (props: Record<string, unknown>) => any },
    memo: (fn: unknown) => fn,
    React: {
      createElement: (
        type: unknown,
        props: unknown,
        ...children: unknown[]
      ) => ({ type, props, children }),
    },
    useState: (initial: unknown) => {
      const index = cursor++;
      if (!(index in slots)) slots[index] = initial;
      return [
        slots[index],
        (next: unknown) => {
          slots[index] = next;
        },
      ];
    },
    useAuiState: () => "thread",
    useChatRuntimeStore: () => "thread",
    useChatProjectScope: () => null,
    markdownSandboxImageSrc: () => null,
    backendPathUrl,
    desktopSignedMediaPath,
    protectedBackendPath,
    useSandboxImage: () => ({ state: sandboxState }),
    HugeiconsIcon: "icon",
    Download01Icon: "download",
  };
  runInNewContext(code, context);
  return (props: Record<string, unknown>) => {
    cursor = 0;
    return context.exports.render(props).children[0].props;
  };
}

test("a replacement image is not hidden by the previous source's decode failure", () => {
  const render = renderer();
  render({ src: "data:image/png;base64,broken" }).onError({});
  assert.match(
    render({ src: "data:image/png;base64,broken" }).className,
    /hidden/,
  );
  assert.doesNotMatch(
    render({ src: "data:image/png;base64,replacement" }).className,
    /hidden/,
  );
});

test("successful image load clears decode failure and forwards image events", () => {
  const render = renderer();
  let errors = 0;
  let loads = 0;
  const props = {
    src: "blob:test",
    onError: () => {
      errors++;
    },
    onLoad: () => {
      loads++;
    },
  };
  render(props).onError({});
  assert.match(render(props).className, /hidden/);
  render(props).onLoad({});
  assert.doesNotMatch(render(props).className, /hidden/);
  assert.equal(errors, 1);
  assert.equal(loads, 1);
});

test("a backend image keeps its own src, never the loopback URL Desktop's img-src blocks", () => {
  const render = renderer((url) => `http://127.0.0.1:8888${url}`);
  const src = "/api/inference/images/gallery/img-1/file-signed?token=t";
  assert.equal(render({ src }).src, src);
});

test("on Desktop a signed media image renders from its fetched blob", () => {
  const src = "/api/inference/images/gallery/img-1/file-signed?token=t";
  const render = renderer(
    (url) => url,
    (url) => url,
    { status: "loaded", url: "blob:media", blob: {} },
  );
  assert.equal(render({ src }).src, "blob:media");
});

test("an unsigned backend image renders from the session fetch, not a bare src that 401s", () => {
  const src = "/api/inference/images/gallery/img-1/file";
  const render = renderer(
    (url) => url,
    () => null,
    { status: "loaded", url: "blob:gallery", blob: {} },
    (url) => url,
  );
  assert.equal(render({ src }).src, "blob:gallery");
});

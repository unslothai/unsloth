// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { runInNewContext } from "node:vm";
import ts from "@typescript/typescript6";

import * as apiBase from "../src/lib/api-base.ts";
import { readSrc } from "./helpers/kit.ts";

function renderer(stubs: Record<string, unknown> = {}) {
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
    // The real URL helpers, so a new one MarkdownImage starts calling is defined here too.
    ...apiBase,
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
    useSandboxImage: () => ({ state: { status: "idle" } }),
    HugeiconsIcon: "icon",
    Download01Icon: "download",
    backendPathUrl: (url: string) => url,
    desktopSignedMediaPath: () => null,
    ...stubs,
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

test("a backend image never takes the loopback src Desktop blocks; a fetched one is a blob", () => {
  const signed = "/api/inference/images/gallery/img-1/file-signed?token=t";
  const loopback = (url: string) => `http://127.0.0.1:8888${url}`;
  const render = renderer({ backendPathUrl: loopback });
  assert.equal(render({ src: signed }).src, signed);
  const state = { status: "loaded", url: "blob:media", blob: {} };
  const useSandboxImage = () => ({ state });
  // Signed media on Desktop is fetched as a blob.
  const desktop = renderer({ useSandboxImage, desktopSignedMediaPath: (u: string) => u });
  assert.equal(desktop({ src: signed }).src, "blob:media");
  // An unsigned gallery file needs the session, so it is fetched too, never handed to <img>.
  const fetched = renderer({ useSandboxImage, backendPathUrl: loopback });
  assert.equal(fetched({ src: "/api/inference/images/gallery/img-1/file" }).src, "blob:media");
  // Any other backend route is not loaded at all (#19197): its query could reach a third party.
  const other = renderer({ useSandboxImage, backendPathUrl: loopback });
  assert.equal(other({ src: "/api/hub/modelscope/api/models?search=x" }).src, undefined);
});

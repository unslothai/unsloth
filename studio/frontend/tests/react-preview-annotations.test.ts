// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import React from "react";
import ts from "@typescript/typescript6";

const read = (path: string) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
function component(source: string, context: Record<string, unknown>) {
  const js = ts.transpileModule(`(${source})`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React },
  }).outputText;
  return vm.runInNewContext(js, { React, useEffect: () => {}, ...context });
}

test("a rendered React file registers its artifact iframe and returns the cleanup", () => {
  const source = read("features/browser/file-view.tsx");
  const cleanup = () => {};
  const frame = {} as HTMLIFrameElement;
  const registered: unknown[][] = [];
  const ReactPreview = () => null;
  const TextFile = component(source.slice(source.indexOf("function TextFile("), source.indexOf("export function FileView(")), {
    useState: (initial: unknown) => [initial === null ? "demo" : initial],
    useBrowserStore: (select: (state: unknown) => unknown) => select({ fileViews: { tab1: { mode: "preview" } } }),
    textFileKind: () => "react",
    useCallback: (callback: unknown) => callback,
    useMemo: (callback: () => unknown) => callback(),
    MAX_HIGHLIGHT_CHARS: 1000,
    truncateAttachmentPreviewText: (text: string) => ({ text }),
    attachmentTextLanguage: () => null,
    cn: () => "",
    stageEditsPrompt: () => {},
    ReactPreview,
    registerArtifactFrame: (...args: unknown[]) => { registered.push(args); return cleanup; },
  });
  const tree = TextFile({ blob: {}, name: "Demo.tsx", tabId: "tab1", scale: 1, reloadNonce: 0 });
  const preview = tree.props.children[0].props.children;
  assert.equal(preview.type, ReactPreview);
  assert.equal(preview.props.onFrameReady(frame), cleanup);
  assert.deepEqual(registered, [["tab1", frame]]);
});

test("ReactPreview passes frame registration through to ArtifactHtmlFrame", () => {
  const source = read("features/chat/artifacts/react-preview/react-preview.tsx");
  const ArtifactHtmlFrame = () => null;
  const onFrameReady = () => () => {};
  const ReactPreview = component(source.slice(source.indexOf("export function ReactPreview")).replace("export function", "function"), {
    useT: () => () => "",
    useState: () => [{ key: "tsx\u0000demo", prepared: { status: "ready", html: "<h1>Demo</h1>" } }],
    ArtifactHtmlFrame,
  });
  const frame = ReactPreview({ source: "demo", lang: "tsx", title: "Demo", onFrameReady });
  assert.equal(frame.type, ArtifactHtmlFrame);
  assert.equal(frame.props.onFrameReady, onFrameReady);
});

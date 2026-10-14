// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";

import type * as TriggerModule from "../src/features/rag/components/rag-document-trigger.tsx";
import { type StubElement, loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

const LocalFileDialog = () => null;
const env = { panel: true, filesInBrowser: true };
const opened: string[] = [];

const { RagDocumentTrigger } = loadWithStubs<typeof TriggerModule>(
  new URL("../src/features/rag/components/rag-document-trigger.tsx", import.meta.url),
  {
    react: React,
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/assistant-ui/attachment-document-dialog": { LocalFileDialog },
    "@/components/file-viewer": { MAX_DOCUMENT_PREVIEW_BYTES: 1 },
    // What the browser's FileView renders by name: documents, text and HTML, not EPUB or RTF.
    "@/features/browser/file-kind": {
      canShowFile: (name: string) => /\.(pdf|docx|md|txt|html)$/.test(name),
    },
    "@/features/browser/panel-availability": { useBrowserPanelAvailable: () => env.panel },
    "@/features/browser/prefs-store": {
      useBrowserPrefsStore: (select: (s: { openFilesInBrowser: boolean }) => unknown) =>
        select({ openFilesInBrowser: env.filesInBrowser }),
    },
    "../lib/open-document": {
      openRagDocumentInBrowser: (_id: string, name: string) => opened.push(`browser:${name}`),
      openRagDocumentExternally: (_id: string, name: string) => opened.push(`tab:${name}`),
      offerRagDocumentOutside: (_id: string, name: string) => opened.push(`offer:${name}`),
    },
    "./source-file": {
      asViewerFile: (file: File) => file,
      fetchSourceFile: async () => null,
      isViewableSource: (name: string) => /\.(pdf|docx|md|txt|rtf)$/.test(name),
    },
  },
);

function route(filename: string): "browser" | "viewer" | "tab" {
  const child = React.createElement("button");
  const element = RagDocumentTrigger({
    documentId: "doc-1",
    filename,
    children: child,
  }) as unknown as StubElement;
  if (element.type === LocalFileDialog) return "viewer";
  opened.length = 0;
  (element.props.onClick as () => void)();
  return opened[0]?.startsWith("browser:") ? "browser" : "tab";
}

test("formats the browser can show open there", () => {
  Object.assign(env, { panel: true, filesInBrowser: true });
  for (const name of ["handbook.pdf", "notes.md", "page.html"]) {
    assert.equal(route(name), "browser", name);
  }
});

test("formats the browser can't show skip it, as they do without the panel", () => {
  Object.assign(env, { panel: true, filesInBrowser: true });
  assert.equal(route("letter.rtf"), "viewer");
  assert.equal(route("book.epub"), "tab");
});

test("without the panel or with files set to open elsewhere, the viewer or a new tab", () => {
  for (const state of [
    { panel: false, filesInBrowser: true },
    { panel: true, filesInBrowser: false },
  ]) {
    Object.assign(env, state);
    assert.equal(route("handbook.pdf"), "viewer", JSON.stringify(state));
    assert.equal(route("page.html"), "tab", JSON.stringify(state));
  }
});

test("a file too large for the attachment viewer is offered outside Studio", async () => {
  Object.assign(env, { panel: false, filesInBrowser: true });
  const element = RagDocumentTrigger({
    documentId: "doc-1",
    filename: "scan.pdf",
    children: React.createElement("button"),
  }) as unknown as StubElement;
  assert.equal(element.type, LocalFileDialog);
  opened.length = 0;
  await assert.rejects((element.props.load as () => Promise<Blob>)());
  assert.deepEqual(opened, ["offer:scan.pdf"]);
});

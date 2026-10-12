// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS,
  RTF_ATTACHMENT_EXTENSIONS,
  SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS,
  isServerDocumentAttachmentName,
} from "../src/features/chat/open-document-accept.ts";
import { TEXT_ATTACHMENT_EXTENSIONS } from "../src/features/chat/text-attachment-accept.ts";
import {
  RAG_DOCUMENT_UPLOAD_ACCEPT,
  RAG_UPLOAD_ACCEPT,
} from "../src/features/rag/types/rag.ts";

const split = (list: string) => list.split(",").map((ext) => ext.trim());

test("the composer takes every document type chat with files parses", () => {
  const browserRead = new Set([
    ".xlsx",
    ".xlsm",
    ".pptx",
    ...split(OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS),
    RTF_ATTACHMENT_EXTENSIONS,
    ...TEXT_ATTACHMENT_EXTENSIONS,
  ]);
  const expected = split(RAG_DOCUMENT_UPLOAD_ACCEPT).filter(
    (ext) => !browserRead.has(ext),
  );
  assert.deepEqual(
    [...split(SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS)].sort(),
    expected.sort(),
  );
  // The base RAG types all have browser readers already (.pdf, .docx, .html, text).
  for (const ext of split(RAG_UPLOAD_ACCEPT)) {
    assert.equal(isServerDocumentAttachmentName(`a${ext}`), false, ext);
  }
});

test("no earlier composer adapter claims a server document by extension", () => {
  const text = new Set(TEXT_ATTACHMENT_EXTENSIONS);
  for (const ext of split(SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS)) {
    assert.equal(text.has(ext), false, ext);
  }
  assert.equal(isServerDocumentAttachmentName("Q3 Report.DOC"), true);
  assert.equal(isServerDocumentAttachmentName("notes.docx"), false);
});

test("server documents are routed ahead of text and tool-only adapters", () => {
  const source = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  const order = (marker: string) => {
    const index = source.indexOf(marker, source.indexOf("new CompositeAttachmentAdapter(["));
    assert.notEqual(index, -1, marker);
    return index;
  };
  assert.ok(order("          server,") < order("new TextAttachmentAdapter()"));
  assert.ok(order("          server,") < order("new ToolOnlyAttachmentAdapter()"));
  for (const adapter of ["PDF", "Docx", "Office", "OpenDocument"]) {
    assert.ok(order(`largeFallback(new ${adapter}AttachmentAdapter())`) > 0);
  }
});

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

test("a failed read cancels the send, and a textless tool-only type falls back to the python tool", () => {
  const source = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  const adapter = source.slice(
    source.indexOf("class ServerDocumentAttachmentAdapter"),
    source.indexOf("class LargeDocumentFallbackAdapter"),
  );
  assert.match(adapter, /text = await pending;\s*\} catch \(cause\) \{[\s\S]*?throw cause;/);
  assert.doesNotMatch(adapter, /pending\.catch\(/);
  assert.match(
    adapter,
    /\?\.name === "NoDocumentTextError" &&\s*isToolOnlyAttachmentName\(file\.name\) &&\s*pythonToolOpensAttachments\(\)/,
  );
  // An oversized scanned PDF is past the PDF originals cap, so it may not fall back to the python tool.
  assert.doesNotMatch(adapter.slice(adapter.indexOf("private read(")), /\\\.pdf\$/);
  // Code can be turned off between attach and send; then the textless file is refused, not sent bare.
  assert.match(adapter, /if \(text === null && !pythonToolOpensAttachments\(\)\) \{[\s\S]*?throw new Error/);
  assert.match(adapter, /only the python tool can read this file/);
  const preview = readFileSync(
    new URL("../src/components/assistant-ui/attachment-preview.tsx", import.meta.url),
    "utf8",
  );
  const webPage = new RegExp(/const WEB_PAGE_NAME = \/(.+)\/i;/.exec(preview)![1]!, "i");
  assert.ok(webPage.test("page.html"));
  assert.ok(!webPage.test("page.xhtml"), "extracted .xhtml text must not render as markup");
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
  assert.ok(order("          server,") < order("new OfficeTemplateAdapter(text, server)"));
  // Ahead of text, which also lists .dot/.pot, so a binary template is sniffed before text refuses it.
  assert.ok(order("new OfficeTemplateAdapter(text, server)") < order("          text,"));
  assert.ok(order("          server,") < order("new ToolOnlyAttachmentAdapter()"));
  for (const adapter of ["PDF", "Docx", "Office", "OpenDocument"]) {
    assert.ok(order(`largeFallback(new ${adapter}AttachmentAdapter())`) > 0);
  }
});

test("Desktop reads the same server-document types the composer does", () => {
  const policy = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const rust = [
    ...(policy
      .match(/SERVER_DOCUMENT_ATTACHMENT_EXTS: &\[&str\] = &\[([^\]]+)\]/)?.[1]
      .matchAll(/"([^"]+)"/g) ?? []),
  ].map((match) => `.${match[1]}`);
  assert.deepEqual(rust.sort(), split(SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS).sort());
});

test("an extraction started by one account never settles in another's composer", () => {
  const reader = readFileSync(
    new URL("../src/features/chat/server-document.ts", import.meta.url),
    "utf8",
  );
  assert.match(reader, /const epoch = getAuthSessionEpoch\(\);/);
  // Retries refuse a changed account, and the answer is checked again before its text is used.
  assert.match(reader, /await extractDocumentText\(file, sameAccount\);\s*sameAccount\(\);/);
  const api = readFileSync(new URL("../src/features/rag/api/rag-api.ts", import.meta.url), "utf8");
  assert.match(api, /\{ method: "POST", body: form \},\s*\{ beforeRetry \},/);
});

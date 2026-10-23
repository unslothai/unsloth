// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("a citation chip's page reaches the PDF view in the browser and the attachment viewer", () => {
  assert.match(
    readSrc("components/assistant-ui/tool-ui-knowledge-base.tsx"),
    /<RagDocumentTrigger[^>]*page=\{pages\?\.\[0\] \?\? citation\.page \?\? undefined\}/,
  );
  assert.match(readSrc("features/browser/file-view.tsx"), /<DocumentView[^>]*pageTarget=\{pageTarget \?\? undefined\}/);
  assert.match(readSrc("components/assistant-ui/attachment-document-dialog.tsx"), /<DocumentView[^>]*pageTarget=\{pageTarget\}/);
  assert.match(readSrc("components/file-viewer/document-view.tsx"), /<PdfView[^>]*pageTarget=\{pageTarget\}/);
  const pdfView = readSrc("components/file-viewer/pdf-view.tsx");
  assert.match(pdfView, /virtualizer\.scrollToIndex\([^;]*;\s*onApplied\?\.\(target\)/);
  // Input while the document loads keeps the target: only one already shown is let go.
  assert.match(pdfView, /if \(applied\.current !== pending\) return;\s*setPending\(undefined\);\s*onPageTargetDone\?\.\(\)/);
  // Once the reader moves on, the browser drops the tab's target, so reopening the panel doesn't replay it.
  assert.match(readSrc("features/browser/file-view.tsx"), /setFileView\(tabId, \{ pageTarget: null \}\)/);
});

test("thread cards and Sources chips carry no page, so they open at the first", () => {
  for (const file of ["features/rag/components/project-source-chip.tsx", "features/rag/components/thread-documents-bar.tsx"]) {
    assert.doesNotMatch(readSrc(file), /<RagDocumentTrigger[^>]*page=/, file);
  }
});

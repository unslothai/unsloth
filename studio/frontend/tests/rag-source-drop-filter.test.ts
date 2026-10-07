// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  RAG_SOURCE_UPLOAD_ACCEPT,
  isSupportedSourceName,
  partitionSupported,
} from "../src/features/rag/components/source-drop-policy.ts";
import { RAG_UPLOAD_ACCEPT } from "../src/features/rag/types/rag.ts";

test("every type the picker offers is accepted from a drop", () => {
  for (const ext of RAG_UPLOAD_ACCEPT.split(",")) {
    assert.equal(isSupportedSourceName(`notes${ext}`), true, ext);
  }
});

test("extensions match case-insensitively", () => {
  assert.equal(isSupportedSourceName("REPORT.PDF"), true);
  assert.equal(isSupportedSourceName("Readme.Md"), true);
});

// A dropped folder arrives as an extension-less entry, and uploading it 400s.
test("folders and unsupported types are rejected", () => {
  assert.equal(isSupportedSourceName("my-documents"), false);
  assert.equal(isSupportedSourceName("archive.zip"), false);
  assert.equal(isSupportedSourceName("model.gguf"), false);
});

test("dotfiles are rejected", () => {
  assert.equal(isSupportedSourceName(".md"), false);
  assert.equal(isSupportedSourceName(".gitignore"), false);
});

test("only the last extension counts", () => {
  assert.equal(isSupportedSourceName("notes.pdf.txt"), true);
  assert.equal(isSupportedSourceName("notes.txt.exe"), false);
});

test("a mixed drop keeps the indexable files and names the rest", () => {
  const { supported, unsupported } = partitionSupported(
    ["notes.pdf", "photo.png", "readme.md", "archive.zip"],
    (name) => name,
  );
  assert.deepEqual(supported, ["notes.pdf", "readme.md"]);
  assert.deepEqual(unsupported, ["photo.png", "archive.zip"]);
});

test("the split reports names, not the entries it was given", () => {
  const { supported, unsupported } = partitionSupported(
    [{ path: "/tmp/a.pdf" }, { path: "/tmp/b.png" }],
    (entry) => entry.path,
  );
  assert.deepEqual(supported, [{ path: "/tmp/a.pdf" }]);
  assert.deepEqual(unsupported, ["/tmp/b.png"]);
});

test("an empty drop yields nothing to upload and nothing to report", () => {
  const { supported, unsupported } = partitionSupported(
    [],
    (name: string) => name,
  );
  assert.equal(supported.length, 0);
  assert.equal(unsupported.length, 0);
});

test("source files the project picker offers are accepted from a drop", () => {
  for (const ext of RAG_SOURCE_UPLOAD_ACCEPT.split(",")) {
    assert.equal(isSupportedSourceName(`app${ext}`), true, ext);
  }
  assert.equal(isSupportedSourceName("index.php"), true);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ARTIFACT_DOWNLOAD_FORMATS,
  buildArtifactDownloadContent,
  buildArtifactDownloadPayload,
  getArtifactDownloadExtension,
  getArtifactDownloadMimeType,
  getArtifactFilename,
  isBinaryArtifactDownloadFormat,
} from "../src/features/chat/artifacts/types.ts";

const CODE = "<h1>hi</h1>";

test("every download format resolves to a distinct extension and mime type", () => {
  const extensions = ARTIFACT_DOWNLOAD_FORMATS.map(
    getArtifactDownloadExtension,
  );
  const mimeTypes = ARTIFACT_DOWNLOAD_FORMATS.map(getArtifactDownloadMimeType);
  assert.equal(new Set(extensions).size, ARTIFACT_DOWNLOAD_FORMATS.length);
  assert.equal(new Set(mimeTypes).size, ARTIFACT_DOWNLOAD_FORMATS.length);
});

test("getArtifactFilename suffixes the slug with the requested format's extension", () => {
  const artifact = { title: "Quarterly Report" };
  assert.equal(getArtifactFilename(artifact, "html"), "quarterly-report.html");
  assert.equal(getArtifactFilename(artifact, "md"), "quarterly-report.md");
  assert.equal(getArtifactFilename(artifact, "txt"), "quarterly-report.txt");
  assert.equal(getArtifactFilename(artifact, "docx"), "quarterly-report.docx");
  assert.equal(getArtifactFilename(artifact, "pptx"), "quarterly-report.pptx");
  assert.equal(getArtifactFilename(artifact, "xlsx"), "quarterly-report.xlsx");
  assert.equal(getArtifactFilename(artifact, "pdf"), "quarterly-report.pdf");
  assert.equal(getArtifactFilename(artifact), "quarterly-report.html");
});

test("markdown downloads wrap the source in a fenced html block", () => {
  const content = buildArtifactDownloadContent(CODE, "md");
  assert.match(content, /^```html\n<h1>hi<\/h1>\n```$/);
});

test("html and text downloads keep the raw source untouched", () => {
  assert.equal(buildArtifactDownloadContent(CODE, "html"), CODE);
  assert.equal(buildArtifactDownloadContent(CODE, "txt"), CODE);
});

test("only html/md/txt are classified as non-binary formats", () => {
  assert.equal(isBinaryArtifactDownloadFormat("html"), false);
  assert.equal(isBinaryArtifactDownloadFormat("md"), false);
  assert.equal(isBinaryArtifactDownloadFormat("txt"), false);
  assert.equal(isBinaryArtifactDownloadFormat("docx"), true);
  assert.equal(isBinaryArtifactDownloadFormat("pptx"), true);
  assert.equal(isBinaryArtifactDownloadFormat("xlsx"), true);
  assert.equal(isBinaryArtifactDownloadFormat("pdf"), true);
});

test("text-format payloads match buildArtifactDownloadContent", async () => {
  const artifact = { title: "Quarterly Report", code: CODE };
  for (const format of ["html", "md", "txt"] as const) {
    const payload = await buildArtifactDownloadPayload(artifact, format);
    assert.equal(payload, buildArtifactDownloadContent(CODE, format));
  }
});

// docx/pptx/xlsx are all zip containers (their signature is the same as any
// zip: "PK"), so this checks the bytes actually produced a zip rather than
// raw HTML saved under the wrong extension — the failure mode the old
// "only text formats" test guarded against.
async function readBlobBytes(blob: Blob): Promise<Uint8Array> {
  return new Uint8Array(await blob.arrayBuffer());
}

test("docx downloads produce a real zip (OOXML) container", async () => {
  const payload = await buildArtifactDownloadPayload(
    { title: "Report", code: CODE },
    "docx",
  );
  assert.ok(payload instanceof Blob);
  const bytes = await readBlobBytes(payload as Blob);
  assert.equal(bytes[0], 0x50); // "P"
  assert.equal(bytes[1], 0x4b); // "K"
});

test("pptx downloads produce a real zip (OOXML) container", async () => {
  const payload = await buildArtifactDownloadPayload(
    { title: "Report", code: CODE },
    "pptx",
  );
  assert.ok(payload instanceof Blob);
  const bytes = await readBlobBytes(payload as Blob);
  assert.equal(bytes[0], 0x50);
  assert.equal(bytes[1], 0x4b);
});

test("xlsx downloads produce a real zip (OOXML) container", async () => {
  const payload = await buildArtifactDownloadPayload(
    { title: "Report", code: CODE },
    "xlsx",
  );
  assert.ok(payload instanceof Blob);
  const bytes = await readBlobBytes(payload as Blob);
  assert.equal(bytes[0], 0x50);
  assert.equal(bytes[1], 0x4b);
});

test("pdf downloads produce a real PDF file", async () => {
  const payload = await buildArtifactDownloadPayload(
    { title: "Report", code: CODE },
    "pdf",
  );
  assert.ok(payload instanceof Blob);
  const bytes = await readBlobBytes(payload as Blob);
  const header = new TextDecoder().decode(bytes.slice(0, 5));
  assert.equal(header, "%PDF-");
});

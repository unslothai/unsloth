// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ARTIFACT_DOWNLOAD_FORMATS,
  buildArtifactDownloadContent,
  getArtifactDownloadExtension,
  getArtifactDownloadMimeType,
  getArtifactFilename,
} from "../src/features/chat/artifacts/types.ts";

const CODE = "<h1>hi</h1>";

test("every download format resolves to a distinct extension and mime type", () => {
  const extensions = ARTIFACT_DOWNLOAD_FORMATS.map(getArtifactDownloadExtension);
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
  assert.equal(getArtifactFilename(artifact, "pages"), "quarterly-report.pages");
  assert.equal(getArtifactFilename(artifact, "numbers"), "quarterly-report.numbers");
});

test("markdown downloads wrap the source in a fenced html block", () => {
  const content = buildArtifactDownloadContent(CODE, "md");
  assert.match(content, /^```html\n<h1>hi<\/h1>\n```$/);
});

test("html, text, and other format downloads keep the raw source untouched", () => {
  assert.equal(buildArtifactDownloadContent(CODE, "html"), CODE);
  assert.equal(buildArtifactDownloadContent(CODE, "txt"), CODE);
  assert.equal(buildArtifactDownloadContent(CODE, "docx"), CODE);
  assert.equal(buildArtifactDownloadContent(CODE, "pptx"), CODE);
  assert.equal(buildArtifactDownloadContent(CODE, "xlsx"), CODE);
  assert.equal(buildArtifactDownloadContent(CODE, "pdf"), CODE);
  assert.equal(buildArtifactDownloadContent(CODE, "pages"), CODE);
  assert.equal(buildArtifactDownloadContent(CODE, "numbers"), CODE);
});

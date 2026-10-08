// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Check thumbnail/original separation in source because the page has no DOM test harness.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function source(path: string): string {
  return readFileSync(
    fileURLToPath(new URL(path, import.meta.url)),
    "utf8",
  ).replace(/\r\n/g, "\n");
}

function between(text: string, start: string, end: string): string {
  const from = text.indexOf(start);
  const to = text.indexOf(end, from + start.length);
  if (from === -1 || to <= from) {
    throw new Error(`markers not found: ${start} / ${end}`);
  }
  return text.slice(from, to);
}

const page = source("../src/features/images/images-page.tsx");

test("strip tiles fetch and show thumbnails, falling back to a cached original", () => {
  const strip = between(
    page,
    "{(thumbById[image.id] ?? srcById[image.id]) ? (",
    "{/* Selection marker",
  );
  assert.ok(strip.includes("src={thumbById[image.id] ?? srcById[image.id]}"));

  const observer = between(page, "const io = new IntersectionObserver(", "return () => io.disconnect();");
  assert.ok(observer.includes("void ensureThumb(image);"));
  assert.ok(!observer.includes("ensureSrc("));
  assert.ok(page.includes("fetchGalleryObjectUrl(galleryThumbnailUrl(image.url))"));
  // The canvas shows the selected thumbnail while its original loads, even with the tile off-screen.
  const thumbPrune = between(page, "galleryCache.thumbById.prune(", ");");
  assert.ok(thumbPrune.includes("galleryCache.selectedId"));
});

test("the canvas, viewer and downloads read only the original", () => {
  assert.ok(page.includes("const selectedSrc = selected ? srcById[selected.id] : undefined;"));
  assert.ok(page.includes("const viewerSrc = viewerImage ? srcById[viewerImage.id] : undefined;"));
  // The selected record is fetched in full whether or not its tile is on screen.
  const preview = between(page, "// The preview is what the user looks at", "// Drop an image from the strip.");
  assert.ok(preview.includes("await ensureSrc(selected);"));

  // The live denoise preview branch comes first while a run is in flight, so the canvas proper is the
  // `selected && selectedSrc` branch that follows it.
  const canvas = between(page, ") : selected && selectedSrc ? (", ") : selected ? (");
  assert.ok(canvas.includes("src={selectedSrc}"));
  for (const format of ["png", "jpeg", "webp"]) {
    assert.ok(canvas.includes(`downloadImage(selectedSrc, selected, "${format}")`));
  }
  assert.ok(!canvas.includes("thumbById"));
  assert.ok(!canvas.includes("selectedThumb"));

  const quickDownload = between(page, "const handleQuickDownload = useCallback(", "const stripReorder");
  assert.ok(quickDownload.includes("srcById[image.id]"));
  assert.ok(quickDownload.includes("fetchGalleryBlob(image.url)"));
  assert.ok(!quickDownload.includes("thumbById"));
});

test("the thumbnail placeholder offers no action that needs the original", () => {
  const placeholder = between(page, ") : selected ? (", ') : busy === "generating" ? null : (');
  assert.ok(placeholder.includes("src={selectedThumb}"));
  for (const action of ["downloadImage", "openViewer", "GalleryItemMenu", "RecipePopover"]) {
    assert.ok(!placeholder.includes(action), action);
  }
});

test("the live denoise preview shows only the in-flight preview and offers no action on it", () => {
  assert.ok(
    page.includes(
      'busy === "generating" && livePreview ? (genStep?.preview ?? undefined) : undefined;',
    ),
  );
  const live = between(page, "{livePreviewSrc ? (", ") : selected && selectedSrc ? (");
  assert.ok(live.includes("src={livePreviewSrc}"));
  for (const forbidden of ["selectedSrc", "thumbById", "selectedThumb", "downloadImage", "openViewer", "GalleryItemMenu"]) {
    assert.ok(!live.includes(forbidden), forbidden);
  }
});

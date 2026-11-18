// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Source-checked because the page has no DOM test harness.

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
  // The strip hands each tile its thumbnail (or the cached original); the tile draws it or a spinner.
  const strip = between(page, "{images.map((image) => (", "{hasMore && (");
  assert.ok(strip.includes("src={thumbById[image.id] ?? srcById[image.id]}"));
  const tile = between(page, "const GalleryStripTile = memo(", "{/* Selection marker");
  assert.ok(tile.includes("{src ? ("));
  assert.ok(tile.includes("src={src}"));

  const observer = between(page, "const io = new IntersectionObserver(", "return () => io.disconnect();");
  assert.ok(observer.includes("void ensureThumb(image);"));
  assert.ok(!observer.includes("ensureSrc("));
  assert.ok(page.includes("fetchGalleryObjectUrl(galleryThumbnailUrl(image.url))"));
  const thumbPrune = between(page, "galleryCache.thumbById.prune(", ");");
  assert.ok(thumbPrune.includes("galleryCache.selectedId"));
});

test("the canvas, viewer and downloads read only the original", () => {
  assert.ok(page.includes("const selectedSrc = selected ? srcById[selected.id] : undefined;"));
  assert.ok(page.includes("const viewerSrc = viewerImage ? srcById[viewerImage.id] : undefined;"));
  const preview = between(page, "// The preview is what the user looks at", "// Drop an image from the strip.");
  assert.ok(preview.includes("await ensureSrc(selected);"));

  // The live denoise preview branch comes first while a run is in flight.
  const canvas = between(page, ") : selected && selectedSrc ? (", ") : selected ? (");
  assert.ok(canvas.includes("src={selectedSrc}"));
  for (const format of ["png", "jpeg", "webp"]) {
    assert.ok(canvas.includes(`downloadImage(selected, "${format}", selectedSrc)`));
  }
  assert.ok(!canvas.includes("thumbById"));
  assert.ok(!canvas.includes("selectedThumb"));

  const quickDownload = between(page, "const handleQuickDownload = useCallback(", "const stripReorder");
  assert.ok(quickDownload.includes("srcById[image.id]"));
  assert.ok(!quickDownload.includes("thumbById"));
  // Without a cached original, the download fetches the full PNG, never a thumbnail.
  const download = between(page, "async function downloadImage(", "\n}\n");
  assert.ok(download.includes("fetchGalleryBlob(image.url)"));
  assert.ok(!download.includes("thumb"));
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
  const live = between(page, "{stagePreviewSrc ? (", ") : selected && selectedSrc ? (");
  assert.ok(live.includes("src={stagePreviewSrc}"));
  for (const forbidden of ["selectedSrc", "thumbById", "selectedThumb", "downloadImage", "openViewer", "GalleryItemMenu"]) {
    assert.ok(!live.includes(forbidden), forbidden);
  }
});

test("an image picked during a run takes the stage from the live preview until the run is followed again", () => {
  assert.ok(page.includes("const stagePreviewSrc = pickedDuringRun ? undefined : livePreviewSrc;"));
  const pick = between(page, "const pickImage = useCallback((id: string) => {", "}, []);");
  assert.ok(pick.includes("galleryCache.selectedId = id;"));
  assert.ok(pick.includes("setSelectedId(id);") && pick.includes("setPickedDuringRun(true);"));
  assert.ok(page.includes("select: pickImage,"));
  assert.ok(page.includes("pickImage(routedItem);"));
  assert.ok(page.includes("() => galleryCache.selectedId,"));
  const liveTile = between(page, 'data-testid="images-live-tile"', "</button>");
  assert.ok(liveTile.includes("onClick={() => setPickedDuringRun(false)}"));
  assert.ok(page.includes("localRun.current = true;\n    setPickedDuringRun(false);"));
  assert.match(
    page,
    /pickedDuringRun &&\s*pickReleased\(\{\s*statusKnown: generateProbed && imageQueue\.polled,\s*generating: busy === "generating",\s*queueBusy: queueBusy\(imageQueue\.jobs\),/,
  );
  assert.ok(page.includes("if (res.images[0] && !pickedDuringRunRef.current) setSelectedId(res.images[0].id);"));
  assert.ok(page.includes('if (finished.some((job) => job.status === "done")) refreshLoadedGallery();'));
  assert.match(page, /setGenStep\(null\);\s*refreshLoadedGallery\(\);\s*void refreshStatus\(\);/);
  const refresh = between(page, "const refreshLoadedGallery = useCallback(() => {", "}, [loadGallery, resyncWindow]);");
  assert.match(
    refresh,
    /attempt < PAGE_MAX_ATTEMPTS && refreshSeq\.current === ticket;[\s\S]*?if \(attempt > 0\) await serializeById\("image-pin", async \(\) => \{\}\);[\s\S]*?if \(await resyncWindow\(galleryCache\.images\.length, fresh\)\) return;/,
  );
  assert.ok(page.includes("selected={!stagePreviewSrc && image.id === selected?.id}"));
});

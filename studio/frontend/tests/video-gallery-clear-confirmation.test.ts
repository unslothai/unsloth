// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const source = readFileSync(
  fileURLToPath(
    new URL("../src/features/video/video-page.tsx", import.meta.url),
  ),
  "utf8",
);

test("video gallery bulk deletion requires an explicit confirmation", () => {
  const clearButton = source.slice(
    source.indexOf("{/* Clear-all, tucked at the end"),
    source.indexOf("<TooltipContent>Clear all videos</TooltipContent>"),
  );
  assert.ok(clearButton.includes("onClick={() => setClearConfirmOpen(true)}"));
  assert.ok(!clearButton.includes("handleClearAll"));

  const dialog = source.slice(
    source.indexOf("<AlertDialog\n        open={clearConfirmOpen}"),
    source.indexOf("<Dialog\n        open={pendingH3Load"),
  );
  assert.ok(
    dialog.includes("<AlertDialogTitle>Clear all videos?</AlertDialogTitle>"),
  );
  assert.ok(
    dialog.includes(
      "This permanently deletes every generated video from the gallery.",
    ),
  );
  assert.ok(dialog.includes("be undone."));
  assert.ok(
    dialog.includes(
      "<AlertDialogCancel disabled={clearingGallery}>Cancel</AlertDialogCancel>",
    ),
  );
  assert.ok(dialog.includes('variant="destructive"'));
  assert.ok(dialog.includes("event.preventDefault();"));
  assert.ok(dialog.includes("void handleClearAll();"));
});

test("video gallery confirmation stays controlled while clearing", () => {
  const handler = source.slice(
    source.indexOf("const handleClearAll = useCallback("),
    source.indexOf("// Load a clip's recipe back into the form inputs."),
  );
  assert.ok(handler.includes("setClearingGallery(true);"));
  assert.ok(handler.includes("await clearVideoGallery();"));
  assert.ok(handler.includes("setClearConfirmOpen(false);"));
  assert.ok(handler.includes("finally {\n      setClearingGallery(false);"));
  assert.ok(
    handler.indexOf("await clearVideoGallery();") <
      handler.indexOf("setClearConfirmOpen(false);"),
  );

  const root = source.slice(
    source.indexOf("<AlertDialog\n        open={clearConfirmOpen}"),
    source.indexOf('<AlertDialogContent size="sm">'),
  );
  assert.ok(root.includes("if (!clearingGallery) setClearConfirmOpen(open);"));
});

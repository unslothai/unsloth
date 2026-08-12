// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Newlines normalized because the assertions below span lines: .gitattributes keeps this
// tree at LF, but a source archive or a stray core.autocrlf hands the same file back with
// CRLF and every multi-line marker would miss.
const source = readFileSync(
  fileURLToPath(
    new URL("../src/features/video/video-page.tsx", import.meta.url),
  ),
  "utf8",
).replace(/\r\n/g, "\n");

/** The source between two markers, asserting both are actually there.
 *
 *  Without the assertion a marker that moves yields an empty (or wildly wrong) slice, and
 *  the negative assertion below -- "the clear button no longer calls handleClearAll" --
 *  passes over it vacuously. A guard that silently stops guarding is worse than none. */
function between(start: string, end: string): string {
  const from = source.indexOf(start);
  const to = source.indexOf(end);
  if (from === -1) {
    throw new Error(`marker not found in video-page.tsx: ${start}`);
  }
  if (to <= from) {
    throw new Error(`marker not found after ${JSON.stringify(start)}: ${end}`);
  }
  return source.slice(from, to);
}

test("video gallery bulk deletion requires an explicit confirmation", () => {
  const clearButton = between(
    "{/* Clear-all, tucked at the end",
    "<TooltipContent>Clear all videos</TooltipContent>",
  );
  assert.ok(clearButton.includes("onClick={() => setClearConfirmOpen(true)}"));
  assert.ok(!clearButton.includes("handleClearAll"));

  const dialog = between(
    "<AlertDialog\n        open={active && clearConfirmOpen}",
    "<Dialog\n        open={pendingH3Load",
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

test("video gallery confirmation stays controlled while clearing and off-route", () => {
  const handler = between(
    "const handleClearAll = useCallback(",
    "// Load a clip's recipe back into the form inputs.",
  );
  assert.ok(handler.includes("setClearingGallery(true);"));
  assert.ok(handler.includes("await clearVideoGallery();"));
  assert.ok(handler.includes("setClearConfirmOpen(false);"));
  assert.ok(handler.includes("finally {\n      setClearingGallery(false);"));
  assert.ok(
    handler.indexOf("await clearVideoGallery();") <
      handler.indexOf("setClearConfirmOpen(false);"),
  );

  const root = between(
    "<AlertDialog\n        open={active && clearConfirmOpen}",
    '<AlertDialogContent size="sm">',
  );
  assert.ok(root.includes("open={active && clearConfirmOpen}"));
  assert.ok(root.includes("if (!clearingGallery) setClearConfirmOpen(open);"));
});

test("leaving the video route closes the confirmation rather than hiding it", () => {
  // The page is mounted persistently, so `active` going false only hides the dialog;
  // Radix does not call onOpenChange for a parent-forced close, so without this reset the
  // same confirm is back on screen the moment the route becomes active again.
  const reset = between(
    "const [clearingGallery, setClearingGallery] = useState(false);",
    "const playCountRef = useRef(0);",
  );
  assert.ok(
    reset.includes(
      "if (!active && clearConfirmOpen) setClearConfirmOpen(false);",
    ),
  );
});

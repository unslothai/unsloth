// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Images and video keep the old pipeline resident while a replacement
// downloads, so the indicator shows the resident row (ejectable) next to the
// incoming one (a spinner). Ejecting the resident row makes the backend cancel
// that replacement, and the page's own listener then tears down its tracking --
// including the load-progress poll, which is the ONLY thing that clears `busy`.
//
// Left set, `busy` locks the page: the picker ignores every choice, Generate
// and Reapply are disabled, and Unload is not even rendered once the status
// read comes back empty. Both pages are mounted for the whole app session, so
// navigating away and back does not reset it either -- only a reload did.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// The runtime name is singular ("image"), the page is not: keep them apart, or
// the listener lookup silently finds nothing and every check passes vacuously.
const PAGES = [
  ["Images", "image", "../src/features/images/images-page.tsx"],
  ["Video", "video", "../src/features/video/video-page.tsx"],
] as const;

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

for (const [page, runtime, path] of PAGES) {
  const SOURCE = read(path);
  const listener = SOURCE.slice(
    SOURCE.indexOf(`subscribeModelEjected("${runtime}"`),
    SOURCE.indexOf(`subscribeModelEjected("${runtime}"`) + 900,
  );

  test(`the ${page} page settles its busy state on an external eject`, () => {
    assert.ok(listener.length > 0, "expected the eject listener");
    assert.match(
      listener,
      /setBusy\(\(prev\) => \(prev === "loading" \? null : prev\)\)/,
      "the listener must clear a load that its own teardown just orphaned",
    );
  });

  test(`the ${page} page still stops the poll it is replacing`, () => {
    // The clear only matters because dropResidentState kills the poll; if that
    // ever stops being true the two lines should be revisited together.
    assert.match(listener, /dropResidentState\(\)/);
    const drop = SOURCE.slice(
      SOURCE.indexOf("const dropResidentState = useCallback("),
      SOURCE.indexOf("const dropResidentState = useCallback(") + 500,
    );
    assert.match(drop, /clearTimeout\(pollTimer\.current\)/);
    assert.doesNotMatch(
      drop,
      /setBusy/,
      "kept in the listener: handleUnload sets busy right after calling this",
    );
  });

  test(`the ${page} page leaves a generation alone`, () => {
    // An unconditional clear would also drop "generating". The backend unload
    // blocks on the generate lock so that is near unreachable, but narrowing it
    // costs nothing.
    assert.doesNotMatch(listener, /setBusy\(null\)/);
  });
}

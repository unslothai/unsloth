// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Ejecting the resident row tears down the load poll, which is the only thing clearing busy.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

// The runtime name is singular, the page is not; a mismatch passes vacuously.
const PAGES = [
  ["Images", "image", "../src/features/images/images-page.tsx"],
  ["Video", "video", "../src/features/video/video-page.tsx"],
] as const;

for (const [page, runtime, path] of PAGES) {
  const SOURCE = readText(path);
  const listener = SOURCE.slice(
    SOURCE.indexOf(`subscribeModelEjected("${runtime}"`),
    // Wide enough for the pending-start fence around the clear.
    SOURCE.indexOf(`subscribeModelEjected("${runtime}"`) + 1800,
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
    assert.match(listener, /dropResidentState\(\)/);
    const drop = SOURCE.slice(
      SOURCE.indexOf("const dropResidentState = useCallback("),
      SOURCE.indexOf(
        "}, [dismissLoadToast,",
        SOURCE.indexOf("const dropResidentState = useCallback("),
      ),
    );
    assert.match(drop, /clearTimeout\(pollTimer\.current\)/);
    assert.doesNotMatch(
      drop,
      /setBusy/,
      "kept in the listener: handleUnload sets busy right after calling this",
    );
  });

  test(`the ${page} page leaves a generation alone`, () => {
    // An unconditional clear would also drop generating.
    assert.doesNotMatch(listener, /setBusy\(null\)/);
  });
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images and Video pages hold their own status and re-read it on tab
// activation and on their own actions, never on a timer. So two reads can be in
// flight across an eject: an activation read that saw the pipeline loaded, and
// the post-eject read that saw it gone. Responses have no order, and the older
// one landing last left the page offering to generate against a freed runtime,
// with no poll coming to correct it.
//
// Asserted by reading the source: both pages pull in the whole media runtime,
// which the node suite cannot mount.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function source(path: string): string {
  return readFileSync(fileURLToPath(new URL(`../src/${path}`, import.meta.url)), "utf8");
}

const PAGES = [
  ["images", "features/images/images-page.tsx", "getDiffusionStatus", "unloadDiffusionModel"],
  ["video", "features/video/video-page.tsx", "getVideoStatus", "unloadVideoModel"],
] as const;

for (const [name, path, read, unload] of PAGES) {
  test(`the ${name} page lets only the newest status read write`, () => {
    const page = source(path);
    assert.match(page, /const statusTicket = useRef\(0\);/);
    assert.match(
      page,
      /if \(ticket === statusTicket\.current\) setStatus\(next\);/,
      "a superseded read must not write",
    );
    // Every writer goes through it, so none can be the one that slips past.
    assert.doesNotMatch(
      page,
      new RegExp(`setStatus\\(await ${read}\\(\\)\\)`),
      "the bare read must not write directly",
    );
    assert.doesNotMatch(
      page,
      new RegExp(`setStatus\\(await ${unload}\\(\\)\\)`),
      "nor the unload",
    );
    assert.equal(
      (page.match(/setStatusIfNewest\(/g) ?? []).length,
      3,
      "the refresh, the load-progress read and the unload all go through it",
    );
  });

  test(`the ${name} page claims its ticket before awaiting, not after`, () => {
    const page = source(path);
    // Claiming after the await would hand every read the newest ticket and
    // defeat the whole thing.
    assert.match(page, /const ticket = \+\+statusTicket\.current;\s*\n\s*try \{/);
  });
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The standalone Downloads overlay sits in the bottom-right corner. Below the
 * md breakpoint the chat composer spans the full width at the bottom, so that
 * corner is the send button (#14146). The overlay keeps its corner on wide
 * screens and lifts above the input bar on narrow ones.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/features/hub/download-manager/download-manager-panel.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("the standalone overlay lifts above the composer below md", () => {
  const positioned = source.match(/positioned\s*\?\s*"([^"]+)"/);
  assert.ok(positioned, "the standalone class list is a string literal");
  const classes = positioned[1].split(/\s+/);
  assert.ok(classes.includes("fixed") && classes.includes("right-4"));
  assert.ok(classes.includes("bottom-4"), "wide screens keep the corner");
  const lifted = classes.find((c) => /^max-md:bottom-\S+$/.test(c));
  assert.ok(lifted, "narrow screens get a max-md bottom offset");
  assert.notEqual(
    lifted,
    "max-md:bottom-4",
    "the offset must actually lift it",
  );
});

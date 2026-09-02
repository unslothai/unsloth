// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);

test("chat lifecycle announcements use a persistent polite status region", () => {
  const rootIndex = source.indexOf("<ThreadPrimitive.Root");
  const regionIndex = source.indexOf("<ChatLiveRegion />", rootIndex);
  const viewportIndex = source.indexOf("<ThreadPrimitive.Viewport", rootIndex);

  assert.ok(rootIndex >= 0);
  assert.ok(regionIndex > rootIndex);
  assert.ok(regionIndex < viewportIndex);
  assert.match(source, /role="status"/);
  assert.match(source, /aria-live="polite"/);
  assert.match(source, /aria-atomic="true"/);
  assert.match(
    source,
    /useAuiEvent\("thread\.runStart", \(\) => setAnnouncement\("Generating response\.\.\."\)\)/,
  );
  assert.match(
    source,
    /useAuiEvent\("thread\.runEnd", \(\) => setAnnouncement\("Response complete\."\)\)/,
  );
  assert.doesNotMatch(source, /setTimeout\(\(\) => \{\s*setAnnouncement/);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const mixer = readSrc("features/audio/components/stem-mixer.tsx");
const sources = readSrc("features/audio/hooks/use-stem-sources.ts");

test("stem audio uses its own pinned cache, not the gallery's LRU", () => {
  assert.match(sources, /new BlobUrlCache\(/);
  assert.doesNotMatch(sources, /galleryCache/);
  assert.match(sources, /cache\.clear\(\)/);
  // The CSP's connect-src refuses fetch() on blob: URLs, so peaks decode from the Blob itself.
  assert.match(sources, /blob\.arrayBuffer\(\)/);
  assert.doesNotMatch(sources, /fetch\(objectUrl\)/);
});

test("a stem that failed to load is left out of playback, so the rest still play", () => {
  assert.match(
    mixer,
    /const playable = useMemo\(\s*\(\) => stems\.filter\(\(stem\) => !stem\.failed\)/,
  );
  assert.match(
    mixer,
    /playable\.map\(\(stem\) => \(\{ id: stem\.clipId, src: stem\.src \}\)\)/,
  );
  assert.match(
    mixer,
    /const loadingCount = playable\.filter\(\(stem\) => !stem\.src\)\.length/,
  );
  assert.match(mixer, /disabled=\{stems\.some\(\(stem\) => !stem\.src\)\}/);
});

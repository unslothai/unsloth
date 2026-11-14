// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("media rails reserve a classic scrollbar out of their right padding, not their content", () => {
  for (const page of ["images/images-page.tsx", "video/video-page.tsx"]) {
    assert.match(
      readSrc(`features/${page}`),
      /px-10 max-sm:px-5 pt-9 pb-6 @\[50rem\]:overflow-y-auto @\[50rem\]:\[scrollbar-gutter:stable\] @\[50rem\]:pr-\[calc\(var\(--spacing\)\*10-var\(--classic-scrollbar-width,0px\)\)\]/,
      page,
    );
  }
  // Audio's rail is narrower-padded (24px), and reserves the scrollbar out of that padding the same way.
  assert.match(
    readSrc("features/audio/audio-page.tsx"),
    /px-6 pt-6 pb-4 @\[50rem\]:mt-\[calc\(60px\*var\(--ui-space-scale,1\)\)\] @\[50rem\]:overflow-y-auto @\[50rem\]:pt-2 @\[50rem\]:\[scrollbar-gutter:stable\] @\[50rem\]:pr-\[calc\(var\(--spacing\)\*6-var\(--classic-scrollbar-width,0px\)\)\]/,
  );
});

test("the classic scrollbar's width is published, and cleared for overlay scrollbars", () => {
  const lib = readSrc("lib/overlay-scrollbar.ts");
  assert.match(lib, /const classic = probe\.offsetWidth - probe\.clientWidth;/);
  assert.match(
    lib,
    /if \(classic > 0\) \{\s*root\.style\.setProperty\(CLASSIC_SCROLLBAR_WIDTH_VAR, `\$\{classic\}px`\);\s*\} else \{\s*root\.style\.removeProperty\(CLASSIC_SCROLLBAR_WIDTH_VAR\);/,
  );
});

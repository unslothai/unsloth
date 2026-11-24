// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

// Audio stays mounted off-route (app/routes/__root.tsx) and these dialogs portal to body.
test("the Delete all confirmations close when the Audio page is hidden", () => {
  const card = readSrc("features/audio/components/clips-card.tsx");
  assert.match(card, /if \(!active\) setOpen\(false\)/);
  assert.match(card, /<AlertDialog open=\{active && open\}/);
  for (const page of ["features/audio/pages/tts-workspace.tsx", "features/audio/pages/separate-page.tsx"]) {
    for (const use of readSrc(page).match(/<ClipsDeleteAll\b[^>]*>/g) ?? []) {
      assert.match(use, /\bactive=\{active\}/, `${page}: ${use}`);
    }
  }
  assert.match(
    readSrc("features/audio/transcript-gallery.tsx"),
    /<AlertDialog open=\{active && confirmingClear\}/,
  );
});

test("Convert decodes both sides once instead of on every switch", () => {
  const source = readSrc("features/audio/components/convert-ab-compare.tsx");
  assert.match(source, /useClipPeaks\(\[barsFor\(a\), barsFor\(b\)\]\)/);
});

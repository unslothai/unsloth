// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// The composer's + menu and every submenu under it (More, Saved prompts, Export chat, Projects)
// are one width, so a submenu never reads as a size bigger or smaller than the menu it opened from.
// Both composers: the chat's own and the one Compare chat uses.
for (const file of ["components/assistant-ui/thread.tsx", "features/chat/shared-composer.tsx"]) {
  test(`every + menu level in ${file} is as wide as the menu itself`, async () => {
    const source = await readSrcAsync(file);
    const widths = [
      ...source.matchAll(/className="unsloth-plus-menu w-\[calc\((\d+)px\*var\(--ui-space-scale,1\)\)\]"/g),
    ].map((match) => Number(match[1]));
    // The + menu, More, Saved prompts, Export chat and Projects.
    assert.equal(widths.length, 5);
    assert.deepEqual(new Set(widths), new Set([244]));
  });
}

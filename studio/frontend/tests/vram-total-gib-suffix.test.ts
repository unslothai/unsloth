// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `memoryTotalGb` is binary, so its label must say GiB.
 *
 * The backend builds it as `props.total_memory / 1024**3`
 * (utils/hardware/hardware.py) and subtracts no budget, reserve or headroom, so
 * the number is GiB whatever the name suggests. Four renderers printed it with a
 * `GB` suffix, which made a 48 GB card read as "45 GB" in the model information
 * sidebar and was reported as broken VRAM detection (issue #9551). 45 GiB is
 * 48.3 GB, so the number was right the whole time.
 *
 * The field name cannot be changed without churning the API, so the suffix is
 * pinned here instead: it is one word, it reads like a typo fix, and it came
 * back on a fourth surface (the video GPU picker) after the first three were
 * spotted.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const SRC = new URL("../src/", import.meta.url);

// Every file that renders a GPU's total memory to the user.
const RENDERERS = [
  "features/model-picker/components/model-config-page.tsx",
  "features/images/images-page.tsx",
  "features/video/video-page.tsx",
  "features/studio/wizard/run-preview-card.tsx",
  "features/hub/hub-page.tsx",
  "features/settings/tabs/about-tab.tsx",
];

// A template chunk that interpolates a binary total and then names a unit.
// `[^`$]*` keeps the match inside one literal segment, so an unrelated "GB"
// later in the file cannot be paired with an earlier total.
const SUFFIXED = /\$\{[^}]*(?:memoryTotalGb|vramTotalGb|systemRamTotalGb)[^}]*\}[^`$]*?\b(GiB|GB)\b/g;

test("VRAM and RAM totals are labelled GiB, not GB", () => {
  const offenders: string[] = [];
  let matched = 0;

  for (const relative of RENDERERS) {
    const source = readFileSync(new URL(relative, SRC), "utf8");
    for (const match of source.matchAll(SUFFIXED)) {
      matched += 1;
      if (match[1] !== "GiB") {
        offenders.push(`${relative}: ${match[0].trim()}`);
      }
    }
  }

  // Guards the regex itself: a rename that stops it matching would otherwise
  // make this test pass by inspecting nothing.
  assert.ok(
    matched >= RENDERERS.length,
    `expected at least one labelled total per renderer, matched ${matched}`,
  );
  assert.deepEqual(
    offenders,
    [],
    `these render a GiB value with a GB suffix:\n${offenders.join("\n")}`,
  );
});

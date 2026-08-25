// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `memoryTotalGb` is binary, so its label must say GiB.
 *
 * The backend builds it as `props.total_memory / 1024**3` and subtracts no
 * budget or reserve, so a 48 GB card printed "45 GB" and was reported as broken
 * VRAM detection (#9551). The field name cannot change without churning the
 * API, so the suffix is pinned here: it reads like a typo fix, and it kept
 * turning up on surfaces the previous pass had missed.
 */

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";

const SRC = new URL("../src/", import.meta.url);

// Files that interpolate the total and name the unit inline.
const RENDERERS = [
  "features/model-picker/components/model-config-page.tsx",
  "features/images/images-page.tsx",
  "features/video/video-page.tsx",
  "features/studio/wizard/run-preview-card.tsx",
  "features/hub/hub-page.tsx",
  "features/settings/tabs/about-tab.tsx",
];

// `[^`$]*` keeps the match inside one literal segment, so an unrelated "GB"
// later in the file cannot be paired with an earlier total.
const SUFFIXED = /\$\{[^}]*(?:memoryTotalGb|vramTotalGb|systemRamTotalGb)[^}]*\}[^`$]*?\b(GiB|GB)\b/g;

test("VRAM and RAM totals are labelled GiB, not GB", () => {
  const offenders: string[] = [];

  for (const relative of RENDERERS) {
    const source = readFileSync(new URL(relative, SRC), "utf8");
    const matches = [...source.matchAll(SUFFIXED)];

    // Per file, not summed: hub-page.tsx matches twice, so an aggregate floor
    // would let one renderer drop to zero matches and still pass.
    assert.ok(matches.length > 0, `${relative}: no labelled total found`);

    for (const match of matches) {
      if (match[1] !== "GiB") {
        offenders.push(`${relative}: ${match[0].trim()}`);
      }
    }
  }

  assert.deepEqual(
    offenders,
    [],
    `these render a GiB value with a GB suffix:\n${offenders.join("\n")}`,
  );
});

// These format through a helper, so the unit choice moves to the call site.
const HELPER_RENDERERS = [
  "components/floating-monitor.tsx",
  "features/settings/tabs/resources-tab.tsx",
];

// The training picker names its unit in the translation, so pin it per locale.
// `{est}` stays decimal GB: estimateLoadingVram divides by 1e9.
const LOCALES = new URL("../src/i18n/locales/", import.meta.url);
// Binary first: GiB has to win the alternation before GB can match its tail.
const UNIT = /\{total\}\s*(GiB|Gio|ГиБ|GB|Go|ГБ)/;
const BINARY_UNITS = ["GiB", "Gio", "ГиБ"];

test("localized GPU totals name a binary unit", () => {
  const files = readdirSync(LOCALES).filter((f) => f.endsWith(".ts"));
  assert.ok(files.length > 0, "no locales found");

  for (const file of files) {
    const source = readFileSync(new URL(file, LOCALES), "utf8");
    for (const line of source.split("\n")) {
      if (!/\b(vramNeeds|vramTight):/.test(line)) continue;
      const unit = line.match(UNIT);
      assert.ok(unit, `${file}: no unit after {total} in ${line.trim()}`);
      assert.ok(
        BINARY_UNITS.includes(unit[1]),
        `${file}: {total} is binary but is labelled ${unit[1]}`,
      );
    }
  }
});

test("helper-formatted totals pick the GiB helper, not the disk one", () => {
  for (const relative of HELPER_RENDERERS) {
    const source = readFileSync(new URL(relative, SRC), "utf8");

    const helper = source.match(/function formatGiB[\s\S]*?\n}/);
    assert.ok(helper, `${relative}: no formatGiB helper`);
    assert.match(helper[0], /`.*GiB`/, `${relative}: formatGiB does not suffix GiB`);

    // formatGb is the decimal disk one, so no memory value may reach it.
    for (const call of source.matchAll(/(?<!function )formatGb\(([^)]*)\)/g)) {
      assert.match(call[1], /disk/i, `${relative}: ${call[0]} formats memory as decimal GB`);
    }
  }
});

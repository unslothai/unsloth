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

// Both memory surfaces name their own units rather than going through
// formatGiB, and both divide by 1024 throughout -- including the gpuGb they
// measure against. Their formatters used to live in two feature modules, one per
// surface, and this guard only knew about one of them: the OTHER divided by
// 1024**3 and labelled the result "GB" across seven figures on the Load Model
// panel, which is exactly the defect this file exists to prevent, sitting
// outside its reach the whole time.
//
// They are now one module, so the check is on that module and covers every
// formatter in it. Listed by name rather than swept, so deleting one is a
// failure here rather than a silent reduction in what is guarded.
const BINARY_FORMATTERS: [string, string][] = [
  ["lib/memory/format.ts", "formatGiB"],
  ["lib/memory/format.ts", "formatBytesGiB"],
  ["lib/memory/format.ts", "formatKvRate"],
];

test("every shared memory formatter names a binary unit", () => {
  for (const [relative, fn] of BINARY_FORMATTERS) {
    const source = readFileSync(new URL(relative, SRC), "utf8");
    const helper = source.match(
      new RegExp(`export function ${fn}[\\s\\S]*?\\n}`),
    );
    assert.ok(helper, `${relative}: no ${fn}`);
    // Any B-suffixed literal that is not preceded by "Ki", "Mi" or "Gi".
    assert.doesNotMatch(
      helper[0],
      /(?<!Ki|Mi|Gi)B["'`]/,
      `${relative}: ${fn} labels a binary value with a decimal unit`,
    );
    assert.match(
      helper[0],
      /(KiB|MiB|GiB)/,
      `${relative}: ${fn} names no binary unit at all`,
    );
  }
});

// The two surfaces reach those formatters through back-compat aliases that kept
// their old names. The aliases are the reason no call site had to change, and
// they are also how a future edit could quietly reintroduce a second
// implementation: redefining formatMemoryGb as a function in either module would
// shadow the shared one and this file would be none the wiser, since the check
// above only reads lib/memory/format.ts.
const ALIAS_ONLY = [
  "lib/model-memory.ts",
  "features/model-picker/model-config/memory-fit.ts",
];

test("the surfaces alias the shared formatters rather than redefining them", () => {
  for (const relative of ALIAS_ONLY) {
    const source = readFileSync(new URL(relative, SRC), "utf8");
    assert.doesNotMatch(
      source,
      /export function formatMemoryGb\b/,
      `${relative}: formatMemoryGb is defined here again. There were once two of ` +
        `these, with the same name and signature but different input units and ` +
        `different labels, and importing the wrong one typechecked cleanly.`,
    );
    assert.doesNotMatch(
      source,
      /export function formatKvRate\b/,
      `${relative}: formatKvRate is defined here again`,
    );
    assert.match(
      source,
      /from "\.\.?\/[^"]*memory\/format\.ts"/,
      `${relative}: no longer sources its formatters from the shared module`,
    );
  }
});

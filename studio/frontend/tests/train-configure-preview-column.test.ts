// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";

import { SIDEBAR_WIDTH_DEFAULT } from "../src/hooks/use-sidebar-width.ts";

// widest window that must still show both columns with the sidebar expanded; the previous 64rem tier needed 1376px
const WINDOW_BUDGET_PX = 1280;

// container queries resolve rem against the root font size, which studio leaves at the browser default
const ROOT_FONT_PX = 16;

const STUDIO_PAGE = new URL(
  "../src/features/studio/studio-page.tsx",
  import.meta.url,
);
const TRAINING_WIZARD = new URL(
  "../src/features/studio/wizard/training-wizard.tsx",
  import.meta.url,
);
const STUDIO_CSS = new URL("../src/index.css", import.meta.url);
const WIZARD_DIR = new URL("../src/features/studio/wizard/", import.meta.url);
const SECTIONS_DIR = new URL(
  "../src/features/studio/sections/",
  import.meta.url,
);

const CONTAINER_SCALE = /--container-([\w-]+):\s*([\d.]+)rem/g;
const SPACING_SCALE = /--spacing:\s*([\d.]+)rem/;
const CLASS_NAME = /className="([^"]+)"/g;
const TWO_COLUMN_RULE =
  /@([\w-]+)\/train-configure:grid-cols-\[minmax\(0,1fr\)_(\d+)px\]/;
// arbitrary variants such as @min-[40rem] are captured too, so an unresolvable tier throws rather than dropping out of the budget
const CONFIGURE_GAP = /@([^/\s"]+)\/train-configure:gap-([\d.]+)/g;
const TRAIN_SECTION_TIER = /@([^/\s"]+)\/train-section:/g;
const PAGE_MAX_WIDTH = /max-w-\[(\d+)px\]/;
const PAGE_PADDING = /(?:^|\s)sm:px-(\d+)(?:\s|$)/;
const CARD_BORDER = /\.elevated-card\s*\{[^}]*?border:\s*(\d+)px/;

/** tailwind's own --container-* and --spacing scales, in px */
async function tailwindScales(): Promise<{
  containers: Map<string, number>;
  spacing: number;
}> {
  const themePath = createRequire(import.meta.url).resolve(
    "tailwindcss/theme.css",
  );
  const theme = await readFile(themePath, "utf8");
  const containers = new Map<string, number>();
  for (const [, name, rem] of theme.matchAll(CONTAINER_SCALE)) {
    containers.set(name, Number(rem) * ROOT_FONT_PX);
  }
  const spacing = SPACING_SCALE.exec(theme);
  if (!spacing) {
    throw new Error("tailwind theme has no --spacing");
  }
  return { containers, spacing: Number(spacing[1]) * ROOT_FONT_PX };
}

/** the single className in source that contains marker */
function classNameContaining(source: string, marker: string): string {
  const found = [...source.matchAll(CLASS_NAME)]
    .map((match) => match[1])
    .filter((value) => value.includes(marker));
  if (found.length !== 1) {
    throw new Error(
      `expected one className with ${marker}, got ${found.length}`,
    );
  }
  return found[0];
}

/** a bare utility such as gap-8 or p-5, in spacing steps */
function baseStep(className: string, utility: string): number {
  const match = new RegExp(`(?:^|\\s)${utility}-(\\d+)(?:\\s|$)`).exec(
    className,
  );
  if (!match) {
    throw new Error(`no base ${utility}-* in "${className}"`);
  }
  return Number(match[1]);
}

/** narrowest @*\/train-section tier in top-level wizard and section files, in px */
async function smallestTrainSectionTier(
  containers: Map<string, number>,
): Promise<number> {
  const widths: number[] = [];
  for (const dir of [WIZARD_DIR, SECTIONS_DIR]) {
    for (const entry of await readdir(dir)) {
      if (!entry.endsWith(".tsx")) {
        continue;
      }
      const source = await readFile(new URL(entry, dir), "utf8");
      for (const [, tier] of source.matchAll(TRAIN_SECTION_TIER)) {
        const px = containers.get(tier);
        if (px === undefined) {
          throw new Error(
            `unresolvable tier @${tier}/train-section in ${entry}`,
          );
        }
        widths.push(px);
      }
    }
  }
  if (widths.length === 0) {
    throw new Error("no train-section container queries found");
  }
  return Math.min(...widths);
}

test("the run preview column reaches a laptop window without dropping the wizard below @md/train-section", async () => {
  const { containers, spacing } = await tailwindScales();
  const page = await readFile(STUDIO_PAGE, "utf8");
  const wizard = await readFile(TRAINING_WIZARD, "utf8");

  // every tier below is inert unless the named containers are still declared
  classNameContaining(page, "@container/train-configure");
  classNameContaining(wizard, "@container/train-section");

  const grid = classNameContaining(page, "grid-cols-[minmax(0,1fr)_");
  const columns = TWO_COLUMN_RULE.exec(grid);
  assert.ok(columns, "no train-configure two-column rule on the grid");
  const [, tierName, previewTrack] = columns;

  const thresholdPx = containers.get(tierName);
  assert.ok(thresholdPx, `unknown container tier @${tierName}`);
  const previewPx = Number(previewTrack);

  // the preview only earns its sticky offset once it is a column, so both switch on the same tier
  const sticky = classNameContaining(page, "/train-configure:sticky");
  assert.match(
    sticky,
    new RegExp(`@${tierName}/train-configure:sticky`),
    `preview sticks at a different tier than the ${tierName} column split`,
  );

  // gap at the narrowest two-column width: the base gap unless a variant at or below the split tier replaces it
  let gapPx = baseStep(grid, "gap") * spacing;
  let gapTierPx = 0;
  for (const [, variantTier, step] of grid.matchAll(CONFIGURE_GAP)) {
    const variantPx = containers.get(variantTier);
    if (variantPx === undefined) {
      throw new Error(
        `unresolvable tier @${variantTier}/train-configure on the gap`,
      );
    }
    if (variantPx <= thresholdPx && variantPx >= gapTierPx) {
      gapPx = Number(step) * spacing;
      gapTierPx = variantPx;
    }
  }

  const wrapper = classNameContaining(page, "mx-auto");
  const maxWidth = PAGE_MAX_WIDTH.exec(wrapper);
  assert.ok(maxWidth, "page wrapper has no explicit max width");
  const padding = PAGE_PADDING.exec(wrapper);
  assert.ok(padding, "page wrapper has no sm:px-* padding");
  const paddingPx = Number(padding[1]) * spacing;

  // a threshold above the page's own width cap can never match, however wide the window gets
  const widestContainerPx = Number(maxWidth[1]) - 2 * paddingPx;
  assert.ok(
    thresholdPx <= widestContainerPx,
    `@${tierName} (${thresholdPx}px) never matches inside a ${widestContainerPx}px container`,
  );

  // the container is min(cap, window - sidebar) - padding, so the tier is really a window-width requirement
  const windowPx = thresholdPx + 2 * paddingPx + SIDEBAR_WIDTH_DEFAULT;
  assert.ok(
    windowPx <= WINDOW_BUDGET_PX,
    `two columns need a ${windowPx}px window, budget is ${WINDOW_BUDGET_PX}px`,
  );

  // and the wizard column still clears the narrowest tier its own sections use
  const card = classNameContaining(wizard, "@container/train-card");
  assert.ok(
    card.includes("elevated-card"),
    "the training card no longer uses elevated-card, so its border is unknown",
  );
  const css = await readFile(STUDIO_CSS, "utf8");
  const border = CARD_BORDER.exec(css);
  assert.ok(border, "elevated-card has no border width to subtract");
  const cardInsetPx = (baseStep(card, "p") * spacing + Number(border[1])) * 2;
  const wizardSectionPx = thresholdPx - gapPx - previewPx - cardInsetPx;
  const smallestSectionTier = await smallestTrainSectionTier(containers);
  assert.ok(
    wizardSectionPx >= smallestSectionTier,
    `wizard section gets ${wizardSectionPx}px, below the ${smallestSectionTier}px train-section tier`,
  );
});

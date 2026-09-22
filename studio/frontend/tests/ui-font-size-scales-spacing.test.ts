// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readdirSync } from "node:fs";
import { join } from "node:path";

import { readSrc } from "./helpers/kit.ts";

/** Every source that can carry a length. */
const SOURCES = (function walk(dir: string): string[] {
  return readdirSync(join(import.meta.dirname, "../src", dir), {
    withFileTypes: true,
  }).flatMap((entry) => {
    const path = dir ? `${dir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) return walk(path);
    return /\.(tsx?|css)$/.test(entry.name) ? [path] : [];
  });
})("");

// UI font size has to move what separates the text too: a 20px label in a row
// padded for 12px reads as cramped. --ui-space-scale is the one multiplier
// every such length goes through.

const CSS = readSrc("index.css");
const SIDEBAR = readSrc("components/app-sidebar.tsx");
const PROVIDER = readSrc("app/provider.tsx");

const SCALED = /calc\([\d.]+(?:px|rem)\s*\*\s*var\(--ui-space-scale,\s*1\)\)/;

test("the spacing scale is the font-size preference, normalised at the default", () => {
  // --ui-font-scale is against a 16px base and the default is 15px, so
  // dividing by the default leaves the shipped spacing untouched.
  assert.match(
    CSS,
    /--ui-space-scale:\s*calc\(var\(--ui-font-scale, 1\) \/ 0\.9375\);/,
  );
  assert.match(CSS, /--ui-font-scale:\s*0\.9375;/);
});

test("every tailwind spacing utility goes through it", () => {
  // p-*, m-*, gap-* and size-* are all calc(var(--spacing) * n), so the step
  // scales the whole layout at once. The @theme passthrough is commented out,
  // so it is not a declaration.
  const declarations = (CSS.match(/--spacing:[^;]+;/g) ?? []).filter(
    (declaration) => !declaration.includes("var(--spacing)"),
  );
  assert.equal(declarations.length, 2, "light and dark each declare --spacing");
  for (const declaration of declarations) {
    assert.match(
      declaration,
      /--spacing:\s*calc\(0\.25rem \* var\(--ui-space-scale, 1\)\);/,
    );
  }
});

test("em lengths are left alone, they already follow the text", () => {
  // rem is against the 16px root, which the preference never touches, so it
  // needs the multiplier. em is against the element's own font size, which
  // does move, so scaling it applies the preference twice.
  const withEm = SOURCES.filter((file) =>
    /[^a-z0-9.]-?[\d.]+em\s*\*\s*var\(--ui-space-scale/.test(readSrc(file)),
  );
  assert.deepEqual(withEm, [], "these em lengths take the scale twice");
  const pickers = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.match(pickers, /mx-\[-0\.1em\]/);
  assert.match(pickers, /ml-\[0\.14em\]/);
});

test("chat chrome that holds text scales, window chrome does not", () => {
  // The header band and the controls in it grow with their labels.
  for (const name of [
    "--studio-chat-header-height",
    "--studio-chat-control-height",
    "--studio-chat-header-padding-top",
  ]) {
    for (const source of [CSS, PROVIDER]) {
      const declaration = new RegExp(
        `${name}"?:\\s*"?(calc\\([^;\\n"]*\\)|[^;,\\n]+)`,
      ).exec(source);
      assert.ok(declaration, `${name} is gone`);
      assert.match(declaration[1], SCALED, `${name} must follow the UI scale`);
    }
  }

  // The title bar and the traffic lights belong to the OS window; a font size
  // preference must not move the UI off them.
  for (const name of [
    "--studio-desktop-titlebar-height",
    "--studio-content-top-inset",
  ]) {
    const declaration = new RegExp(`${name}":\\s*"([^"]+)"`).exec(PROVIDER);
    if (declaration) {
      assert.doesNotMatch(declaration[1], /--ui-space-scale/, name);
    }
  }
});

test("the sidebar's hand-set spacing follows the scale", () => {
  // The rows use one-off px values (gap-[8.5px], pl-[39px]), which is exactly
  // the spacing that used to stay put while the labels grew.
  const bare = [
    ...SIDEBAR.matchAll(
      /(?<![\w-])-?(?:p|px|py|pt|pb|pl|pr|m|mx|my|mt|mb|ml|mr|gap|gap-x|gap-y)-\[(\d*\.?\d+)px\]/g,
    ),
  ].filter((match) => Number(match[1]) > 1);
  assert.deepEqual(
    bare.map((match) => match[0]),
    [],
    "these sidebar paddings ignore the UI font size",
  );
});

test("fixed slots that hold scaled content scale with it", () => {
  // These heights live in JS, so the CSS variable cannot reach them. A slot
  // left at its 15px value clips or overlaps the row it holds.
  const HOOK = "useUiSpaceScale";
  const rows = readSrc("features/hub/catalog/models-catalog-rows.tsx");
  assert.ok(rows.includes(HOOK), "the virtualizer ignores the UI font size");
  assert.match(rows, /estimateSize: \(\) => slotHeight/);
  assert.match(rows, /height: `\$\{slotHeight\}px`/);
  // estimateSize is cached, so a new scale has to invalidate the sizes.
  assert.match(rows, /virtualizer\.measure\(\)/);

  const carousel = readSrc("features/hub/catalog/hub-section-row.tsx");
  assert.ok(carousel.includes(HOOK), "the carousel slot ignores it");
  assert.match(carousel, /itemHeight=\{cardHeight\}/);
  // The stride and the arrow's centre line are measured in JS against gap-4
  // and pt-2, which scale, so an unscaled stride stops short of the next card.
  const strip = readSrc("features/hub/catalog/card-carousel.tsx");
  assert.ok(strip.includes(HOOK), "the carousel stride ignores it");
  assert.match(strip, /const gapPx = CARD_GAP_PX \* scale;/);
  assert.match(strip, /const topPaddingPx = CAROUSEL_TOP_PADDING_PX \* scale;/);

  const lists = readSrc("features/hub/catalog/models-catalog-lists.tsx");
  assert.ok(lists.includes(HOOK), "the pinned grid ignores it");
  // The pinned block sits directly above the virtualized rows in the same
  // lanes, so a gutter of its own would step the card widths.
  assert.match(
    lists,
    /Math\.round\(CATALOG_COLUMN_GAP_PX \* pinnedScale\)/,
  );
  assert.doesNotMatch(lists, /columnGap: 12,/);
});

test("the titlebar reserves room for its controls, which stay in the band", () => {
  const titlebar = readSrc("components/tauri/window-titlebar.tsx");
  // The band is a fixed 34px and clips nothing, so a grown button would hang
  // over the page and take its clicks.
  assert.match(titlebar, /inline-flex size-\[30px\] shrink-0/);
  // The spacer stands in for one of those buttons while the navbar renders
  // its own trigger, so it holds the same fixed width.
  assert.match(titlebar, /aria-hidden="true" className="size-\[30px\] shrink-0"/);
  assert.match(titlebar, /inline-flex h-\[26px\] w-\[26px\] shrink-0/);
  // The padding and gaps around them still scale, so the drag region has to
  // start further out or it covers the last button.
  // max(), because the buttons are fixed: the slot may grow with the padding
  // around them but must never fall under their own width.
  assert.match(
    titlebar,
    /max\(7rem, calc\(7rem \* var\(--ui-space-scale, 1\)\)\)/,
  );
});

test("the composer's one-row clamp is one row at any size", () => {
  // The editor box wins over the input's own min-height, so all three clamps
  // and the JS floor have to move together or an empty composer clips.
  const clamps = CSS.match(/calc\(40px \* var\(--ui-space-scale, 1\)\)/g) ?? [];
  assert.equal(clamps.length, 3, "a 40px composer clamp is still fixed");
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /const oneRowHeight = Math\.round\(40 \* uiSpaceScale\);/);
  assert.doesNotMatch(thread, /Math\.max\(40, editorHeight\)/);
});

test("overlays that scale cannot outgrow the screen", () => {
  // w-* follows --spacing too, so a w-72 popover is 384px at the 20px
  // setting, past a 375px phone. Each floating surface caps itself.
  for (const file of [
    "components/ui/popover.tsx",
    "components/ui/dropdown-menu.tsx",
    "components/ui/select.tsx",
  ]) {
    assert.match(
      readSrc(file),
      /max-w-\[calc\(100vw-32px\)\]/,
      `${file} can overflow a narrow viewport`,
    );
  }
});

test("the artifact panel's vertical budget adds up at any size", () => {
  // The shell's mb-8 scales. If the offsets above and below it do not, the
  // panel outgrows its pane and the bottom is clipped.
  const surface = readSrc("features/chat/artifacts/artifact-surface.tsx");
  assert.match(surface, /\bmb-8\b/, "the panel no longer carries mb-8");
  // The titlebar band inside the 90 is window chrome and stays fixed, so it
  // comes out of the offset before the rest is scaled and goes back in after.
  assert.match(
    surface,
    /marginTop:\s*\n?\s*"calc\(var\(--studio-content-top-inset, 0px\) \+ \(90px - var\(--studio-content-top-inset, 0px\)\) \* var\(--ui-space-scale, 1\) \+ var\(--studio-chat-notice-height, 0px\)\)"/,
  );
  assert.match(
    surface,
    /height:\s*\n?\s*"calc\(100% - var\(--studio-content-top-inset, 0px\) - \(122px - var\(--studio-content-top-inset, 0px\)\) \* var\(--ui-space-scale, 1\) - var\(--studio-chat-notice-height, 0px\)\)"/,
  );
  // 90 above plus the 32 of mb-8 is the 122 taken off the pane.
  assert.equal(90 + 8 * 4, 122);
});

test("the stylesheet's comments stay comments", () => {
  // A "*/" inside the prose ends the comment early and postcss then reads the
  // rest of the sentence as a declaration, taking the next one with it.
  const stripped = CSS.replace(/\/\*[\s\S]*?\*\//g, "");
  assert.doesNotMatch(
    stripped,
    /\butility follows\b|\bTailwind's\b/,
    "prose leaked out of a comment",
  );
  assert.ok(
    stripped.includes("--spacing: calc(0.25rem * var(--ui-space-scale, 1));"),
    "the spacing step is no longer a declaration of its own",
  );
});

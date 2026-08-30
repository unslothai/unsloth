// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

const PICKERS = read(
  "../src/features/model-picker/components/model-selector/pickers.tsx",
);
const MODELS_TABLE = read("../src/features/hub/catalog/models-table.tsx");
const SIDEBAR = read("../src/components/app-sidebar.tsx");
const CSS = read("../src/index.css");

test("a downloaded row is marked the way the Hub marks one", () => {
  const start = PICKERS.indexOf("function DownloadedBadge()");
  const badge = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  // A download arrow read as "click to fetch this" on the one row that needs
  // no fetching. The Hub already had the right answer.
  assert.ok(!badge.includes("Download01Icon"), "no download glyph");
  assert.match(badge, /size-\[5px\] rounded-full bg-status-success/);
  assert.match(badge, /aria-label="On device"/);
  assert.ok(
    MODELS_TABLE.includes("bg-status-success"),
    "and the Hub still uses that dot, so the two agree",
  );
});

test("the download glyph is gone from the picker entirely", () => {
  // Left behind, it would still be imported for nothing.
  assert.ok(!PICKERS.includes("Download01Icon"));
});

test("the scoped badge column reserves the wider on-device marker", () => {
  // Video can show one 18px capability, a 4px gap and the 14px marker. If the
  // fixed width remains 34px, min-w-min expands only those rows and shifts all
  // metadata columns after the badge slot.
  assert.ok(PICKERS.includes('badgeMid: "min-w-min min-[560px]:w-[36px]"'));
});

test("the unscoped badge column is sized per list, not to the union of both", () => {
  // One slot sized for both lists left ~44px of empty column on every On Device row. 26px is the
  // vision badge measured, so rows with and without one keep the same column positions.
  assert.ok(PICKERS.includes('badgeDevice: "min-w-min min-[560px]:w-[26px]"'));
  assert.ok(PICKERS.includes('badgeWide: "min-w-min min-[560px]:w-[36px]"'));
  assert.match(
    PICKERS,
    /alignMeta === "device"\n\s*\? META_COLUMN\.badgeDevice\n\s*: META_COLUMN\.badgeWide/,
  );
});

test("a row's leading dot starts where its section label does", () => {
  // Section labels sit at px-2.5. The dot is centred in a 14px hover target, so its slot starts
  // at 10 - (14 - 5) / 2 = 5.5px for the dot to land on 10px. px-2 put it at 12.5px.
  assert.match(PICKERS, /py-1\.5 pl-\[5\.5px\] pr-2 text-left text-sm/);
  const label = PICKERS.slice(PICKERS.indexOf("flex items-center justify-between gap-1 px-2.5"));
  assert.ok(label.startsWith("flex items-center justify-between gap-1 px-2.5"));
  // The 14px hover target is what makes 5.5 the right number; shrinking it would move the dot.
  assert.ok(PICKERS.includes('className="flex size-[14px] shrink-0 items-center justify-center"'));
});

test("the parameter and size columns are sized to the ink they hold", () => {
  // The widest size formatBytes writes ("128GB", no space) is 29.5px, not the ~40px a spaced
  // "536 MB" would need.
  assert.ok(PICKERS.includes('size: "min-w-min min-[560px]:w-[3.2em]"'));
  // The no-space format is what makes 3.2em enough; a spaced size would need ~4.2em again.
  assert.match(
    PICKERS,
    /No space: "145MB" reads as one value beside the quant chip\./,
  );
});

test("the parameter chip hugs its label so the gap to the modality mark is the row's own", () => {
  // A fixed right-aligned column spends its leftover in front of the chip, and that leftover is
  // the gap to the modality mark, which no gap setting can undo. Hugging plus -ml-0.5 gives 2px.
  assert.ok(PICKERS.includes('param: "min-w-min -ml-0.5"'));
  // Hub keeps a column: its labels run to "2779.5B".
  assert.ok(PICKERS.includes('paramWide: "min-w-min min-[560px]:w-[5.2em]"'));
});

test("an over budget row dims instead of putting a pill on every line", () => {
  // Recommended is mostly over budget on a normal GPU, so a pill per row was a wall of colour.
  // The row dims and the pill is painted only while that row is hovered or focused.
  assert.ok(PICKERS.includes("group/row flex w-full flex-col items-stretch"));
  assert.ok(
    PICKERS.includes(
      "rounded opacity-0 transition-opacity group-hover/row:opacity-100 group-focus-visible/row:opacity-100",
    ),
  );
  assert.ok(
    PICKERS.includes(
      '"opacity-60 transition-opacity group-hover/row:opacity-100 group-focus-visible/row:opacity-100"',
    ),
  );
  // The pill is hidden, not removed, and its slot keeps a width, so revealing it cannot reflow.
  assert.ok(PICKERS.includes('vram: "min-w-min min-[560px]:w-[4em]"'));
  // TIGHT is rare enough to stay put; only the OOM pill hides.
  const tight = PICKERS.slice(PICKERS.indexOf('if (status === "tight")'));
  assert.ok(!tight.slice(0, 300).includes("opacity-0"));
});

test("aligned meta slots spend their slack on the name", () => {
  // Centring a lone glyph splits the slack either side of it, reading as a gap on both sides.
  assert.ok(
    PICKERS.includes(
      '"flex shrink-0 items-center justify-end gap-1 text-ui-10"',
    ),
  );
});

test("every select-model surface shares that one badge", () => {
  // Images, Video and Audio render the same ModelSelector, so there is no
  // second copy of the badge to keep in step.
  const copies = [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
    "../src/features/audio/audio-page.tsx",
  ];
  for (const path of copies) {
    const src = read(path);
    assert.ok(
      src.includes("@/features/model-picker/components/model-selector"),
      `${path} uses the shared selector`,
    );
    assert.ok(
      !src.includes("DownloadedBadge"),
      `${path} has no badge of its own`,
    );
  }
});

test("list header actions end where a hovered row's action does", () => {
  // A row action is `right-0 pr-1.5` inside a pill the list inset by
  // unrailedRowPadding: 12px in normally, 11px under the desktop titlebar.
  assert.match(
    CSS,
    /\.sidebar-row-action \{\n\t\t@apply absolute top-0 bottom-0 right-0[^;]*pr-1\.5/,
  );
  const label = CSS.slice(CSS.indexOf(".sidebar-sticky-label {"));
  assert.match(label.slice(0, 400), /pl-\[16px\] pr-3 /);

  assert.ok(
    CSS.includes(
      ".sidebar-sticky-label.sidebar-sticky-label-desktop {\n\t\tpadding-right: 11px;",
    ),
  );
  assert.ok(
    CSS.includes(
      ".sidebar-sticky-label.sidebar-sticky-label-desktop-recents {\n\t\tpadding-right: 13px;",
    ),
  );

  assert.match(
    SIDEBAR,
    /const unrailedRowPadding = usesDesktopTitlebar \? "px-\[5px\]" : "px-1\.5";/,
  );
  assert.ok(
    SIDEBAR.includes(
      'const headerRightPadding = usesDesktopTitlebar\n    ? "sidebar-sticky-label-desktop"\n    : null;',
    ),
  );
  // Recents is nudged 2px right there and carries its padding with it.
  assert.ok(
    SIDEBAR.includes(
      'const recentsHeaderRightPadding = usesDesktopTitlebar\n    ? "sidebar-sticky-label-desktop-recents"\n    : null;',
    ),
  );
});

test("all three list headers take the same alignment", () => {
  // Pinned and Projects share one class string; Recents has its own because of
  // the translate. Two of the first, one of the second.
  const shared =
    SIDEBAR.split(
      '"sidebar-sticky-label sidebar-sticky-label-following group/sidebar-header gap-1", headerRightPadding,',
    ).length - 1;
  assert.equal(shared, 2, "Pinned and Projects");
  assert.ok(
    SIDEBAR.includes("recentsHeaderRightPadding,"),
    "and Recents applies its own",
  );
});

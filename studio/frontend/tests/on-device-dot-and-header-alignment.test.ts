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

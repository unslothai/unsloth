// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const APP_SIDEBAR = readSrc("components/app-sidebar.tsx");

const INDEX = readSrc("index.css");

// The pinned top rows run an action rather than open a page, so neither may
// mark itself active: nav rows paint one pill for both states, and an active
// action row therefore sits there looking permanently hovered.

/** The props of the NavItem carrying `label`, from its tag to the next one. */
function navItemFor(source: string, label: string): string {
  const rows = source.split("<NavItem").slice(1);
  const row = rows.find((chunk) => chunk.includes(label));
  assert.ok(row, `no NavItem renders ${label}`);
  return row;
}

test("New chat never marks itself active", async () => {
  const row = navItemFor(APP_SIDEBAR, "shell.navigation.newChat");
  assert.match(row, /active=\{false\}/);
});

test("Search never marks itself active either", async () => {
  const row = navItemFor(APP_SIDEBAR, 'label={t("shell.navigation.search")}');
  assert.match(row, /active=\{false\}/);
});

test("a nav row paints the same pill when active and when hovered", async () => {
  // The reason the rows above pass false. If active ever gets its own
  // background, that reason is gone and this can be revisited.
  const rule = /([^}]*)\{\s*background-color: var\(--nav-surface-hover\)/.exec(
    INDEX,
  );
  assert.ok(rule, "no rule paints a nav row with --nav-surface-hover");
  assert.match(rule[1], /\.sidebar-nav-btn:hover/);
  assert.match(rule[1], /\.sidebar-nav-btn\[data-active="true"\]/);
});

test("desktop branding clears the titlebar actions", async () => {
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /shrink-0 p-0 pt-\[calc\(var\(--studio-desktop-titlebar-height,34px\)\+17px\)\]/,
  );
});

test("desktop branding keeps an 11px gap above New chat", async () => {
  const source = APP_SIDEBAR;
  assert.match(source, /usesDesktopTitlebar \? "pt-\[11px\]" : "pt-\[9px\]"/);
});

test("footer profile sits 11px above the sidebar edge", async () => {
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /relative pb-\[11px\] group-data-\[collapsible=icon\]:px-0/,
  );
});

test("navigation rows align while the profile footer ignores the scroll rail", async () => {
  // A recent chat's pill has to match New Chat's. The rows inside the scroller
  // lose the rail's width, so New Chat adds it back and both end on one edge.
  // The profile footer is unrelated to that list and must keep its full width
  // when the scrollbar appears. Logical sides, since the rail moves under rtl.
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /const rowPadding = usesDesktopTitlebar\s*\?\s*"ps-\[5px\] pe-\[calc\(var\(--sidebar-rail,0px\)\+5px\)\]"\s*:\s*"ps-1\.5 pe-\[calc\(var\(--sidebar-rail,0px\)\+6px\)\]"/,
  );
  assert.match(
    source,
    /const unrailedRowPadding = usesDesktopTitlebar \? "px-\[5px\]" : "px-1\.5"/,
  );
  // New Chat is the only outside row that aligns with the scroller's rail.
  assert.equal(source.match(/(?<!const )rowPadding[,}]/g)?.length, 1);
  // Nav rows, pinned chats, Projects, Recents, and training runs sit inside the
  // scroller; the footer is the sixth unrailed use outside it.
  assert.equal(source.match(/unrailedRowPadding[,}]/g)?.length, 6);

  const footer = source
    .split("<SidebarFooter")[1]
    ?.split("</SidebarFooter>")[0];
  assert.ok(footer, "no sidebar footer");
  const footerProps = footer.split(">\n")[0];
  assert.match(footerProps, /unrailedRowPadding/);
  assert.doesNotMatch(footerProps, /var\(--sidebar-rail/);
  assert.equal(source.match(/"pl-2 pr-\[5px\]"/g), null);
});

test("the sidebar list measures its scroll rail", async () => {
  // 0 where scrollbars overlay, the platform's thin rail where they are
  // classic. Read off the scroller and written to the DOM: state loops (#185).
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /const rail = el\.offsetWidth - el\.clientWidth;[\s\S]*el\.parentElement\?\.style\.setProperty\(\s*"--sidebar-rail",\s*`\$\{rail\}px`,?\s*\)/,
  );
  // A callback ref, not an effect: the Sheet unmounts on close and the
  // breakpoint swaps subtrees, so the scroller is a new node each time.
  assert.match(source, /ref=\{attachScroller\}/);
  assert.match(
    source,
    /const attachScroller = useCallback\(\s*\(el: HTMLDivElement \| null\) => \{/,
  );
  assert.equal(/useLayoutEffect/.test(source), false);
  // Old observer goes first, or a detached node keeps one.
  assert.match(source, /railObserverRef\.current\?\.disconnect\(\);/);
  // Cache is per node: a new parent has no variable even at the same width.
  assert.match(source, /railWidthRef\.current = null;/);
  // Measured on attach, or a list overflowing on arrival stays misaligned
  // until something fires a scroll.
  assert.match(source, /if \(!el\) return;\s*measureScrollRail\(el\);/);
  // Then off the box, not off renders: the Images disclosure and the project
  // toggles change the row count without rendering AppSidebar, and a scrollbar
  // appearing shrinks the content box.
  assert.match(
    source,
    /const observer = new ResizeObserver\(\(\) => measureScrollRail\(el\)\);\s*observer\.observe\(el\);\s*railObserverRef\.current = observer;/,
  );
  // Writes a variable, never state: that pairing is what looped.
  assert.equal(/new ResizeObserver\([^)]*set[A-Z]/.test(source), false);
  // And only on a change, so it cannot re-trigger itself.
  assert.match(source, /if \(rail === railWidthRef\.current\) return;/);
  // The fade stops at the rail too: the thumb ends its travel in that band.
  assert.match(
    source,
    /absolute start-0 end-\[var\(--sidebar-rail,0px\)\] bottom-full/,
  );
  // Only the Windows auto reset may set a width; hiding the rail is what a
  // width override caused before.
  const railWidthDecls = (
    INDEX.match(
      /\.sidebar-scroll-fade[^{]*\{[^}]*scrollbar-width:\s*[^;}]+/g,
    ) ?? []
  ).map((rule) => /scrollbar-width:\s*([^;}]+)/.exec(rule)?.[1].trim());
  assert.deepEqual(railWidthDecls, ["auto"]);
  assert.match(INDEX, /:root\.client-windows \.sidebar-scroll-fade,/);
  assert.equal(
    /\.sidebar-scroll-fade::-webkit-scrollbar \{/.test(INDEX),
    false,
  );
  // Thumb stays hidden until the list is hovered, as the other lists do.
  assert.match(INDEX, /\.sidebar-scroll-fade:hover::-webkit-scrollbar-thumb,/);
  // A mask covers the scrollbar, so the top fade keeps the rail column opaque.
  assert.match(
    INDEX,
    /mask-image: linear-gradient\(to bottom, transparent 0, #000 14px\),\s*linear-gradient\(to left, #000 var\(--sidebar-rail, 0px\), transparent 0\);/,
  );
  assert.match(
    INDEX,
    /\[dir="rtl"\] \.sidebar-scroll-fade\.is-scrolled \{[\s\S]*linear-gradient\(to right, #000 var\(--sidebar-rail, 0px\), transparent 0\);/,
  );
});

test("Tauri chat Recents label keeps its 2px shift", async () => {
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /scrolled && "is-scrolled",\s*usesDesktopTitlebar && "translate-x-\[2px\]"/,
  );
});

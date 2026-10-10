// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { atDefaultUiScale, readSrc } from "./helpers/kit.ts";

const APP_SIDEBAR = atDefaultUiScale(readSrc("components/app-sidebar.tsx"));

const INDEX = atDefaultUiScale(readSrc("index.css"));

// Action rows must not mark active: nav rows paint one pill for active and hover.

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

test("custom titlebar branding centers on the chat header's model picker", async () => {
  const header = APP_SIDEBAR.split("<SidebarHeader")[1].split("</SidebarHeader>")[0];
  assert.match(
    header,
    /usesCustomTitlebar\s*\?\s*"shrink-0 p-0 pt-\[calc\(var\(--studio-content-top-inset,0px\)\+var\(--studio-chat-header-padding-top,11px\)\)\]"/,
  );
  assert.match(
    header,
    /usesCustomTitlebar && "h-\[var\(--studio-chat-control-height,34px\)\]"/,
  );
});

test("desktop branding keeps an 11px gap above New chat", async () => {
  const source = APP_SIDEBAR;
  assert.match(source, /usesDesktopTitlebar \? "pt-\[11px\]" : "pt-\[7px\]"/);
});

test("footer profile sits 11px above the sidebar edge", async () => {
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /relative pb-\[11px\] group-data-\[collapsible=icon\]:px-0/,
  );
});

test("navigation rows align while the profile footer ignores the scroll rail", async () => {
  // Rows in the scroller lose the rail width, so New Chat adds it back to align.
  // The footer keeps full width. Logical sides, since the rail moves under rtl.
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /const rowPadding = usesDesktopTitlebar\s*\?\s*"ps-\[5px\] pe-\[calc\(var\(--sidebar-rail,0px\)\+5px\*var\(--ui-space-scale,1\)\)\]"\s*:\s*"ps-1\.5 pe-\[calc\(var\(--sidebar-rail,0px\)\+6px\*var\(--ui-space-scale,1\)\)\]"/,
  );
  assert.match(
    source,
    /const unrailedRowPadding = usesDesktopTitlebar \? "px-\[5px\]" : "px-1\.5"/,
  );
  assert.equal(source.match(/(?<!const )rowPadding[,}]/g)?.length, 1);
  assert.equal(source.match(/unrailedRowPadding[,}]/g)?.length, 7);

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
  // 0 for overlay scrollbars, else the thin rail width; written to the DOM to avoid state loops.
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /const rail = el\.offsetWidth - el\.clientWidth;[\s\S]*el\.parentElement\?\.style\.setProperty\(\s*"--sidebar-rail",\s*`\$\{rail\}px`,?\s*\)/,
  );
  // A callback ref: the Sheet unmounts on close, so the scroller is a new node each time.
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
  // Measured on attach, or a list overflowing on arrival stays misaligned.
  assert.match(source, /if \(!el\) return;\s*measureScrollRail\(el\);/);
  // Observe the box, not renders: row counts change without rendering AppSidebar.
  assert.match(
    source,
    /const observer = new ResizeObserver\(\(\) => \{\s*measureScrollRail\(el\);\s*syncFade\(el\);\s*\}\);\s*observer\.observe\(el\);/,
  );
  // Writes the DOM, never state: that pairing is what looped.
  assert.equal(/new ResizeObserver\([^)]*set[A-Z]/.test(source), false);
  assert.match(source, /if \(fade\.dataset\.visible !== visible\) fade\.dataset\.visible = visible;/);
  assert.equal(/setCanScrollDown/.test(source), false);
  assert.match(source, /for \(const section of el\.children\) observer\.observe\(section\);/);
  assert.match(source, /sections\.observe\(el, \{ childList: true \}\);/);
  // Only on a change, so it cannot re-trigger itself.
  assert.match(source, /if \(rail === railWidthRef\.current\) return;/);
  assert.match(
    source,
    /absolute start-0 end-\[var\(--sidebar-rail,0px\)\] bottom-full/,
  );
  // Only the Windows-wide auto reset may set a width; a width override hid the rail.
  const railWidthDecls = (
    INDEX.match(
      /\.sidebar-scroll-fade[^{]*\{[^}]*scrollbar-width:\s*[^;}]+/g,
    ) ?? []
  ).map((rule) => /scrollbar-width:\s*([^;}]+)/.exec(rule)?.[1].trim());
  assert.deepEqual(railWidthDecls, []);
  assert.match(
    INDEX,
    /:root\.client-windows \*,\s*:root\.client-windows \*:hover \{\s*scrollbar-width: auto;/,
  );
  assert.equal(
    /\.sidebar-scroll-fade::-webkit-scrollbar \{/.test(INDEX),
    false,
  );
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

test("Tauri chat Recents label takes the shared header inset, not a shift", async () => {
  const source = APP_SIDEBAR;
  assert.match(source, /headerInset,\s*scrolled && "is-scrolled",\s*!chatOpen/);
  assert.doesNotMatch(source, /translate-x-\[2px\]/);
});

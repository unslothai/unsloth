// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The pinned top rows run an action rather than open a page, so neither may
// mark itself active: nav rows paint one pill for both states, and an active
// action row therefore sits there looking permanently hovered.

async function sidebarSource(): Promise<string> {
  return readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
}

/** The props of the NavItem carrying `label`, from its tag to the next one. */
function navItemFor(source: string, label: string): string {
  const rows = source.split("<NavItem").slice(1);
  const row = rows.find((chunk) => chunk.includes(label));
  assert.ok(row, `no NavItem renders ${label}`);
  return row;
}

test("New chat never marks itself active", async () => {
  const row = navItemFor(await sidebarSource(), "shell.navigation.newChat");
  assert.match(row, /active=\{false\}/);
});

test("Search never marks itself active either", async () => {
  const row = navItemFor(
    await sidebarSource(),
    'label={t("shell.navigation.search")}',
  );
  assert.match(row, /active=\{false\}/);
});

test("a nav row paints the same pill when active and when hovered", async () => {
  // The reason the rows above pass false. If active ever gets its own
  // background, that reason is gone and this can be revisited.
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  const rule = /([^}]*)\{\s*background-color: var\(--nav-surface-hover\)/.exec(
    css,
  );
  assert.ok(rule, "no rule paints a nav row with --nav-surface-hover");
  assert.match(rule[1], /\.sidebar-nav-btn:hover/);
  assert.match(rule[1], /\.sidebar-nav-btn\[data-active="true"\]/);
});

test("desktop branding clears the titlebar actions", async () => {
  const source = await sidebarSource();
  assert.match(
    source,
    /shrink-0 p-0 pt-\[calc\(var\(--studio-desktop-titlebar-height,34px\)\+17px\)\]/,
  );
});

test("desktop branding keeps an 11px gap above New chat", async () => {
  const source = await sidebarSource();
  assert.match(source, /usesDesktopTitlebar \? "pt-\[11px\]" : "pt-\[9px\]"/);
});

test("footer profile sits 11px above the sidebar edge", async () => {
  const source = await sidebarSource();
  assert.match(
    source,
    /relative pb-\[11px\] group-data-\[collapsible=icon\]:px-0/,
  );
});

test("every sidebar row pill sits in one shared box", async () => {
  // A recent chat's pill has to match New Chat's. The rows inside the list
  // scroller lose the scrollbar rail's width, so the rows outside it add that
  // same width back and both end on one edge. Right pad matches left, so a
  // pill sits the same distance from the scrollbar as from the near edge.
  const source = await sidebarSource();
  assert.match(
    source,
    /const rowPadding = usesDesktopTitlebar\s*\?\s*"pl-\[5px\] pr-\[calc\(var\(--sidebar-rail,0px\)\+5px\)\]"\s*:\s*"pl-1\.5 pr-\[calc\(var\(--sidebar-rail,0px\)\+6px\)\]"/,
  );
  assert.match(
    source,
    /const scrollRowPadding = usesDesktopTitlebar\s*\?\s*"pl-\[5px\] pr-\[5px\]"\s*:\s*"pl-1\.5 pr-1\.5"/,
  );
  // New Chat and the footer sit outside the scroller.
  assert.equal(source.match(/(?<!const )rowPadding[,}]/g)?.length, 2);
  // Nav rows, pinned projects, Recents, training runs sit inside it.
  assert.equal(source.match(/scrollRowPadding[,}]/g)?.length, 4);
  assert.equal(source.match(/"pl-2 pr-\[5px\]"/g), null);
});

test("the sidebar list measures its scroll rail", async () => {
  // 0 where scrollbars overlay, 8px where they are classic. Read off the
  // scroller and written to the DOM: state here loops (React #185).
  const source = await sidebarSource();
  assert.match(
    source,
    /const rail = el\.offsetWidth - el\.clientWidth;[\s\S]*el\.parentElement\?\.style\.setProperty\(\s*"--sidebar-rail",\s*`\$\{rail\}px`,?\s*\)/,
  );
  // Measured before paint, or a list that overflows on arrival stays
  // misaligned until something happens to fire a scroll.
  assert.match(
    source,
    /useLayoutEffect\(\(\) => \{\s*const el = scrollRef\.current;\s*if \(!el\) return;\s*measureScrollRail\(el\);/,
  );
  // Then off the box, not off renders: the Images disclosure and the project
  // toggles change the row count without rendering AppSidebar or firing a
  // collapsible animation, and a scrollbar appearing shrinks the content box.
  assert.match(
    source,
    /const observer = new ResizeObserver\(\(\) => measureScrollRail\(el\)\);\s*observer\.observe\(el\);\s*return \(\) => observer\.disconnect\(\);/,
  );
  // Writes a variable, never state: that pairing is what looped.
  assert.equal(
    /new ResizeObserver\([^)]*set[A-Z]/.test(source),
    false,
  );
  // And only on a change, so it cannot re-trigger itself.
  assert.match(
    source,
    /if \(rail === railWidthRef\.current\) return;/,
  );
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  // No width override: the rail keeps the 8px the rest of the app uses, and
  // hiding it outright is what took the scrollbar away.
  assert.equal(/\.sidebar-scroll-fade[^{]*\{[^}]*scrollbar-width/.test(css), false);
  assert.equal(
    /\.sidebar-scroll-fade::-webkit-scrollbar \{/.test(css),
    false,
  );
  // Thumb stays hidden until the list is hovered, as the other lists do.
  assert.match(css, /\.sidebar-scroll-fade:hover::-webkit-scrollbar-thumb,/);
});

test("Tauri chat Recents label keeps its 2px shift", async () => {
  const source = await sidebarSource();
  assert.match(
    source,
    /scrolled && "is-scrolled",\s*usesDesktopTitlebar && "translate-x-\[2px\]"/,
  );
});

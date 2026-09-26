// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("submenu chevrons sit as far from the right edge as leading icons from the left", () => {
  // The arrow is drawn against its box's right edge, inset as a leading icon's stroke is on the
  // left, so a box flush on the row's padding lines up at any icon size: no pixel pull.
  const chevrons = readSrc("lib/chevron-icons.ts");
  assert.match(chevrons, /MenuChevronRightIcon[\s\S]*?d: "M16 6L22 12L16 18"/);
  for (const file of ["dropdown-menu", "context-menu", "menubar"]) {
    const source = readSrc(`components/ui/${file}.tsx`);
    assert.match(
      source,
      /icon=\{MenuChevronRightIcon\}\n\s*strokeWidth=\{1\.5\}\n\s*className="ml-auto size-\[calc\(12px\*var\(--ui-space-scale,1\)\)\]"\n\s*\/>\n\s*<\/\w+Primitive\.SubTrigger>/,
      `${file} chevron`,
    );
    assert.doesNotMatch(source, /-mr-\[calc\([\d.]+px\*var\(--ui-space-scale/, `${file} pulls a glyph by pixels`);
    assert.doesNotMatch(source, /lucide-react/, `${file} mixes in another chevron`);
  }
});

test("trailing ticks and shortcuts sit as far in as the row's leading edge", () => {
  // The tick is drawn inset like a leading icon, and its box sits on the row's own padding.
  assert.match(readSrc("lib/tick-icon.ts"), /MenuTickIcon[\s\S]*?d: "M7\.227 13\.299L11\.758 17\.829L21\.873 7\.714"/);
  for (const [file, rows] of [
    ["dropdown-menu", 2],
    ["select", 1],
    ["combobox", 1],
  ] as const) {
    const source = readSrc(`components/ui/${file}.tsx`);
    assert.match(source, /py-2 pr-9 pl-3|pl-3[^"]*aria-selected=true\]\]:pr-9/, `${file} row keeps room for the tick`);
    assert.equal(source.match(/absolute right-3 flex/g)?.length, rows, `${file} tick box off the row padding`);
    assert.match(source, /icon=\{MenuTickIcon\}/, `${file} tick`);
    assert.doesNotMatch(source, /icon=\{Tick02Icon\}/, `${file} still uses the centred tick`);
  }
  const context = readSrc("components/ui/context-menu.tsx");
  assert.match(context, /py-2 pr-9 pl-3 text-sm[\s\S]*?absolute right-3 pointer-events-none/);
  assert.match(context, /pr-8 pl-2 text-sm[\s\S]*?absolute right-2 pointer-events-none/);
  // tracking-widest leaves 0.1em after the last letter; the shortcut takes it back.
  for (const file of ["dropdown-menu", "context-menu", "menubar", "command"]) {
    assert.match(readSrc(`components/ui/${file}.tsx`), /-mr-\[0\.1em\][^"]*tracking-widest|tracking-widest[^"]*-mr-\[0\.1em\]/, `${file} shortcut`);
  }
  // Ticks a call site sets at the end of a row use the same trailing tick.
  for (const file of ["features/chat/shared-composer.tsx", "features/chat/mcp-composer-button.tsx", "features/chat/permission-mode-select.tsx"]) {
    assert.doesNotMatch(readSrc(file), /icon=\{Tick02Icon\}\s+strokeWidth=\{2\}\s+className="ml-auto/, file);
  }
});

test("menu rows keep a small gap between them", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /\[data-slot="context-menu-sub-content"\],[\s\S]*?\[role="menuitemradio"\]\n\s*\) \{\n\s*margin-block: 2px;/,
  );
});

// A menu's padding and trigger-aligning margin scale with the UI (8px is 7.47px at font size
// 14). Chromium draws a row's rounded hover pill on whole CSS pixels while the menu keeps its
// fraction, so the pill sat a device pixel nearer one edge; Firefox does the same for a
// fractional width. Every menu surface rounds them as it opens.
test("menu surfaces keep whole-pixel padding, margin and width, so the hover pill sits centred", async () => {
  const { snapInlinePadding } = await import("../src/lib/snap-padding.ts");
  const computed: Record<string, string> = {
    paddingLeft: "7.466px",
    paddingRight: "7.466px",
    marginLeft: "-2.8px",
    marginRight: "0px",
  };
  const element = { style: {} as Record<string, string> };
  const saved = { window: globalThis.window, getComputedStyle: globalThis.getComputedStyle };
  Object.assign(globalThis, { window: {}, getComputedStyle: () => computed });
  try {
    snapInlinePadding(element as unknown as HTMLElement);
  } finally {
    Object.assign(globalThis, saved);
  }
  assert.deepEqual(element.style, { paddingLeft: "7px", paddingRight: "7px", marginLeft: "-3px" });

  for (const [file, surfaces] of [
    ["dropdown-menu", 2],
    ["context-menu", 2],
    ["menubar", 2],
    ["select", 1],
  ] as const) {
    const source = readSrc(`components/ui/${file}.tsx`);
    const snapped = (source.match(/ref=\{snappedRef\}/g)?.length ?? 0) + (source.match(/snapInlinePadding\(element\);/g)?.length ?? 0);
    assert.equal(snapped, surfaces, `${file} surfaces rounded as they mount`);
  }
  assert.match(
    readSrc("components/ui/dropdown-menu.tsx"),
    /w-\[round\(calc\(var\(--radix-dropdown-menu-trigger-width\)_\+_6px\*var\(--ui-space-scale,1\)\),1px\)\]/,
  );
  assert.match(readSrc("components/app-sidebar.tsx"), /app-user-menu sidebar-menu[^"]*w-\[round\(calc\(16rem\*var\(--ui-space-scale,1\)\),1px\)\]/);
});

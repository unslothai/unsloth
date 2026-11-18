// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * The flags that replaced `HOST:has(> .chat-wallpaper...) X`, `:is(p, div):has(> a:nth-of-type(2) > img)`
 * and the Windows frame :has(): every flag the CSS reads is set by code, and the code is wired.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { WALLPAPER_HOST_FLAGS } from "../src/features/chat-background/host-flags.ts";
import { readSrc } from "./helpers/kit.ts";

const css = readSrc("index.css");

test("every wallpaper host flag the CSS reads is mirrored from the wallpaper layer", () => {
  for (const [name] of WALLPAPER_HOST_FLAGS)
    assert.ok(
      css.includes(`[${name}]`),
      `${name} is set but index.css never reads it`,
    );
  for (const [, selector] of WALLPAPER_HOST_FLAGS)
    assert.match(selector, /^:scope > \.chat-wallpaper/);
  const rules = css.replace(/\/\*[\s\S]*?\*\//g, "");
  assert.doesNotMatch(
    rules,
    /:has\(\s*>\s*\.chat-wallpaper/,
    "wallpaper CSS reads the host flags, not a :has()",
  );
});

test("both wallpaper layers keep their host's flags in step", () => {
  for (const file of [
    "features/chat-background/background.tsx",
    "features/chat-background/backdrop.tsx",
  ]) {
    const source = readSrc(file);
    assert.match(source, /useWallpaperHostFlags\(host\);/, file);
    assert.match(
      source,
      /ref=\{host\}[\s\S]{0,80}className="chat-wallpaper"/,
      file,
    );
  }
});

test("hub README badge rows are marked, not matched with :has()", () => {
  const hub = readSrc("features/hub/hub.css");
  assert.doesNotMatch(hub, /:has\(> a:nth-of-type/);
  assert.equal(
    hub.match(/\.hub-readme-prose :is\(p, div\)\[data-badge-row\]/g)?.length,
    4,
  );
  assert.match(
    readSrc("features/hub/lib/badge-rows.ts"),
    /":scope > a:nth-of-type\(2\) > img"/,
  );
  assert.match(
    readSrc("features/hub/catalog/model-readme.tsx"),
    /<div ref=\{badgeRowsRef\} className=\{cn\("hub-readme-prose", PROSE\)\}>/,
  );
});

test("the Windows page blur learns about the sidebar frame from the root, not a :has()", () => {
  assert.match(
    css,
    /\[data-windows-wallpaper-page\] > \[data-slot="page-wallpaper-blur"\]\[data-sidebar-frame\] \{\n {2}border-top-left-radius: var\(--radius-md\);/,
  );
  const root = readSrc("app/routes/__root.tsx");
  assert.match(
    root,
    /data-slot="page-wallpaper-blur"\s+\/\/[^\n]*\n\s+data-sidebar-frame=\{\(pinned && !mobileShell\) \|\| undefined\}/,
  );
  // The flag mirrors AppSidebar's own condition (windowsWallpaperPage already implies the in-page titlebar,
  // which Windows and Linux share).
  assert.match(
    readSrc("components/app-sidebar.tsx"),
    /data-windows-glass-frame=\{\(pinned && usesCustomTitlebar\) \|\| undefined\}/,
  );
});

test("the info hint re-checks keyboard focus on key and pointer input, not only on focus moves", () => {
  const hint = readSrc("components/ui/info-hint.tsx");
  // A mouse-focused slider nudged with an arrow key gains :focus-visible without a focus event.
  assert.match(hint, /\["focusin", "focusout", "keydown", "pointerdown"\]/);
  assert.match(hint, /for \(const type of events\) top\.addEventListener\(type, later\);/);
  assert.match(hint, /for \(const type of events\) top\.removeEventListener\(type, later\);/);
});

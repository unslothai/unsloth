// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import postcss from "postcss";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const GATE = "html.tauri.render-linux.webkit-legacy-compositor[data-glass]";
const MAIN = readSrc("main.tsx");
const SIDEBAR = `${GATE}[data-chat-wallpaper-workspace-ready] [data-windows-glass-frame] > [data-slot="sidebar-inner"]`;
const FRAME_BEFORE = "html[data-glass][data-chat-wallpaper-workspace-ready] [data-windows-glass-frame]::before";
const STRIP = '[data-slot="sidebar-container"][data-windows-glass-frame]::before';
const CLEAR =
  'html:is([data-glass], [data-glass-clear])[data-chat-wallpaper-workspace-ready] [data-windows-glass-frame] > [data-slot="sidebar-inner"]';

function decls(selector: string): Map<string, string> {
  const out = new Map<string, string>();
  postcss.parse(CSS).walkRules((rule) => {
    if (!rule.selectors.includes(selector)) return;
    rule.walkDecls((d) => {
      out.set(d.prop, d.value);
    });
  });
  return out;
}

test("under WebKitGTK's TextureMapper compositor the pinned sidebar blurs for itself, with the strip's fill", () => {
  const frame = decls(FRAME_BEFORE);
  const own = decls(SIDEBAR);
  for (const prop of ["backdrop-filter", "-webkit-backdrop-filter", "background-color"]) {
    assert.ok(frame.get(prop), `the strip sets ${prop}`);
    assert.equal(own.get(prop), frame.get(prop), prop);
  }
});

test("under it the strip leaves the sidebar column to the sidebar, so the column keeps one layer of fill", () => {
  const shared = decls(`html:is([data-glass], [data-glass-clear])[data-chat-wallpaper-workspace-ready] ${STRIP}`).get("mask");
  const legacy = decls(`${GATE}[data-chat-wallpaper-workspace-ready] ${STRIP}`).get("mask");
  assert.match(shared ?? "", /left top \/ var\(--sidebar-width\) 100%/);
  assert.ok(legacy, "the legacy strip sets its own mask");
  assert.doesNotMatch(legacy, /var\(--sidebar-width\) 100%/, "no sidebar column under the self-blurred sidebar");
  assert.match(legacy, /var\(--sidebar-width\) top \/ 100% var\(--studio-custom-titlebar-height\)/);
});

test("the self blur comes after the rule that clears it and stays off every other engine", () => {
  assert.ok(CSS.indexOf(`${SIDEBAR} {`) > CSS.indexOf(`${CLEAR} {`));
  assert.equal(decls(CLEAR).get("backdrop-filter"), "none", "other engines keep the one shared strip");
  postcss.parse(CSS).walkRules((rule) => {
    for (const selector of rule.selectors) {
      if (!selector.endsWith('[data-windows-glass-frame] > [data-slot="sidebar-inner"]')) continue;
      let blur = false;
      rule.walkDecls(/backdrop-filter$/, (d) => {
        if (d.value !== "none") blur = true;
      });
      if (blur) assert.ok(selector.startsWith(GATE), `${selector} blurs outside the legacy compositor gate`);
    }
  });
});

test("the gate follows the shell's compositor answer, not CPU painting", () => {
  assert.match(MAIN, /if \(status\.legacyCompositor\) \{\s*document\.documentElement\.classList\.add\("webkit-legacy-compositor"\);/);
  assert.doesNotMatch(CSS, /webkit-cpu-paint\[data-glass\][^{]*sidebar-inner/);
});

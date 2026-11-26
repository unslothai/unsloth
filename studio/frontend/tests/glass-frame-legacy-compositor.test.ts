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
  const shared = decls(`html:is([data-glass], [data-glass-clear])[data-chat-wallpaper-workspace-ready] ${STRIP.replace("::before", "")}`).get("--studio-glass-strip-mask");
  assert.equal(decls(`html:is([data-glass], [data-glass-clear])[data-chat-wallpaper-workspace-ready] ${STRIP}`).get("mask"), "var(--studio-glass-strip-mask)");
  // On the frame, so its ::before and its lens (index.css, [data-windows-glass-frame] > .glass-lens) both take the caption shape.
  const legacy = decls(`${GATE}[data-chat-wallpaper-workspace-ready] ${STRIP.replace("::before", "")}`).get("--studio-glass-strip-mask");
  assert.equal(decls(`${GATE}[data-chat-wallpaper-workspace-ready] ${STRIP}`).get("mask"), undefined, "the ::before reads the var");
  assert.match(shared ?? "", /left top \/ var\(--sidebar-width\) 100%/);
  assert.ok(legacy, "the legacy frame sets its own strip mask");
  assert.doesNotMatch(legacy, /var\(--sidebar-width\) 100%/, "no sidebar column under the self-blurred sidebar");
  assert.match(legacy, /var\(--sidebar-width\) top \/ 100% var\(--studio-custom-titlebar-height\)/);
  // Docked Run settings keeps its own glass below the caption, in the strip's fill, sheen and no edge line.
  const panel = decls(`${GATE}[data-sidebar-glass-frame][data-chat-wallpaper-workspace-ready] aside[data-slot="chat-settings-panel"]`);
  assert.equal(panel.get("box-shadow"), "none");
  assert.match(panel.get("background-color") ?? "", /^color-mix\(in srgb, var\(--glass-sidebar-fill\)/);
  assert.equal(panel.get("background"), undefined, "a shorthand would drop the liquid sheen");
  assert.equal(panel.get("background-attachment"), "fixed");
  assert.ok(CSS.includes('html:is([data-glass], [data-glass-clear])[data-sidebar-glass-frame][data-chat-wallpaper-workspace-ready] aside[data-slot="chat-settings-panel"] > [data-chat-settings-content] {\n  border-left-color: transparent;'));
  // Run settings keeps its own glass here, so the caption rounds the chat's top right corner as well as its left.
  assert.match(legacy, /radial-gradient\(circle at left bottom, transparent calc\(var\(--studio-run-settings-fillet\) - 1px\), black var\(--studio-run-settings-fillet\)\)\s+calc\(100vw - var\(--studio-chat-settings-docked-width, 0px\) - var\(--studio-run-settings-fillet\)\) var\(--studio-custom-titlebar-height\)/);
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

test("under it the sidebar and Run settings blur past the caption's edge and mask back, so no step shows where they meet", () => {
  const own = decls(SIDEBAR);
  assert.equal(own.get("--studio-legacy-glass-bleed"), "calc(3 * var(--glass-blur, 12px))");
  assert.equal(own.get("width"), "calc(100% + var(--studio-legacy-glass-bleed))");
  assert.equal(own.get("padding-right"), "var(--studio-legacy-glass-bleed)", "the content keeps the sidebar's width");
  assert.equal(own.get("flex"), "none", "the frame would shrink it back to the sidebar's width");
  assert.equal(own.get("mask"), "linear-gradient(black 0 0) left top / calc(100% - var(--studio-legacy-glass-bleed)) 100% no-repeat");
  // The masked bleed lies over the conversation, so it passes clicks through; the lens stays inert.
  assert.equal(own.get("pointer-events"), "none");
  assert.equal(decls(`${SIDEBAR} > :not(.glass-lens)`).get("pointer-events"), "auto");
  const top = 'html:is([data-glass], [data-glass-clear])[data-sidebar-glass-frame][data-chat-wallpaper-workspace-ready] aside[data-slot="chat-settings-panel"]';
  assert.equal(decls(top).get("--chat-settings-top"), "0px", "Run settings runs to the window top on every engine");
  assert.equal(decls(top).get("padding-top"), "var(--studio-custom-titlebar-height, 0px)");
  const panel = decls(`${GATE}[data-sidebar-glass-frame][data-chat-wallpaper-workspace-ready] aside[data-slot="chat-settings-panel"]`);
  assert.equal(panel.get("mask"), "linear-gradient(transparent var(--studio-custom-titlebar-height, 0px), black var(--studio-custom-titlebar-height, 0px))", "the caption draws its own band");
});

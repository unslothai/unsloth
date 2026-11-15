// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import postcss from "postcss";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const MATERIAL = readSrc("features/chat-background/material.ts");

function ruleAt(selectorStart: string): string {
  const at = CSS.indexOf(selectorStart);
  assert.ok(at > 0, `rule ${selectorStart} exists`);
  return CSS.slice(at, CSS.indexOf("}", at) + 1);
}

/** Every rule keyed on the WebKitGTK flag, with the properties it sets. */
function cpuRules(): { selector: string; props: string[]; topLevel: boolean }[] {
  const found: { selector: string; props: string[]; topLevel: boolean }[] = [];
  postcss.parse(CSS).walkRules((rule) => {
    if (!rule.selector.includes("[data-glass-cpu]")) return;
    const props: string[] = [];
    rule.walkDecls((decl) => {
      props.push(decl.prop);
    });
    found.push({ selector: rule.selector, props, topLevel: rule.parent?.type === "root" });
  });
  return found;
}

test("the CPU glass flag is WebKitGTK only and follows the layer flag's material and scope gate", () => {
  assert.match(MATERIAL, /import \{ isWebKitGTK \} from "@\/components\/ui\/theme-view-transition";/);
  assert.match(
    MATERIAL,
    /const WEBKIT_GTK =\s*typeof navigator !== "undefined" &&\s*isWebKitGTK\(navigator\.platform, navigator\.userAgent\);/,
  );
  assert.match(
    MATERIAL,
    /toggleAttribute\(\s*"data-glass-cpu",\s*WEBKIT_GTK && material !== "solid" && material !== "transparent" && here,\s*\)/,
    "solid, Transparent and out-of-scope routes never carry the flag",
  );
});

test("WebKitGTK glass keeps its live blur, its fills and its strength", () => {
  const rules = cpuRules();
  // The flag only swaps paint costs: nothing under it may touch a blur, a fill or the strength variables.
  const allowed = new Set(["background-image", "--glass-rim", "--glass-glow", "--glass-lift", "--popup-shadow"]);
  for (const rule of rules) {
    for (const prop of rule.props) assert.ok(allowed.has(prop), `${rule.selector} sets ${prop}`);
  }
  const at = CSS.indexOf("[data-glass-cpu]");
  assert.doesNotMatch(CSS.slice(at), /\[data-glass-cpu\][^{]*\{[^}]*backdrop-filter/, "no blur is turned off on that path");
  assert.doesNotMatch(CSS, /--glass-opacity-set/, "the strength slider is not floored");
  assert.doesNotMatch(CSS, /--glass-(?:pane|composer|menu)-floor: 110%/, "no fill is pinned at its cap");
});

test("WebKitGTK Frosted drops only its grain tile", () => {
  const rule = ruleAt(
    'html[data-glass="frosted"][data-glass-cpu] :is([data-slot="sidebar-inner"], .unsloth-composer-surface',
  );
  assert.match(
    rule,
    /background-image: none !important;/,
    "the Frosted grain (an SVG turbulence tile) is rendered again on every paint there",
  );
  assert.doesNotMatch(rule, /backdrop-filter|background-color/);
  // Every surface the grain rule paints must be covered: the title bar frame is a pseudo-element the :is() list cannot hold.
  const grain = postcss.parse(CSS).nodes.find(
    (node): node is postcss.Rule => node.type === "rule" && node.toString().includes("feTurbulence"),
  );
  assert.ok(grain, "the Frosted grain rule exists");
  for (const selector of grain.selectors) {
    const target = selector.includes("[data-windows-glass-frame]::before")
      ? "[data-windows-glass-frame]::before"
      : /(\[data-slot="sidebar-inner"\]|\.wallpaper-preview-composer|\.unsloth-composer-surface)/.exec(selector)?.[1];
    assert.ok(target, `known grain surface: ${selector}`);
    assert.ok(rule.includes(target), `the WebKitGTK rule clears the grain on ${target}`);
  }
  assert.ok(grain.selectors.some((selector) => selector.includes(".unsloth-files-panel")) && rule.includes(".unsloth-files-panel"));
});

test("WebKitGTK liquid 3D draws its wide inner glows as gradients, not blurred shadows", () => {
  for (const sel of [
    'html[data-glass="liquid-3d"][data-glass-cpu] {',
    'html.dark[data-glass="liquid-3d"][data-glass-cpu] {',
    'html[data-glass="liquid-3d"][data-glass-cpu] :is(.unsloth-composer-surface',
    'html.dark[data-glass="liquid-3d"][data-glass-cpu] :is(.unsloth-composer-surface',
  ]) {
    const rule = ruleAt(sel);
    const rim = /--glass-rim: ([^;]*);/.exec(rule)?.[1] ?? "";
    assert.ok(rim, `${sel} sets the rim`);
    for (const shadow of rim.split(/,(?![^(]*\))/)) {
      const blur = /inset\s+-?[\d.]+(?:px)?\s+-?[\d.]+(?:px)?\s+([\d.]+)px/.exec(shadow)?.[1];
      assert.ok(!blur || Number(blur) <= 2, `${sel}: no wide blurred inset (${shadow.trim()})`);
    }
    assert.match(rule, /--glass-glow: linear-gradient\(/);
  }
  const glowUses = CSS.match(/background-image: var\(--glass-glow, none\), var\(--glass-sheen\);/g) ?? [];
  assert.equal(glowUses.length, 3, "composer, menus and dialogs; elsewhere --glass-glow is unset, so the layer is none");
});

test("WebKit liquid glass drops the sheen, which bands into stripes without dithering", () => {
  // data-glass-layers is the WebKit flag (macOS and Linux apps, Safari); Chromium dithers the gradient and keeps it.
  assert.match(MATERIAL, /"data-glass-layers",\s*WEBKIT_LAYERS && material !== "solid" && material !== "transparent" && here,/);
  const rule = ruleAt('html[data-glass^="liquid"][data-glass-layers],');
  assert.ok(
    rule.includes('html[data-glass^="liquid"][data-glass-layers] :is(.unsloth-composer-surface, .unsloth-files-panel, .wallpaper-preview-composer) {'),
    "the composer sets its own sheen, so it is cleared there too",
  );
  assert.match(rule, /--glass-sheen: none;/);
  // Every sheen definition must come before the override: equal specificity, so the later rule wins.
  const at = CSS.indexOf('html[data-glass^="liquid"][data-glass-layers],');
  assert.doesNotMatch(CSS.slice(at), /--glass-sheen: linear-gradient/);
  assert.equal((CSS.slice(0, at).match(/--glass-sheen: linear-gradient/g) ?? []).length, 8);
});

test("Firefox drops the sheen too, with a selector as specific as the dark sheens it overrides", () => {
  let found = 0;
  postcss.parse(CSS).walkAtRules("supports", (at) => {
    if (at.params !== "(-moz-appearance: none)") return;
    at.walkRules((rule) => {
      if (!rule.selector.startsWith('html[data-glass^="liquid"][data-glass]')) return;
      found += 1;
      assert.equal(at.parent?.type, "root", "unlayered");
      assert.deepEqual(
        rule.selectors,
        [
          'html[data-glass^="liquid"][data-glass]',
          'html[data-glass^="liquid"][data-glass] :is(.unsloth-composer-surface, .unsloth-files-panel, .wallpaper-preview-composer)',
        ],
        "two attributes on html, like html.dark[data-glass=\"liquid-3d\"]",
      );
      assert.match(rule.toString(), /--glass-sheen: none;/);
      assert.doesNotMatch(CSS.slice(CSS.indexOf(rule.toString())), /--glass-sheen: linear-gradient/, "after every sheen definition");
    });
  });
  assert.equal(found, 1);
});

test("the WebKitGTK rules are unlayered top-level rules", () => {
  const rules = cpuRules();
  for (const rule of rules) assert.ok(rule.topLevel, `${rule.selector} must not sit inside @layer or another block`);
  assert.equal(rules.length, 5, "the Frosted grain rule and the four Liquid 3D glow rules");
});

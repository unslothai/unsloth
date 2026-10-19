// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  CODE_BLOCK_CONTAINMENT_ATTRIBUTE,
  CODE_BLOCK_CONTAINMENT_ON,
  FENCE_HEIGHT_PROPERTY,
} from "../src/components/assistant-ui/code-block-containment-mode.ts";

import { readSrc } from "./helpers/kit.ts";

const INDEX_CSS = readSrc("index.css");
const DEFER = readSrc("components/assistant-ui/code-fence-defer.tsx");
const MAIN = readSrc("main.tsx");
const CONTAINMENT = readSrc("components/assistant-ui/code-block-containment.ts");
const GLUE = readSrc("components/assistant-ui/progressive-messages.tsx");

const GATE = `html[${CODE_BLOCK_CONTAINMENT_ATTRIBUTE}="${CODE_BLOCK_CONTAINMENT_ON}"]`;

const layerBounds = (css: string, name: string): [number, number] => {
  const open = css.indexOf(`@layer ${name} {`);
  assert.ok(open >= 0, `PRECONDITION: @layer ${name} exists`);
  let depth = 0;
  for (let i = css.indexOf("{", open); i < css.length; i += 1) {
    if (css[i] === "{") depth += 1;
    else if (css[i] === "}") {
      depth -= 1;
      if (depth === 0) return [open, i];
    }
  }
  throw new Error(`@layer ${name} never closes`);
};

const printBlocks = (css: string): string[] => {
  const blocks: string[] = [];
  for (let start = css.indexOf("@media print"); start >= 0; start = css.indexOf("@media print", start + 1)) {
    let depth = 0;
    for (let i = css.indexOf("{", start); i < css.length; i += 1) {
      if (css[i] === "{") depth += 1;
      else if (css[i] === "}") {
        depth -= 1;
        if (depth === 0) {
          blocks.push(css.slice(start, i + 1));
          break;
        }
      }
    }
  }
  return blocks;
};

test("one rule, gated on the html attribute, inside the utilities layer", () => {
  const rules = INDEX_CSS.split(GATE).slice(1);
  assert.equal(rules.length, 1, "exactly one gated rule");
  const at = INDEX_CSS.indexOf(GATE);
  const [open, close] = layerBounds(INDEX_CSS, "utilities");
  assert.ok(at > open && at < close, "inside @layer utilities, where the visible override lives");
  const rule = rules[0].slice(0, rules[0].indexOf("}"));
  assert.ok(rule.includes(".aui-thread-root"), "scoped to the chat thread");
  assert.ok(rule.includes('[data-streamdown="code-block"]:not([data-incomplete])'), "settled fences only");
  assert.ok(rule.includes(':not([data-incomplete]) > [data-unsloth-fence-windowed] {'), "the box around the windowed body: a skipped scroller drops its scrollbar in Chromium and its padding in Firefox, so the body itself would come back taller than it was skipped");
  assert.ok(rule.includes("content-visibility: auto;"), "the declaration under test");
  assert.ok(rule.includes(`contain-intrinsic-size: auto var(${FENCE_HEIGHT_PROPERTY})`), "sized by the component's measurement");
  assert.equal(rule.includes(":has("), false, "never :has()");
  assert.equal(rule.includes("!important"), false, "the windowed box is not the element the visible override targets");
});

test("the wrapper keeps the visible override and the body keeps the DOM contract", () => {
  assert.ok(
    INDEX_CSS.includes('.aui-thread-root [data-streamdown="code-block"] {\n\t\tcontent-visibility: visible !important;'),
    "the flicker rule on the wrapper stays",
  );
  const body = DEFER.slice(DEFER.indexOf("export const FenceBody = memo("));
  assert.ok(/<div\s*data-unsloth-fence-windowed=\{lineWindow === null \|\| !measured \? undefined : "true"\}\s*ref=\{wrapper\}\s*>\s*<div\s*className=\{joinClasses\([\s\S]*?\)\}\s*data-language=\{language \?\? undefined\}\s*data-streamdown="code-block-body"\s*ref=\{surface\}/.test(body), "the window hook sits on a box that does not scroll, around a body that keeps streamdown's markup and the measuring ref");
  assert.ok(!/contentVisibility:\s*"hidden"|display:\s*"none"/.test(body), "no line may be hidden from the engine");
});

test("the component writes the measured height before paint and only when it changes", () => {
  assert.ok(DEFER.includes('import { FENCE_HEIGHT_PROPERTY } from "./code-block-containment-mode";'));
  assert.match(DEFER, /const declared = `\$\{height\}px`;\s*if \(region\.style\.getPropertyValue\(FENCE_HEIGHT_PROPERTY\) !== declared\) \{\s*region\.style\.setProperty\(FENCE_HEIGHT_PROPERTY, declared\);/, "compared with the element's own declaration, so a box that remounts is written again");
  assert.match(DEFER, /insets: Math\.max\(0, surface\.getBoundingClientRect\(\)\.height - lineCount \* lineHeight\)/, "the box holds the whole body, so its border, padding and horizontal scrollbar are part of the height a skipped box reproduces");
  assert.match(DEFER, /const height =\s*Math\.round\(\s*\(lines\.current \* known\.lineHeight \+ known\.insets\) \* measureIntrinsicScale\(region\) \* 1000,\s*\) \/ 1000;/, "scaled where WebKit lays out contain-intrinsic-size lengths without the page zoom");
  assert.match(DEFER, /contain:size;contain-intrinsic-size:0 1000px/);
  assert.match(DEFER, /CSS\.supports\("contain-intrinsic-size", "0 1000px"\);\s*const measureIntrinsicScale = \(beside: HTMLElement\): number => \{\s*if \(!intrinsicSizeSupported\) return 1;/, "an engine without contain-intrinsic-size probes nothing, since its probe measures 0 and would be inserted again on every measure");
  assert.match(DEFER, /window\.devicePixelRatio,\s*measureIntrinsicScale\(beside\),\s*\]\.join\("\|"\);/, "a page zoom WebKit hides from devicePixelRatio still re-probes the pitch");
  const measure = DEFER.slice(DEFER.indexOf("measure.current = () => {"), DEFER.indexOf("useLayoutEffect(() => {", DEFER.indexOf("measure.current = () => {")));
  assert.ok(measure.indexOf("setProperty(FENCE_HEIGHT_PROPERTY") < measure.indexOf("const reach ="), "written before the far-away early return, so a fence measured once always carries its height");
});

test("the measured height does not inherit, so a streamed line restyles only the box it sizes", () => {
  assert.ok(new RegExp(`@property ${FENCE_HEIGHT_PROPERTY} \\{\\s*syntax: "<length>";\\s*inherits: false;\\s*initial-value: 0px;\\s*\\}`).test(INDEX_CSS), "registered, so a write restyles the box and not every line inside it");
});

test("the window measure never reads inside a body that may be skipped", () => {
  const measure = DEFER.slice(DEFER.indexOf("measure.current = () => {"), DEFER.indexOf("useLayoutEffect(() => {", DEFER.indexOf("measure.current = () => {")));
  assert.ok(!/(node|body|code\.current)\.getBoundingClientRect|rectDuringFrame\(body\)/.test(measure), "descendant rects are read by readFenceMetrics on registration and resize only");
  assert.match(measure, /rectDuringFrame\(region\)/, "the box around the body is what the window is read against");
  assert.match(measure, /\} else if \(metricsStale\.current && !skippedByContainment\(body\)\) \{/, "a resize re-reads a skipped fence's metrics once it is laid out again, instead of forcing its layout");
  assert.match(DEFER, /const FAR_VIEWPORTS = OVERSCAN_VIEWPORTS \+ HYSTERESIS_VIEWPORTS \+ 1;/, "the far band is derived from the window's own margins");
});

test("the line pitch is measured from layout once per metric set, beside the box around the body", () => {
  const metrics = DEFER.slice(DEFER.indexOf("const measureLinePitch = "), DEFER.indexOf("const readFenceGeometry = ("));
  assert.match(metrics, /const pitch = measureLinePitch\(region\);/);
  assert.match(metrics, /pitch > 0\s*\? pitch/, "the laid-out pitch wins over a computed line-height the engine truncates to 1/64 px per line");
  assert.match(metrics, /const context = getComputedStyle\(parent\);/, "keyed on the wrapper, never on the body that may be skipped");
  assert.equal(/getComputedStyle\((node|code)\b/.test(metrics), false);
  assert.match(metrics, /beside\.before\(probe\);/, "the probe sits beside the box, outside the skipped subtree");
  assert.match(metrics, /line\.className = LINE_CLASS;/, "built like the fence's own lines, inside a pre and a code");
  assert.match(metrics, /\(lines\[PITCH_PROBE_LINES - 1\]\.getBoundingClientRect\(\)\.top - lines\[0\]\.getBoundingClientRect\(\)\.top\)\s*\/ \(PITCH_PROBE_LINES - 1\)/, "the mean advance across the probe's lines, so no padding or border of the probe can leak in and no single line's rounding is multiplied");
  const probeLines = Number(/const PITCH_PROBE_LINES = (\d+);/.exec(DEFER)?.[1] ?? 0);
  assert.ok(probeLines >= 32, "enough lines to average Firefox's per-line rounding");
  assert.match(metrics, /getPropertyValue\("--custom-code-font-size"\)/, "a custom Code font size gets its own entry");
  assert.match(metrics, /window\.devicePixelRatio/);
  assert.match(metrics, /probe\.remove\(\);/);
});

test("a print clears the containment on every windowed fence, through the media query", () => {
  const block = printBlocks(INDEX_CSS).find((candidate) => candidate.includes("[data-unsloth-fence-windowed]")) ?? "";
  assert.notEqual(block, "", "a @media print block names the windowed box");
  assert.ok(block.includes("content-visibility: visible !important"));
  assert.ok(block.includes("contain-intrinsic-size: none !important"));
});

test("startup arms the attribute and watches the console override, after the maths one", () => {
  assert.ok(MAIN.includes('} from "./components/assistant-ui/code-block-containment";'));
  const math = MAIN.indexOf("watchMathBlockContainmentOverride();");
  const apply = MAIN.indexOf("applyCodeBlockContainment();");
  const watch = MAIN.indexOf("watchCodeBlockContainmentOverride();");
  assert.ok(math >= 0 && apply > math && watch > apply, "in that order, before the first render");
  assert.ok(apply < MAIN.indexOf("function renderApp()"), "before renderApp");
  assert.ok(CONTAINMENT.includes('import { engineFindsSkippedContent } from "./math-block-containment";'), "the same engine gate as the maths blocks");
  assert.ok(CONTAINMENT.includes("import.meta.env.VITE_UNSLOTH_CODE_BLOCK_CONTAINMENT"));
  assert.ok(CONTAINMENT.includes("root.removeAttribute(CODE_BLOCK_CONTAINMENT_ATTRIBUTE)"), "off presents the same DOM as an install that never heard of it");
});

test("the widening anchor stops at a skipped subtree instead of forcing it to render", () => {
  const pick = GLUE.slice(GLUE.indexOf("function pickAnchorRow("), GLUE.indexOf("function sampleAnchor("));
  assert.match(pick, /anchor\.checkVisibility\(\{ contentVisibilityAuto: true \}\)/);
  assert.ok(pick.indexOf("checkVisibility") < pick.indexOf("for (const child of anchor.children)"), "checked before any child rect is read");
  const loop = pick.slice(pick.indexOf("for (const child of anchor.children)"));
  assert.ok(loop.indexOf("child.checkVisibility({ contentVisibilityAuto: true })") < loop.indexOf("child.getBoundingClientRect()"), "the child is asked before its rect is read, because the skipping element itself passes the check");
  assert.match(loop, /if \(child\.checkVisibility\(\)\) return anchor;\s*continue;/, "a skipped child makes its parent the anchor; a boxless child is skipped");
});

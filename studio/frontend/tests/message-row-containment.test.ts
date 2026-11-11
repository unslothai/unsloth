// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";
import ts from "@typescript/typescript6";

import {
  MESSAGE_ROW_CONTAINMENT,
  MESSAGE_ROW_CONTAINMENT_ATTRIBUTE,
  MESSAGE_ROW_CONTAINMENT_ON,
} from "../src/components/assistant-ui/message-row-containment-mode.ts";
import { readSrc } from "./helpers/kit.ts";

const source = readFileSync(new URL("../src/components/assistant-ui/distant-rows.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } });

const INDEX_CSS = readSrc("index.css");
const THREAD = readSrc("components/assistant-ui/thread.tsx");
const MAIN = readSrc("main.tsx");
const CONTAINMENT = readSrc("components/assistant-ui/message-row-containment.ts");
const ROWS = ".aui-thread-viewport [data-role][data-message-id]";

class Row {
  attributes = new Set<string>();
  scrollWidth = 600;
  clientWidth = 600;
  top: number;
  height: number;
  viewport: Viewport;
  constructor(viewport: Viewport, top: number, height: number) {
    this.viewport = viewport;
    this.top = top;
    this.height = height;
    viewport.rows.push(this);
  }
  closest(selector: string) { return selector === ".aui-thread-viewport" ? this.viewport : null; }
  hasAttribute(name: string) { return this.attributes.has(name); }
  removeAttribute(name: string) { this.attributes.delete(name); }
  toggleAttribute(name: string, on: boolean) {
    const was = this.attributes.has(name);
    if (on) this.attributes.add(name); else this.attributes.delete(name);
    // Releasing a row lays it out at the current width; skipping it uses its remembered size.
    if (was && !on) this.viewport.relayout(this);
    if (!was && on) this.viewport.skip(this);
  }
  numberedEquation = false;
  querySelector(selector: string) { return this.numberedEquation && selector.includes(".eqn-num") ? {} : null; }
  get isConnected() { return true; }
  getBoundingClientRect() {
    const top = this.viewport.offsetOf(this) - this.viewport.scrollTop;
    return { top, bottom: top + this.height };
  }
}

class Viewport {
  rows: Row[] = [];
  scrollTop = 0;
  /** Height a row takes once released; set to simulate a narrower column than when it was skipped. */
  releasedHeight = new Map<Row, number>();
  offsetOf(row: Row) {
    let y = 0;
    for (const r of this.rows) {
      if (r === row) return y;
      y += r.height;
    }
    return y;
  }
  relayout(row: Row) {
    const next = this.releasedHeight.get(row);
    if (next !== undefined) row.height = next;
  }
  /** Height a row takes once skipped, when the engine remembered a different one. */
  skippedHeight = new Map<Row, number>();
  skip(row: Row) {
    const next = this.skippedHeight.get(row);
    if (next !== undefined) row.height = next;
  }
  getBoundingClientRect() { return { top: 0, bottom: 800 }; }
  querySelectorAll() { return this.rows; }
  scrollTo({ top }: { top: number }) { this.scrollTop = top; }
}

type FakeObserver = { margin: string; observed: Set<Row>; disconnected: boolean };

function world(gate = true, innerHeight = 900) {
  let intersect: (entries: unknown[]) => void = () => {};
  const observers: FakeObserver[] = [];
  const resizeListeners: Array<() => void> = [];
  let frames: Array<() => void> = [];
  const keyListeners: Array<(event: unknown) => void> = [];
  const win = {
    innerHeight,
    addEventListener: (type: string, cb: () => void) => { if (type === "resize") resizeListeners.push(cb); },
  };
  const exports: {
    observeDistantRow?: (row: Row, adjust?: (delta: number) => void) => (() => void) | undefined;
    rowReachPx?: (height: number) => number;
  } = {};
  vm.runInNewContext(outputText, {
    exports,
    window: win,
    document: {
      documentElement: { hasAttribute: (name: string) => gate && name === MESSAGE_ROW_CONTAINMENT_ATTRIBUTE },
      addEventListener: (type: string, cb: (event: unknown) => void) => { if (type === "keydown") keyListeners.push(cb); },
    },
    requestAnimationFrame: (cb: () => void) => { frames.push(cb); return frames.length; },
    IntersectionObserver: class {
      own: FakeObserver;
      constructor(cb: typeof intersect, options: { rootMargin: string }) {
        intersect = cb;
        this.own = { margin: options.rootMargin, observed: new Set(), disconnected: false };
        observers.push(this.own);
      }
      observe(row: Row) { this.own.observed.add(row); }
      unobserve(row: Row) { this.own.observed.delete(row); }
      disconnect() { this.own.disconnected = true; this.own.observed.clear(); }
    },
  });
  const rootBounds = { top: -1500, bottom: 2300 };
  const entry = (row: Row) => ({ target: row, rootBounds, boundingClientRect: row.getBoundingClientRect() });
  const frame = () => { const run = frames; frames = []; run.forEach((cb) => cb()); };
  return {
    observe: exports.observeDistantRow!,
    rowReachPx: exports.rowReachPx!,
    report: (...rows: Row[]) => intersect(rows.map(entry)),
    /** Two frames: the settle before a distant row is skipped. */
    settle: () => { frame(); frame(); },
    get observed() { return observers.at(-1)?.observed ?? new Set<Row>(); },
    observers,
    resize: (height: number) => { win.innerHeight = height; resizeListeners.forEach((cb) => cb()); },
    resizeListeners,
    keydown: (event: { key: string; metaKey?: boolean; ctrlKey?: boolean; target?: unknown }) =>
      keyListeners.forEach((cb) => cb({ metaKey: false, ctrlKey: false, target: null, ...event })),
  };
}

test("the reach outruns the engine's own content-visibility margin, which grows with the window", () => {
  const w = world();
  assert.equal(w.rowReachPx(700), 1500, "a short window keeps the 1500 px floor");
  assert.equal(w.rowReachPx(900), 1800);
  assert.equal(w.rowReachPx(2200), 4400, "Chromium renders 150% of the window height out (3300 px here), so 1500 px released rows after it had re-wrapped them");
  assert.equal(w.rowReachPx(Number.NaN), 1500);
});

test("a resize to a new window height swaps the observer once, keeping every row; the same height does nothing", () => {
  const w = world(true, 900);
  const viewport = new Viewport();
  const rows = Array.from({ length: 4 }, () => new Row(viewport, 0, 600));
  const stops = rows.map((row) => w.observe(row)!);
  assert.equal(w.observers.length, 1, "one observer per thread viewport");
  assert.equal(w.observers[0].margin, "1800px 0px");
  assert.equal(w.resizeListeners.length, 1, "one resize listener for every viewport");
  w.resize(900);
  assert.equal(w.observers.length, 1, "same height, same reach: no new observer");
  w.resize(2200);
  assert.equal(w.observers.length, 2);
  assert.equal(w.observers[0].disconnected, true);
  assert.equal(w.observers[1].margin, "4400px 0px");
  assert.deepEqual([...w.observers[1].observed], rows, "the new observer watches every row the old one did");
  w.resize(2200);
  assert.equal(w.observers.length, 2, "idempotent");
  stops.forEach((stop) => stop());
  assert.equal(w.observers[1].disconnected, true, "the last row leaving disconnects its viewport's observer");
  const again = new Row(viewport, 0, 600);
  w.observe(again);
  assert.equal(w.observers.length, 3, "a remounted thread gets a fresh observer at the current reach");
  assert.equal(w.observers[2].margin, "4400px 0px");
});

test("off when the gate is off: no observer, no attribute", () => {
  const w = world(false);
  const viewport = new Viewport();
  const row = new Row(viewport, 0, 500);
  assert.equal(w.observe(row), undefined);
  assert.equal(w.observed.size, 0);
});

test("only rows outside the reach that fit the column are marked, and cleanup unmarks", () => {
  const w = world();
  const viewport = new Viewport();
  const rows = Array.from({ length: 12 }, (_, i) => new Row(viewport, i, 600));
  viewport.scrollTop = 6600; // rows 0-7 lie more than 1500 px above the viewport top
  const stops = rows.map((row) => w.observe(row)!);
  rows[1].scrollWidth = 900; // a wide row owns horizontal scroll range
  rows[3].numberedEquation = true; // its equation numbers count on into later messages
  w.report(...rows);
  assert.equal(rows.some((r) => r.hasAttribute("data-row-distant")), false, "nothing is skipped in the frame it was first seen");
  w.settle();
  assert.deepEqual(rows.map((r) => r.hasAttribute("data-row-distant")), [true, false, true, false, true, true, true, true, false, false, false, false]);
  stops[0]();
  assert.equal(rows[0].hasAttribute("data-row-distant"), false);
  assert.equal(w.observed.has(rows[0]), false);
});

test("a row released at a new width keeps the first visible row where the reader left it", () => {
  const w = world();
  const viewport = new Viewport();
  const rows = Array.from({ length: 12 }, () => new Row(viewport, 0, 600));
  viewport.scrollTop = 6600;
  rows.forEach((row) => w.observe(row, (delta) => { viewport.scrollTop += delta; }));
  w.report(...rows);
  w.settle();
  assert.equal(rows[7].hasAttribute("data-row-distant"), true);
  // The column narrowed while row 7 was skipped; scrolling up brings it inside the reach.
  viewport.releasedHeight.set(rows[7], 1000);
  viewport.scrollTop = 5800;
  const anchor = rows[9]; // first visible row after the scroll
  const before = anchor.getBoundingClientRect().top;
  w.report(rows[7]);
  assert.equal(rows[7].hasAttribute("data-row-distant"), false);
  assert.equal(anchor.getBoundingClientRect().top, before, "the 400 px the released row grew is scrolled away");
  assert.equal(viewport.scrollTop, 6200);
});

test("select all releases every skipped row before the browser selects, keeping the reader in place", () => {
  const w = world();
  const viewport = new Viewport();
  const rows = Array.from({ length: 12 }, () => new Row(viewport, 0, 600));
  viewport.scrollTop = 6600;
  rows.forEach((row) => w.observe(row, (delta) => { viewport.scrollTop += delta; }));
  w.report(...rows);
  w.settle();
  // The column narrowed while rows 0-7 were skipped: each is 1000 px once laid out again.
  for (const row of rows.slice(0, 8)) viewport.releasedHeight.set(row, 1000);
  const anchor = rows[11];
  const before = anchor.getBoundingClientRect().top;
  // Typing select-all in an input is the input's own; nothing is released.
  w.keydown({ key: "a", metaKey: true, target: { closest: () => ({}) } });
  assert.equal(rows[0].hasAttribute("data-row-distant"), true);
  w.keydown({ key: "a", ctrlKey: true });
  assert.equal(rows.some((r) => r.hasAttribute("data-row-distant")), false);
  assert.equal(anchor.getBoundingClientRect().top, before, "the 3200 px the released rows grew is scrolled away");
});

test("no correction when the release changed nothing, and none without a released row", () => {
  const w = world();
  const viewport = new Viewport();
  const rows = Array.from({ length: 12 }, () => new Row(viewport, 0, 600));
  viewport.scrollTop = 6600;
  let calls = 0;
  rows.forEach((row) => w.observe(row, () => { calls += 1; }));
  w.report(...rows);
  w.settle();
  viewport.scrollTop = 5800;
  w.report(rows[7]);
  w.report(rows[11]);
  assert.equal(calls, 0);
});

test("a row skipped at a different height than it rendered does not move the reader", () => {
  const w = world();
  const viewport = new Viewport();
  const rows = Array.from({ length: 12 }, () => new Row(viewport, 0, 600));
  viewport.scrollTop = 6600;
  rows.forEach((row) => w.observe(row, (delta) => { viewport.scrollTop += delta; }));
  viewport.skippedHeight.set(rows[2], 300);
  const anchor = rows[11];
  const before = anchor.getBoundingClientRect().top;
  w.report(...rows);
  w.settle();
  assert.equal(rows[2].hasAttribute("data-row-distant"), true);
  assert.equal(anchor.getBoundingClientRect().top, before, "the 300 px it lost above the reader is scrolled back");
});

test("a row that comes back inside the reach before it settles is never skipped", () => {
  const w = world();
  const viewport = new Viewport();
  const rows = Array.from({ length: 12 }, () => new Row(viewport, 0, 600));
  viewport.scrollTop = 6600;
  rows.forEach((row) => w.observe(row));
  w.report(...rows);
  viewport.scrollTop = 4000; // rows 5-7 are back inside the reach before the settle
  w.report(rows[5], rows[6], rows[7]);
  w.settle();
  assert.deepEqual(rows.slice(0, 8).map((r) => r.hasAttribute("data-row-distant")), [true, true, true, true, true, false, false, false]);
});

test("every row records its size ungated; skipping is gated, marked, and exempt for print", () => {
  const sizing = INDEX_CSS.indexOf(`\t${ROWS} {\n\t\tcontain-intrinsic-size: auto 600px;\n\t}`);
  assert.ok(sizing >= 0, "auto, so the engine remembers the rendered height before any row is skipped");
  const gate = `html[${MESSAGE_ROW_CONTAINMENT_ATTRIBUTE}="${MESSAGE_ROW_CONTAINMENT_ON}"] ${ROWS}[data-row-distant] {\n\t\tcontent-visibility: auto;\n\t\tflex-shrink: 0;\n\t}`;
  assert.ok(INDEX_CSS.includes(gate), "content-visibility: auto only on marked rows under the gate, which must not shrink as flex items");
  assert.equal(INDEX_CSS.includes(`html[data-interface-zooming] ${ROWS}`), false, "a typography change must not lay out the whole thread; stale heights are corrected on release");
  const print = INDEX_CSS.indexOf("@media print", sizing);
  const block = INDEX_CSS.slice(print, INDEX_CSS.indexOf("}\n\t}", print));
  assert.ok(block.includes(ROWS) && block.includes("content-visibility: visible !important") && block.includes("contain-intrinsic-size: none !important"));
});

test("gated on the engine finding skipped content, off on WebKitGTK, overridable at runtime", () => {
  assert.equal(MESSAGE_ROW_CONTAINMENT.mode(undefined, "", true), "contain");
  assert.equal(MESSAGE_ROW_CONTAINMENT.mode(undefined, "", false), "off");
  assert.equal(MESSAGE_ROW_CONTAINMENT.mode("contain", "", false), "contain");
  assert.equal(MESSAGE_ROW_CONTAINMENT.mode("off", "", true), "off");
  assert.match(CONTAINMENT, /engineFindsSkippedContent\(\) && !webKitGTK\(\)/);
  assert.match(MAIN, /applyMessageRowContainment\(\);\s*watchMessageRowContainmentOverride\(\);/);
});

test("both message rows are observed through the autoscroll owner", () => {
  assert.match(THREAD, /return observeDistantRow\(replyRef\.current, adjustAbove\);/);
  assert.match(THREAD, /return observeDistantRow\(promptRef\.current, adjustAbove\);/);
});

test("idle fence ranking reads a skipped row's box, never a fence inside it", () => {
  const DEFER = readSrc("components/assistant-ui/code-fence-defer.tsx");
  const rank = DEFER.slice(DEFER.indexOf("const rankGates"), DEFER.indexOf("const SPECULATION_IN_FLIGHT"));
  assert.match(rank, /const row = gate\.node\.closest\(`\[\$\{ROW_DISTANT_ATTRIBUTE\}\]`\);/);
  assert.ok(rank.indexOf("row.getBoundingClientRect()") < rank.indexOf("rect ??= gate.node.getBoundingClientRect()"), "the fence is read only outside a skipped row");
});

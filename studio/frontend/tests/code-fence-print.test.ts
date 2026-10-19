// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { planPrintTokenization } from "../src/components/assistant-ui/code-fence-print.ts";

import { readSrc } from "./helpers/kit.ts";

test("the smallest fences are planned first and the plan stops at the first one that does not fit", () => {
  assert.deepEqual(planPrintTokenization([42_000, 45_000, 300_000], 200_000), [0, 1]);
  assert.deepEqual(planPrintTokenization([150_000, 150_000], 200_000), [0]);
  assert.deepEqual(planPrintTokenization([5, 1, 3], 4), [1, 2]);
  assert.deepEqual(planPrintTokenization([], 200_000), []);
  assert.deepEqual(planPrintTokenization([300_000], 200_000), []);
  assert.deepEqual(planPrintTokenization([100, 100], 200), [0, 1]);
});

test("the plan module stays plain TypeScript the test runner can execute", () => {
  const source = readSrc("components/assistant-ui/code-fence-print.ts");
  assert.equal(/<\/?[a-z]+[\s>]/i.test(source), false);
  assert.equal(source.includes("from \"react\""), false);
});

test("a print tokenizes what the worker still owes, synchronously and within a budget", () => {
  const defer = readSrc("components/assistant-ui/code-fence-defer.tsx");
  assert.ok(defer.includes("const PRINT_TOKENIZE_CHARS = 10 * MAX_HIGHLIGHT_CHARS;"));
  assert.ok(defer.includes("export const awaitWorker = ("));
  assert.ok(defer.includes('import { planPrintTokenization } from "./code-fence-print";'));
  const printing = defer.slice(defer.indexOf("const setPrinting = "), defer.indexOf("const scheduleRemeasure = "));
  assert.ok(printing.indexOf("if (value) tokenizeForPrint();") < printing.indexOf("flushSync(remeasureWindows);"));
  const tokenize = defer.slice(defer.indexOf("const tokenizeForPrint = "), defer.indexOf("const remeasureWindows = "));
  assert.match(tokenize, /flushSync\(\(\) => \{\s*for \(const index of order\) pending\[index\]\.tokenizeNow\(\);/);
  assert.ok(defer.includes("return overCap && !printing && !findBarOpen() ? EMPTY_WINDOW : NO_WINDOW;"));
  assert.ok(defer.includes("const EMPTY_WINDOW = { window: EMPTY_LINE_WINDOW, pins: null, measured: false } as const;"));
  const markdown = readSrc("components/assistant-ui/markdown-text.tsx");
  const hook = markdown.slice(markdown.indexOf("function useFenceTokens("), markdown.indexOf("function StreamingFenceBlock("));
  assert.equal((hook.match(/awaitWorker\(/g) ?? []).length, 1);
  assert.ok(hook.indexOf("requestFullHighlight(") < hook.indexOf("awaitWorker("));
  assert.match(hook, /requestFullHighlight\(\s*body,\s*languageToken,\s*\(result\) => \{\s*if \(result === null\) \{[^}]*\}\s*return;\s*\}\s*release\?\.\(\);\s*release = null;/, "the worker's tokens end what the print entry stands for; a failed reply keeps it, so a print still colours the fence");
  const probes = [...hook.matchAll(/highlight\("", /g)];
  assert.ok(probes.length >= 2);
  for (const probe of probes) {
    assert.ok(hook.startsWith("() => {})", (probe.index ?? 0) + probe[0].length));
  }
});

test("a drag freezes the line window, and a selection pins only the lines holding its boundaries", () => {
  const defer = readSrc("components/assistant-ui/code-fence-defer.tsx");
  const measure = defer.slice(defer.indexOf("measure.current = () => {"), defer.indexOf("useLayoutEffect(() => {", defer.indexOf("measure.current = () => {")));
  const flying = measure.indexOf("flyingDuringFrame(");
  const dragging = measure.indexOf("if (current.current !== null && draggingDuringFrame(body)) return;");
  assert.ok(flying >= 0 && dragging > flying && dragging < measure.indexOf("selectLineWindow({"));
  assert.ok(measure.indexOf("pinBoundaryLines(") > measure.indexOf("selectLineWindow({"), "the window keeps following the reader while the boundary lines stay as they were");
  assert.match(measure, /if \(next === current\.current && samePins\(pins, pinned\.current\)\) return;/);
  assert.match(defer, /while \(line !== null && line\.parentNode !== codeNode\) line = line\.parentNode;/, "a boundary maps to the line span under code");
  assert.match(defer, /windowed=\{lineRendered\(lineWindow, pins, index\)\}/);
  assert.equal(defer.includes("heldDuringFrame"), false);
  assert.equal(defer.includes("intersectsNode"), false);
  assert.ok(defer.includes("selection.rangeCount > 0 && !selection.isCollapsed"));
  assert.ok(defer.includes('if (event.pointerType === "touch" || event.button !== 0) return;'));
  for (const type of ["pointerdown", "pointerup", "pointercancel", "dragend", "drop", "pointermove", "blur", "selectionchange"]) {
    assert.ok(defer.includes(`addEventListener("${type}",`), `watches ${type}`);
    assert.ok(defer.includes(`removeEventListener("${type}",`), `unwatches ${type}`);
  }
  assert.match(defer, /window\.addEventListener\("dragend", releasePointer,/, "WebKit ends a text drag with dragend and never sends pointerup or pointercancel");
  assert.match(defer, /window\.addEventListener\("drop", releasePointer,/);
  assert.match(defer, /const onPointerMove = \(event: PointerEvent\): void => \{\s*if \(pointerHeld !== null && event\.buttons === 0\) releasePointer\(\);/, "and a move with no button down ends any hold an engine forgot to release");
  assert.match(defer, /const onSelectionChange = \(\): void => \{\s*if \(heldLastFrame\) scheduleRemeasure\(\);/);
});

test("the scroller walk is memoized for one synchronous stack and nowhere longer", () => {
  const defer = readSrc("components/assistant-ui/code-fence-defer.tsx");
  const predicate = defer.slice(defer.indexOf("const isScrollable"), defer.indexOf("const scrollerOf"));
  assert.ok(predicate.includes("scrollableNow.get(el)"));
  assert.ok(predicate.includes("queueMicrotask(forgetScrollable)"));
  for (const token of ['"auto"', '"scroll"', '"overlay"', "scrollHeight > el.clientHeight"]) {
    assert.ok(predicate.includes(token), token);
  }
  assert.match(defer, /region\.style\.setProperty\(FENCE_HEIGHT_PROPERTY, declared\);\s*forgetScrollable\(\);/);
  assert.match(defer, /const remeasureWindows = \(\): void => \{\s*windowFrame = 0;\s*forgetScrollable\(\);/);
  assert.equal(defer.includes("requestAnimationFrame(forgetScrollable"), false);
  assert.equal(defer.includes("frameScrollable"), false);
  assert.equal(defer.includes("scrollableDuringFrame"), false);
  assert.ok(defer.includes("!isScrollable(known.scroller)"));
});

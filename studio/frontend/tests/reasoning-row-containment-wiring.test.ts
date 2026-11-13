// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  REASONING_ROW_CONTAINMENT_ATTRIBUTE,
  REASONING_ROW_CONTAINMENT_ON,
  CHUNK_ESTIMATE_PROPERTY,
  ROW_SETTLED_ATTRIBUTE,
} from "../src/components/assistant-ui/reasoning-row-containment-mode.ts";

import { readSrc } from "./helpers/kit.ts";

const INDEX_CSS = readSrc("index.css");
const TRANSCRIPT = readSrc("components/assistant-ui/reasoning-transcript.tsx");
const ANCHOR = readSrc("components/assistant-ui/reasoning-reading-anchor.ts");
const MAIN = readSrc("main.tsx");
const CONTAINMENT = readSrc("components/assistant-ui/reasoning-row-containment.ts");

const GATE = `html[${REASONING_ROW_CONTAINMENT_ATTRIBUTE}="${REASONING_ROW_CONTAINMENT_ON}"]`;
const ROWS = '.aui-thread-root [data-slot="reasoning-transcript"] > [data-reasoning-chunk]';

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

test("every chunk records its size before any chunk can be skipped", () => {
  const at = INDEX_CSS.indexOf(`${ROWS} {`);
  assert.ok(at >= 0, "an ungated rule on every transcript chunk");
  const rule = INDEX_CSS.slice(at, INDEX_CSS.indexOf("}", at));
  assert.ok(rule.includes(`contain-intrinsic-size: auto var(${CHUNK_ESTIMATE_PROPERTY})`), "auto, so the engine remembers the rendered size, with the controller's estimate until then");
  assert.equal(rule.includes("content-visibility"), false, "sizing alone: the skipping is gated separately");
  const [open, close] = layerBounds(INDEX_CSS, "utilities");
  assert.ok(at > open && at < close);
});

test("the chunk estimate does not inherit, so a growing chunk restyles only itself", () => {
  assert.ok(new RegExp(`@property ${CHUNK_ESTIMATE_PROPERTY} \\{\\s*syntax: "<length>";\\s*inherits: false;\\s*initial-value: 0px;\\s*\\}`).test(INDEX_CSS), "registered, so a write restyles the chunk and not every element inside it");
});

test("only settled rows are skipped, and only where the engine finds skipped content", () => {
  const rules = INDEX_CSS.split(GATE).slice(1).map((rule) => rule.slice(0, rule.indexOf("}")));
  assert.equal(rules.length, 2, "two gated rules");
  const [box, rule] = rules;
  assert.ok(box.startsWith(` ${ROWS} {`), "every chunk, settled or not");
  assert.match(box, /^\s*\S[^{]*\{\s*display: flow-root;\s*$/, "is its own formatting context from the start, so settling changes no margin and no height");
  assert.ok(rule.startsWith(` ${ROWS}[${ROW_SETTLED_ATTRIBUTE}]`), "the settled chunk, a direct child of the transcript");
  assert.ok(rule.includes("content-visibility: auto;"));
  for (const gated of rules) {
    assert.equal(gated.includes(":has("), false);
    assert.equal(gated.includes("!important"), false);
  }
  const at = INDEX_CSS.indexOf(GATE);
  const [open, close] = layerBounds(INDEX_CSS, "utilities");
  assert.ok(at > open && at < close, "inside @layer utilities");
  const printAt = INDEX_CSS.indexOf("@media print", at);
  assert.ok(printAt > at, "a print override follows");
  const print = INDEX_CSS.slice(printAt, INDEX_CSS.indexOf("}", INDEX_CSS.indexOf("}", printAt) + 1));
  assert.ok(print.includes(ROWS), "and names the transcript chunks");
  assert.ok(print.includes("content-visibility: visible !important") && print.includes("contain-intrinsic-size: none !important"));
});

test("the transcript mounts every fragment and never unmounts one for scrolling", () => {
  assert.equal(TRANSCRIPT.includes("@tanstack/react-virtual"), false, "no virtualizer");
  assert.equal(/translateY|getTotalSize|measureElement|rangeExtractor/.test(TRANSCRIPT), false, "no absolute row positioning");
  assert.match(TRANSCRIPT, /const shown = covered \? fragments\.length : limit;/, "rows render in order up to the widening limit");
  assert.match(TRANSCRIPT, /limitRef\.current = isCovered\(rows, all\.length\)\s*\? Number\.POSITIVE_INFINITY\s*: rows;/, "and the limit is dropped for good once it covers the trace");
  assert.match(TRANSCRIPT, /if \(streamingRef\.current\) setLimit\(value\);\s*else startTransition\(\(\) => setLimit\(value\)\);/, "widening yields to input on a settled trace and cannot be starved by a stream");
  assert.match(TRANSCRIPT, /if \(widenFrame === 0\) widenFrame = requestAnimationFrame\(widenAll\);/, "one shared frame drives every open transcript");
  assert.match(TRANSCRIPT, /let characters = Math\.ceil\(WIDEN_CHARACTERS_PER_FRAME \* widenScale\);/, "under one shared budget of characters per frame");
  const movers = TRANSCRIPT.replace(/useState\(\(\) => \{[\s\S]*?\n {2}\}\);/, "").replace(/setLimit\(isCovered\(value, fragments\.length\) \? Number\.POSITIVE_INFINITY : value\);/, "").replace(/requestedAt\.current = performance\.now\(\);\s*setLimit\(current\);/, "");
  assert.equal(/setLimit\((?!value\))/.test(movers), false, "no other path moves the limit, except the bounded restart for a replaced document and the re-issue of a starved request at its own value");
  assert.match(TRANSCRIPT, /if \(mount\.origin !== origin\) \{[\s\S]*?setMount\(\{\s*origin,\s*limit:/, "a replaced document restarts the bounded mount instead of mounting the new trace whole");
  assert.match(TRANSCRIPT, /const setLimit = useCallback\(\(value: number\) => \{\s*setMount\(\{ origin: originRef\.current, limit: value \}\);\s*\}, \[\]\);/, "every limit carries the document it was computed for, so a widening, print or re-issue queued for the previous document resolves to a restart, and one for the new document wins even when React replays it on a stale base");
  assert.match(TRANSCRIPT, /const islands = islandState\.origin === origin \? islandState\.list : NO_ISLANDS;/, "islands belong to the document they were made for");
  assert.equal(/setMountedFor|setIslands\(NO_ISLANDS\)/.test(TRANSCRIPT), false, "the restart needs no reset in render beyond the tagged limit");
  assert.match(TRANSCRIPT, /const restarted = originRef\.current !== origin;\s*if \(restarted\) \{\s*originRef\.current = origin;\s*mountedAt\.current = performance\.now\(\);\s*islandsRef\.current = NO_ISLANDS;\s*limitRef\.current = limit;\s*\}/, "and the requested limit follows it down once");
  assert.match(TRANSCRIPT, /const carried = restarted && carry\.current\(previous\);/, "a rewrite that is not an append re-keys every row, so the reader's passage is found again by its place in the trace and mounted at once");
  assert.match(TRANSCRIPT, /if \(restarted && !carried\) demandNow\.current\(\);/, "and a still reader gets rows under the viewport at once instead of a spacer until the next scroll");
  assert.match(TRANSCRIPT, /if \(limit > limitRef\.current\) limitRef\.current = limit;/, "a committed limit never moves the requested one backwards");
});

test("a row settles one frame after it has been laid out, so the engine remembers its real size", () => {
  assert.match(TRANSCRIPT, /const settleQueue = createSettleQueue<HTMLElement>\(\s*\(callback\) => requestAnimationFrame\(callback\),\s*\(row\) => \{\s*if \(!row\.isConnected\) return;\s*row\.setAttribute\(ROW_SETTLED_ATTRIBUTE, ""\);\s*const transcript = row\.parentElement;\s*if \(transcript && hiddenTranscripts\.has\(transcript\)\)\s*blindTranscripts\.add\(transcript\);\s*\},\s*\);/, "rows settle through the frame-counted queue: a whole frame lays each row out and records it before it may be skipped, and a row settled while its transcript has no box marks the transcript");
  assert.match(TRANSCRIPT, /const changed = new MutationObserver\(\(\) => \{\s*if \(!element\.hasAttribute\(ROW_SETTLED_ATTRIBUTE\)\) settleQueue\.add\(element\);\s*\}\);/, "a late render inside a closed chunk that has not settled yet restarts its frame count, so it is never skipped on the size it had before; a settled chunk is never unsettled, so no growth lands outside a commit");
  assert.match(TRANSCRIPT, /changed\.observe\(element, \{\s*subtree: true,\s*childList: true,\s*characterData: true,\s*\}\);/, "every row change inside a closed chunk is seen");
  assert.match(TRANSCRIPT, /getSnapshotBeforeUpdate\(\) \{\s*this\.props\.before\(\);\s*return null;\s*\}\s*componentDidUpdate\(\) \{\s*this\.props\.after\(\);\s*\}/, "a trace's growth is measured across its own commit, where no scroll by the reader can land");
  assert.match(TRANSCRIPT, /<CommitBounds before=\{beforeCommit\} after=\{afterCommit\}>\s*\{rendered\}\s*<\/CommitBounds>/);
  assert.match(TRANSCRIPT, /if \(!laidOut\(\)\) return;\s*reading \?\?= visibleAnchor\(\);\s*if \(reading\) \{\s*rootBefore = element\.getBoundingClientRect\(\)\.top;\s*return;\s*\}\s*if \(hasPendingProgressiveMounts\(\)\) return;\s*const bounds = scroll\.getBoundingClientRect\(\);\s*const box = element\.getBoundingClientRect\(\);\s*if \(box\.top < bounds\.top && box\.bottom <= bounds\.bottom\)\s*edge = box\.bottom - bounds\.top;/, "a reader who arrived since the last frame is anchored before the commit changes the DOM, so the commit's own hold corrects it; a trace that starts above the viewport and ends in it, with none of its rows on screen and no progressive message window correcting the same growth, measures its edge, so the reply below it stays put");
  assert.match(TRANSCRIPT, /const keepOnScreen = \(\) => \{\s*if \(reading && rootBefore !== null\)\s*reading\.offset \+= rootBefore - element\.getBoundingClientRect\(\)\.top;\s*rootBefore = null;\s*\};/, "a rewrite near the end of a trace shrank it to estimates, the browser clamped scrollTop, and the hold, which measures from the transcript, never saw it: the reader was thrown to the bottom and followed again");
  assert.match(TRANSCRIPT, /carry\.current = \(previous\) => \{\s*keepOnScreen\(\);/, "the restart commit's own movement");
  assert.match(TRANSCRIPT, /setIslands\(next\);\s*carrying = true;\s*return true;/);
  assert.match(TRANSCRIPT, /after: \(\) => \{\s*if \(carrying\) \{\s*carrying = false;\s*keepOnScreen\(\);\s*\}/, "and the island commit's, where native anchoring on a reply below kept the page at the clamped bottom");
  assert.match(TRANSCRIPT, /if \(Math\.abs\(shift\) >= SHIFT_EPSILON_PX\) adjustAbove\(Math\.round\(shift\)\);/, "undoes its growth, which WebKit's anchoring leaves to the page, so a reader below it stays put; native anchoring has already zeroed the shift elsewhere");
  assert.match(TRANSCRIPT, /return \(\) => \{\s*changed\.disconnect\(\);\s*remeasureQueue\.forget\(element\);\s*settleQueue\.forget\(element\);\s*\};/, "a chunk that unmounts or reopens leaves both queues");
  assert.equal(/data-settled=\{/.test(TRANSCRIPT), false, "settling is never a React prop");
  assert.equal((TRANSCRIPT.match(/removeAttribute\(ROW_SETTLED_ATTRIBUTE\)/g) ?? []).length, 1, "and the only unsettle is the relayout after a width change or a blind reveal");
  assert.match(TRANSCRIPT, /if \(blindTranscripts\.delete\(element\) && !widened\) \{\s*if \(settleTimer !== 0\) clearTimeout\(settleTimer\);\s*settleTimer = window\.setTimeout\(resettleLater, RESETTLE_DELAY_MS\);\s*\}/, "a trace shown again at its width after rows settled while it was hidden is laid out again, as after a width change, and a plain reopen costs nothing extra");
  assert.match(TRANSCRIPT, /data-reasoning-chunk=""/, "closed chunks of rows are the skip units");
  assert.match(TRANSCRIPT, /\[CHUNK_ESTIMATE_PROPERTY\]: `\$\{estimate\}px`/, "each chunk carries its estimate for the time before it has rendered");
  assert.match(TRANSCRIPT, /useSettledRow\(box, closed\);/, "a chunk settles once it is closed, never while rows can still join it");
  assert.match(TRANSCRIPT, /const Chunk = memo\(function Chunk\(\{[\s\S]*?\}, closedChunkUnchanged\);/, "every streaming commit re-rendered every mounted chunk, row and code group of the trace");
  assert.match(TRANSCRIPT, /streaming:\s*streaming &&\s*\(group\.code\s*\? fragments\[first\]\.code!\.incomplete\s*: first === fragments\.length - 1\),/, "a slice says whether it is still being written, so a closed chunk holding a fence's copy button renders again when the fence ends");
  assert.equal((TRANSCRIPT.match(/useSettledRow\(/g) ?? []).length, 2, "rows and code groups no longer settle on their own");
  assert.match(TRANSCRIPT, /closed=\{piece\.whole && \(!piece\.last \|\| sealed\)\}/, "only a whole grid chunk settles, and the last one only once the trace is complete and mounted");
  assert.match(TRANSCRIPT, /const sealed = covered && !streaming;/);
});

test("the DOM the rest of the app reads is unchanged", () => {
  for (const token of [
    'data-slot="reasoning-transcript"',
    'data-slot="reasoning-code-fragment"',
    'data-reasoning-code-row=""',
    "data-reasoning-fragment={fragment.key}",
    'style={{ overflowAnchor: "none" }}',
    "SearchImagesEnabledContext.Provider value={false}",
    "aui-reasoning-prose-fragment",
  ]) {
    assert.ok(TRANSCRIPT.includes(token), `kept: ${token}`);
  }
  assert.equal(TRANSCRIPT.split("data-reasoning-fragment={fragment.key}").length - 1, 2, "prose rows and code rows");
  assert.match(TRANSCRIPT, /if \(touches\) detach\(\);/, "a selection or focus inside the trace still parks the autoscroll");
  assert.match(TRANSCRIPT, /for \(const \{ from, to, reserve \} of mountPlan\(\s*shown,\s*islands,\s*fragments\.length,\s*\)\) \{/, "every unmounted stretch keeps its estimated height, laid out by the plan");
  assert.match(TRANSCRIPT, /data-reasoning-reserve=""\s*style=\{\{ height: grid\.heights\[to\] - grid\.heights\[from\] \}\}/, "as a spacer that shrinks as rows mount, so the thread below never jumps");
  assert.match(TRANSCRIPT, /const key = `reserve:\$\{reserve\}`;\s*rendered\.push\(\s*<div\s*key=\{key\}/, "and each spacer keeps one element for its whole life, so WebKit never lays the thread out without it mid-commit");
  assert.equal(/reserve:\$\{to\}|reserved > 0/.test(TRANSCRIPT), false, "no spacer is keyed by a moving row or dropped when its stretch fills");
  assert.match(TRANSCRIPT, /adjustAbove\(passage\.getBoundingClientRect\(\)\.top - pending\.top\)/, "the threshold handover still restores the reading passage");
  assert.match(TRANSCRIPT, /const top = passage\.getBoundingClientRect\(\)\.top;\s*anchor\.offset = top - box\.top;\s*adjustAbove\(top - anchor\.top\);/, "and a width change still re-finds it, against the viewport so a prompt above that re-wraps is undone too");
  assert.match(TRANSCRIPT, /settleTimer = window\.setTimeout\(resettleLater, RESETTLE_DELAY_MS\);/, "after a width change every chunk is laid out again before it may be skipped, once the drag has settled");
  assert.match(TRANSCRIPT, /resettleRows\(element\);\s*const next = measureGeometry\(element, codeProbe\.current\);\s*if \(next\) setGeometry\(keepGeometry\(next\)\);\s*if \(reading\) hold\(reading\);/, "and the relayout that follows keeps the reading passage where it was, with the spacers re-estimated for the new width");
  assert.match(TRANSCRIPT, /const rootTop = element\.getBoundingClientRect\(\)\.top;\s*const was = rootTop \+ anchor\.offset;\s*const bounds = scroll\.getBoundingClientRect\(\);\s*if \(was < bounds\.top - bounds\.height \|\| was > bounds\.bottom \+ bounds\.height\)\s*return false;\s*const passage = passageOf\(anchor\);/, "an anchor the reader has scrolled away from is dropped before anything inside a skipped chunk is measured");
  assert.match(TRANSCRIPT, /const top = passage\.getBoundingClientRect\(\)\.top;\s*const shift = top - was;/, "a shift is measured inside the transcript, which has native anchoring off, so a reader's own scroll is never undone");
  assert.match(TRANSCRIPT, /const chunks = element\.querySelectorAll<HTMLElement>\(\s*":scope > \[data-reasoning-chunk\]",\s*\);/, "the fold is found among chunk boxes, whose size is known while they are skipped");
  assert.match(TRANSCRIPT, /const rows = chunk\.querySelectorAll<HTMLElement>\(\s*"\[data-reasoning-fragment\]",\s*\);/, "and rows are measured only inside the chunk at the fold, never inside a skipped one");
  assert.match(TRANSCRIPT, /if \(Math\.abs\(shift\) < SHIFT_EPSILON_PX\) \{\s*residue = shift;\s*return true;\s*\}\s*anchor\.offset = top - rootTop;/, "sub-pixel residue accumulates instead of being dropped");
  assert.match(TRANSCRIPT, /if \(applied !== 0 \|\| \(acted && rounded !== 0\)\) \{\s*anchor\.offset -= shift - applied;\s*residue = shift - applied;\s*\}/, "and so does the part of a correction that whole-pixel scrolling could not apply");
  assert.match(TRANSCRIPT, /reading = visibleAnchor\(\);\s*if \(held && reading\) reading\.offset -= residue;/, "an anchor captured after a hold inherits that residue, so Firefox and WebKit do not creep a fraction of a pixel per commit");
  assert.match(TRANSCRIPT, /if \(!widened\) \{\s*if \(anchor && !atBottom\(\)\) hold\(anchor\);\s*return;\s*\}/, "a height change holds the passage unless the reader sits at the bottom");
  assert.match(TRANSCRIPT, /passageRange \?\?= document\.createRange\(\);\s*return reasoningTextRange\(row, anchor\.text, anchor\.occurrence, passageRange\);/, "passage lookups on every scrolled frame reuse one Range, which Blink would otherwise walk on every node removal");
  assert.match(TRANSCRIPT, /if \(committedRef\.current !== current\) \{\s*if \(performance\.now\(\) - requestedAt\.current < STARVED_MS\)\s*return NOTHING_SPENT;\s*requestedAt\.current = performance\.now\(\);\s*setLimit\(current\);\s*return NOTHING_SPENT;\s*\}/, "widening waits for the previous limit to commit, and a transition another reply's stream keeps discarding is re-issued through the sync path after STARVED_MS");
  assert.match(TRANSCRIPT, /const STARVED_MS = 200;/);
  assert.match(TRANSCRIPT, /return NOTHING_SPENT;\s*\}\s*if \(isCovered\(current, all\.length\)\) return null;/, "the final step waits for its own commit too, so a starved last transition is re-issued instead of leaving the tail a spacer");
  assert.equal(TRANSCRIPT.includes("afterprint"), false, "nothing unmounts rows after a print");
  assert.match(TRANSCRIPT, /let characters = Math\.ceil\(WIDEN_CHARACTERS_PER_FRAME \* widenScale\);\s*let rows = Math\.ceil\(WIDEN_FRAGMENTS_PER_FRAME \* widenScale\);\s*const pending = \[\.\.\.wideners\];/, "one frame's rows are shared by every open transcript, so their transitions land as one bounded commit");
  assert.match(TRANSCRIPT, /const remaining = pending\.length - at;/, "and a transcript that spends nothing passes its share on");
  assert.match(TRANSCRIPT, /Math\.min\(share, allowed\.fragments\),/, "and a transcript's rows never exceed its share of the frame");
  assert.match(TRANSCRIPT, /requestedAt\.current = performance\.now\(\);\s*limitRef\.current = isCovered\(rows, all\.length\)/, "the request time is what the starvation check measures from");
  assert.match(TRANSCRIPT, /if \(spent === null\) wideners\.delete\(widen\);/, "only a covered transcript leaves the shared frame");
  assert.match(TRANSCRIPT, /if \(next === islandsRef\.current\) return;\s*islandsRef\.current = next;\s*setIslands\(next\);/, "a reader who scrolls into an unmounted stretch gets an island of rows around the viewport at once");
  assert.match(TRANSCRIPT, /if \(end > start\) next = addIsland\(next, \{ start, end \}, prefix\);/, "islands only ever grow, so rows a reader saw stay mounted when they jump elsewhere");
  assert.match(TRANSCRIPT, /if \(!rect\.height \|\| rect\.bottom <= above \|\| rect\.top >= below\) continue;/, "a spacer with no box, as in a hidden folded round, never asks for rows");
  assert.match(TRANSCRIPT, /let rows = chunkEndAt\(gridRef\.current, budgeted\.rows\);/, "the prefix grows by whole grid chunks, so it never meets an island inside a chunk");
  assert.match(TRANSCRIPT, /const start = chunkStartAt\(\s*gridRef\.current,/, "and islands start and end on chunk boundaries too");
  assert.match(TRANSCRIPT, /window\.matchMedia\?\.\("print"\)\?\.addEventListener\?\.\(\s*"change",\s*\(event\) => \{\s*if \(event\.matches\) printTranscripts\(\);\s*\},\s*\{ capture: true \},\s*\);/, "the print media query is a door of its own, for page.pdf() and print emulation");
  assert.match(TRANSCRIPT, /for \(const skip of islandsRef\.current\)\s*if \(rows >= skip\.start && rows < skip\.end\) rows = skip\.end;/, "the bounded prefix then fills the gaps and steps over rows the islands already hold");
  assert.match(TRANSCRIPT, /for \(const piece of chunkPieces\(grid, fragmentKey, from, to\)\)/, "chunk boundaries come from one grid over the whole trace, so a range edge never re-keys the chunks on either side of it");
  assert.match(TRANSCRIPT, /registry\.set\(key, \{ element, from, to \}\);\s*return \(\) => \{\s*if \(registry\.get\(key\)\?\.element === element\) registry\.delete\(key\);/, "each spacer registers the stretch it stands for, so demand reads committed geometry");
  assert.match(TRANSCRIPT, /if \(grew \|\| mountedIslands\.current !== islands\) \{\s*mountedIslands\.current = islands;\s*if \(!carried\) recapture\.current\(\);\s*\}/, "the reading anchor is captured as soon as rows land under a viewport that had none, so a reflow still has a passage to restore");
  assert.match(TRANSCRIPT, /const laidOut = \(\) => \{\s*const now = element\.getBoundingClientRect\(\)\.width;\s*return now !== 0 && now === width;\s*\};/, "a transcript inside a hidden ancestor has no box, so it neither captures nor demands rows");
  assert.match(TRANSCRIPT, /const rounded = Math\.round\(shift\);\s*const acted = adjustAbove\(rounded\);\s*const applied = scroll\.scrollTop - before;\s*if \(applied !== 0 \|\| \(acted && rounded !== 0\)\) \{\s*anchor\.offset -= shift - applied;/, "the part of a correction the engine rounded away or clamped is carried to the next one, so the passage never creeps and a clamped correction lands once the scroll range allows it");
  assert.match(TRANSCRIPT, /recapture\.current = \(fresh = false\) => \{\s*if \(!laidOut\(\)\) return;\s*if \(!fresh && reading && hold\(reading\)\) return;\s*reading = visibleAnchor\(\);/, "a commit undoes its own shift in the task that made it, before a scrolled frame could capture the shifted layout as the new anchor");
  assert.match(TRANSCRIPT, /adjustAbove\(passage\.getBoundingClientRect\(\)\.top - pending\.top\);\s*recapture\.current\(true\);/, "and re-read right after the reopen restore moves the viewport");
  assert.match(TRANSCRIPT, /const mounted = committedRef\.current;\s*const prefix = isCovered\(mounted, all\) \? all : mounted;/, "demand measures the spacer the DOM has, not the limit a pending transition will give it");
  assert.match(TRANSCRIPT, /const capture = \(\) => \{\s*frame = 0;\s*if \(!laidOut\(\)\) return;\s*const held = reading !== undefined && hold\(reading\);\s*demand\(\);\s*reading = visibleAnchor\(\);/, "a scroll the engine's own anchoring fires inside the resize frame must not replace the pre-reflow anchor, nor demand rows for a position the width handler is about to undo");
  assert.match(TRANSCRIPT, /if \(committedRef\.current === value\) return;\s*limitRef\.current = value;\s*setLimit\(value\);/, "a print mounts the whole trace even while the last widening transition is still pending");
  assert.match(TRANSCRIPT, /const printTranscripts = \(\): void => \{\s*flushSync\(\(\) => \{\s*for \(const print of printers\) print\(\);\s*\}\);\s*\};/, "every open transcript lands in one synchronous commit; its code is plain, so there is no fence left to latch");
  assert.match(TRANSCRIPT, /for \(let i = firstEndingBelow\(rows, bounds\.top\); i < rows\.length; i \+= 1\) \{\s*const row = rows\[i\];\s*if \(row\.getBoundingClientRect\(\)\.top >= bounds\.bottom\) return undefined;/, "a fold row without capturable text falls through to the next visible row");
  assert.match(TRANSCRIPT, /const chunk = chunks\[at\];\s*if \(chunk\.getBoundingClientRect\(\)\.top >= bounds\.bottom\) break;/, "and on into the next chunk while it is still on screen");
  assert.match(TRANSCRIPT, /row\.removeAttribute\(ROW_SETTLED_ATTRIBUTE\);\s*settleQueue\.add\(row\);/, "re-settling goes through the same queue as the first settle");
  assert.match(TRANSCRIPT, /"aui-reasoning-code-fragment relative isolate min-w-0"/, "a code group positions its own copy actions; chunks settle now, so no containment around the group gives them a box");
  assert.match(TRANSCRIPT, /useState\(\(\) => cachedReasoningTranscriptIndex\(indexKey\)\)/, "a reopened transcript keeps the index it already parsed");
});

test("the per-frame reading capture reuses one Range", () => {
  assert.match(ANCHOR, /const range = \(probe \?\?= document\.createRange\(\)\);/, "one live Range for every capture: the engine walks every attached Range on each node removal, so a Range per text node per scrolled frame made the fence windows' span churn an order of magnitude dearer");
  assert.equal((ANCHOR.match(/document\.createRange\(\)/g) ?? []).length, 2, "the only other Range is the one a caller asked for by position");
});

test("startup arms the attribute after the code-block one", () => {
  const code = MAIN.indexOf("watchCodeBlockContainmentOverride();");
  const apply = MAIN.indexOf("applyReasoningRowContainment();");
  const watch = MAIN.indexOf("watchReasoningRowContainmentOverride();");
  assert.ok(code >= 0 && apply > code && watch > apply && watch < MAIN.indexOf("function renderApp()"));
  assert.ok(CONTAINMENT.includes('import { engineFindsSkippedContent } from "./math-block-containment";'));
  assert.ok(CONTAINMENT.includes("import.meta.env.VITE_UNSLOTH_REASONING_ROW_CONTAINMENT"));
  assert.ok(CONTAINMENT.includes("root.removeAttribute(REASONING_ROW_CONTAINMENT.attribute)"));
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  INITIAL_VIEWPORTS,
  SETTLE_MS,
  WIDEN_CHARACTERS_PER_FRAME,
  WIDEN_FRAGMENTS_PER_FRAME,
  estimateFragmentHeight,
  estimateFragmentHeights,
  closedChunkUnchanged,
  frameBudget,
  buildChunkGrid,
  chunkEndAt,
  chunkPieces,
  chunkStartAt,
  addIsland,
  mountedRanges,
  mountPlan,
  type MountItem,
  NO_ISLANDS,
  TAIL_RESERVE,
  CHUNK_CHARACTERS,
  CHUNK_MAX_GROUPS,
  initialRows,
  isCovered,
  anchorAtSamePlace,
  widenBudget,
} from "../src/components/assistant-ui/reasoning-mount-controller.ts";
import { ReasoningTranscriptIndex } from "../src/components/assistant-ui/reasoning-transcript-index.ts";
import {
  REASONING_ROW_CONTAINMENT,
  REASONING_ROW_CONTAINMENT_ATTRIBUTE,
  ROW_SETTLED_ATTRIBUTE,
  CHUNK_ESTIMATE_PROPERTY,
  SHIP_DEFAULT,
} from "../src/components/assistant-ui/reasoning-row-containment-mode.ts";
import { defineContainmentFlag } from "../src/components/assistant-ui/containment-flag.ts";

import { readSrc } from "./helpers/kit.ts";

const geometry = { width: 640, lineHeight: 24, fontPixels: 15, codeLineHeight: 18.9, codeFontPixels: 12.2, codeFooter: 12.25 };

test("a hidden fragment has no height and prose loses its paragraph gap", () => {
  assert.equal(estimateFragmentHeight({ text: "x", first: true, last: false, hidden: true }, geometry), 0);
  const prose = estimateFragmentHeight({ text: "one\ntwo\nthree\nfour", first: false, last: false }, geometry);
  assert.equal(prose, 2 * 24, "four lines of prose estimate as the lines minus the two blank separators");
  const first = estimateFragmentHeight({ text: "one\ntwo\nthree\nfour", first: true, last: false }, geometry);
  assert.equal(first, prose + 16, "the first fragment carries the pane's top padding");
});

test("a code fragment is sized per line of the code block, with its header and footer", () => {
  const code = { source: "a\nb", incomplete: false, language: "ts", lines: [] };
  const middle = estimateFragmentHeight({ text: "a\nb", first: false, last: false, code }, geometry);
  assert.equal(middle, 2 * 18.9, "code lines at the prose line height reserved 13% too much, and a reader following a stream was clamped up when the rows mounted");
  const whole = estimateFragmentHeight({ text: "a\nb", first: true, last: true, code }, geometry);
  assert.equal(whole, 2 * 18.9 + 40 + 12.25, "and the footer is the measured one, never more");
  const transcript = readSrc("components/assistant-ui/reasoning-transcript.tsx");
  assert.match(transcript, /<div\s+ref=\{codeProbe\}\s+aria-hidden\s+className="aui-reasoning-code-fragment aui-reasoning-code-last pointer-events-none invisible absolute"\s*>\s*<div className="aui-reasoning-code-lines" \/>\s*<\/div>/, "the metrics come from the code block's own rules, chosen font size included");
  assert.equal((transcript.match(/measureGeometry\(element, codeProbe\.current\)/g) ?? []).length, 2);
});

test("long lines wrap into the estimate at the measured column count", () => {
  const columns = Math.floor(640 / (15 * 0.48));
  const text = "x".repeat(columns * 6);
  const single = estimateFragmentHeight({ text: "x", first: false, last: false }, geometry);
  const wrapped = estimateFragmentHeight({ text, first: false, last: false }, geometry);
  assert.ok(wrapped > single, "a wrapped line costs more than a short one");
});

test("a streaming commit estimates only the rows it has not seen at this geometry", () => {
  let reads = 0;
  const row = (text: string) => ({
    get text() {
      reads += 1;
      return text;
    },
    first: false,
    last: false,
  });
  const kept = row("one\ntwo\nthree");
  const first = estimateFragmentHeights([kept, row("x")], geometry);
  assert.deepEqual(first, [estimateFragmentHeight(kept, geometry), estimateFragmentHeight({ text: "x", first: false, last: false }, geometry)]);
  reads = 0;
  const grown = row("x y");
  assert.deepEqual(estimateFragmentHeights([kept, grown], geometry), [first[0], estimateFragmentHeight(grown, geometry)]);
  assert.equal(reads, 2, "every commit re-split the text of every row in the trace");
  reads = 0;
  estimateFragmentHeights([kept], { ...geometry, width: 320 });
  assert.equal(reads, 1, "a new width estimates every row again");
});

test("the first commit covers the budget and always includes the anchor row", () => {
  const estimates = [100, 100, 100, 100, 100, 100, 100, 100];
  assert.equal(initialRows(estimates, 250, -1), 3, "three rows cover 250px");
  assert.equal(initialRows(estimates, 250, 5), 8, "the anchor row, everything above it, and the budget below it so the reader's position survives the hand-over");
  assert.equal(initialRows(estimates, 250, 7), 8, "an anchor near the end mounts to the end");
  assert.equal(initialRows(estimates, 10_000, -1), 8, "and never more than there are");
  assert.equal(initialRows(estimates, 10_000, -1, Array(8).fill(1500)), 3, "and never more characters than one frame can parse, so a reopen stays inside the collapse scroll lock");
  assert.equal(initialRows(estimates, 10_000, -1, Array(8).fill(9000)), 1, "one oversized fragment still mounts alone");
  assert.equal(initialRows(estimates, 10_000, 6, Array(8).fill(9000)), 7, "unless the anchor row needs more");
  assert.equal(initialRows(estimates, 0, -1), 1, "at least one row, so an opened pane is never empty");
  assert.equal(initialRows([], 250, -1), 0, "an empty trace has nothing to mount");
});

test("widening covers the trace and stops at the end", () => {
  assert.equal(isCovered(10, 10), true);
  assert.equal(isCovered(9, 10), false);
  const sizes = Array(40).fill(100);
  let mounted = initialRows(sizes, 2 * 900, -1);
  let steps = 0;
  while (!isCovered(mounted, 40)) {
    mounted = widenBudget(sizes, mounted, 300).rows;
    steps += 1;
    assert.ok(steps < 100, "widening converges");
  }
  assert.equal(mounted, 40);
  assert.ok(steps <= Math.ceil(40 / 3) + 1);
  assert.equal(INITIAL_VIEWPORTS, 1);
});

test("widening stays light while the collapse animation and its scroll lock run", () => {
  const settling = frameBudget(0);
  const steady = frameBudget(SETTLE_MS);
  assert.ok(settling.characters < steady.characters && settling.fragments < steady.fragments);
  assert.equal(steady.characters, WIDEN_CHARACTERS_PER_FRAME);
  assert.equal(steady.fragments, WIDEN_FRAGMENTS_PER_FRAME);
  assert.ok(SETTLE_MS >= 200, "at least the collapsible's animation");
});

test("a frame's widening is budgeted by characters, never by fragment count", () => {
  const sizes = [200, 200, 8000, 8000, 200, 200];
  assert.deepEqual(widenBudget(sizes, 0, 1000), { rows: 2, spent: 400 }, "small fragments fill the budget together");
  assert.deepEqual(widenBudget(sizes, 2, 1000), { rows: 3, spent: 8000 }, "an oversized fragment still mounts, alone");
  assert.deepEqual(widenBudget(sizes, 3, 24_000), { rows: 6, spent: 8400 }, "the rest fits in one frame");
  assert.deepEqual(widenBudget(sizes, 6, 24_000), { rows: 6, spent: 0 }, "nothing left");
  assert.ok(WIDEN_CHARACTERS_PER_FRAME >= 8_192, "at least one full fragment per frame");
  const tiny = Array(500).fill(10);
  assert.deepEqual(widenBudget(tiny, 0, 24_000), { rows: 64, spent: 640 }, "many tiny fragments are capped per frame too");
  assert.deepEqual(widenBudget(tiny, 0, 24_000, 8), { rows: 8, spent: 80 });
});

test("the reasoning-row flag is a containment flag with its own names", () => {
  assert.equal(SHIP_DEFAULT, "contain");
  assert.equal(REASONING_ROW_CONTAINMENT_ATTRIBUTE, "data-reasoning-row-containment");
  assert.equal(REASONING_ROW_CONTAINMENT.global, "__UNSLOTH_REASONING_ROW_CONTAINMENT__");
  assert.equal(ROW_SETTLED_ATTRIBUTE, "data-settled");
  assert.equal(REASONING_ROW_CONTAINMENT.resolve(undefined, ""), "contain");
  assert.equal(REASONING_ROW_CONTAINMENT.resolve(undefined, "typo"), "off");
  assert.equal(REASONING_ROW_CONTAINMENT.mode(undefined, "", false), "off", "gated on the engine");
  assert.equal(REASONING_ROW_CONTAINMENT.mode(true, "", false), "contain", "unless forced from the console");
});

test("the flag factory resolves, gates and watches independently per instance", () => {
  const a = defineContainmentFlag({ attribute: "data-a", global: "__A__" });
  const b = defineContainmentFlag({ attribute: "data-b", global: "__B__", shipDefault: "off" });
  assert.equal(a.resolve(undefined, ""), "contain");
  assert.equal(b.resolve(undefined, ""), "off");
  assert.equal(b.resolve("contain", ""), "contain");
  assert.equal(a.resolve(false, "contain"), "off");
  assert.equal(a.mode("contain", "", false), "contain");
  assert.equal(a.mode(undefined, "contain", false), "off");
  const scope: Record<string, unknown> = {};
  const applied: string[] = [];
  assert.equal(a.installWatcher(scope, () => { applied.push("a"); return "off"; }), true);
  assert.equal(b.installWatcher(scope, () => { applied.push("b"); return "off"; }), true);
  scope.__A__ = "off";
  scope.__B__ = true;
  assert.deepEqual(applied, ["a", "b"]);
  assert.equal(scope.__A__, "off");
  assert.equal(scope.__B__, true);
  assert.equal(a.installWatcher(Object.freeze({}) as Record<string, unknown>, () => "off"), false);
});

test("the controller and flag modules are plain TypeScript", () => {
  for (const file of [
    "components/assistant-ui/reasoning-mount-controller.ts",
    "components/assistant-ui/reasoning-row-containment-mode.ts",
    "components/assistant-ui/containment-flag.ts",
  ]) {
    const source = readSrc(file);
    assert.ok(!/<\/?[a-z]+[\s>]/i.test(source.replace(/^\s*[/*].*$/gm, "")), `${file}: no JSX`);
    assert.ok(!/\bfrom\s+["']react["']/.test(source), `${file}: no react import`);
    assert.ok(!source.includes("import.meta"), `${file}: no import.meta`);
  }
});

type GridInput = { key: string; text: string; document: number; start: number; code?: { source: string; incomplete: boolean; language: string | null; lines: { line: number; column: number; text: string }[] } };
const prose = (key: string, chars: number, start = 0): GridInput => ({ key, text: "x".repeat(chars), document: 0, start });
const code = (fence: number, line: number, chars: number): GridInput => ({
  key: `0:0:${fence}:code:${line}:0`,
  text: "y".repeat(chars),
  document: 0,
  start: fence,
  code: { source: "", incomplete: false, language: "js", lines: [] },
});

test("the grid groups a fence's fragments, closes chunks by characters or count, and keys each by its first group", () => {
  assert.equal(CHUNK_ESTIMATE_PROPERTY, "--unsloth-chunk-estimate");
  assert.ok(CHUNK_CHARACTERS >= 4_096 && CHUNK_MAX_GROUPS >= 8);
  const fragments = [prose("a", 3000), prose("b", 3000), prose("c", 3000), code(9000, 0, 2000), code(9000, 1, 2000), prose("e", 10)];
  const grid = buildChunkGrid(fragments, fragments.map(() => 10), 8_192, 12);
  assert.deepEqual(grid.groups.map((g) => [g.key, g.first, g.end, g.code]), [
    ["a", 0, 1, false], ["b", 1, 2, false], ["c", 2, 3, false], ["0:9000", 3, 5, true], ["e", 5, 6, false],
  ]);
  assert.deepEqual(grid.chunks.map((c) => [c.key, c.first, c.end]), [["a", 0, 3], ["0:9000", 3, 6]], "a chunk takes groups until it already holds the budget");
  assert.deepEqual([...grid.heights], [0, 10, 20, 30, 40, 50, 60]);
  const many = Array.from({ length: 30 }, (_, i) => prose(`g${i}`, 10));
  assert.deepEqual(buildChunkGrid(many, many.map(() => 1), 8_192, 12).chunks.map((c) => c.end - c.first), [12, 12, 6]);
  assert.deepEqual(buildChunkGrid([], []).chunks, []);
});

test("a long fence splits into chunk-sized groups at line starts, keyed so its growth never re-keys one", () => {
  const row = (line: number, chars: number, column = 0): GridInput => ({
    key: `0:0:7:code:${line}:${column}`,
    text: "y".repeat(chars),
    document: 0,
    start: 7,
    code: { source: "", incomplete: true, language: "js", lines: [{ line, column, text: "" }] },
  });
  const fence = Array.from({ length: 12 }, (_, i) => row(i * 16, 1000));
  const grid = buildChunkGrid(fence, fence.map(() => 1), 8_192, 12);
  assert.deepEqual(grid.groups.map((g) => [g.key, g.first, g.end, g.code]), [
    ["0:7", 0, 9, true], [fence[9].key, 9, 12, true],
  ]);
  assert.deepEqual(grid.chunks.map((c) => [c.key, c.first, c.end]), [["0:7", 0, 9], [fence[9].key, 9, 12]]);
  assert.equal(chunkEndAt(grid, 3), 9, "the mount budget can stop inside a long fence");
  const grown = [...fence, row(192, 1000)];
  assert.deepEqual(buildChunkGrid(grown, grown.map(() => 1), 8_192, 12).groups.slice(0, 2).map((g) => [g.key, g.first]), grid.groups.map((g) => [g.key, g.first]));
  const continued = fence.map((fragment, i) => (i === 9 ? row(128, 1000, 4000) : fragment));
  assert.deepEqual(buildChunkGrid(continued, continued.map(() => 1), 8_192, 12).groups.map((g) => [g.first, g.end]), [[0, 10], [10, 12]], "a row that continues a line stays with the line it continues");
});

test("a growing last group never moves to another chunk, and appended groups never move a boundary", () => {
  const base = [prose("a", 6000), prose("b", 1000), prose("tail", 100)];
  const grown = [prose("a", 6000), prose("b", 1000), prose("tail", 20_000)];
  const key = (fragments: GridInput[]) => buildChunkGrid(fragments, fragments.map(() => 1)).chunks.map((c) => [c.key, c.first, c.end]);
  assert.deepEqual(key(grown), key(base), "membership is decided by what the chunk held before the group");
  const before = Array.from({ length: 18 }, (_, i) => prose(`g${i}`, 700));
  const after = [...before, ...Array.from({ length: 7 }, (_, i) => prose(`h${i}`, 700))];
  const a = buildChunkGrid(before, before.map(() => 1)).chunks;
  const b = buildChunkGrid(after, after.map(() => 1)).chunks;
  for (let i = 0; i < a.length - 1; i += 1) assert.deepEqual(b[i], a[i]);
  assert.equal(b[a.length - 1].key, a.at(-1)!.key, "the open chunk keeps its key while it grows and when it closes");
});

test("a range cuts grid chunks into pieces whose keys stay unique and stable as the range moves", () => {
  const fragments = [prose("p0", 3000), prose("p1", 3000), code(6000, 0, 2000), code(6000, 1, 2000), code(6000, 2, 2000), prose("p5", 3000), prose("p6", 3000)];
  const grid = buildChunkGrid(fragments, fragments.map(() => 5), 8_192, 12);
  const key = (at: number) => fragments[at].key;
  const whole = chunkPieces(grid, key, 0, fragments.length);
  assert.deepEqual(whole.map((p) => [p.key, p.first, p.end, p.whole, p.last]), grid.chunks.map((c, i) => [c.key, c.first, c.end, true, i === grid.chunks.length - 1]));
  const prefix = chunkPieces(grid, key, 0, 3);
  const island = chunkPieces(grid, key, 4, 7);
  const keys = [...prefix, ...island].map((p) => p.key);
  assert.equal(new Set(keys).size, keys.length, "a fence split by a range edge never gives two sibling chunks one key");
  assert.equal(prefix[0].key, whole[0].key, "the piece at a chunk's start keeps the chunk's key");
  assert.deepEqual(island[0].groups[0], { index: 2, first: 4, end: 5 }, "the cut fence keeps only its lines inside the range");
  assert.equal(prefix.at(-1)?.whole, false, "a piece missing part of its chunk stays open");
  const grownPrefix = chunkPieces(grid, key, 0, 4);
  assert.deepEqual(grownPrefix.map((p) => p.key), prefix.map((p) => p.key), "a prefix growing inside a chunk keeps every key");
});

test("a closed chunk re-renders only when a prop, a slice or one of its rows changes", () => {
  const rows = [{}, {}, {}, {}];
  const slice = { key: "a", code: false, first: 1, end: 2, streaming: false };
  const chunk = (over: Record<string, unknown> = {}) => ({
    slices: [slice, { key: "b", code: true, first: 2, end: 3, streaming: false }],
    fragments: rows,
    closed: true,
    estimate: 120,
    messageId: "m",
    ...over,
  });
  assert.equal(closedChunkUnchanged(chunk(), chunk({ fragments: [...rows, {}] })), true, "a commit that only grew the trace past it leaves it alone");
  assert.equal(closedChunkUnchanged(chunk(), chunk({ closed: false })), false);
  assert.equal(closedChunkUnchanged(chunk({ closed: false }), chunk()), false, "the commit that closes it renders it");
  assert.equal(closedChunkUnchanged(chunk(), chunk({ estimate: 140 })), false);
  assert.equal(closedChunkUnchanged(chunk(), chunk({ messageId: "n" })), false, "any other prop is compared too");
  assert.equal(closedChunkUnchanged(chunk(), chunk({ extra: 1 })), false);
  assert.equal(closedChunkUnchanged(chunk(), chunk({ fragments: [rows[0], {}, rows[2], rows[3]] })), false, "a replaced row inside it");
  assert.equal(closedChunkUnchanged(chunk(), chunk({ fragments: [rows[0], rows[1], rows[2], {}] })), false, "or the row after it, whose first line decides the group's trailing line break");
  assert.equal(closedChunkUnchanged(chunk(), chunk({ fragments: [{}, rows[1], rows[2], rows[3]] })), true, "a row before it is not its business");
  assert.equal(
    closedChunkUnchanged(chunk(), chunk({ slices: [slice, { key: "b", code: true, first: 2, end: 3, streaming: true }] })),
    false,
    "a fence still being written keeps its copy button disabled, and the commit that finishes it enables the button with the whole source",
  );
});

test("islands only grow, keep their identity, and stay listed after the prefix passes them", () => {
  let islands = addIsland(NO_ISLANDS, { start: 50, end: 60 }, 10);
  assert.deepEqual(islands, [{ id: 50, start: 50, end: 60 }]);
  const same = addIsland(islands, { start: 52, end: 58 }, 10);
  assert.equal(same, islands, "a covered request returns the same array, so nothing commits");
  islands = addIsland(islands, { start: 90, end: 95 }, 10);
  assert.deepEqual(islands, [{ id: 50, start: 50, end: 60 }, { id: 90, start: 90, end: 95 }], "a jump elsewhere keeps the rows already mounted");
  islands = addIsland(islands, { start: 60, end: 70 }, 10);
  assert.deepEqual(islands, [{ id: 50, start: 50, end: 70 }, { id: 90, start: 90, end: 95 }], "an adjacent request grows the island it touches");
  islands = addIsland(islands, { start: 40, end: 92 }, 10);
  assert.deepEqual(islands, [{ id: 50, start: 40, end: 92 }, { id: 90, start: 90, end: 95 }], "a request across two islands grows the first, and the second keeps its identity");
  assert.equal(addIsland(islands, { start: 60, end: 94 }, 10), islands, "rows held by overlapping islands count as covered");
  assert.equal(addIsland(islands, { start: 0, end: 20 }, 30), islands, "rows the prefix already holds are never an island");
  assert.deepEqual(addIsland(islands, { start: 100, end: 110 }, 96), [...islands, { id: 100, start: 100, end: 110 }], "islands the prefix passed stay listed, so their spacers keep their place");
  assert.deepEqual(addIsland(addIsland(NO_ISLANDS, { start: 5, end: 15 }, 0), { start: 5, end: 40 }, 30).map((island) => island.id), [5, 30], "an island is named by the first row it mounts, so a request clipped by the prefix cannot reuse another island's name");
});

test("a rewrite that re-keys every row still finds each row at its place in the trace", () => {
  const code = Array.from({ length: 600 }, (_, i) => `const line${i} = ${i};`).join("\n");
  const trace = `First thought.\n\n\`\`\`javascript\n${code}\n\`\`\`\n\nLast thought.`;
  const index = new ReasoningTranscriptIndex();
  const before = index.update([`${trace} </thi`]);
  const after = index.update([trace]);
  const prose = before[0];
  const codeRow = before.find((fragment) => (fragment.code?.lines[0]?.line ?? 0) > 0);
  assert.ok(codeRow, "the fence spans several code rows");
  for (const row of [prose, codeRow]) {
    const place = anchorAtSamePlace(before, after, row, "const", 2);
    assert.ok(place);
    assert.notEqual(after[place.index].key, row.key, "a split closing tag rewrote the trace, so every key changed");
    assert.equal(after[place.index].text, row.text);
    assert.equal(place.occurrence, 2);
  }
  assert.equal(anchorAtSamePlace(before, after, { ...prose, document: 1 }, "First", 0), null);
});

test("a streamed fence's rows keep the reader's place when the rewrite parses the fence as one prose row", () => {
  const code = Array.from({ length: 40 }, (_, i) => `const line${i} = ${i};`).join("\n");
  const trace = `First thought.\n\n\`\`\`javascript\n${code}\n\`\`\`\n\nSecond thought.\n\nThird thought.`;
  const index = new ReasoningTranscriptIndex();
  for (let end = 8; end < trace.length; end += 8) index.update([trace.slice(0, end)]);
  const before = index.update([`${trace} </thi`]);
  const after = index.update([trace]);
  const fence = after.find((fragment) => fragment.text.startsWith("```"));
  assert.ok(fence && !fence.code, "PRECONDITION: the full parse holds the fence as prose");
  const codeRows = before.filter((fragment) => fragment.code);
  assert.ok(codeRows.length > 1, "PRECONDITION: the stream held the fence as several code rows");
  for (const row of codeRows) assert.equal(after[anchorAtSamePlace(before, after, row, "const", 0)!.index], fence);
  const third = codeRows[2];
  assert.equal(third.code?.lines[0]?.line, 32, "PRECONDITION: code rows hold 16 lines");
  assert.deepEqual(
    anchorAtSamePlace(before, after, third, "const", 3),
    { index: after.indexOf(fence), occurrence: 35 },
    "the passage on line 35 is the fourth match in its code row but the 36th in the whole fence, so the reader was moved up to line 3",
  );
  assert.deepEqual(
    anchorAtSamePlace(after, before, fence, "const", 35),
    { index: before.indexOf(third), occurrence: 3 },
    "a fence held as one prose row finds the code row that holds the passage",
  );
  for (const row of before.filter((fragment) => !fragment.code)) {
    const place = anchorAtSamePlace(before, after, row, "thought", 0);
    assert.ok(place, `no place for ${JSON.stringify(row.text)}`);
    assert.ok(after[place.index].text.includes(row.text.trim().replace(" </thi", "")));
  }
});

test("mounted ranges merge the prefix with the islands and clip to the trace", () => {
  assert.deepEqual(mountedRanges(10, NO_ISLANDS, 100), [[0, 10]]);
  assert.deepEqual(mountedRanges(10, [{ start: 40, end: 50 }, { start: 80, end: 120 }], 100), [[0, 10], [40, 50], [80, 100]]);
  assert.deepEqual(mountedRanges(45, [{ start: 40, end: 50 }], 100), [[0, 50]], "an island the prefix reached joins it");
  assert.deepEqual(mountedRanges(10, [{ start: 10, end: 20 }], 100), [[0, 20]]);
  assert.deepEqual(mountedRanges(0, [{ start: 5, end: 9 }], 100), [[5, 9]]);
  assert.deepEqual(mountedRanges(Number.POSITIVE_INFINITY, [{ start: 5, end: 9 }], 30), [[0, 30]]);
});

test("the prefix and islands snap to whole grid chunks, so no chunk is ever split across two pieces", () => {
  const fragments = Array.from({ length: 30 }, (_, i) => prose(`r${i}`, 1000));
  const grid = buildChunkGrid(fragments, fragments.map(() => 1), 4_096, 12);
  assert.deepEqual(grid.chunks.map((c) => [c.first, c.end]), [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 30]]);
  assert.equal(chunkEndAt(grid, 0), 0);
  assert.equal(chunkEndAt(grid, 1), 5);
  assert.equal(chunkEndAt(grid, 5), 5);
  assert.equal(chunkEndAt(grid, 6), 10);
  assert.equal(chunkEndAt(grid, 30), 30);
  assert.equal(chunkEndAt(grid, Number.POSITIVE_INFINITY), Number.POSITIVE_INFINITY);
  assert.equal(chunkStartAt(grid, 0), 0);
  assert.equal(chunkStartAt(grid, 7), 5);
  assert.equal(chunkStartAt(grid, 10), 10);
  assert.equal(chunkStartAt(grid, 30), 30);
  const key = (at: number) => fragments[at].key;
  const pieces = [...chunkPieces(grid, key, 0, chunkEndAt(grid, 7)), ...chunkPieces(grid, key, chunkStartAt(grid, 17), chunkEndAt(grid, 22))];
  assert.ok(pieces.every((piece) => piece.whole), "aligned ranges only ever render whole chunks");
  assert.deepEqual(pieces.map((piece) => piece.key), ["r0", "r5", "r15", "r20"]);
});


test("the mount plan gives every unmounted stretch to the island below it, or to the tail", () => {
  assert.deepEqual(mountPlan(10, NO_ISLANDS, 100), [{ from: 0, to: 10 }, { from: 10, to: 100, reserve: TAIL_RESERVE }]);
  assert.deepEqual(mountPlan(100, NO_ISLANDS, 100), [{ from: 0, to: 100 }, { from: 100, to: 100, reserve: TAIL_RESERVE }], "a covered trace keeps its tail spacer, at zero height");
  const middle = [{ id: 40, start: 40, end: 50 }];
  assert.deepEqual(mountPlan(10, middle, 100), [{ from: 0, to: 10 }, { from: 10, to: 40, reserve: "40" }, { from: 40, to: 50 }, { from: 50, to: 100, reserve: TAIL_RESERVE }]);
  assert.deepEqual(mountPlan(60, middle, 100), [{ from: 0, to: 40 }, { from: 40, to: 40, reserve: "40" }, { from: 40, to: 60 }, { from: 60, to: 100, reserve: TAIL_RESERVE }], "a closed gap keeps its spacer where it closed");
  const overlapping = [{ id: 50, start: 40, end: 92 }, { id: 90, start: 90, end: 95 }];
  assert.deepEqual(mountPlan(10, overlapping, 100), [{ from: 0, to: 10 }, { from: 10, to: 40, reserve: "50" }, { from: 40, to: 90 }, { from: 90, to: 90, reserve: "90" }, { from: 90, to: 95 }, { from: 95, to: 100, reserve: TAIL_RESERVE }]);
  assert.deepEqual(mountPlan(10, middle, 30), [{ from: 0, to: 10 }, { from: 10, to: 10, reserve: "40" }, { from: 10, to: 30, reserve: TAIL_RESERVE }], "an island the trace no longer reaches keeps its spacer, at zero height, before the tail");
});

test("a reader at the bottom gets an island while the remainder's spacer stays the same element", () => {
  assert.deepEqual(mountPlan(8, NO_ISLANDS, 100), [{ from: 0, to: 8 }, { from: 8, to: 100, reserve: TAIL_RESERVE }]);
  const islands = addIsland(NO_ISLANDS, { start: 90, end: 100 }, 8);
  assert.deepEqual(mountPlan(12, islands, 100), [{ from: 0, to: 12 }, { from: 12, to: 90, reserve: "90" }, { from: 90, to: 100 }, { from: 100, to: 100, reserve: TAIL_RESERVE }], "the island and its spacer are inserted before the tail, which only shrinks");
});

const planElements = (plan: MountItem[]): string[] =>
  plan.flatMap((item) => (item.reserve === undefined ? Array.from({ length: item.to - item.from }, (_, i) => `row:${item.from + i}`) : [`reserve:${item.reserve}`]));

test("no plan step removes or reorders an element, so WebKit never lays the thread out without a spacer", () => {
  let seed = 7;
  const random = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  for (let trial = 0; trial < 300; trial += 1) {
    let total = 20 + Math.floor(random() * 180);
    let prefix = 1 + Math.floor(random() * 8);
    let islands = NO_ISLANDS;
    let previous = mountPlan(prefix, islands, total);
    for (let step = 0; step < 50; step += 1) {
      const roll = random();
      if (roll < 0.4) prefix = Math.min(total, prefix + 1 + Math.floor(random() * 12));
      else if (roll < 0.85) {
        const start = Math.floor(random() * total);
        islands = addIsland(islands, { start, end: Math.min(total, start + 1 + Math.floor(random() * 12)) }, prefix);
      } else if (roll < 0.95) {
        const covered = prefix >= total;
        total += Math.floor(random() * 8);
        if (covered) prefix = total;
      } else {
        total = Math.max(1, total - 1 - Math.floor(random() * 6));
        prefix = Math.min(prefix, total);
      }
      const ids = islands.map((island) => island.id);
      assert.equal(new Set(ids).size, ids.length, "island names stay unique, so no two spacers share a key");
      for (const island of islands) assert.ok(island.start <= island.id && island.id < island.end, "an island is named by one of its own rows");
      const plan = mountPlan(prefix, islands, total);
      let cursor = 0;
      for (const item of plan) {
        assert.equal(item.from, cursor, "the plan walks the trace in order with no gap or overlap");
        assert.ok(item.to >= item.from);
        cursor = item.to;
      }
      assert.equal(cursor, total, "and accounts for every row");
      const mounted = new Set<number>();
      for (let row = 0; row < Math.min(prefix, total); row += 1) mounted.add(row);
      for (const island of islands) for (let row = island.start; row < Math.min(island.end, total); row += 1) mounted.add(row);
      const rows = plan.filter((item) => item.reserve === undefined).reduce((sum, item) => sum + item.to - item.from, 0);
      assert.equal(rows, mounted.size, "rows are mounted exactly where the prefix and the islands reach");
      const before = planElements(previous).filter((element) => !element.startsWith("row:") || Number(element.slice(4)) < total);
      const after = planElements(plan);
      const kept = new Set(after);
      for (const element of before) assert.ok(kept.has(element), `${element} was removed by a later plan`);
      const earlier = new Set(before);
      assert.deepEqual(after.filter((element) => earlier.has(element)), before, "elements that stay keep their order, so none is moved");
      previous = plan;
    }
  }
});

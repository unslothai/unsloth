// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReasoningFragment } from "./reasoning-transcript-index.ts";

export type FragmentGeometry = {
  width: number;
  lineHeight: number;
  fontPixels: number;
  codeLineHeight: number;
  codeFontPixels: number;
  codeFooter: number;
};

export const INITIAL_VIEWPORTS = 1;
export const INITIAL_CHARACTERS = 4_096;
export const WIDEN_CHARACTERS_PER_FRAME = 24_000;
export const WIDEN_FRAGMENTS_PER_FRAME = 64;
export const SETTLE_MS = 300;
export const SETTLING_DIVISOR = 4;

export const frameBudget = (
  sinceMountMs: number,
): { characters: number; fragments: number } =>
  sinceMountMs < SETTLE_MS
    ? {
        characters: Math.ceil(WIDEN_CHARACTERS_PER_FRAME / SETTLING_DIVISOR),
        fragments: Math.ceil(WIDEN_FRAGMENTS_PER_FRAME / SETTLING_DIVISOR),
      }
    : {
        characters: WIDEN_CHARACTERS_PER_FRAME,
        fragments: WIDEN_FRAGMENTS_PER_FRAME,
      };

type EstimatedFragment = Pick<
  ReasoningFragment,
  "text" | "code" | "first" | "last" | "hidden"
>;

export const estimateFragmentHeight = (
  fragment: EstimatedFragment,
  geometry: FragmentGeometry,
): number => {
  if (fragment.hidden) return 0;
  const code = fragment.code;
  const columns = Math.max(
    12,
    Math.floor(
      code
        ? (geometry.width - 32) / (geometry.codeFontPixels * 0.58)
        : geometry.width / (geometry.fontPixels * 0.48),
    ),
  );
  const lines = fragment.text
    .split("\n")
    .reduce(
      (sum, line) => sum + Math.max(1, Math.ceil(line.length / columns)),
      0,
    );
  return code
    ? lines * geometry.codeLineHeight +
        (fragment.first ? 40 : 0) +
        (fragment.last ? geometry.codeFooter : 0)
    : Math.max(1, lines - 2) * geometry.lineHeight +
        (fragment.first ? 16 : 0);
};

const estimated = new WeakMap<
  FragmentGeometry,
  WeakMap<EstimatedFragment, number>
>();

export const estimateFragmentHeights = (
  fragments: readonly EstimatedFragment[],
  geometry: FragmentGeometry,
): number[] => {
  let known = estimated.get(geometry);
  if (known === undefined) {
    known = new WeakMap();
    estimated.set(geometry, known);
  }
  return fragments.map((fragment) => {
    let height = known.get(fragment);
    if (height === undefined) {
      height = estimateFragmentHeight(fragment, geometry);
      known.set(fragment, height);
    }
    return height;
  });
};

export const initialRows = (
  estimates: readonly number[],
  budgetPx: number,
  mustInclude: number,
  sizes?: readonly number[],
  budgetChars: number = INITIAL_CHARACTERS,
): number => {
  if (estimates.length === 0) return 0;
  let rows = Math.min(estimates.length, Math.max(0, mustInclude));
  let covered = 0;
  let spent = 0;
  while (
    rows < estimates.length
    && (rows <= mustInclude || (covered < budgetPx && spent < budgetChars))
  ) {
    covered += estimates[rows];
    spent += sizes?.[rows] ?? 0;
    rows += 1;
  }
  return Math.max(1, rows);
};

export const widenBudget = (
  sizes: readonly number[],
  mounted: number,
  budget: number,
  maxRows: number = WIDEN_FRAGMENTS_PER_FRAME,
): { rows: number; spent: number } => {
  let rows = mounted;
  let spent = 0;
  while (
    rows < sizes.length
    && (spent === 0 || (spent + sizes[rows] <= budget && rows - mounted < maxRows))
  ) {
    spent += sizes[rows];
    rows += 1;
  }
  return { rows, spent };
};

export const isCovered = (mounted: number, total: number): boolean =>
  mounted >= total;

export const CHUNK_CHARACTERS = 8_192;
export const CHUNK_MAX_GROUPS = 12;

export type FragmentGroup = {
  readonly key: string;
  readonly code: boolean;
  readonly first: number;
  readonly end: number;
  readonly chars: number;
};

export type GridChunk = {
  readonly key: string;
  readonly groupStart: number;
  readonly groupEnd: number;
  readonly first: number;
  readonly end: number;
};

export type ChunkGrid = {
  readonly groups: readonly FragmentGroup[];
  readonly chunks: readonly GridChunk[];
  readonly groupOf: Int32Array;
  readonly chunkOf: Int32Array;
  readonly heights: Float64Array;
};

type GridFragment = Pick<ReasoningFragment, "key" | "text" | "code" | "document" | "start">;

const groupKeyOf = (fragment: GridFragment): string =>
  fragment.code ? `${fragment.document}:${fragment.start}` : fragment.key;

type PlacedFragment = Pick<ReasoningFragment, "text" | "code" | "document" | "start" | "end">;

const rowAtSamePlace = (
  fragments: readonly PlacedFragment[],
  row: PlacedFragment,
): number => {
  const line = row.code?.lines[0]?.line ?? 0;
  const column = row.code?.lines[0]?.column ?? 0;
  const same = fragments.findIndex(
    (fragment) =>
      fragment.document === row.document
      && fragment.start === row.start
      && (fragment.code?.lines[0]?.line ?? 0) === line
      && (fragment.code?.lines[0]?.column ?? 0) === column,
  );
  if (same >= 0) return same;
  const at = row.code
    ? row.start
    : row.start + row.text.length - row.text.trimStart().length;
  return fragments.findIndex(
    (fragment) =>
      fragment.document === row.document
      && fragment.start <= at
      && at < fragment.end,
  );
};

const occurrencesIn = (row: PlacedFragment, needle: string): number => {
  if (!needle) return 0;
  let found = 0;
  for (const { text } of row.code?.lines ?? []) {
    for (let at = text.indexOf(needle); at >= 0; at = text.indexOf(needle, at + 1))
      found += 1;
  }
  return found;
};

const fenceRows = (
  fragments: readonly PlacedFragment[],
  fence: PlacedFragment,
): number[] =>
  fragments.flatMap((fragment, index) =>
    fragment.code
    && fragment.document === fence.document
    && fragment.start === fence.start
      ? [index]
      : [],
  );

export const anchorAtSamePlace = (
  previous: readonly PlacedFragment[],
  fragments: readonly PlacedFragment[],
  row: PlacedFragment,
  text: string,
  occurrence: number,
): { index: number; occurrence: number } | null => {
  const index = rowAtSamePlace(fragments, row);
  if (index < 0) return null;
  if (Boolean(row.code) === Boolean(fragments[index].code))
    return { index, occurrence };
  if (row.code) {
    let before = 0;
    for (const at of fenceRows(previous, row)) {
      if (previous[at] === row) break;
      before += occurrencesIn(previous[at], text);
    }
    return { index, occurrence: occurrence + before };
  }
  let left = occurrence;
  for (const at of fenceRows(fragments, fragments[index])) {
    const here = occurrencesIn(fragments[at], text);
    if (left < here) return { index: at, occurrence: left };
    left -= here;
  }
  return { index, occurrence };
};

const startsLine = (fragment: GridFragment): boolean =>
  (fragment.code?.lines[0]?.column ?? 0) === 0;

export const buildChunkGrid = (
  fragments: readonly GridFragment[],
  estimates: readonly number[],
  maxChars: number = CHUNK_CHARACTERS,
  maxGroups: number = CHUNK_MAX_GROUPS,
): ChunkGrid => {
  const groups: FragmentGroup[] = [];
  const groupOf = new Int32Array(fragments.length);
  const heights = new Float64Array(fragments.length + 1);
  for (let i = 0; i < fragments.length; i += 1) {
    const fragment = fragments[i];
    heights[i + 1] = heights[i] + (estimates[i] ?? 0);
    const key = groupKeyOf(fragment);
    const previous = groups[groups.length - 1];
    const sameFence =
      previous !== undefined
      && previous.end === i
      && groupKeyOf(fragments[previous.first]) === key;
    if (sameFence && (previous.chars < maxChars || !startsLine(fragment))) {
      groups[groups.length - 1] = {
        ...previous,
        end: i + 1,
        chars: previous.chars + fragment.text.length,
      };
    } else {
      groups.push({
        key: sameFence ? fragment.key : key,
        code: Boolean(fragment.code),
        first: i,
        end: i + 1,
        chars: fragment.text.length,
      });
    }
    groupOf[i] = groups.length - 1;
  }
  const chunks: GridChunk[] = [];
  const chunkOf = new Int32Array(groups.length);
  let start = 0;
  let chars = 0;
  for (let g = 0; g < groups.length; g += 1) {
    if (g > start && (g - start >= maxGroups || chars >= maxChars)) {
      chunks.push({
        key: groups[start].key,
        groupStart: start,
        groupEnd: g,
        first: groups[start].first,
        end: groups[g - 1].end,
      });
      start = g;
      chars = 0;
    }
    chars += groups[g].chars;
    chunkOf[g] = chunks.length;
  }
  if (start < groups.length) {
    chunks.push({
      key: groups[start].key,
      groupStart: start,
      groupEnd: groups.length,
      first: groups[start].first,
      end: groups[groups.length - 1].end,
    });
  }
  return { groups, chunks, groupOf, chunkOf, heights };
};

export const chunkEndAt = (grid: ChunkGrid, rows: number): number => {
  if (rows <= 0) return 0;
  if (rows >= grid.groupOf.length) return rows;
  return grid.chunks[grid.chunkOf[grid.groupOf[rows - 1]]].end;
};

export const chunkStartAt = (grid: ChunkGrid, row: number): number => {
  if (row <= 0) return 0;
  if (row >= grid.groupOf.length) return row;
  return grid.chunks[grid.chunkOf[grid.groupOf[row]]].first;
};

export type ChunkPiece = {
  readonly key: string;
  readonly first: number;
  readonly end: number;
  readonly whole: boolean;
  readonly last: boolean;
  readonly groups: readonly { index: number; first: number; end: number }[];
};

export const chunkPieces = (
  grid: ChunkGrid,
  fragmentKey: (index: number) => string,
  from: number,
  to: number,
): ChunkPiece[] => {
  const pieces: ChunkPiece[] = [];
  for (let at = from; at < to; ) {
    const c = grid.chunkOf[grid.groupOf[at]];
    const chunk = grid.chunks[c];
    const end = Math.min(to, chunk.end);
    const groups: { index: number; first: number; end: number }[] = [];
    for (let g = grid.groupOf[at]; g < chunk.groupEnd; g += 1) {
      const group = grid.groups[g];
      if (group.first >= end) break;
      groups.push({
        index: g,
        first: Math.max(group.first, at),
        end: Math.min(group.end, end),
      });
    }
    pieces.push({
      key: at === chunk.first ? chunk.key : `${chunk.key}>${fragmentKey(at)}`,
      first: at,
      end,
      whole: at === chunk.first && end === chunk.end,
      last: c === grid.chunks.length - 1,
      groups,
    });
    at = end;
  }
  return pieces;
};

export type ChunkSlice = {
  readonly key: string;
  readonly code: boolean;
  readonly first: number;
  readonly end: number;
  readonly streaming: boolean;
};

type ChunkRender = {
  readonly slices: readonly ChunkSlice[];
  readonly fragments: readonly unknown[];
  readonly closed: boolean;
  readonly [prop: string]: unknown;
};

const sameSlice = (a: ChunkSlice, b: ChunkSlice): boolean =>
  a.key === b.key
  && a.code === b.code
  && a.first === b.first
  && a.end === b.end
  && a.streaming === b.streaming;

export const closedChunkUnchanged = (
  previous: ChunkRender,
  next: ChunkRender,
): boolean => {
  const props = Object.keys(next);
  if (
    !previous.closed
    || !next.closed
    || Object.keys(previous).length !== props.length
  )
    return false;
  for (const prop of props) {
    if (prop === "slices" || prop === "fragments") continue;
    if (!Object.is(previous[prop], next[prop])) return false;
  }
  const slices = next.slices;
  if (
    slices.length === 0
    || previous.slices.length !== slices.length
    || !slices.every((slice, at) => sameSlice(previous.slices[at], slice))
  )
    return false;
  for (let at = slices[0].first; at <= slices[slices.length - 1].end; at += 1) {
    if (previous.fragments[at] !== next.fragments[at]) return false;
  }
  return true;
};

export type Island = {
  readonly id: number;
  readonly start: number;
  readonly end: number;
};

export const NO_ISLANDS: readonly Island[] = [];

const coversRows = (
  islands: readonly Island[],
  start: number,
  end: number,
): boolean => {
  let reach = start;
  for (const island of islands) {
    if (island.start > reach) break;
    if (island.end > reach) reach = island.end;
    if (reach >= end) return true;
  }
  return reach >= end;
};

export const addIsland = (
  islands: readonly Island[],
  add: { readonly start: number; readonly end: number },
  prefix: number,
): readonly Island[] => {
  const start = Math.max(add.start, prefix);
  if (add.end <= start || coversRows(islands, start, add.end)) return islands;
  const at = islands.findIndex(
    (island) => island.start <= add.end && island.end >= start,
  );
  const next = islands.slice();
  if (at < 0) next.push({ id: start, start, end: add.end });
  else
    next[at] = {
      id: islands[at].id,
      start: Math.min(islands[at].start, start),
      end: Math.max(islands[at].end, add.end),
    };
  return next.sort((a, b) => a.start - b.start);
};

export const mountedRanges = (
  prefix: number,
  islands: readonly { readonly start: number; readonly end: number }[],
  total: number,
): [number, number][] => {
  const ranges: [number, number][] = [];
  const shown = Math.min(prefix, total);
  if (shown > 0) ranges.push([0, shown]);
  for (const island of islands) {
    const from = Math.max(island.start, shown);
    const to = Math.min(island.end, total);
    if (to <= from) continue;
    const last = ranges[ranges.length - 1];
    if (last !== undefined && from <= last[1]) last[1] = Math.max(last[1], to);
    else ranges.push([from, to]);
  }
  return ranges;
};

export type MountItem = {
  readonly from: number;
  readonly to: number;
  readonly reserve?: string;
};

export const TAIL_RESERVE = "tail";

export const mountPlan = (
  prefix: number,
  islands: readonly Island[],
  total: number,
): MountItem[] => {
  const items: MountItem[] = [];
  const rowOf = (at: number) => Math.min(islands[at].start, total);
  let anchor = 0;
  let covered = 0;
  for (const [from, to] of mountedRanges(prefix, islands, total)) {
    while (anchor < islands.length && rowOf(anchor) <= from) {
      const opens = rowOf(anchor) === from;
      items.push({
        from: covered,
        to: opens ? from : covered,
        reserve: String(islands[anchor].id),
      });
      if (opens) covered = from;
      anchor += 1;
    }
    let cursor = from;
    while (anchor < islands.length && rowOf(anchor) < to) {
      const row = rowOf(anchor);
      if (row > cursor) items.push({ from: cursor, to: row });
      items.push({ from: row, to: row, reserve: String(islands[anchor].id) });
      cursor = row;
      anchor += 1;
    }
    if (to > cursor) items.push({ from: cursor, to });
    covered = to;
  }
  for (; anchor < islands.length; anchor += 1)
    items.push({
      from: covered,
      to: covered,
      reserve: String(islands[anchor].id),
    });
  items.push({
    from: covered,
    to: Math.max(covered, total),
    reserve: TAIL_RESERVE,
  });
  return items;
};

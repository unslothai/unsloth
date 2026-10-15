// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Words are whitespace-separated tokens: punctuation stays on its word.

/** The diff is quadratic: past this many words on either side it is not computed. */
export const EDIT_DIFF_MAX_WORDS = 400;

type DiffOpKind = "equal" | "delete" | "insert";

interface DiffOp {
  kind: DiffOpKind;
  word: string;
}

export interface EditChange {
  kind: "replace" | "delete" | "insert";
  old: string[];
  new: string[];
  /** The transcript word right after the change; null at the end. */
  before: string | null;
  index: number;
}

interface DiffSegment {
  kind: DiffOpKind;
  text: string;
  /** Screen-reader prefix, so colour and decoration never carry the meaning alone. */
  srLabel: "inserted " | "deleted " | null;
}

export function tokenizeWords(text: string): string[] {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/) : [];
}

/** LCS word diff; null past the word cap. In a changed run removed words precede added ones. */
export function diffWords(a: string, b: string): DiffOp[] | null {
  const left = tokenizeWords(a);
  const right = tokenizeWords(b);
  if (left.length > EDIT_DIFF_MAX_WORDS || right.length > EDIT_DIFF_MAX_WORDS)
    return null;
  const n = left.length;
  const m = right.length;
  const width = m + 1;
  const lcs = new Uint16Array((n + 1) * width);
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      lcs[i * width + j] =
        left[i] === right[j]
          ? lcs[(i + 1) * width + j + 1] + 1
          : Math.max(lcs[(i + 1) * width + j], lcs[i * width + j + 1]);
    }
  }
  const ops: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (left[i] === right[j]) {
      ops.push({ kind: "equal", word: left[i] });
      i += 1;
      j += 1;
    } else if (lcs[(i + 1) * width + j] >= lcs[i * width + j + 1]) {
      ops.push({ kind: "delete", word: left[i] });
      i += 1;
    } else {
      ops.push({ kind: "insert", word: right[j] });
      j += 1;
    }
  }
  for (; i < n; i += 1) ops.push({ kind: "delete", word: left[i] });
  for (; j < m; j += 1) ops.push({ kind: "insert", word: right[j] });
  return ops;
}

export function groupChanges(ops: readonly DiffOp[]): EditChange[] {
  const changes: EditChange[] = [];
  let originalIndex = 0;
  let k = 0;
  while (k < ops.length) {
    if (ops[k].kind === "equal") {
      originalIndex += 1;
      k += 1;
      continue;
    }
    const start = originalIndex;
    const old: string[] = [];
    const added: string[] = [];
    while (k < ops.length && ops[k].kind !== "equal") {
      if (ops[k].kind === "delete") {
        old.push(ops[k].word);
        originalIndex += 1;
      } else {
        added.push(ops[k].word);
      }
      k += 1;
    }
    changes.push({
      kind:
        old.length && added.length
          ? "replace"
          : old.length
            ? "delete"
            : "insert",
      old,
      new: added,
      before: k < ops.length ? ops[k].word : null,
      index: start,
    });
  }
  return changes;
}

export function changesBetween(a: string, b: string): EditChange[] | null {
  const ops = diffWords(a, b);
  return ops ? groupChanges(ops) : null;
}

export function countChanges(a: string, b: string): number | null {
  return changesBetween(a, b)?.length ?? null;
}

export function diffSegments(a: string, b: string): DiffSegment[] | null {
  const ops = diffWords(a, b);
  if (!ops) return null;
  const segments: DiffSegment[] = [];
  const push = (kind: DiffOpKind, words: string[]) => {
    if (words.length === 0) return;
    segments.push({
      kind,
      text: words.join(" "),
      srLabel:
        kind === "insert" ? "inserted " : kind === "delete" ? "deleted " : null,
    });
  };
  let k = 0;
  while (k < ops.length) {
    const equal: string[] = [];
    while (k < ops.length && ops[k].kind === "equal") {
      equal.push(ops[k].word);
      k += 1;
    }
    push("equal", equal);
    const deleted: string[] = [];
    const inserted: string[] = [];
    while (k < ops.length && ops[k].kind !== "equal") {
      (ops[k].kind === "delete" ? deleted : inserted).push(ops[k].word);
      k += 1;
    }
    push("delete", deleted);
    push("insert", inserted);
  }
  return segments;
}

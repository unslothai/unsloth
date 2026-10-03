// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The word diff between a recording's transcript and the user's edited copy. One diff drives
// both the highlighting in ② and the changes sent to the model, so what the user sees is what
// gets edited. Words are whitespace-separated tokens: punctuation stays on its word. Free of app
// imports so the node test runner can load it directly.

/** Past this many words on either side the diff is not computed (it is quadratic). */
export const EDIT_DIFF_MAX_WORDS = 400;

export type DiffOpKind = "equal" | "delete" | "insert";

export interface DiffOp {
  kind: DiffOpKind;
  word: string;
}

export type EditChangeKind = "replace" | "delete" | "insert";

/** One contiguous change: the words it removes from the transcript and the words it adds. */
export interface EditChange {
  kind: EditChangeKind;
  old: string[];
  new: string[];
  /** The transcript word right after the change, or null when the change ends the sentence. */
  before: string | null;
  /** Where the change starts, as an index into the transcript's words. */
  index: number;
}

export interface DiffSegment {
  kind: DiffOpKind;
  text: string;
  /** Screen-reader prefix, so colour and decoration never carry the meaning alone. */
  srLabel: "inserted " | "deleted " | null;
}

export function tokenizeWords(text: string): string[] {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/) : [];
}

/** The word-level diff of two texts (longest common subsequence), or null past the word cap.
 *  Within a changed run the removed words come before the added ones. */
export function diffWords(a: string, b: string): DiffOp[] | null {
  const left = tokenizeWords(a);
  const right = tokenizeWords(b);
  if (left.length > EDIT_DIFF_MAX_WORDS || right.length > EDIT_DIFF_MAX_WORDS)
    return null;
  const n = left.length;
  const m = right.length;
  const width = m + 1;
  // lcs[i * width + j] is the LCS length of left[i..] and right[j..].
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

/** Each run of changed words as one change: removed and added words together are a replace. */
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

/** The changes between two texts, or null past the word cap. */
export function changesBetween(a: string, b: string): EditChange[] | null {
  const ops = diffWords(a, b);
  return ops ? groupChanges(ops) : null;
}

/** How many separate changes the edit makes, or null past the word cap. */
export function countChanges(a: string, b: string): number | null {
  return changesBetween(a, b)?.length ?? null;
}

/** The diff as runs to render: equal words, then each change's deleted and inserted words.
 *  Null past the word cap. */
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

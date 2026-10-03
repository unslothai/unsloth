// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RagDocument, isLinkedFolderManaged } from "../types/rag";
import { toSortTime } from "./document-format";

/** Search, ordering and collapse rules for the project sources list. Pure so the
 * node:test suite can cover them without rendering the panel. */

export type SourceSortMode = "uploaded" | "name" | "size";

export const SOURCE_SORT_LABELS: Record<SourceSortMode, string> = {
  uploaded: "Recently added",
  name: "Name",
  size: "Size",
};

/** Documents past this count are hidden behind "Show all" so a project with
 * dozens of sources does not push the rest of the page off-screen. */
export const SOURCES_COLLAPSED_LIMIT = 25;

/** Case-insensitive substring match on the filename. An all-whitespace query
 * matches everything, so the field behaves as empty until something is typed. */
export function filterSources<T extends Pick<RagDocument, "filename">>(
  documents: readonly T[],
  query: string,
): T[] {
  const trimmed = query.trim().toLowerCase();
  if (trimmed === "") return documents.slice();
  return documents.filter((doc) =>
    doc.filename.toLowerCase().includes(trimmed),
  );
}

type SortableDocument = Pick<RagDocument, "filename" | "createdAt"> & {
  id?: string;
  status?: RagDocument["status"];
  sizeBytes?: number | null;
};

/** An upload that has not finished indexing: either an optimistic chip with no
 * server id yet, or a real row whose ingestion job is still running. */
function isInFlight(doc: SortableDocument): boolean {
  if (doc.id?.startsWith("pending_")) return true;
  return doc.status === "pending" || doc.status === "running";
}

/** Returns a new array; the input order is left alone.
 *
 * Unknown values sort last in every mode rather than clustering at the top: a
 * document still indexing has no size yet, and `createdAt` is optional on the
 * type. `toSortTime` maps missing dates to 0, which is already last under a
 * descending sort. Ties fall back to the filename so the order is stable across
 * renders instead of depending on the fetch order. */
export function sortSources<T extends SortableDocument>(
  documents: readonly T[],
  mode: SourceSortMode,
): T[] {
  const byName = (a: T, b: T) => a.filename.localeCompare(b.filename);
  return documents.slice().sort((a, b) => {
    // An upload in flight pins to the top whatever the mode. It has no
    // createdAt or sizeBytes yet, so every mode would otherwise sort it last
    // and the collapse limit would hide the row the user just created, making
    // the upload look like it never started.
    const pending = Number(isInFlight(b)) - Number(isInFlight(a));
    if (pending !== 0) return pending;
    if (mode === "name") return byName(a, b);
    if (mode === "size") {
      const left = a.sizeBytes ?? -1;
      const right = b.sizeBytes ?? -1;
      return right - left || byName(a, b);
    }
    return toSortTime(b.createdAt) - toSortTime(a.createdAt) || byName(a, b);
  });
}

/** The slice to render. Searching always spans every source — a match hidden
 * behind the collapse limit would make the search look broken. */
export function visibleSources<T>(
  documents: readonly T[],
  options: { expanded: boolean; searching: boolean },
): T[] {
  if (options.expanded || options.searching) return documents.slice();
  return documents.slice(0, SOURCES_COLLAPSED_LIMIT);
}

/** Whether a document may be removed as part of a bulk selection.
 *
 * Three kinds are excluded. An optimistic chip has no server id yet. A row that
 * is still indexing has a live ingestion worker writing to it, so deleting it
 * would race the job rather than remove a finished source. Linked-folder rows
 * are owned by folder sync, and the DELETE route answers 409 for them.
 *
 * DocumentStatusChip hides its checkbox on exactly these, so "Select all" must
 * use the same predicate or it would select rows the user cannot select
 * individually. */
export function isBulkRemovable(
  doc: Pick<RagDocument, "id" | "status" | "managed" | "linkedFolderId">,
): boolean {
  if (doc.id.startsWith("pending_")) return false;
  if (doc.status === "pending" || doc.status === "running") return false;
  return !isLinkedFolderManaged(doc as RagDocument);
}

/** Whether a "Show all" toggle is warranted for the current view. */
export function hasHiddenSources(
  count: number,
  options: { expanded: boolean; searching: boolean },
): boolean {
  if (options.expanded || options.searching) return false;
  return count > SOURCES_COLLAPSED_LIMIT;
}

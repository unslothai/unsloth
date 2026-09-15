// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LibrarySort =
  | "default"
  | "updated"
  | "created"
  | "oldest"
  | "alphabetical";
export interface LibraryFilters {
  query: string;
  type: string;
  project: string;
  sort: LibrarySort;
}
export const DEFAULT_LIBRARY_FILTERS: LibraryFilters = {
  query: "",
  type: "all",
  project: "all",
  sort: "updated",
};
export interface LibraryItem {
  id: string;
  title: string;
  createdAt: number;
  updatedAt?: number;
  type?: string;
  projectId?: string | null;
}

export interface LibraryProjectLabels {
  noProject: string;
  unavailableProject: string;
}

function searchable(value: string): string {
  return value.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();
}

export function filterLibraryItems<T extends LibraryItem>(
  items: readonly T[],
  filters: LibraryFilters,
  projects: ReadonlyMap<string, string> = new Map(),
  labels?: LibraryProjectLabels,
  locale?: string,
): T[] {
  const terms = searchable(filters.query).trim().split(/\s+/).filter(Boolean);
  const timestamp = (item: T) => {
    const value =
      filters.sort === "updated"
        ? (item.updatedAt ?? item.createdAt)
        : item.createdAt;
    return Number.isFinite(value) ? value : 0;
  };
  const matching = items.filter((item) => {
    if (filters.type !== "all" && item.type !== filters.type) return false;
    if (filters.project === "none" && item.projectId) return false;
    if (
      filters.project.startsWith("project:") &&
      item.projectId !== filters.project.slice(8)
    )
      return false;
    const project = item.projectId
      ? (projects.get(item.projectId) ?? labels?.unavailableProject ?? "")
      : (labels?.noProject ?? "");
    const haystack = searchable(`${item.title} ${project}`);
    return terms.every((term) => haystack.includes(term));
  });
  if (filters.sort === "default") return matching;
  const compare = new Intl.Collator(locale).compare;
  return matching.sort((a, b) => {
    const byTitle = compare(a.title, b.title);
    if (filters.sort === "alphabetical") return byTitle || compare(a.id, b.id);
    const byTime =
      (timestamp(b) - timestamp(a)) * (filters.sort === "oldest" ? -1 : 1);
    return byTime || byTitle || compare(a.id, b.id);
  });
}

export function groupLibraryItems<T extends LibraryItem>(
  items: readonly T[],
  projects: ReadonlyMap<string, string>,
  labels: LibraryProjectLabels,
) {
  const groups = new Map<string, { id: string; name: string; items: T[] }>();
  for (const item of items) {
    const id = item.projectId ?? "";
    let group = groups.get(id);
    if (!group) {
      group = {
        id,
        name: id
          ? (projects.get(id) ?? labels.unavailableProject)
          : labels.noProject,
        items: [],
      };
      groups.set(id, group);
    }
    group.items.push(item);
  }
  return [...groups.values()];
}

export function formatLibraryDate(ms: number, locale?: string): string {
  const date = new Date(ms);
  return Number.isFinite(date.getTime())
    ? date.toLocaleString(locale, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : "";
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chats tab data model. Reads chat history, so chats never enter the Library item list.

export const CHATS_SECTIONS = ["all", "chats", "projects", "sections", "archived"] as const;
export type ChatsSection = (typeof CHATS_SECTIONS)[number];

export interface ChatEntry {
  id: string;
  type: "single" | "compare";
  threadIds?: string[];
  title: string;
  createdAt: number;
  updatedAt: number;
  isFork?: boolean;
  projectId?: string | null;
  /** Last rename, move or archive, when later than `updatedAt`. */
  modifiedAt?: number;
}

export interface ProjectEntry {
  id: string;
  name: string;
  instructions?: string;
  createdAt: number;
  updatedAt: number;
}

/** The one date column's choices: created, last active, last modified. */
export const DATE_FIELDS = ["created", "updated", "modified"] as const;
export type DateField = (typeof DATE_FIELDS)[number];

export type ChatSortKey = "name" | DateField;
export type ChatSort = { key: ChatSortKey; desc: boolean };
export type ProjectSortKey = "name" | DateField | "chats";
export type ProjectSort = { key: ProjectSortKey; desc: boolean };
/** "updated" is a section's latest chat; "chats" is how many it holds. */
export type SectionSortKey = "name" | "updated" | "chats";
export type SectionSort = { key: SectionSortKey; desc: boolean };
export type ChatGroupBy = "none" | "project" | "section" | "date";

export type ChatFlag = "favorite" | "pinned" | "forks" | "compare";
export const CHAT_FLAGS: ChatFlag[] = ["favorite", "pinned", "forks", "compare"];

/** Filter value for chats outside any project. */
export const NO_PROJECT = "none";
/** Filter value for chats outside any section. */
export const NO_SECTION = "none";

export interface ChatFilters {
  flags: Set<ChatFlag>;
  projects: Set<string>;
  sections: Set<string>;
  models: Set<string>;
}

export const EMPTY_CHAT_FILTERS: ChatFilters = {
  flags: new Set(),
  projects: new Set(),
  sections: new Set(),
  models: new Set(),
};

export function chatFiltersActive(filters: ChatFilters): boolean {
  return (
    filters.flags.size > 0 ||
    filters.projects.size > 0 ||
    filters.sections.size > 0 ||
    filters.models.size > 0
  );
}

function searchable(value: string): string {
  return value.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();
}

export function searchTerms(query: string): string[] {
  return searchable(query).trim().split(/\s+/).filter(Boolean);
}

export function matchesTerms(terms: string[], ...fields: (string | undefined | null)[]): boolean {
  if (terms.length === 0) return true;
  const haystack = searchable(fields.filter(Boolean).join(" "));
  return terms.every((term) => haystack.includes(term));
}

export interface ChatContext {
  pinned: ReadonlySet<string>;
  favorites?: ReadonlySet<string>;
  projectNames: ReadonlyMap<string, string>;
  /** Model ids per chat row, from its pane threads. */
  models: ReadonlyMap<string, string[]>;
  /** Chat row id -> id of the sidebar section it is filed in. */
  sectionOf?: ReadonlyMap<string, string>;
  sectionNames?: ReadonlyMap<string, string>;
}

/** Flags must all match; projects, sections and models each match any ticked value. */
export function filterChats<T extends ChatEntry>(
  chats: readonly T[],
  query: string,
  filters: ChatFilters,
  context: ChatContext,
): T[] {
  const terms = searchTerms(query);
  return chats.filter((chat) => {
    if (filters.flags.has("favorite") && !context.favorites?.has(chat.id)) return false;
    if (filters.flags.has("pinned") && !context.pinned.has(chat.id)) return false;
    if (filters.flags.has("forks") && !chat.isFork) return false;
    if (filters.flags.has("compare") && chat.type !== "compare") return false;
    if (filters.projects.size > 0 && !filters.projects.has(chat.projectId || NO_PROJECT)) {
      return false;
    }
    const sectionId = context.sectionOf?.get(chat.id);
    if (filters.sections.size > 0 && !filters.sections.has(sectionId || NO_SECTION)) {
      return false;
    }
    const models = context.models.get(chat.id) ?? [];
    if (filters.models.size > 0 && !models.some((model) => filters.models.has(model))) {
      return false;
    }
    const project = chat.projectId ? context.projectNames.get(chat.projectId) : undefined;
    const section = sectionId ? context.sectionNames?.get(sectionId) : undefined;
    return matchesTerms(terms, chat.title, project, section, ...models);
  });
}

export function chatTime(chat: ChatEntry, field: DateField): number {
  if (field === "created") return chat.createdAt;
  if (field === "modified") return Math.max(chat.updatedAt, chat.modifiedAt ?? 0);
  return chat.updatedAt;
}

/** "updated" is the latest chat in it; "modified" the project's own last edit. */
export function projectTime(
  project: ProjectEntry,
  stats: ProjectStats | undefined,
  field: DateField,
): number {
  if (field === "created") return project.createdAt;
  if (field === "modified") return project.updatedAt;
  return stats?.lastActive ?? project.updatedAt;
}

export function compareChats(
  sort: ChatSort,
  locale?: string,
): (a: ChatEntry, b: ChatEntry) => number {
  const collate = new Intl.Collator(locale, { numeric: true, sensitivity: "base" }).compare;
  const ascending = (a: ChatEntry, b: ChatEntry) => {
    if (sort.key === "name") return collate(a.title, b.title);
    return (chatTime(a, sort.key) || 0) - (chatTime(b, sort.key) || 0);
  };
  return (a, b) => {
    const primary = sort.desc ? ascending(b, a) : ascending(a, b);
    return primary || b.updatedAt - a.updatedAt || a.id.localeCompare(b.id);
  };
}

/** Pinned chats sort first whatever the order, as in the sidebar. */
export function sortChats<T extends ChatEntry>(
  chats: readonly T[],
  sort: ChatSort,
  pinned: ReadonlySet<string>,
  pinnedFirst: boolean,
  locale?: string,
): T[] {
  const compare = compareChats(sort, locale);
  return [...chats].sort((a, b) => {
    if (pinnedFirst) {
      const byPin = Number(pinned.has(b.id)) - Number(pinned.has(a.id));
      if (byPin) return byPin;
    }
    return compare(a, b);
  });
}

export interface ProjectStats {
  chats: number;
  archived: number;
  lastActive: number;
}

export function projectStats(
  projects: readonly ProjectEntry[],
  chats: readonly ChatEntry[],
  archived: readonly ChatEntry[],
): Map<string, ProjectStats> {
  const stats = new Map<string, ProjectStats>(
    projects.map((project) => [project.id, { chats: 0, archived: 0, lastActive: project.updatedAt }]),
  );
  for (const chat of chats) {
    const entry = chat.projectId ? stats.get(chat.projectId) : undefined;
    if (!entry) continue;
    entry.chats += 1;
    entry.lastActive = Math.max(entry.lastActive, chat.updatedAt);
  }
  for (const chat of archived) {
    const entry = chat.projectId ? stats.get(chat.projectId) : undefined;
    if (entry) entry.archived += 1;
  }
  return stats;
}

export function sortProjects<T extends ProjectEntry>(
  projects: readonly T[],
  sort: ProjectSort,
  stats: ReadonlyMap<string, ProjectStats>,
  pinned: ReadonlySet<string>,
  locale?: string,
): T[] {
  const collate = new Intl.Collator(locale, { numeric: true, sensitivity: "base" }).compare;
  const value = (project: T) => {
    const entry = stats.get(project.id);
    if (sort.key === "chats") return entry?.chats ?? 0;
    if (sort.key === "name") return 0;
    return projectTime(project, entry, sort.key);
  };
  return [...projects].sort((a, b) => {
    const byPin = Number(pinned.has(b.id)) - Number(pinned.has(a.id));
    if (byPin) return byPin;
    const ascending = sort.key === "name" ? collate(a.name, b.name) : value(a) - value(b);
    return (sort.desc ? -ascending : ascending) || collate(a.name, b.name);
  });
}

export interface SectionStats {
  chats: number;
  projects: number;
  /** Latest chat time; 0 when empty. */
  lastActive: number;
}

/** Per section: live chats and still-existing projects filed in it. */
export function sectionStats(
  sections: readonly { id: string }[],
  chats: readonly ChatEntry[],
  sectionOf: ReadonlyMap<string, string>,
  projectSectionOf: ReadonlyMap<string, string>,
  projectIds: ReadonlySet<string>,
): Map<string, SectionStats> {
  const out = new Map<string, SectionStats>(
    sections.map((section) => [section.id, { chats: 0, projects: 0, lastActive: 0 }]),
  );
  for (const chat of chats) {
    const stats = out.get(sectionOf.get(chat.id) ?? "");
    if (!stats) continue;
    stats.chats += 1;
    stats.lastActive = Math.max(stats.lastActive, chat.updatedAt);
  }
  for (const [projectId, sectionId] of projectSectionOf) {
    const stats = out.get(sectionId);
    if (stats && projectIds.has(projectId)) stats.projects += 1;
  }
  return out;
}

export function sortSections<T extends { id: string; name: string }>(
  sections: readonly T[],
  sort: SectionSort,
  stats: ReadonlyMap<string, SectionStats>,
  locale?: string,
): T[] {
  const collate = new Intl.Collator(locale, { numeric: true, sensitivity: "base" }).compare;
  const value = (section: T) => {
    const entry = stats.get(section.id);
    return sort.key === "chats" ? (entry?.chats ?? 0) : (entry?.lastActive ?? 0);
  };
  return [...sections].sort((a, b) => {
    const ascending = sort.key === "name" ? collate(a.name, b.name) : value(a) - value(b);
    return (sort.desc ? -ascending : ascending) || collate(a.name, b.name);
  });
}

export type DateBucket =
  | { kind: "today" }
  | { kind: "yesterday" }
  | { kind: "week" }
  | { kind: "month" }
  | { kind: "older"; year: number; month: number };

const DAY_MS = 86_400_000;

function startOfDay(ms: number): number {
  const date = new Date(ms);
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

export function dateBucket(ts: number, now: number = Date.now()): DateBucket {
  // Rounded: DST days are 23 or 25 hours long.
  const days = Math.round((startOfDay(now) - startOfDay(ts)) / DAY_MS);
  if (days <= 0) return { kind: "today" };
  if (days === 1) return { kind: "yesterday" };
  if (days < 7) return { kind: "week" };
  if (days < 30) return { kind: "month" };
  const date = new Date(ts);
  return { kind: "older", year: date.getFullYear(), month: date.getMonth() };
}

export function dateBucketKey(bucket: DateBucket): string {
  return bucket.kind === "older" ? `older:${bucket.year}-${bucket.month}` : bucket.kind;
}

export interface ChatGroup<T> {
  key: string;
  items: T[];
  /** Set for project groups; null is the "no project" group. */
  projectId?: string | null;
  /** Set for section groups; null is the "no section" group. */
  sectionId?: string | null;
  bucket?: DateBucket;
  /** The leading group of pinned chats. */
  pinned?: boolean;
}

/** Keeps input order within groups. Date groups run newest first (or `oldestFirst`) regardless
 *  of sort; other groups follow their first chat, "none" last. `pinned` adds a leading group. */
export function groupChats<T extends ChatEntry>(
  chats: readonly T[],
  by: ChatGroupBy,
  options: {
    time?: DateField;
    oldestFirst?: boolean;
    now?: number;
    pinned?: ReadonlySet<string>;
    sectionOf?: ReadonlyMap<string, string>;
  } = {},
): ChatGroup<T>[] {
  if (by === "none") return chats.length ? [{ key: "all", items: [...chats] }] : [];
  const { time = "updated", oldestFirst = false, now = Date.now(), pinned, sectionOf } = options;
  const pinnedItems = pinned ? chats.filter((chat) => pinned.has(chat.id)) : [];
  const rest = pinnedItems.length ? chats.filter((chat) => !pinned?.has(chat.id)) : chats;
  const groups = new Map<string, { group: ChatGroup<T>; latest: number }>();
  for (const chat of rest) {
    const ts = chatTime(chat, time);
    let key: string;
    let group: Omit<ChatGroup<T>, "items">;
    if (by === "project") {
      const projectId = chat.projectId || null;
      key = `project:${projectId ?? NO_PROJECT}`;
      group = { key, projectId };
    } else if (by === "section") {
      const sectionId = sectionOf?.get(chat.id) || null;
      key = `section:${sectionId ?? NO_SECTION}`;
      group = { key, sectionId };
    } else {
      const bucket = dateBucket(ts, now);
      key = dateBucketKey(bucket);
      group = { key, bucket };
    }
    let existing = groups.get(key);
    if (!existing) {
      existing = { group: { ...group, items: [] }, latest: ts };
      groups.set(key, existing);
    }
    existing.latest = Math.max(existing.latest, ts);
    existing.group.items.push(chat);
  }
  const out = [...groups.values()];
  if (by === "date") {
    out.sort((a, b) => (oldestFirst ? a.latest - b.latest : b.latest - a.latest));
  } else {
    // The "none" group goes last.
    const outside = ({ group }: { group: ChatGroup<T> }) =>
      Number(by === "project" ? group.projectId === null : group.sectionId === null);
    out.sort((a, b) => outside(a) - outside(b));
  }
  const grouped = out.map(({ group }) => group);
  return pinnedItems.length
    ? [{ key: "pinned", pinned: true, items: pinnedItems }, ...grouped]
    : grouped;
}

/** Distinct models across the chats, most used first. */
export function modelFacets(
  chats: readonly ChatEntry[],
  models: ReadonlyMap<string, string[]>,
): { model: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const chat of chats) {
    for (const model of new Set(models.get(chat.id) ?? [])) {
      counts.set(model, (counts.get(model) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([model, count]) => ({ model, count }))
    .sort((a, b) => b.count - a.count || a.model.localeCompare(b.model));
}

/** Model ids per chat row; a compare pair is keyed by its pair id. */
export function modelsByChat(
  threads: readonly { id: string; pairId?: string | null; modelId?: string | null }[],
): Map<string, string[]> {
  const out = new Map<string, string[]>();
  for (const thread of threads) {
    if (!thread.modelId) continue;
    const key = thread.pairId || thread.id;
    const list = out.get(key) ?? [];
    if (!list.includes(thread.modelId)) list.push(thread.modelId);
    out.set(key, list);
  }
  return out;
}

export function validateChatsSection(value: unknown): ChatsSection | undefined {
  return CHATS_SECTIONS.find((section) => section === value);
}

/** What a chat holds, for the Contents column. */
export interface ChatContents {
  messages: number;
}

/** Messages along the shown branch: from the newest message back through its parents. */
export function summarizeChatMessages(
  messages: readonly {
    id: string;
    parentId?: string | null;
    role?: string;
    createdAt: number;
  }[],
): ChatContents {
  const byId = new Map(messages.map((message) => [message.id, message]));
  const branched = messages.some((message) => message.parentId);
  let path: (typeof messages)[number][] = [...messages];
  if (branched && messages.length > 0) {
    path = [];
    const seen = new Set<string>();
    let at = messages.reduce((a, b) => (b.createdAt > a.createdAt ? b : a)) as
      | (typeof messages)[number]
      | undefined;
    while (at && !seen.has(at.id)) {
      seen.add(at.id);
      path.push(at);
      at = at.parentId ? byId.get(at.parentId) : undefined;
    }
  }
  return {
    messages: path.filter((message) => message.role === "user" || message.role === "assistant")
      .length,
  };
}

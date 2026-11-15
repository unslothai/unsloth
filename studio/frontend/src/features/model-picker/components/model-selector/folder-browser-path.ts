// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface Crumb {
  label: string;
  value: string;
}

export function splitBreadcrumb(path: string): Crumb[] {
  if (!path) return [];
  // Detect path style before normalizing: `\` is a valid POSIX filename char.
  const isWindowsDrive =
    /^[A-Za-z]:[\\/]/.test(path) || /^[A-Za-z]:$/.test(path);
  const isUnc = /^\\\\/.test(path);
  const isWindows = isWindowsDrive || isUnc;
  const normalized = isWindows ? path.replace(/\\/g, "/") : path;
  const segments = normalized.split("/");
  const parts: Crumb[] = [];

  if (segments[0] === "") {
    parts.push({ label: "/", value: "/" });
    let cur = "";
    for (const seg of segments.slice(1)) {
      if (!seg) continue;
      cur = `${cur}/${seg}`;
      parts.push({ label: seg, value: cur });
    }
    return parts;
  }

  // Use `C:/` so the crumb goes to the drive root (`C:` alone is CWD-on-C).
  if (/^[A-Za-z]:$/.test(segments[0])) {
    const driveRoot = `${segments[0]}/`;
    let cur = driveRoot;
    parts.push({ label: segments[0], value: driveRoot });
    for (const seg of segments.slice(1)) {
      if (!seg) continue;
      cur = cur.endsWith("/") ? `${cur}${seg}` : `${cur}/${seg}`;
      parts.push({ label: seg, value: cur });
    }
    return parts;
  }

  return [{ label: path, value: path }];
}

export function joinChild(current: string, name: string): string {
  return /[\\/]$/.test(current) ? `${current}${name}` : `${current}/${name}`;
}

/** Text that names a place rather than narrowing the list: `~/code`, `/srv`, `C:\x`, `src/app`. */
export function isPathLike(text: string): boolean {
  const value = text.trim();
  return (
    /^(~|\/|\\|[A-Za-z]:)/.test(value) || value.includes("/") || value === ".."
  );
}

/** The absolute or home-relative path typed text names, read against the open folder. */
export function resolveTypedPath(current: string, text: string): string {
  const value = text.trim();
  if (/^(~|\/|\\|[A-Za-z]:)/.test(value)) return value;
  return joinChild(current, value);
}

/** `~/code/uns` -> `~/code` plus `uns`, so a half-typed last name can narrow its parent. */
export function splitTypedPath(
  path: string,
): { parent: string; leaf: string } | null {
  const trimmed = path.replace(/[\\/]+$/, "");
  const cut = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  if (cut < 0) return null;
  const leaf = trimmed.slice(cut + 1);
  if (!leaf) return null;
  const parent = trimmed.slice(0, cut);
  // The root keeps its separator: `/us` narrows `/`, `C:/us` narrows `C:/`.
  if (parent === "" || /^[A-Za-z]:$/.test(parent)) {
    return { parent: trimmed.slice(0, cut + 1), leaf };
  }
  return { parent, leaf };
}

/** Names starting with the query first, then names containing it; case-insensitive. */
export function filterEntries<T extends { name: string }>(
  entries: T[],
  query: string,
): T[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return entries;
  const starts: T[] = [];
  const contains: T[] = [];
  for (const entry of entries) {
    const name = entry.name.toLowerCase();
    if (name.startsWith(needle)) starts.push(entry);
    else if (name.includes(needle)) contains.push(entry);
  }
  return [...starts, ...contains];
}

/** The trail cut to its last `keep` crumbs; `hidden` holds the ones folded behind the ellipsis. */
export function collapseCrumbs(
  crumbs: Crumb[],
  keep = 3,
): { hidden: Crumb[]; shown: Crumb[] } {
  if (crumbs.length <= keep + 1) return { hidden: [], shown: crumbs };
  return {
    hidden: crumbs.slice(0, crumbs.length - keep),
    shown: crumbs.slice(crumbs.length - keep),
  };
}

export function folderName(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const cut = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return trimmed.slice(cut + 1) || path;
}

export const RECENT_FOLDERS_KEY = "unsloth_folder_browser_recent";
const RECENT_FOLDERS_MAX = 5;

type RecentStore = Pick<Storage, "getItem" | "setItem">;

export function readRecentFolders(store: RecentStore | undefined): string[] {
  try {
    const parsed: unknown = JSON.parse(
      store?.getItem(RECENT_FOLDERS_KEY) ?? "[]",
    );
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

/** Most recent first, without repeats. */
export function rememberRecentFolder(
  store: RecentStore | undefined,
  path: string,
): string[] {
  const next = [
    path,
    ...readRecentFolders(store).filter((item) => item !== path),
  ].slice(0, RECENT_FOLDERS_MAX);
  try {
    store?.setItem(RECENT_FOLDERS_KEY, JSON.stringify(next));
  } catch {
    // Private mode or a full quota: the list is a convenience.
  }
  return next;
}

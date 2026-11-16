// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ProjectChangeStatus,
  ProjectFileEntry,
  ProjectFileListing,
} from "../api/chat-api";

export function formatFileSize(bytes: number | null | undefined): string {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) {
    return "";
  }
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const shown =
    value < 10
      ? value.toFixed(1).replace(/\.0$/, "")
      : String(Math.round(value));
  return `${shown} ${units[unit]}`;
}

export function joinProjectPath(parent: string, name: string): string {
  return parent ? `${parent}/${name}` : name;
}

export function splitProjectPath(path: string): { dir: string; name: string } {
  const trimmed = path.replace(/\/+$/, "");
  const slash = trimmed.lastIndexOf("/");
  return slash < 0
    ? { dir: "", name: trimmed }
    : { dir: trimmed.slice(0, slash), name: trimmed.slice(slash + 1) };
}

export function addToChatText(path: string): string {
  return `File: ${path}`;
}

export const CHANGE_STATUS_LABELS: Record<ProjectChangeStatus, string> = {
  modified: "Modified",
  added: "Added",
  deleted: "Deleted",
  renamed: "Renamed",
  untracked: "Untracked",
  conflicted: "Conflicted",
};

export type ChangeTone = "amber" | "green" | "red" | "neutral";

export function changeStatusTone(status: ProjectChangeStatus): ChangeTone {
  switch (status) {
    case "modified":
      return "amber";
    case "added":
    case "untracked":
      return "green";
    case "deleted":
    case "conflicted":
      return "red";
    default:
      return "neutral";
  }
}

export type DirState =
  | { status: "loading" }
  | { status: "ready"; listing: ProjectFileListing }
  | { status: "error"; message: string };

export type FileTreeRow =
  | {
      type: "entry";
      path: string;
      entry: ProjectFileEntry;
      depth: number;
      open: boolean;
    }
  | { type: "loading"; path: string; depth: number }
  | { type: "empty"; path: string; depth: number }
  | { type: "truncated"; path: string; depth: number; count: number }
  | { type: "error"; path: string; depth: number; message: string };

export function toggleOpenPath(
  open: ReadonlySet<string>,
  path: string,
): Set<string> {
  const next = new Set(open);
  if (next.has(path)) next.delete(path);
  else next.add(path);
  return next;
}

export function closeSubtree(
  open: ReadonlySet<string>,
  path: string,
): Set<string> {
  const prefix = `${path}/`;
  return new Set([...open].filter((p) => p !== path && !p.startsWith(prefix)));
}

/** Own keys only: folder names such as `constructor` or `__proto__` must not hit Object.prototype. */
export function dirStateAt(
  dirs: Readonly<Record<string, DirState>>,
  path: string,
): DirState | undefined {
  return Object.hasOwn(dirs, path) ? dirs[path] : undefined;
}

export function pathsToLoad(
  dirs: Readonly<Record<string, DirState>>,
  open: ReadonlySet<string>,
): string[] {
  return ["", ...open].filter((path) => !Object.hasOwn(dirs, path));
}

export function flattenFileTree(
  dirs: Readonly<Record<string, DirState>>,
  open: ReadonlySet<string>,
  path = "",
  depth = 0,
): FileTreeRow[] {
  const state = dirStateAt(dirs, path);
  if (!state || state.status === "loading") {
    return [{ type: "loading", path, depth }];
  }
  if (state.status === "error") {
    return [{ type: "error", path, depth, message: state.message }];
  }
  const { entries, truncated } = state.listing;
  if (entries.length === 0) return [{ type: "empty", path, depth }];
  const rows: FileTreeRow[] = [];
  for (const entry of entries) {
    const child = joinProjectPath(path, entry.name);
    const isOpen = entry.kind === "dir" && open.has(child);
    rows.push({ type: "entry", path: child, entry, depth, open: isOpen });
    if (isOpen) rows.push(...flattenFileTree(dirs, open, child, depth + 1));
  }
  if (truncated) {
    rows.push({ type: "truncated", path, depth, count: entries.length });
  }
  return rows;
}

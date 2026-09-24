// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState } from "react";
import { type LibraryDisk, type LibraryItem, getLibrary } from "./api";
import { fileKind } from "./file-kind";
import type { LibrarySearch } from "./search";
import { includedBySettings, useLibrarySettingsStore } from "./settings-store";

export type StorageCategory = "files" | "images" | "videos" | "audio" | "fineTunes";

export interface StorageUsage {
  category: StorageCategory;
  /** Where Manage storage opens it, sorted by size. */
  link: LibrarySearch;
  bytes: number;
  count: number;
}

export interface LibraryStorage {
  status: "loading" | "ready" | "error";
  totalBytes: number;
  /** The share of totalBytes on `disk`, which the bar draws. */
  diskBytes: number;
  categories: StorageUsage[];
  disk: LibraryDisk | null;
}

const CATEGORY_LINKS: [StorageCategory, LibrarySearch][] = [
  ["files", { show: "all", filter: "files", sort: "size" }],
  ["images", { show: "images", sort: "size" }],
  ["videos", { show: "videos", sort: "size" }],
  ["audio", { show: "audio", sort: "size" }],
  ["fineTunes", { show: "models", sort: "size" }],
];

const KIND_CATEGORIES: Partial<Record<string, StorageCategory>> = {
  image: "images",
  video: "videos",
  audio: "audio",
  model: "fineTunes",
};

/** The id prefix `disk.sources` names: the source, and for a model where it came from. */
function diskSource(id: string): string {
  const [source, origin] = id.split(":", 2);
  return source === "model" ? `${source}:${origin}` : source!;
}

const STALE_EVENT = "unsloth:library-storage-stale";

/** Measure again wherever storage is on screen, after files left outside the Library (a chat clear). */
export function refreshLibraryStorage(): void {
  window.dispatchEvent(new Event(STALE_EVENT));
}

/**
 * What the Library holds on disk, by category. Sources hidden in Content settings are left out,
 * so each category link lands on exactly what it counted. Empty categories are left out too.
 */
export function useLibraryStorage(): LibraryStorage {
  const settings = useLibrarySettingsStore();
  const [snapshot, setSnapshot] = useState<{
    status: LibraryStorage["status"];
    items: LibraryItem[];
    disk: LibraryDisk | null;
  }>({ status: "loading", items: [], disk: null });
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const stale = () => setVersion((current) => current + 1);
    window.addEventListener(STALE_EVENT, stale);
    return () => window.removeEventListener(STALE_EVENT, stale);
  }, []);
  useEffect(() => {
    let cancelled = false;
    getLibrary().then(
      ({ items, disk }) => !cancelled && setSnapshot({ status: "ready", items, disk: disk ?? null }),
      () => !cancelled && setSnapshot((current) => ({ ...current, status: "error" })),
    );
    return () => {
      cancelled = true;
    };
  }, [version]);
  return useMemo(() => {
    const totals = new Map<StorageCategory, { bytes: number; count: number }>();
    const onDisk = snapshot.disk?.sources ? new Set(snapshot.disk.sources) : null;
    let diskBytes = 0;
    for (const item of snapshot.items) {
      if (!includedBySettings(item.id, settings)) continue;
      if (!onDisk || onDisk.has(diskSource(item.id))) {
        diskBytes += item.sizeBytes ?? 0;
      }
      const category = KIND_CATEGORIES[fileKind(item)] ?? "files";
      const total = totals.get(category) ?? { bytes: 0, count: 0 };
      total.bytes += item.sizeBytes ?? 0;
      total.count += 1;
      totals.set(category, total);
    }
    const categories = CATEGORY_LINKS.flatMap(([category, link]) => {
      const total = totals.get(category);
      return total ? [{ category, link, ...total }] : [];
    }).sort((a, b) => b.bytes - a.bytes);
    return {
      status: snapshot.status,
      totalBytes: categories.reduce((sum, entry) => sum + entry.bytes, 0),
      diskBytes,
      categories,
      disk: snapshot.disk,
    };
  }, [snapshot, settings]);
}

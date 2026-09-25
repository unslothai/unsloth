// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import { useEffect, useMemo, useState } from "react";
import { type LibraryDisk, type LibraryItem, type LibraryLocation, getLibrary } from "./api";
import { fileKind } from "./file-kind";
import type { LibrarySearch } from "./search";
import { includedBySettings, useLibrarySettingsStore } from "./settings-store";

type StorageCategory = "files" | "images" | "videos" | "audio" | "fineTunes";

/** Settings names for storage categories and Library locations. */
export const STORAGE_LABELS: Record<StorageCategory | LibraryLocation["key"], TranslationKey> = {
  files: "settings.data.filesSection",
  uploads: "settings.library.locationUploads",
  images: "settings.library.categoryImages",
  videos: "settings.library.categoryVideos",
  audio: "library.tabs.audio",
  fineTunes: "settings.library.categoryFineTunes",
  exports: "settings.library.locationExports",
};

interface StorageUsage {
  category: StorageCategory;
  /** Where Manage storage opens it, sorted by size. */
  link: LibrarySearch;
  bytes: number;
  count: number;
}

interface LibraryStorage {
  status: "loading" | "ready" | "error";
  /** Everything on disk, hidden sources included. */
  totalBytes: number;
  /** The part of totalBytes that Content settings hide, so no category counts it. */
  hiddenBytes: number;
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

/** What the Library holds on disk, by category, largest first. Hidden sources count toward the
 *  total and the bar but no category, so each link lands on exactly what it counted. */
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
    let hiddenBytes = 0;
    for (const item of snapshot.items) {
      if (!onDisk || onDisk.has(diskSource(item.id))) {
        diskBytes += item.sizeBytes ?? 0;
      }
      if (!includedBySettings(item.id, settings)) {
        hiddenBytes += item.sizeBytes ?? 0;
        continue;
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
      totalBytes: categories.reduce((sum, entry) => sum + entry.bytes, hiddenBytes),
      hiddenBytes,
      diskBytes,
      categories,
      disk: snapshot.disk,
    };
  }, [snapshot, settings]);
}

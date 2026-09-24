// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { getLibrary } from "./api";
import { fileKind } from "./file-kind";
import type { LibraryTab } from "./search";

export type StorageCategory = "files" | "images" | "videos" | "audio" | "fineTunes";

export interface StorageUsage {
  category: StorageCategory;
  /** Where Manage storage opens it, sorted by size. */
  tab: LibraryTab;
  bytes: number;
  count: number;
}

export interface LibraryStorage {
  status: "loading" | "ready" | "error";
  totalBytes: number;
  categories: StorageUsage[];
}

const CATEGORY_TABS: [StorageCategory, LibraryTab][] = [
  ["files", "all"],
  ["images", "images"],
  ["videos", "videos"],
  ["audio", "audio"],
  ["fineTunes", "models"],
];

const KIND_CATEGORIES: Partial<Record<string, StorageCategory>> = {
  image: "images",
  video: "videos",
  audio: "audio",
  model: "fineTunes",
};

/** What the Library holds on disk, by category. Empty categories are left out. */
export function useLibraryStorage(): LibraryStorage {
  const [storage, setStorage] = useState<LibraryStorage>({
    status: "loading",
    totalBytes: 0,
    categories: [],
  });
  useEffect(() => {
    let cancelled = false;
    getLibrary().then(
      ({ items }) => {
        const totals = new Map<StorageCategory, { bytes: number; count: number }>();
        for (const item of items) {
          const category = KIND_CATEGORIES[fileKind(item)] ?? "files";
          const total = totals.get(category) ?? { bytes: 0, count: 0 };
          total.bytes += item.sizeBytes ?? 0;
          total.count += 1;
          totals.set(category, total);
        }
        const categories = CATEGORY_TABS.flatMap(([category, tab]) => {
          const total = totals.get(category);
          return total ? [{ category, tab, ...total }] : [];
        });
        if (!cancelled) {
          setStorage({
            status: "ready",
            totalBytes: categories.reduce((sum, entry) => sum + entry.bytes, 0),
            categories,
          });
        }
      },
      () => !cancelled && setStorage((current) => ({ ...current, status: "error" })),
    );
    return () => {
      cancelled = true;
    };
  }, []);
  return storage;
}

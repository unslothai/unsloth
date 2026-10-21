// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLocale, useT } from "@/i18n";
import type { LibraryDisk } from "../api";
import { formatSize } from "../format";

export function LibraryStorageBar({
  libraryBytes,
  disk,
}: {
  libraryBytes: number;
  disk: LibraryDisk | null;
}) {
  const t = useT();
  const locale = useLocale();
  if (!disk || disk.totalBytes <= 0) return null;
  const otherBytes = Math.max(0, disk.totalBytes - disk.freeBytes - libraryBytes);
  const share = (bytes: number) => `${(bytes / disk.totalBytes) * 100}%`;
  return (
    <>
      <div className="flex h-2 overflow-hidden rounded-full bg-muted">
        <div className="min-w-1 bg-foreground" style={{ width: share(libraryBytes) }} />
        <div className="bg-muted-foreground/35" style={{ width: share(otherBytes) }} />
      </div>
      <p className="text-xs text-muted-foreground">
        {t("settings.library.storageDisk", {
          free: formatSize(disk.freeBytes, locale, t) ?? "",
          total: formatSize(disk.totalBytes, locale, t) ?? "",
        })}
      </p>
    </>
  );
}

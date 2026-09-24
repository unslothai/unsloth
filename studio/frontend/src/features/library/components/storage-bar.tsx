// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSystemInfo } from "@/hooks";
import { useT } from "@/i18n";
import { formatSize } from "../format";

const GB = 1e9;

/** The Library, then everything else on its disk, then free space. Nothing when the disk is unknown. */
export function LibraryStorageBar({ libraryBytes }: { libraryBytes: number }) {
  const t = useT();
  const { disk } = useSystemInfo();
  const diskBytes = disk.total_gb > 0 ? disk.total_gb * GB : 0;
  if (!diskBytes) return null;
  const otherBytes = Math.max(0, diskBytes - disk.free_gb * GB - libraryBytes);
  const share = (bytes: number) => `${(bytes / diskBytes) * 100}%`;
  return (
    <>
      {/* Only the track is rounded, so the segments join flush. */}
      <div className="flex h-2 overflow-hidden rounded-full bg-muted">
        <div className="min-w-1 bg-foreground" style={{ width: share(libraryBytes) }} />
        <div className="bg-muted-foreground/35" style={{ width: share(otherBytes) }} />
      </div>
      <p className="text-xs text-muted-foreground">
        {t("settings.library.storageDisk", {
          free: formatSize(disk.free_gb * GB) ?? "",
          total: formatSize(diskBytes) ?? "",
        })}
      </p>
    </>
  );
}

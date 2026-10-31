// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect } from "react";
import { checkDiskSpace, setLowDiskNotifier } from "../low-disk-check";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Decimal GB like /api/system; the copy interpolates the unit bare. */
function formatGb(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "?";
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} GB`;
}

/** Owner only: "resources" is in OWNER_ONLY_SETTINGS_TABS, so others could not act on it. */
export function useLowDiskNotice(): void {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const openDialog = useSettingsDialogStore((state) => state.openDialog);

  useEffect(() => {
    if (!isOwner) return;
    setLowDiskNotifier((level, disk) => {
      toast.warning(
        level === "critical"
          ? t("settings.resources.storage.lowDisk.criticalTitle")
          : t("settings.resources.storage.lowDisk.title"),
        {
          description: t("settings.resources.storage.lowDisk.description", {
            free: formatGb(disk.free_gb),
            total: formatGb(disk.total_gb),
          }),
          action: {
            label: t("settings.resources.storage.lowDisk.action"),
            onClick: () =>
              openDialog("resources", { scrollTarget: "resources-caches" }),
          },
        },
      );
    });
    // force: a disk already full before Studio started is exactly who this is for.
    void checkDiskSpace({ force: true });
    return () => setLowDiskNotifier(null);
  }, [isOwner, openDialog, t]);
}

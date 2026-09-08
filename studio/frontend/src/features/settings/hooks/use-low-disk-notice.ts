// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { subscribeSystemInfo } from "@/hooks/use-system";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect } from "react";
import { observeDiskPressure } from "../low-disk";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Disk figures come from /api/system in decimal GB, so format them the same way. */
function formatGb(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "?";
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

/**
 * Warn once when free disk crosses a low or critical threshold, and offer the
 * caches as the thing to trim.
 *
 * Mounted at the app level rather than inside Settings. The first version lived
 * in the resources tab and hung off the poll that tab already ran, which meant
 * the warning only ever appeared to someone who had already opened Settings and
 * navigated to the one screen that shows the disk figure. That is the audience
 * least in need of telling.
 *
 * It still adds no polling: subscribeSystemInfo attaches to the single system
 * poll the app already runs, and observeDiskPressure holds the notified level in
 * module scope for the browser session, so a threshold is announced when it is
 * CROSSED, not once per poll while the disk sits below it.
 */
export function useLowDiskNotice(): void {
  const t = useT();
  const openDialog = useSettingsDialogStore((state) => state.openDialog);

  useEffect(() => {
    return subscribeSystemInfo((info) => {
      if (!info.disk) return;
      const level = observeDiskPressure(info.disk);
      if (level === null) return;
      toast.warning(
        level === "critical"
          ? t("settings.resources.storage.lowDisk.criticalTitle")
          : t("settings.resources.storage.lowDisk.title"),
        {
          description: t("settings.resources.storage.lowDisk.description", {
            free: formatGb(info.disk.free_gb),
            total: formatGb(info.disk.total_gb),
          }),
          action: {
            label: t("settings.resources.storage.lowDisk.action"),
            onClick: () =>
              openDialog("resources", { scrollTarget: "resources-caches" }),
          },
        },
      );
    });
  }, [openDialog, t]);
}

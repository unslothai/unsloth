// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect } from "react";
import { checkDiskSpace, setLowDiskNotifier } from "../low-disk-check";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Decimal GB, as the disk route reports it and as the Resources tab renders it.
 * The unit belongs in the value: the description interpolates it bare. */
function formatGb(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "?";
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} GB`;
}

/**
 * Warn when free disk crosses a low or critical threshold, and offer the caches as the thing to
 * trim.
 *
 * This hook owns the WORDING, not the schedule. It registers a notifier with low-disk-check and
 * takes one reading when the app shell mounts; every other reading is taken by the download
 * path, where the disk is about to be written to.
 *
 * There is deliberately no interval. An earlier version polled /api/system every 60 s from the
 * app shell, which runs GPU enumeration, package-metadata reads and a CPU sample forever in
 * every open tab, to answer a question whose answer only changes when something writes to the
 * disk. /api/system/disk is one syscall and is asked for on demand instead.
 *
 * observeDiskPressure holds the notified level in module scope for the browser session, so a
 * threshold is announced when it is CROSSED, not on every reading while the disk sits below it.
 *
 * Owner only. The shell mounts this for every authenticated user, but "resources" is in
 * OWNER_ONLY_SETTINGS_TABS, so resolveSettingsTab sends a managed account to General: the
 * action would be a dead end, and the caches it names are install-wide state that account
 * cannot clear anyway. A warning nobody can act on is worse than no warning, so the notice
 * is not registered at all rather than shown with its action removed.
 */
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
    // force: the first reading of the session has no earlier one to collapse against, and a
    // disk that was already full before Studio started is exactly who this is for.
    void checkDiskSpace({ force: true });
    return () => setLowDiskNotifier(null);
  }, [isOwner, openDialog, t]);
}

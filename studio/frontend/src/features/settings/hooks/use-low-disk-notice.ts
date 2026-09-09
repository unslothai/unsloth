// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSystemInfo } from "@/hooks/use-system";
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

/** Slow on purpose: a disk fills over hours. */
const LOW_DISK_POLL_MS = 60_000;

/**
 * Warn once when free disk crosses a low or critical threshold, and offer the
 * caches as the thing to trim.
 *
 * Mounted in the app shell, and owning a real poll, because both of those are
 * what reach the person who has not opened Settings: subscribeSystemInfo only
 * adds a callback to a Set, and its only two callers are the floating monitor
 * and the resources tab, both gated on being open. observeDiskPressure holds
 * the notified level for the browser session, so a threshold is announced when
 * it is CROSSED rather than once per poll.
 */
export function useLowDiskNotice(): void {
  const t = useT();
  const openDialog = useSettingsDialogStore((state) => state.openDialog);
  const systemInfo = useSystemInfo({ pollMs: LOW_DISK_POLL_MS });

  useEffect(() => {
    if (systemInfo.status !== "ready" || !systemInfo.disk) return;
    const level = observeDiskPressure(systemInfo.disk);
    if (level === null) return;
    toast.warning(
      level === "critical"
        ? t("settings.resources.storage.lowDisk.criticalTitle")
        : t("settings.resources.storage.lowDisk.title"),
      {
        description: t("settings.resources.storage.lowDisk.description", {
          free: formatGb(systemInfo.disk.free_gb),
          total: formatGb(systemInfo.disk.total_gb),
        }),
        action: {
          label: t("settings.resources.storage.lowDisk.action"),
          onClick: () =>
            openDialog("resources", { scrollTarget: "resources-caches" }),
        },
      },
    );
  }, [openDialog, systemInfo, t]);
}

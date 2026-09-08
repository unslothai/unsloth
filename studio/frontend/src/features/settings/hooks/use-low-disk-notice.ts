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

/**
 * How often the notice asks for a disk reading. Slow on purpose: a disk fills
 * over hours, and the two screens that show live system figures poll at 3 s and
 * 5 s only while they are open.
 */
const LOW_DISK_POLL_MS = 60_000;

/**
 * Warn once when free disk crosses a low or critical threshold, and offer the
 * caches as the thing to trim.
 *
 * Mounted in the app shell, not in Settings. Two earlier placements were wrong
 * and both failed the same way, by warning only people who were already looking
 * at the disk figure:
 *
 *   1. In the resources tab, where it hung off that tab's own poll.
 *   2. In StudioPage, which is the /studio training route rather than a shell,
 *      so anyone sitting in Chat, Images or Audio never subscribed at all.
 *
 * It also has to do its own fetching, which the second version did not.
 * subscribeSystemInfo only adds a callback to a Set: it never requests
 * /api/system and never replays the cached reading to a new subscriber, so a
 * subscriber alone is notified only if some other caller happens to make an
 * uncached request. The only two callers are the floating monitor and the
 * resources tab, both lazily mounted and both gated on being open, which left
 * this notice silent for exactly the user it exists for. useSystemInfo owns a
 * real interval, so the reading arrives whether or not anything else is on
 * screen.
 *
 * observeDiskPressure holds the notified level in module scope for the browser
 * session, so a threshold is announced when it is CROSSED, not once per poll
 * while the disk sits below it.
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

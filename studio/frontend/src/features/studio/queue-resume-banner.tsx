// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  useTrainingQueueActions,
  useTrainingQueueStore,
} from "@/features/training";
import { AlertCircleIcon, PlayCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { useT } from "@/i18n";

export function QueueResumeBanner(): ReactElement | null {
  const t = useT();
  const paused = useTrainingQueueStore((s) => s.paused);
  const pausedReason = useTrainingQueueStore((s) => s.pausedReason);
  const pendingCount = useTrainingQueueStore((s) => s.pendingCount);
  const dismissed = useTrainingQueueStore((s) => s.restartBannerDismissed);
  const dismiss = useTrainingQueueStore((s) => s.dismissRestartBanner);
  const { resume } = useTrainingQueueActions();

  if (!paused || pausedReason !== "restart" || pendingCount === 0 || dismissed) {
    return null;
  }

  return (
    <div className="mb-4 flex flex-wrap items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3">
      <HugeiconsIcon
        icon={AlertCircleIcon}
        className="size-4 shrink-0 text-amber-600 dark:text-amber-400"
      />
      <p className="min-w-0 flex-1 text-sm text-amber-800 dark:text-amber-300">
        {t("studio.training.queue.restartBanner", { count: String(pendingCount) })}
      </p>
      <div className="flex shrink-0 items-center gap-2">
        <Button size="sm" onClick={() => void resume()}>
          <HugeiconsIcon icon={PlayCircleIcon} className="size-4" />
          {t("studio.training.queue.resume")}
        </Button>
        <Button variant="ghost" size="sm" onClick={dismiss}>
          {t("studio.training.queue.dismiss")}
        </Button>
      </div>
    </div>
  );
}

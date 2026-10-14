// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { staleDescription, useReembedStore } from "@/features/rag";
import { useT } from "@/i18n";
import { type ReactElement, useEffect, useState } from "react";
import { SettingsRow } from "./settings-row";

/** Offers to re-embed documents indexed under a previous embedding model; hidden when none are. */
export function ReembedDocumentsRow({
  needsDownload = false,
}: {
  needsDownload?: boolean;
}): ReactElement | null {
  const t = useT();
  const status = useReembedStore((s) => s.status);
  const [stoppingJobId, setStoppingJobId] = useState<string | null>(null);

  useEffect(() => {
    void useReembedStore.getState().load();
  }, []);

  const job = status?.job ?? null;
  const running = job?.status === "running";

  const stale = status ? staleDescription(status) : null;
  if (!running && !stale) return null;

  return (
    <SettingsRow
      label={t("settings.general.rag.reembed.label")}
      description={
        running && job ? (
          <span className="flex flex-col gap-1.5">
            <span>
              {t("settings.general.rag.reembed.running", {
                done: job.done + job.failed,
                total: job.done + job.failed + job.remaining,
              })}
            </span>
            <Progress
              value={Math.round(job.progress * 100)}
              className="h-1.5 w-[calc(160px*var(--ui-space-scale,1))]"
            />
          </span>
        ) : (
          stale
        )
      }
    >
      {running ? (
        <Button
          variant="outline"
          size="sm"
          disabled={stoppingJobId === job?.id}
          onClick={() => {
            setStoppingJobId(job?.id ?? null);
            void useReembedStore
              .getState()
              .cancel()
              .then((sent) => sent || setStoppingJobId(null));
          }}
        >
          {t("settings.general.rag.reembed.stop")}
        </Button>
      ) : (
        <Button
          variant="outline"
          size="sm"
          disabled={needsDownload}
          onClick={() => void useReembedStore.getState().start()}
        >
          {t("settings.general.rag.reembed.action")}
        </Button>
      )}
    </SettingsRow>
  );
}

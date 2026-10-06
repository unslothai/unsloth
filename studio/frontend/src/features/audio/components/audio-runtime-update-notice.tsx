// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useIsAccountOwner } from "@/features/auth";
import { useLlamaUpdateCheck } from "@/hooks/use-llama-update-check";
import { toast } from "@/lib/toast";
import { Alert02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useRef, useState } from "react";
import { audioRuntimeNoticeMode } from "../audio-page-policy";

/** The managed audio runtime is not the release this Studio pins. The owner updates it in place
 *  through the shared update job, which also installs any pending llama.cpp or whisper.cpp update;
 *  the app-wide update card follows that job and resyncs the loaded model. */
export function AudioRuntimeUpdateNotice({
  update,
  onUpdated,
}: {
  update: { installed: string; expected: string };
  onUpdated: () => Promise<unknown>;
}): ReactElement {
  const isOwner = useIsAccountOwner();
  const { status, applying, apply } = useLlamaUpdateCheck({
    enabled: isOwner,
  });
  const offered = Boolean(status?.audio?.update_available);
  // Refresh the page once a job this notice followed settles (the card can run it too), or when
  // the offer disappears without one (another tab updated): a stale notice would send the owner
  // to the CLI. The offer is gone as soon as the job lands, so stay on "updating" until the page
  // has re-read the runtime.
  const followedJob = useRef(false);
  const wasOffered = useRef(false);
  const [refreshing, setRefreshing] = useState(false);
  useEffect(() => {
    if (applying) {
      followedJob.current = true;
      return;
    }
    const offerWithdrawn = wasOffered.current && !offered;
    wasOffered.current = offered;
    if (followedJob.current || offerWithdrawn) {
      followedJob.current = false;
      setRefreshing(true);
      void onUpdated().finally(() => setRefreshing(false));
    }
  }, [applying, offered, onUpdated]);
  const mode = audioRuntimeNoticeMode({
    isOwner,
    offered,
    applying: applying || refreshing,
    checked: status !== null,
  });

  async function handleUpdate() {
    const result = await apply();
    if (result.ok) {
      // The job says what each phase did, including a release lookup that kept the old tree.
      toast.success(
        result.message?.trim() || `audio.cpp updated to ${update.expected}.`,
      );
    } else {
      toast.error(
        `audio.cpp update failed: ${result.error ?? "unknown error"}`,
      );
    }
  }

  return (
    <div
      role="status"
      className="mt-1 flex items-start gap-1.5 text-ui-12 leading-snug text-foreground"
    >
      <HugeiconsIcon
        icon={Alert02Icon}
        className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
      />
      <div className="grid justify-items-start gap-1.5">
        <span>
          Your audio runtime is{" "}
          <span className="whitespace-nowrap">{update.installed}</span>; Unsloth
          expects <span className="whitespace-nowrap">{update.expected}</span>.{" "}
          {mode === "ask_owner" ? (
            <>
              Some models may not work until it is updated. Ask the Unsloth
              Studio owner to update it.
            </>
          ) : mode === "updating" || mode === "checking" ? null : (
            <>Some models may not work until you update. </>
          )}
          {mode === "cli" ? (
            <>
              Stop Unsloth Studio, run{" "}
              <code className="font-mono">unsloth studio update</code>, then
              start it again.
            </>
          ) : null}
        </span>
        {mode === "update" || mode === "updating" ? (
          <Button
            size="xs"
            onClick={handleUpdate}
            disabled={mode === "updating"}
          >
            {mode === "updating" ? "Updating..." : "Update"}
          </Button>
        ) : null}
      </div>
    </div>
  );
}

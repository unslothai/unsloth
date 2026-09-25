// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

import { ModelLoadDescription } from "@/features/chat";
import { formatBytes, type StagedDownloadProgress } from "@/features/hub";
import {
  createPickToast,
  type PickToast,
  type PickToastPhase,
  type PickToastProgress,
} from "@/lib/diffusion-pick-toast";

// Match the load toast's styling for a seamless handoff.
const PICK_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
};

function describePick(phase: PickToastPhase, progress: PickToastProgress | null) {
  if (phase === "downloading") {
    return (
      <ModelLoadDescription
        title="Downloading model…"
        message="The model loads once its files are downloaded."
        progressPercent={
          progress ? (progress.downloadedBytes / progress.totalBytes) * 100 : null
        }
        progressLabel={
          progress
            ? `${formatBytes(progress.downloadedBytes)} of ${formatBytes(progress.totalBytes)}`
            : null
        }
      />
    );
  }
  if (phase === "queued") {
    return (
      <ModelLoadDescription
        title="Download queued"
        message="It starts once the current download finishes."
      />
    );
  }
  if (phase === "waiting") {
    return (
      <ModelLoadDescription
        title="Model downloaded"
        message="It loads once the other downloads finish."
      />
    );
  }
  if (phase === "ready") {
    return (
      <ModelLoadDescription
        title="Model downloaded"
        message="It loads when you return to this page."
      />
    );
  }
  return (
    <ModelLoadDescription
      title="Preparing download…"
      message="Checking which files this model needs."
    />
  );
}

/** One pick toast per page, dropped with the page. */
export function useDiffusionPickToast(): PickToast {
  const [pickToast] = useState(() =>
    createPickToast({ describe: describePick, classNames: PICK_TOAST_CLASSNAMES }),
  );
  useEffect(() => () => pickToast.dismissAll(), [pickToast]);
  return pickToast;
}

/** Bind progress separately because staging is initialized after the pick toast. */
export function usePickToastProgress(
  pickToast: PickToast,
  progress: StagedDownloadProgress | null,
): void {
  useEffect(() => {
    pickToast.progress(progress);
  }, [pickToast, progress]);
}

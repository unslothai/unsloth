// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DownloadStartOutcome } from "./transport-conflict";

export type StagedDownloadFeedback = {
  tone: "error" | "info";
  title: string;
  description: string;
};

export function stagedDownloadStartFeedback(
  outcome: DownloadStartOutcome,
  error?: string | null,
): StagedDownloadFeedback | null {
  if (outcome === "error") {
    return {
      tone: "error",
      title: "Could not start the download",
      description:
        error || "Check the connection, then select the model again.",
    };
  }
  if (outcome === "conflict") {
    return {
      tone: "info",
      title: "Resume this download from Models",
      description:
        "An earlier partial download used a different transport. Open the Model hub tab to resume or restart it.",
    };
  }
  if (outcome === "busy") {
    return {
      tone: "info",
      title: "Download already in progress",
      description:
        "Reselect this model once the running download finishes to load it.",
    };
  }
  return null;
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";

export function showForkCreatedToast(
  containerSnapshotWarning: string | null,
  title?: string,
): void {
  if (containerSnapshotWarning) {
    toast.info("Fork created", { description: containerSnapshotWarning });
  } else {
    // An untitled fork names itself on its first reply, so there is nothing to show yet.
    const description = title && title !== "New Chat" ? title : undefined;
    toast.success("Fork created", { description });
  }
}

/** One wording for every fork surface: a chat still generating is a refusal, not a failure. */
export function showForkFailedToast(
  error: unknown,
  failedTitle = "Failed to fork",
): void {
  const message = error instanceof Error ? error.message : "";
  const tagged = error as {
    unslothForkRefused?: boolean;
    status?: unknown;
  } | null;
  if (tagged?.unslothForkRefused || message.includes("still generating")) {
    toast.info(
      message || "This chat is still generating. Fork it once it finishes.",
    );
    return;
  }
  if (tagged?.status === 404) {
    toast.error(failedTitle, {
      description: message.startsWith("Message ")
        ? "This message is no longer available to fork."
        : "This chat is no longer available to fork.",
    });
    return;
  }
  toast.error(failedTitle, { description: message || undefined });
}

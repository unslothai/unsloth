// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";

export async function downloadTranscript(
  text: string,
  title: string,
): Promise<boolean> {
  try {
    await downloadFile(
      text,
      `${
        title
          .replace(/\.[^.]+$/, "")
          .replace(/[<>:"/\\|?*]/g, "_")
          .replace(/\p{Cc}/gu, "_") || "transcript"
      }.txt`,
      "text/plain;charset=utf-8",
    );
    return true;
  } catch (error) {
    if (!isDownloadCancelled(error))
      toast.error(
        error instanceof Error
          ? error.message
          : "Could not download transcript.",
      );
    return false;
  }
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { exportFileName } from "./transcript-export";

/** false when it failed or the user cancelled the save dialog. */
export async function downloadTranscriptFile(
  text: string,
  title: string,
  ext: string,
  mime: string,
): Promise<boolean> {
  try {
    await downloadFile(text, exportFileName(title, ext), mime);
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

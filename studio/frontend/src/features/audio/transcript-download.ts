// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import {
  type TranscriptExport,
  type TranscriptExportFormat,
  exportFileName,
  exportTranscript,
} from "./transcript-export";

export async function downloadTranscript(
  format: TranscriptExportFormat,
  input: TranscriptExport,
): Promise<boolean> {
  const { content: text, mime } = exportTranscript(format, input);
  try {
    await downloadFile(text, exportFileName(input.title, format), mime);
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

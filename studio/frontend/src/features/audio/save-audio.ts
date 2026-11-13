// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import {
  downloadBlobStreaming,
  downloadFile,
  downloadUrl,
  isDownloadCancelled,
} from "@/lib/native-files";
import { toast } from "@/lib/toast";

/** desktop reloads blob: URLs because the page CSP blocks fetch() and the native save dialog needs bytes. */
export async function saveAudio(
  filename: string,
  url: string | null,
  load: (() => Promise<Blob>) | null,
): Promise<void> {
  try {
    let savedFileName: string | undefined;
    if (load && (!url || (isTauri && url.startsWith("blob:")))) {
      const blob = await load();
      savedFileName = isTauri
        ? await downloadBlobStreaming(blob, filename)
        : await downloadFile(blob, filename, blob.type || "audio/wav");
    } else if (url) {
      savedFileName = await downloadUrl(url, filename);
    }
    if (savedFileName !== undefined) toast.success("Audio saved", { description: savedFileName });
  } catch (error) {
    if (isDownloadCancelled(error)) return;
    toast.error("Could not save audio", {
      description: error instanceof Error ? error.message : undefined,
    });
  }
}

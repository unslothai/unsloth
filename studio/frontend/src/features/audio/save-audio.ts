// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import {
  downloadFile,
  downloadUrl,
  isDownloadCancelled,
} from "@/lib/native-files";
import { toast } from "@/lib/toast";

/** Desktop saves through its dialog (it drops a plain anchor download); the web downloads as before.
 *  `url` is audio already on the page (blob: or data:), `load` reads the bytes again. The dialog needs
 *  the bytes and the page CSP refuses fetch() on blob: URLs, so on desktop a blob: URL is loaded again. */
export async function saveAudio(
  filename: string,
  url: string | null,
  load: (() => Promise<Blob>) | null,
): Promise<void> {
  try {
    if (load && (!url || (isTauri && url.startsWith("blob:")))) {
      const blob = await load();
      await downloadFile(blob, filename, blob.type || "audio/wav");
    } else if (url) {
      await downloadUrl(url, filename);
    }
    if (isTauri) toast.success("Audio saved", { description: filename });
  } catch (error) {
    if (isDownloadCancelled(error)) return;
    toast.error("Could not save audio", {
      description: error instanceof Error ? error.message : undefined,
    });
  }
}

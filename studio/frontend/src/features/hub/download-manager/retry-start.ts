// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import type { ManagedDownload } from "./download-manager-types";
import { retryRequestFor } from "./retry-download";
import { requestStart, startRefusedAsBusy } from "./transport-conflict";

/** No dismiss first: a refused start must leave the failed row in place. */
export async function retryDownload(job: ManagedDownload): Promise<void> {
  const request = retryRequestFor(job);
  if (startRefusedAsBusy(request)) {
    toast.info("Download already in progress", {
      description:
        "Another download of this model is running. Retry once it finishes.",
    });
    return;
  }
  // The panel has no conflict dialog; the model's Hub card shows the stored one.
  if ((await requestStart(request)) === "conflict") {
    toast.info("Resume this download from Models", {
      description:
        "An earlier partial download used a different transport. Open this model in the Model hub to resume or restart it.",
    });
  }
}

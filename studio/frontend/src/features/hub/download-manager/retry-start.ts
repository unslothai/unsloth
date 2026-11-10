// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import type { ManagedDownload } from "./download-manager-types";
import { retryRequestFor } from "./retry-download";
import { requestStart } from "./transport-conflict";

/** No dismiss first: a refused start must leave the failed row in place. */
export async function retryDownload(job: ManagedDownload): Promise<void> {
  // The panel has no conflict dialog; the model's Hub card shows the stored one.
  if ((await requestStart(retryRequestFor(job))) === "conflict") {
    toast.info("Resume this download from Models", {
      description:
        "An earlier partial download used a different transport. Open this model in the Model hub to resume or restart it.",
    });
  }
}

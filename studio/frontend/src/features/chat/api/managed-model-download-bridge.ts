// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  downloadManager,
  jobKeyOf,
  subscribeJobListeners,
} from "@/features/hub";
import {
  type ManagedModelDownloadRequest,
  type ManagedModelDownloadResult,
  coordinateManagedModelDownload,
} from "./managed-model-download";

export function downloadModelWithManager(
  request: ManagedModelDownloadRequest,
  signal: AbortSignal,
): Promise<ManagedModelDownloadResult> {
  return coordinateManagedModelDownload(request, signal, {
    requestStart: downloadManager.requestStart,
    cancel: downloadManager.cancel,
    subscribe: subscribeJobListeners,
    jobKey: jobKeyOf,
  });
}
